// ontos_view: thin Phase 3 stream consumer (JoltViewer pattern). Plays back an
// ontos v2 gravity stream and renders the bodies as instanced billboarded
// quads plus a faint 2x2 region grid. Zero coupling to the meshlet/.vgeo core.

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

#include <vulkan/vulkan.h>
#if defined(__APPLE__) && __has_include(<vulkan/vulkan_metal.h>)
#include <vulkan/vulkan_metal.h>
#endif
#include <GLFW/glfw3.h>
#if defined(__APPLE__)
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3native.h>
#endif
#include <shaderc/shaderc.hpp>

#if defined(__APPLE__)
#import <Cocoa/Cocoa.h>
#import <QuartzCore/CAMetalLayer.h>
#include <AudioToolbox/AudioToolbox.h>
#endif

#if defined(__linux__)
#include <alsa/asoundlib.h>
#include <atomic>
#include <thread>
#endif

// Realtime contact audio exists on macOS (CoreAudio AudioQueue) and Linux
// (ALSA); other platforms compile the viewer silent.
#if defined(__APPLE__) || defined(__linux__)
#define ONTOS_VIEW_REALTIME_AUDIO 1
#else
#define ONTOS_VIEW_REALTIME_AUDIO 0
#endif

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using f64 = double;

namespace {

struct StreamBody {
    u32 id = 0;
    u8 region = 0;
    u8 level = 0;
    f64 x = 0, y = 0, vx = 0, vy = 0, mass = 0;
};

struct StreamContact {
    u64 tick = 0;
    u32 a = 0, b = 0;
    f64 jn = 0, cx = 0, cy = 0;
};

// Spec section 24 static contactant pseudo body-ids: collapsed-region
// monopoles and walls are not stream bodies; contacts against them encode the
// contactant in b as one of these bases.
constexpr u32 kContactMonopoleBase = 0xFF000000u;
constexpr u32 kContactWallBase = 0xFFFFFF00u;

// Contact record validation (identical rules to ontos_stream_dump):
// body_a is always a real fine body; body_b is either a real body
// (a < b, both < body_count) or one of the static contactant pseudo ids
// -- a collapsed-region monopole 0xFF000000 + r (r < 4), or, when walls
// mode is on, one of the four wall ids 0xFFFFFF00 + 0..3. Anything else
// would index bodies[a] or region_collapse_mass[b - base] out of bounds.
bool contact_pair_valid(u32 a, u32 b, u32 body_count, bool walls_on) {
    if (a >= body_count) {
        return false;
    }
    if (b < kContactMonopoleBase) {
        return b < body_count && a < b;
    }
    if (b < kContactWallBase) {
        return b - kContactMonopoleBase < 4;
    }
    return walls_on && b - kContactWallBase < 4;
}

struct StreamFrame {
    u64 tick = 0;
    u64 fine = 0;
    u64 coarse = 0;
    u8 region_level[4] = {1, 1, 1, 1};
    std::vector<StreamBody> bodies;
    std::vector<StreamContact> contacts;
};

struct Stream {
    u32 body_count = 0;
    // Collapse mass per region (spec 19 RegionCollapsed records); needed to
    // resolve monopole-contactant reduced masses like ontos_stream_dump does.
    f64 region_collapse_mass[4] = {0.0, 0.0, 0.0, 0.0};
    std::vector<StreamFrame> frames;
};

struct ByteView {
    const u8* data = nullptr;
    std::size_t size = 0;
};

// Contactant mass resolution for static pseudo body-ids (spec 24), matching
// ontos_stream_dump: a wall contactant uses the moving body's mass (its audio
// rule is mu = m_a), a collapsed-region monopole uses the region's collapse
// mass, anything else is a real stream body.
f64 contactant_mass(const Stream& stream, const StreamFrame& bodies_frame, u32 b, f64 mass_a) {
    if (b >= kContactWallBase) {
        return mass_a;
    }
    if (b >= kContactMonopoleBase) {
        return stream.region_collapse_mass[b - kContactMonopoleBase];
    }
    return bodies_frame.bodies[b].mass;
}

f64 contact_reduced_mass(const Stream& stream, const StreamFrame& bodies_frame,
                         const StreamContact& c) {
    const f64 mass_a = bodies_frame.bodies[c.a].mass;
    if (c.b >= kContactWallBase) {
        return mass_a;
    }
    const f64 mass_b = contactant_mass(stream, bodies_frame, c.b, mass_a);
    return (mass_a * mass_b) / (mass_a + mass_b);
}

bool take_u32(ByteView d, std::size_t& off, u32& out) {
    if (d.size - off < 4) return false;
    out = static_cast<u32>(d.data[off]) | (static_cast<u32>(d.data[off + 1]) << 8) |
          (static_cast<u32>(d.data[off + 2]) << 16) | (static_cast<u32>(d.data[off + 3]) << 24);
    off += 4;
    return true;
}

bool take_u64(ByteView d, std::size_t& off, u64& out) {
    if (d.size - off < 8) return false;
    u64 lo = 0;
    u64 hi = 0;
    for (int i = 0; i < 4; ++i) {
        lo |= static_cast<u64>(d.data[off + i]) << (8 * i);
        hi |= static_cast<u64>(d.data[off + 4 + i]) << (8 * i);
    }
    out = lo | (hi << 32);
    off += 8;
    return true;
}

bool take_f64(ByteView d, std::size_t& off, f64& out) {
    u64 bits = 0;
    if (!take_u64(d, off, bits)) return false;
    std::memcpy(&out, &bits, 8);
    return true;
}

u32 le32_at(ByteView d, std::size_t off) {
    return static_cast<u32>(d.data[off]) | (static_cast<u32>(d.data[off + 1]) << 8) |
           (static_cast<u32>(d.data[off + 2]) << 16) | (static_cast<u32>(d.data[off + 3]) << 24);
}

[[noreturn]] void stream_error(const char* what, std::size_t offset) {
    std::ostringstream message;
    message << what << " at offset " << offset;
    throw std::runtime_error(message.str());
}

// Spec section 16: a timestamped record's tick field repeats the TickHeader
// tick of the tick the record belongs to. Immediate records (CellFlipped,
// RegionState, BodyState, TotalsState) belong to the section they appear
// in; boundary records (RegionCollapsed, RegionMultipole, Contact,
// RegionRadial, RegionShells) belong to the next tick -- the verifier
// applies them at the next TickHeader, and boundary records queued before
// the first TickHeader target tick 1. One shared check mirrors those
// ordering rules (BodyState inlines the immediate check; Contact keeps its
// queue validation at the next TickHeader).
void check_record_tick(const char* what, u64 record_tick, u64 applicable_tick,
                       std::size_t rec_start) {
    if (record_tick != applicable_tick) {
        std::ostringstream message;
        message << what << " record tick " << record_tick << " does not match applicable tick "
                << applicable_tick;
        stream_error(message.str().c_str(), rec_start);
    }
}

void finalize_frame(StreamFrame& frame, bool has_snapshot, u64 snapshot_population,
                    u32 body_count, const u8 region_level[4]) {
    if (has_snapshot && snapshot_population != frame.bodies.size()) {
        std::ostringstream message;
        message << "tick " << frame.tick << " snapshot population " << snapshot_population
                << " != body records " << frame.bodies.size();
        throw std::runtime_error(message.str());
    }
    if (frame.bodies.size() != body_count) {
        std::ostringstream message;
        message << "tick " << frame.tick << " has " << frame.bodies.size()
                << " body records, expected " << body_count;
        throw std::runtime_error(message.str());
    }
    std::memcpy(frame.region_level, region_level, sizeof frame.region_level);
}

// Parsing logic ported from tools/ontos/ontos_stream_dump.cpp (record walk
// only; records are trusted, no re-simulation). The file is mmap'd read-only
// for the walk and unmapped again; only the parsed frames stay in RAM.
Stream parse_stream(const std::filesystem::path& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("cannot open stream file: " + path.string());
    }
    struct stat st {};
    if (::fstat(::fileno(f), &st) != 0 || st.st_size <= 0) {
        std::fclose(f);
        throw std::runtime_error("cannot stat stream file: " + path.string());
    }
    const std::size_t file_size = static_cast<std::size_t>(st.st_size);
    void* mapped = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, ::fileno(f), 0);
    std::fclose(f);
    if (mapped == MAP_FAILED) {
        throw std::runtime_error("cannot mmap stream file: " + path.string());
    }
    ByteView data{static_cast<const u8*>(mapped), file_size};

    if (data.size < 20 || std::memcmp(data.data, "ONTO", 4) != 0) {
        ::munmap(mapped, file_size);
        throw std::runtime_error("not an ontos v2 stream (bad magic or truncated header)");
    }
    const u32 version = le32_at(data, 4);
    const u32 world_w = le32_at(data, 8);
    const u32 world_h = le32_at(data, 12);
    const u32 body_count = le32_at(data, 16);
    if (version != 2) {
        ::munmap(mapped, file_size);
        std::ostringstream message;
        message << "unsupported stream version " << version << " (ontos_view requires v2)";
        throw std::runtime_error(message.str());
    }
    if (world_w != 128 || world_h != 128) {
        ::munmap(mapped, file_size);
        std::ostringstream message;
        message << "unsupported world size " << world_w << "x" << world_h;
        throw std::runtime_error(message.str());
    }
    if (body_count == 0 || body_count > 100000) {
        ::munmap(mapped, file_size);
        std::ostringstream message;
        message << "implausible body count " << body_count;
        throw std::runtime_error(message.str());
    }

    Stream stream;
    stream.body_count = body_count;
    u8 region_level[4] = {1, 1, 1, 1};
    bool has_snapshot = false;
    u64 snapshot_population = 0;
    std::vector<StreamContact> pending_contacts;
    bool params_seen = false;
    bool walls_on = false;
    u64 last_tick = 0;
    std::size_t off = 20;

    try {
            while (off < data.size) {
            const std::size_t rec_start = off;
            const u8 tag = data.data[off++];
            switch (tag) {
                case 1: {
                    u64 t = 0;
                    if (!take_u64(data, off, t)) stream_error("truncated TickHeader", rec_start);
                    if (!stream.frames.empty()) {
                        finalize_frame(stream.frames.back(), has_snapshot, snapshot_population,
                                       body_count, region_level);
                    }
                    StreamFrame frame;
                    frame.tick = t;
                    last_tick = t;
                    for (StreamContact& c : pending_contacts) {
                        if (c.tick != t) stream_error("Contact tick mismatch", rec_start);
                        frame.contacts.push_back(c);
                    }
                    pending_contacts.clear();
                    stream.frames.push_back(std::move(frame));
                    has_snapshot = false;
                    snapshot_population = 0;
                } break;
                case 2: {
                    u64 p = 0;
                    if (!take_u64(data, off, p)) stream_error("truncated Snapshot", rec_start);
                    has_snapshot = true;
                    snapshot_population = p;
                } break;
                case 3: {
                    u64 t = 0;
                    u32 x = 0;
                    u32 y = 0;
                    if (!take_u64(data, off, t) || !take_u32(data, off, x) || !take_u32(data, off, y)) {
                        stream_error("truncated CellFlipped", rec_start);
                    }
                    check_record_tick("CellFlipped", t,
                                      stream.frames.empty() ? 0 : stream.frames.back().tick,
                                      rec_start);
                } break;
                case 4: {
                    u32 rx = 0;
                    u32 ry = 0;
                    if (!take_u32(data, off, rx) || !take_u32(data, off, ry) ||
                        data.size - off < 1) {
                        stream_error("truncated RegionLevel", rec_start);
                    }
                    const u8 lv = data.data[off++];
                    if (rx > 1 || ry > 1 || lv > 2) {
                        stream_error("bad RegionLevel", rec_start);
                    }
                    region_level[ry * 2 + rx] = lv;
                } break;
                case 5: {
                    u64 t = 0;
                    u32 rx = 0;
                    u32 ry = 0;
                    u64 p = 0;
                    u64 h = 0;
                    if (!take_u64(data, off, t) || !take_u32(data, off, rx) ||
                        !take_u32(data, off, ry) || data.size - off < 1) {
                        stream_error("truncated RegionState", rec_start);
                    }
                    const u8 lv = data.data[off++];
                    if (!take_u64(data, off, p) || !take_u64(data, off, h)) {
                        stream_error("truncated RegionState", rec_start);
                    }
                    if (rx > 1 || ry > 1 || lv > 2) {
                        stream_error("bad RegionState", rec_start);
                    }
                    if (stream.frames.empty()) {
                        stream_error("RegionState before any TickHeader", rec_start);
                    }
                    check_record_tick("RegionState", t, stream.frames.back().tick, rec_start);
                    region_level[ry * 2 + rx] = lv;
                } break;
                case 6: {
                    u64 t = 0;
                    u32 bid = 0;
                    u8 reg = 0;
                    u8 lv = 0;
                    f64 x = 0, y = 0, vx = 0, vy = 0, mass = 0;
                    if (!take_u64(data, off, t) || !take_u32(data, off, bid) ||
                        data.size - off < 2) {
                        stream_error("truncated BodyState", rec_start);
                    }
                    reg = data.data[off++];
                    lv = data.data[off++];
                    if (!take_f64(data, off, x) || !take_f64(data, off, y) ||
                        !take_f64(data, off, vx) || !take_f64(data, off, vy) ||
                        !take_f64(data, off, mass)) {
                        stream_error("truncated BodyState", rec_start);
                    }
                    if (stream.frames.empty()) {
                        stream_error("BodyState before any TickHeader", rec_start);
                    }
                    StreamFrame& frame = stream.frames.back();
                    if (bid != frame.bodies.size() || bid >= body_count || lv > 2 ||
                        (reg > 3 && reg != 255) || t != frame.tick) {
                        stream_error("bad BodyState", rec_start);
                    }
                    StreamBody body;
                    body.id = bid;
                    body.region = reg;
                    body.level = lv;
                    body.x = x;
                    body.y = y;
                    body.vx = vx;
                    body.vy = vy;
                    body.mass = mass;
                    frame.bodies.push_back(body);
                } break;
                case 7: {
                    u64 t = 0;
                    u64 fine = 0;
                    u64 cn = 0;
                    f64 mass = 0, tpx = 0, tpy = 0, energy = 0;
                    if (!take_u64(data, off, t) || !take_u64(data, off, fine) ||
                        !take_u64(data, off, cn) || !take_f64(data, off, mass) ||
                        !take_f64(data, off, tpx) || !take_f64(data, off, tpy) ||
                        !take_f64(data, off, energy)) {
                        stream_error("truncated TotalsState", rec_start);
                    }
                    if (stream.frames.empty()) {
                        stream_error("TotalsState before any TickHeader", rec_start);
                    }
                    check_record_tick("TotalsState", t, stream.frames.back().tick, rec_start);
                    stream.frames.back().fine = fine;
                    stream.frames.back().coarse = cn;
                } break;
                case 8: {
                    // Spec 19 RegionCollapsed: the viewer does not replay the
                    // collapse, but records the mass so monopole-contactant
                    // reduced masses resolve like ontos_stream_dump's.
                    u64 t = 0;
                    u32 rx = 0;
                    u32 ry = 0;
                    u64 n = 0;
                    f64 mass = 0, com_x = 0, com_y = 0, pxt = 0, pyt = 0, energy = 0;
                    if (!take_u64(data, off, t) || !take_u32(data, off, rx) ||
                        !take_u32(data, off, ry) || !take_u64(data, off, n) ||
                        !take_f64(data, off, mass) || !take_f64(data, off, com_x) ||
                        !take_f64(data, off, com_y) || !take_f64(data, off, pxt) ||
                        !take_f64(data, off, pyt) || !take_f64(data, off, energy)) {
                        stream_error("truncated RegionCollapsed", rec_start);
                    }
                    if (rx > 1 || ry > 1) {
                        stream_error("bad RegionCollapsed", rec_start);
                    }
                    check_record_tick("RegionCollapsed", t,
                                      (stream.frames.empty() ? 0 : stream.frames.back().tick) + 1,
                                      rec_start);
                    stream.region_collapse_mass[ry * 2 + rx] = mass;
                } break;
                case 9: {
                    // Spec 20 RegionMultipole: validated, not rendered.
                    u64 t = 0;
                    u32 rx = 0;
                    u32 ry = 0;
                    f64 mx = 0, my = 0, qxx = 0, qxy = 0, qyy = 0;
                    if (!take_u64(data, off, t) || !take_u32(data, off, rx) ||
                        !take_u32(data, off, ry) || !take_f64(data, off, mx) ||
                        !take_f64(data, off, my) || !take_f64(data, off, qxx) ||
                        !take_f64(data, off, qxy) || !take_f64(data, off, qyy)) {
                        stream_error("truncated RegionMultipole", rec_start);
                    }
                    if (rx > 1 || ry > 1) {
                        stream_error("bad RegionMultipole", rec_start);
                    }
                    check_record_tick("RegionMultipole", t,
                                      (stream.frames.empty() ? 0 : stream.frames.back().tick) + 1,
                                      rec_start);
                } break;
                case 10: {
                    StreamContact c;
                    if (!take_u64(data, off, c.tick) || !take_u32(data, off, c.a) ||
                        !take_u32(data, off, c.b) || !take_f64(data, off, c.jn) ||
                        !take_f64(data, off, c.cx) || !take_f64(data, off, c.cy)) {
                        stream_error("truncated Contact", rec_start);
                    }
                    // Static contactants (spec 24) encode a monopole/wall
                    // pseudo id in b; see contact_pair_valid for the exact
                    // bounds rules (same rules as ontos_stream_dump).
                    if (!contact_pair_valid(c.a, c.b, body_count, walls_on) || c.tick == 0) {
                        stream_error("bad Contact", rec_start);
                    }
                    pending_contacts.push_back(c);
                } break;
                case 11: {
                    // Spec 23 RegionRadial: validated, not rendered.
                    u64 t = 0;
                    u32 rx = 0;
                    u32 ry = 0;
                    f64 binding = 0;
                    if (!take_u64(data, off, t) || !take_u32(data, off, rx) ||
                        !take_u32(data, off, ry) || !take_f64(data, off, binding)) {
                        stream_error("truncated RegionRadial", rec_start);
                    }
                    if (rx > 1 || ry > 1) {
                        stream_error("bad RegionRadial", rec_start);
                    }
                    check_record_tick("RegionRadial", t,
                                      (stream.frames.empty() ? 0 : stream.frames.back().tick) + 1,
                                      rec_start);
                } break;
                case 12: {
                    // Spec 24 ContactParams: at most one, before the first
                    // tick; gates the extended contact parsing modes (static
                    // contactants, walls). Validated, not rendered.
                    f64 restitution = 0;
                    f64 friction = 0;
                    if (!take_f64(data, off, restitution) ||
                        !take_f64(data, off, friction) || data.size - off < 1) {
                        stream_error("truncated ContactParams", rec_start);
                    }
                    const u8 walls = data.data[off++];
                    if (params_seen || last_tick != 0 || walls > 1 ||
                        !(restitution >= 0.0 && restitution <= 1.0) || friction < 0.0) {
                        stream_error("bad ContactParams", rec_start);
                    }
                    params_seen = true;
                    walls_on = walls == 1;
                } break;
                case 13: {
                    // Spec 25 RegionShells: validated, not rendered.
                    u64 t = 0;
                    u32 rx = 0;
                    u32 ry = 0;
                    f64 binding = 0, b0 = 0, b1 = 0, b2 = 0, b3 = 0;
                    if (!take_u64(data, off, t) || !take_u32(data, off, rx) ||
                        !take_u32(data, off, ry) || !take_f64(data, off, binding) ||
                        !take_f64(data, off, b0) || !take_f64(data, off, b1) ||
                        !take_f64(data, off, b2) || !take_f64(data, off, b3)) {
                        stream_error("truncated RegionShells", rec_start);
                    }
                    if (rx > 1 || ry > 1) {
                        stream_error("bad RegionShells", rec_start);
                    }
                    check_record_tick("RegionShells", t,
                                      (stream.frames.empty() ? 0 : stream.frames.back().tick) + 1,
                                      rec_start);
                } break;
                default: {
                    std::ostringstream message;
                    message << "unknown record tag " << tag << " at offset " << rec_start;
                    throw std::runtime_error(message.str());
                }
            }
            }
    } catch (...) {
        ::munmap(mapped, file_size);
        throw;
    }
    ::munmap(mapped, file_size);

    // The format has no terminal flush: contacts queue until the next
    // TickHeader validates their tick, so records left in the queue at EOF
    // would attach to a tick that never arrives. The verifier has rejected
    // this since LS-60; the viewer rejects it too instead of silently
    // dropping the records.
    if (!pending_contacts.empty()) {
        std::ostringstream message;
        message << "stream ends with " << pending_contacts.size()
                << " dangling Contact record(s) (no following TickHeader)";
        throw std::runtime_error(message.str());
    }

    if (stream.frames.empty()) {
        throw std::runtime_error("stream contains no ticks");
    }
    finalize_frame(stream.frames.back(), has_snapshot, snapshot_population, body_count,
                   region_level);
    return stream;
}

void hsv_to_rgb(float h, float s, float v, float out[3]) {
    const float hp = h * 6.0f;
    const int hi = static_cast<int>(std::floor(hp)) % 6;
    const float f = hp - std::floor(hp);
    const float p = v * (1.0f - s);
    const float q = v * (1.0f - f * s);
    const float t = v * (1.0f - (1.0f - f) * s);
    switch (hi) {
        case 0: out[0] = v; out[1] = t; out[2] = p; break;
        case 1: out[0] = q; out[1] = v; out[2] = p; break;
        case 2: out[0] = p; out[1] = v; out[2] = t; break;
        case 3: out[0] = p; out[1] = q; out[2] = v; break;
        case 4: out[0] = t; out[1] = p; out[2] = v; break;
        default: out[0] = v; out[1] = p; out[2] = q; break;
    }
}

struct BodyInstance {
    float x, y;
    float half_extent, shape;
    float r, g, b, a;
};

struct ViewPush {
    float sx, sy, tx, ty;
    float z;
};

struct Camera2D {
    double cx = 64.0;
    double cy = 64.0;
    double zoom = 5.0;
};

struct WindowState {
    bool resized = false;
    Camera2D camera;
    int fb_width = 1280;
    int fb_height = 720;
};

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                              VkDebugUtilsMessageTypeFlagsEXT types,
                                              const VkDebugUtilsMessengerCallbackDataEXT* data,
                                              void* user_data) {
    (void)types;
    (void)user_data;
    const char* tag = (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0
                          ? "VALIDATION ERROR"
                          : "VALIDATION WARNING";
    std::fprintf(stderr, "%s: %s\n", tag, data->pMessage);
    return VK_FALSE;
}

void configure_macos_moltenvk_environment() {
#if defined(__APPLE__)
    if (std::getenv("VK_ICD_FILENAMES") == nullptr) {
        constexpr const char* candidates[] = {
            "/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json",
            "/usr/local/etc/vulkan/icd.d/MoltenVK_icd.json",
        };
        for (const char* candidate : candidates) {
            if (std::filesystem::exists(candidate)) {
                setenv("VK_ICD_FILENAMES", candidate, 0);
                break;
            }
        }
    }

    if (std::getenv("VK_LAYER_PATH") == nullptr) {
        constexpr const char* candidates[] = {
            "/opt/homebrew/opt/vulkan-validationlayers/share/vulkan/explicit_layer.d",
            "/usr/local/opt/vulkan-validationlayers/share/vulkan/explicit_layer.d",
        };
        for (const char* candidate : candidates) {
            const std::filesystem::path manifest =
                std::filesystem::path(candidate) / "VkLayer_khronos_validation.json";
            const std::filesystem::path library =
                std::filesystem::path(candidate) / ".." / ".." / ".." / "lib" /
                "libVkLayer_khronos_validation.dylib";
            if (!std::filesystem::exists(manifest)) {
                continue;
            }
            if (std::filesystem::exists(library)) {
                // Homebrew ships a bare library_path the loader cannot dlopen
                // outside default search paths; shadow it with an absolute path.
                std::ifstream input(manifest);
                std::ostringstream buffer;
                buffer << input.rdbuf();
                std::string text = buffer.str();
                const std::string from = "\"library_path\": \"libVkLayer_khronos_validation.dylib\"";
                const std::string to =
                    "\"library_path\": \"" + std::filesystem::absolute(library).string() + "\"";
                if (text.find(from) != std::string::npos) {
                    std::error_code ec;
                    const std::filesystem::path shadow_dir =
                        std::filesystem::temp_directory_path(ec) / "ontos-view-vklayer";
                    if (!std::filesystem::exists(shadow_dir)) {
                        std::filesystem::create_directories(shadow_dir, ec);
                    }
                    if (!ec) {
                        const std::filesystem::path shadow = shadow_dir / manifest.filename();
                        std::ofstream output(shadow, std::ios::trunc);
                        output << text.replace(text.find(from), from.size(), to);
                        setenv("VK_LAYER_PATH", shadow_dir.c_str(), 0);
                        break;
                    }
                }
            }
            setenv("VK_LAYER_PATH", candidate, 0);
            break;
        }
    }
#endif
}

std::filesystem::path resolve_shader_path(const char* name) {
    std::filesystem::path exe_dir;
#if defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::string exe_path(size, '\0');
    std::error_code ec;
    if (_NSGetExecutablePath(exe_path.data(), &size) == 0) {
        const auto canonical = std::filesystem::canonical(exe_path, ec);
        if (!ec) exe_dir = canonical.parent_path();
    }
#endif
    std::vector<std::filesystem::path> candidates;
    if (!exe_dir.empty()) {
        candidates.push_back(exe_dir / ".." / "shaders" / name);
        candidates.push_back(exe_dir / "shaders" / name);
    }
    candidates.push_back(std::filesystem::current_path() / ".." / "shaders" / name);
    candidates.push_back(std::filesystem::path("shaders") / name);
    for (const auto& p : candidates) {
        if (std::filesystem::exists(p)) return p;
    }
    throw std::runtime_error(std::string("shader not found: ") + name);
}

std::string load_shader_source(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open shader: " + path.string());
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

std::vector<uint32_t> compile_glsl(const std::string& source, shaderc_shader_kind kind,
                                   const char* name) {
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;
    options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
    const shaderc::SpvCompilationResult result =
        compiler.CompileGlslToSpv(source, kind, name, options);
    if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
        throw std::runtime_error(std::string("shader compilation failed for ") + name + ": " +
                                 result.GetErrorMessage());
    }
    return {result.cbegin(), result.cend()};
}

struct GpuBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize size = 0;
};

uint32_t find_memory_type(VkPhysicalDevice physical_device, uint32_t type_bits,
                          VkMemoryPropertyFlags required) {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physical_device, &properties);
    for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
        const bool supported = (type_bits & (1u << i)) != 0;
        const bool has_flags = (properties.memoryTypes[i].propertyFlags & required) == required;
        if (supported && has_flags) return i;
    }
    return std::numeric_limits<uint32_t>::max();
}

void create_host_buffer(VkPhysicalDevice physical_device, VkDevice device, const void* data,
                        VkDeviceSize size, VkBufferUsageFlags usage, bool persistent_map,
                        GpuBuffer& out) {
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult result = vkCreateBuffer(device, &buffer_info, nullptr, &out.buffer);
    if (result != VK_SUCCESS) throw std::runtime_error("vkCreateBuffer failed");

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, out.buffer, &requirements);
    VkMemoryAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex =
        find_memory_type(physical_device, requirements.memoryTypeBits,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (allocate.memoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("no host-visible memory type");
    }
    result = vkAllocateMemory(device, &allocate, nullptr, &out.memory);
    if (result != VK_SUCCESS) throw std::runtime_error("vkAllocateMemory failed");
    result = vkBindBufferMemory(device, out.buffer, out.memory, 0);
    if (result != VK_SUCCESS) throw std::runtime_error("vkBindBufferMemory failed");
    out.size = size;
    if (data != nullptr || persistent_map) {
        result = vkMapMemory(device, out.memory, 0, size, 0, &out.mapped);
        if (result != VK_SUCCESS) throw std::runtime_error("vkMapMemory failed");
        if (data != nullptr) {
            std::memcpy(out.mapped, data, static_cast<std::size_t>(size));
        }
        if (!persistent_map) {
            vkUnmapMemory(device, out.memory);
            out.mapped = nullptr;
        }
    }
}

void destroy_buffer(VkDevice device, GpuBuffer& buffer) {
    if (buffer.mapped != nullptr) vkUnmapMemory(device, buffer.memory);
    if (buffer.buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, buffer.buffer, nullptr);
    if (buffer.memory != VK_NULL_HANDLE) vkFreeMemory(device, buffer.memory, nullptr);
    buffer = {};
}

struct Viewer {
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t graphics_family = 0;
    VkQueue queue = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat swapchain_format = VK_FORMAT_B8G8R8A8_UNORM;
    VkFormat depth_format = VK_FORMAT_D32_SFLOAT;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    VkExtent2D extent{};


    std::vector<VkImage> images;
    std::vector<VkImageView> image_views;
    std::vector<VkFramebuffer> framebuffers;
    VkImage msaa_color = VK_NULL_HANDLE;
    VkDeviceMemory msaa_color_memory = VK_NULL_HANDLE;
    VkImageView msaa_color_view = VK_NULL_HANDLE;
    VkImage depth_image = VK_NULL_HANDLE;
    VkDeviceMemory depth_memory = VK_NULL_HANDLE;
    VkImageView depth_view = VK_NULL_HANDLE;
    VkRenderPass render_pass = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline body_pipeline = VK_NULL_HANDLE;
    VkPipeline line_pipeline = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkSemaphore image_available = VK_NULL_HANDLE;
    std::vector<VkSemaphore> render_finished_per_image;
    VkFence in_flight = VK_NULL_HANDLE;
    GpuBuffer quad_buffer;
    GpuBuffer line_buffer;
    GpuBuffer line_instance_buffer;
    GpuBuffer minor_line_buffer;
    GpuBuffer minor_line_instance_buffer;
    GpuBuffer hud_line_buffer;
    GpuBuffer hud_panel_buffer;
    GpuBuffer hud_line_instance_buffer;
    GpuBuffer hud_quad_instance_buffer;
    // Consecutive frames write alternating slots; the single in-flight
    // fence serializes submits, so the slot picked below is never read by
    // still-running GPU work when it is refilled.
    GpuBuffer body_instance_buffers[2];
    GpuBuffer contact_instance_buffers[2];

    void destroy_extent_resources() {
        if (device != VK_NULL_HANDLE) vkDeviceWaitIdle(device);
        for (VkFramebuffer fb : framebuffers) vkDestroyFramebuffer(device, fb, nullptr);
        framebuffers.clear();
        if (msaa_color_view != VK_NULL_HANDLE) vkDestroyImageView(device, msaa_color_view, nullptr);
        msaa_color_view = VK_NULL_HANDLE;
        if (msaa_color != VK_NULL_HANDLE) vkDestroyImage(device, msaa_color, nullptr);
        msaa_color = VK_NULL_HANDLE;
        if (msaa_color_memory != VK_NULL_HANDLE) vkFreeMemory(device, msaa_color_memory, nullptr);
        msaa_color_memory = VK_NULL_HANDLE;
        if (depth_view != VK_NULL_HANDLE) vkDestroyImageView(device, depth_view, nullptr);
        depth_view = VK_NULL_HANDLE;
        if (depth_image != VK_NULL_HANDLE) vkDestroyImage(device, depth_image, nullptr);
        depth_image = VK_NULL_HANDLE;
        if (depth_memory != VK_NULL_HANDLE) vkFreeMemory(device, depth_memory, nullptr);
        depth_memory = VK_NULL_HANDLE;
    }

    void destroy_swapchain_views() {
        destroy_extent_resources();
        for (VkImageView view : image_views) vkDestroyImageView(device, view, nullptr);
        image_views.clear();
        images.clear();
        if (swapchain != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(device, swapchain, nullptr);
            swapchain = VK_NULL_HANDLE;
        }
    }

    void destroy() {
        if (device != VK_NULL_HANDLE) vkDeviceWaitIdle(device);
        destroy_swapchain_views();
        destroy_buffer(device, contact_instance_buffers[0]);
        destroy_buffer(device, contact_instance_buffers[1]);
        destroy_buffer(device, body_instance_buffers[0]);
        destroy_buffer(device, body_instance_buffers[1]);
        destroy_buffer(device, line_instance_buffer);
        destroy_buffer(device, minor_line_buffer);
        destroy_buffer(device, minor_line_instance_buffer);
        destroy_buffer(device, hud_line_buffer);
        destroy_buffer(device, hud_panel_buffer);
        destroy_buffer(device, hud_line_instance_buffer);
        destroy_buffer(device, hud_quad_instance_buffer);
        destroy_buffer(device, line_buffer);
        destroy_buffer(device, quad_buffer);
        if (in_flight != VK_NULL_HANDLE) vkDestroyFence(device, in_flight, nullptr);
        for (VkSemaphore semaphore : render_finished_per_image) {
            vkDestroySemaphore(device, semaphore, nullptr);
        }
        render_finished_per_image.clear();
        if (image_available != VK_NULL_HANDLE) vkDestroySemaphore(device, image_available, nullptr);
        if (command_pool != VK_NULL_HANDLE) vkDestroyCommandPool(device, command_pool, nullptr);
        if (line_pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, line_pipeline, nullptr);
        if (body_pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, body_pipeline, nullptr);
        if (pipeline_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        }
        if (render_pass != VK_NULL_HANDLE) vkDestroyRenderPass(device, render_pass, nullptr);
        if (device != VK_NULL_HANDLE) vkDestroyDevice(device, nullptr);
        if (surface != VK_NULL_HANDLE) vkDestroySurfaceKHR(instance, surface, nullptr);
        if (debug_messenger != VK_NULL_HANDLE) {
            auto destroy_messenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
            if (destroy_messenger != nullptr) {
                destroy_messenger(instance, debug_messenger, nullptr);
            }
        }
        if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, nullptr);
    }
};

struct GpuImage {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
};

void create_image(Viewer& v, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                  GpuImage& out) {
    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = format;
    image_info.extent = {v.extent.width, v.extent.height, 1};
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.samples = v.samples;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage = usage;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult result = vkCreateImage(v.device, &image_info, nullptr, &out.image);
    if (result != VK_SUCCESS) throw std::runtime_error("vkCreateImage failed");

    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(v.device, out.image, &requirements);
    VkMemoryAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex =
        find_memory_type(v.physical_device, requirements.memoryTypeBits,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (allocate.memoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("no device-local memory type");
    }
    result = vkAllocateMemory(v.device, &allocate, nullptr, &out.memory);
    if (result != VK_SUCCESS) throw std::runtime_error("vkAllocateMemory failed");
    result = vkBindImageMemory(v.device, out.image, out.memory, 0);
    if (result != VK_SUCCESS) throw std::runtime_error("vkBindImageMemory failed");

    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = out.image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = format;
    view_info.subresourceRange.aspectMask = aspect;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.layerCount = 1;
    result = vkCreateImageView(v.device, &view_info, nullptr, &out.view);
    if (result != VK_SUCCESS) throw std::runtime_error("vkCreateImageView failed");
}
// First supported composite alpha mode, preferring opaque (mirrors the
// main renderer's LS-37 policy). The spec guarantees at least one bit in
// supportedCompositeAlpha; INHERIT is the final fallback and is always
// legal.
VkCompositeAlphaFlagBitsKHR choose_composite_alpha(VkCompositeAlphaFlagsKHR supported) {
    if ((supported & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) != 0) {
        return VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    }
    if ((supported & VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR) != 0) {
        return VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR;
    }
    if ((supported & VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR) != 0) {
        return VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR;
    }
    return VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
}

void create_swapchain(Viewer& v, GLFWwindow* window) {
    VkSurfaceCapabilitiesKHR capabilities{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(v.physical_device, v.surface, &capabilities);

    uint32_t format_count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(v.physical_device, v.surface, &format_count, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(format_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(v.physical_device, v.surface, &format_count,
                                         formats.data());
    VkSurfaceFormatKHR surface_format = formats.front();
    for (const VkSurfaceFormatKHR& format : formats) {
        if (format.format == VK_FORMAT_B8G8R8A8_UNORM &&
            format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            surface_format = format;
            break;
        }
    }
    if (surface_format.format != v.swapchain_format) {
        throw std::runtime_error("surface does not expose B8G8R8A8_UNORM sRGB");
    }

    uint32_t present_mode_count = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(v.physical_device, v.surface, &present_mode_count,
                                              nullptr);
    std::vector<VkPresentModeKHR> present_modes(present_mode_count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(v.physical_device, v.surface, &present_mode_count,
                                              present_modes.data());
    VkPresentModeKHR present_mode = present_modes.front();
    for (VkPresentModeKHR mode : present_modes) {
        if (mode == VK_PRESENT_MODE_FIFO_KHR) {
            present_mode = mode;
            break;
        }
    }

    VkExtent2D extent{};
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        extent = capabilities.currentExtent;
    } else {
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        extent.width = std::clamp(static_cast<uint32_t>(width), capabilities.minImageExtent.width,
                                  capabilities.maxImageExtent.width);
        extent.height = std::clamp(static_cast<uint32_t>(height),
                                   capabilities.minImageExtent.height,
                                   capabilities.maxImageExtent.height);
    }

    uint32_t image_count = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 && image_count > capabilities.maxImageCount) {
        image_count = capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    create_info.surface = v.surface;
    create_info.minImageCount = image_count;
    create_info.imageFormat = surface_format.format;
    create_info.imageColorSpace = surface_format.colorSpace;
    create_info.imageExtent = extent;
    create_info.imageArrayLayers = 1;
    create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    create_info.preTransform = capabilities.currentTransform;
    create_info.compositeAlpha = choose_composite_alpha(capabilities.supportedCompositeAlpha);
    create_info.presentMode = present_mode;
    create_info.clipped = VK_TRUE;
    create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult result = vkCreateSwapchainKHR(v.device, &create_info, nullptr, &v.swapchain);
    if (result != VK_SUCCESS) throw std::runtime_error("vkCreateSwapchainKHR failed");

    uint32_t count = 0;
    vkGetSwapchainImagesKHR(v.device, v.swapchain, &count, nullptr);
    v.images.resize(count);
    vkGetSwapchainImagesKHR(v.device, v.swapchain, &count, v.images.data());
    v.extent = extent;

    v.image_views.resize(v.images.size());
    for (size_t i = 0; i < v.images.size(); ++i) {
        VkImageViewCreateInfo view_info{};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image = v.images[i];
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = v.swapchain_format;
        view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view_info.subresourceRange.levelCount = 1;
        view_info.subresourceRange.layerCount = 1;
        result = vkCreateImageView(v.device, &view_info, nullptr, &v.image_views[i]);
        if (result != VK_SUCCESS) throw std::runtime_error("vkCreateImageView failed");
    }
}

// Creates the extent-sized attachments (MSAA color resolve source + depth)
// and the per-swapchain-image framebuffers. Must run after create_swapchain
// and again after every resize.
void create_extent_resources(Viewer& v) {
    if (v.samples != VK_SAMPLE_COUNT_1_BIT) {
        GpuImage msaa{};
        create_image(v, v.swapchain_format,
                     VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                     VK_IMAGE_ASPECT_COLOR_BIT, msaa);
        v.msaa_color = msaa.image;
        v.msaa_color_memory = msaa.memory;
        v.msaa_color_view = msaa.view;
    }
    {
        GpuImage depth{};
        create_image(v, v.depth_format,
                     VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                     VK_IMAGE_ASPECT_DEPTH_BIT, depth);
        v.depth_image = depth.image;
        v.depth_memory = depth.memory;
        v.depth_view = depth.view;
    }

    v.framebuffers.resize(v.images.size());
    for (size_t i = 0; i < v.images.size(); ++i) {
        const VkImageView attachments[3] = {v.msaa_color_view, v.image_views[i], v.depth_view};
        const uint32_t attachment_count = v.samples != VK_SAMPLE_COUNT_1_BIT ? 3 : 2;
        const VkImageView* used_attachments =
            v.samples != VK_SAMPLE_COUNT_1_BIT ? attachments : attachments + 1;
        VkFramebufferCreateInfo fb_info{};
        fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb_info.renderPass = v.render_pass;
        fb_info.attachmentCount = attachment_count;
        fb_info.pAttachments = used_attachments;
        fb_info.width = v.extent.width;
        fb_info.height = v.extent.height;
        fb_info.layers = 1;
        const VkResult result =
            vkCreateFramebuffer(v.device, &fb_info, nullptr, &v.framebuffers[i]);
        if (result != VK_SUCCESS) throw std::runtime_error("vkCreateFramebuffer failed");
    }
}

VkPipeline create_pipeline(Viewer& v, VkPrimitiveTopology topology, bool depth_write) {
    const std::string vert_source = load_shader_source(resolve_shader_path("ontos_bodies.vert"));
    const std::string frag_source = load_shader_source(resolve_shader_path("ontos_bodies.frag"));
    const std::vector<uint32_t> vert_spirv =
        compile_glsl(vert_source, shaderc_vertex_shader, "ontos_bodies.vert");
    const std::vector<uint32_t> frag_spirv =
        compile_glsl(frag_source, shaderc_fragment_shader, "ontos_bodies.frag");

    VkShaderModuleCreateInfo module_info{};
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = vert_spirv.size() * sizeof(uint32_t);
    module_info.pCode = vert_spirv.data();
    VkShaderModule vert_module = VK_NULL_HANDLE;
    VkResult result = vkCreateShaderModule(v.device, &module_info, nullptr, &vert_module);
    if (result != VK_SUCCESS) throw std::runtime_error("vkCreateShaderModule failed");
    module_info.codeSize = frag_spirv.size() * sizeof(uint32_t);
    module_info.pCode = frag_spirv.data();
    VkShaderModule frag_module = VK_NULL_HANDLE;
    result = vkCreateShaderModule(v.device, &module_info, nullptr, &frag_module);
    if (result != VK_SUCCESS) throw std::runtime_error("vkCreateShaderModule failed");

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert_module;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag_module;
    stages[1].pName = "main";

    VkVertexInputBindingDescription bindings[2]{};
    bindings[0].binding = 0;
    bindings[0].stride = sizeof(float) * 2;
    bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    bindings[1].binding = 1;
    bindings[1].stride = sizeof(BodyInstance);
    bindings[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

    VkVertexInputAttributeDescription attributes[4]{};
    attributes[0].location = 0;
    attributes[0].binding = 0;
    attributes[0].format = VK_FORMAT_R32G32_SFLOAT;
    attributes[0].offset = 0;
    attributes[1].location = 1;
    attributes[1].binding = 1;
    attributes[1].format = VK_FORMAT_R32G32_SFLOAT;
    attributes[1].offset = offsetof(BodyInstance, x);
    attributes[2].location = 2;
    attributes[2].binding = 1;
    attributes[2].format = VK_FORMAT_R32G32_SFLOAT;
    attributes[2].offset = offsetof(BodyInstance, half_extent);
    attributes[3].location = 3;
    attributes[3].binding = 1;
    attributes[3].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attributes[3].offset = offsetof(BodyInstance, r);

    VkPipelineVertexInputStateCreateInfo vertex_input{};
    vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input.vertexBindingDescriptionCount = 2;
    vertex_input.pVertexBindingDescriptions = bindings;
    vertex_input.vertexAttributeDescriptionCount = 4;
    vertex_input.pVertexAttributeDescriptions = attributes;

    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = topology;
    // Draws are non-indexed, so restart cannot trigger; MoltenVK warns when
    // it is disabled for strips because Metal has no such switch, while list
    // topologies forbid enabling it without primitiveTopologyListRestart.
    input_assembly.primitiveRestartEnable =
        topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP ? VK_TRUE : VK_FALSE;

    VkPipelineViewportStateCreateInfo viewport_state{};
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterization{};
    rasterization.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterization.polygonMode = VK_POLYGON_MODE_FILL;
    rasterization.cullMode = VK_CULL_MODE_NONE;
    rasterization.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = v.samples;

    // Both pipelines depth-test LESS with z pushed per pipeline: grid lines
    // sit at z ~ 0.5, body discs slightly nearer, so blended bodies always
    // layer over the grid regardless of draw order (bodies never write depth,
    // keeping alpha blending commutative with painter's order).
    VkPipelineDepthStencilStateCreateInfo depth_stencil{};
    depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth_stencil.depthTestEnable = VK_TRUE;
    depth_stencil.depthWriteEnable = depth_write ? VK_TRUE : VK_FALSE;
    depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState attachment{};
    attachment.blendEnable = VK_TRUE;
    attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    attachment.colorBlendOp = VK_BLEND_OP_ADD;
    attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    attachment.alphaBlendOp = VK_BLEND_OP_ADD;
    attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo color_blend{};
    color_blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blend.attachmentCount = 1;
    color_blend.pAttachments = &attachment;

    const VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{};
    dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamic_states;

    VkGraphicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_info.stageCount = 2;
    pipeline_info.pStages = stages;
    pipeline_info.pVertexInputState = &vertex_input;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &rasterization;
    pipeline_info.pMultisampleState = &multisample;
    pipeline_info.pDepthStencilState = &depth_stencil;
    pipeline_info.pColorBlendState = &color_blend;
    pipeline_info.pDynamicState = &dynamic;
    pipeline_info.layout = v.pipeline_layout;
    pipeline_info.renderPass = v.render_pass;
    pipeline_info.subpass = 0;

    VkPipeline pipeline = VK_NULL_HANDLE;
    result = vkCreateGraphicsPipelines(v.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr,
                                       &pipeline);
    vkDestroyShaderModule(v.device, vert_module, nullptr);
    vkDestroyShaderModule(v.device, frag_module, nullptr);
    if (result != VK_SUCCESS) throw std::runtime_error("vkCreateGraphicsPipelines failed");
    return pipeline;
}

float body_half_extent(f64 mass) {
    const float log_size = 0.55f + 1.15f * static_cast<float>(std::log2(mass));
    return std::clamp(log_size, 0.35f, 2.4f);
}

void region_fill_color(u8 region, u8 level, float rgb[3]) {
    static const float region_hue[4] = {0.62f, 0.08f, 0.33f, 0.78f};
    if (region <= 3) {
        const float s = level == 1 ? 0.85f : 0.22f;
        const float val = level == 1 ? 1.0f : 0.60f;
        hsv_to_rgb(region_hue[region], s, val, rgb);
    } else {
        const float g = level == 1 ? 0.75f : 0.45f;
        rgb[0] = g;
        rgb[1] = g;
        rgb[2] = g;
    }
}

void fill_body_instances(const StreamFrame& frame, BodyInstance* instances) {
    for (size_t i = 0; i < frame.bodies.size(); ++i) {
        const StreamBody& body = frame.bodies[i];
        BodyInstance& inst = instances[i];
        float rgb[3];
        region_fill_color(body.region, body.level, rgb);
        inst.x = static_cast<float>(body.x);
        inst.y = static_cast<float>(body.y);
        inst.half_extent = body_half_extent(body.mass);
        inst.shape = 1.0f;
        inst.r = rgb[0];
        inst.g = rgb[1];
        inst.b = rgb[2];
        inst.a = 0.95f;
    }
}

ViewPush compute_view_push(const Camera2D& camera, VkExtent2D extent, float z) {
    const float sx = static_cast<float>(2.0 * camera.zoom / extent.width);
    const float sy = static_cast<float>(2.0 * camera.zoom / extent.height);
    ViewPush push{};
    push.sx = sx;
    push.sy = sy;
    push.tx = static_cast<float>(-camera.cx * sx);
    push.ty = static_cast<float>(-camera.cy * sy);
    push.z = z;
    return push;
}

// Screen-space transform for HUD primitives: pixel coordinates (y down,
// origin top-left) to NDC, independent of the world camera. Vulkan NDC +y
// points down the framebuffer, so the slope is positive.
ViewPush compute_screen_push(VkExtent2D extent, float z) {
    ViewPush push{};
    push.sx = 2.0f / static_cast<float>(extent.width);
    push.sy = 2.0f / static_cast<float>(extent.height);
    push.tx = -1.0f;
    push.ty = -1.0f;
    push.z = z;
    return push;
}

// Spec section 22 (normative: ontos docs/STREAM_SPEC.md): modal audio as a
// pure function of the stream. Offline deterministic render — the v1 audio
// output path; realtime device output is a later concern.
struct AudioSpec {
    static constexpr u64 kSamplesPerTick = 64;
    static constexpr int kRing = 16384;
    static constexpr u64 kTailBlocks = 260;
    static constexpr f64 kOmega0 = 0.0004448824124529259;
    static constexpr f64 kPartial[3] = {1.0, 4.0, 9.0};
    static constexpr f64 kRho[3] = {0.9990, 0.9985, 0.9980};
    static constexpr f64 kAmp[3] = {0.5, 0.3, 0.2};
};

u64 fnv1a64_bytes(const u8* data, std::size_t len) {
    u64 h = 0xcbf29ce484222325ULL;
    for (std::size_t i = 0; i < len; ++i) {
        h = (h ^ data[i]) * 0x100000001b3ULL;
    }
    return h;
}

u64 render_contact_audio(const Stream& stream, const char* wav_path, bool write_file) {
    const u64 final_tick = stream.frames.back().tick;
    const std::size_t n = static_cast<std::size_t>((final_tick + AudioSpec::kTailBlocks) *
                                                   AudioSpec::kSamplesPerTick);
    std::vector<f64> buf(n, 0.0);
    const StreamFrame& last = stream.frames.back();
    for (const StreamFrame& frame : stream.frames) {
        for (const StreamContact& c : frame.contacts) {
            const std::size_t e =
                static_cast<std::size_t>((c.tick + 1) * AudioSpec::kSamplesPerTick);
            const f64 mu = contact_reduced_mass(stream, last, c);
            for (int k = 0; k < 3; ++k) {
                const f64 omega = AudioSpec::kOmega0 * AudioSpec::kPartial[k] / mu;
                const f64 a = (2.0 - omega) * AudioSpec::kRho[k];
                const f64 b = AudioSpec::kRho[k] * AudioSpec::kRho[k];
                const f64 s0 = AudioSpec::kAmp[k] * c.jn;
                f64 s_prev = s0;
                f64 s_prev2 = 0.0;
                for (int i = 0; i < AudioSpec::kRing; ++i) {
                    const f64 s = i == 0 ? s0 : (i == 1 ? a * s0 : a * s_prev - b * s_prev2);
                    buf[e + static_cast<std::size_t>(i)] += s;
                    s_prev2 = s_prev;
                    s_prev = s;
                }
            }
        }
    }
    std::vector<u8> pcm(n * 2);
    for (std::size_t i = 0; i < n; ++i) {
        f64 v = buf[i];
        if (v < -1.0) {
            v = -1.0;
        } else if (v > 1.0) {
            v = 1.0;
        }
        const short sample = static_cast<short>(std::floor(v * 32767.0 + 0.5));
        const u16 bits = static_cast<u16>(sample);
        pcm[i * 2] = static_cast<u8>(bits & 0xff);
        pcm[i * 2 + 1] = static_cast<u8>(bits >> 8);
    }
    if (write_file) {
        std::ofstream out(wav_path, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) {
            throw std::runtime_error(std::string("cannot write wav: ") + wav_path);
        }
        const auto put_u32 = [](std::vector<u8>& v, u32 x) {
            v.push_back(static_cast<u8>(x & 0xff));
            v.push_back(static_cast<u8>((x >> 8) & 0xff));
            v.push_back(static_cast<u8>((x >> 16) & 0xff));
            v.push_back(static_cast<u8>((x >> 24) & 0xff));
        };
        const auto put_u16 = [](std::vector<u8>& v, u16 x) {
            v.push_back(static_cast<u8>(x & 0xff));
            v.push_back(static_cast<u8>(x >> 8));
        };
        std::vector<u8> wav;
        wav.reserve(44 + pcm.size());
        wav.insert(wav.end(), {'R', 'I', 'F', 'F'});
        put_u32(wav, 36 + static_cast<u32>(pcm.size()));
        wav.insert(wav.end(), {'W', 'A', 'V', 'E'});
        wav.insert(wav.end(), {'f', 'm', 't', ' '});
        put_u32(wav, 16);
        put_u16(wav, 1);  // PCM
        put_u16(wav, 1);  // mono
        put_u32(wav, 65536);
        put_u32(wav, 131072);
        put_u16(wav, 2);  // block align
        put_u16(wav, 16);
        wav.insert(wav.end(), {'d', 'a', 't', 'a'});
        put_u32(wav, static_cast<u32>(pcm.size()));
        wav.insert(wav.end(), pcm.begin(), pcm.end());
        out.write(reinterpret_cast<const char*>(wav.data()),
                  static_cast<std::streamsize>(wav.size()));
        out.flush();
        if (!out) {
            throw std::runtime_error(std::string("wav write failed: ") + wav_path);
        }
    }
    return fnv1a64_bytes(pcm.data(), pcm.size());
}

#if ONTOS_VIEW_REALTIME_AUDIO

// Realtime companion to the offline --wav render: the same spec-22 resonator
// samples, scheduled live. Each contact spawns a voice when playback crosses
// its tick; the device callback advances the identical recurrence (same
// constants, same operands, same coefficient math — voices are constructed
// from the final-frame masses like the offline mix) and sums active voices
// into a stereo stream. Sample content stays a pure function of the stream;
// wall-clock only decides when a ring starts. VoiceBank is the shared,
// platform-neutral pool+mixer; the per-platform RealtimeAudio backends only
// differ in how they pull mixed frames out (AudioQueue callback vs ALSA
// write thread).
struct ContactVoice {
    bool active = false;
    u64 emitted = 0;
    f64 s0[3] = {};
    f64 s_prev[3] = {};
    f64 s_prev2[3] = {};
    f64 a[3] = {};
    f64 b[3] = {};
    f64 gl = 1.0, gr = 1.0;
};

struct VoiceBank {
    static constexpr int kVoiceCount = 48;

    std::mutex mutex;
    ContactVoice voices[kVoiceCount] = {};

    // Advances every active voice exactly one spec-rate sample and sums,
    // clamps into one stereo frame. Caller must hold the mutex.
    void mix_frame(f64& out_l, f64& out_r) {
        f64 left = 0.0;
        f64 right = 0.0;
        for (ContactVoice& voice : voices) {
            if (!voice.active) continue;
            f64 mono = 0.0;
            for (int k = 0; k < 3; ++k) {
                f64 s = 0.0;
                if (voice.emitted == 0) {
                    s = voice.s0[k];
                } else if (voice.emitted == 1) {
                    s = voice.a[k] * voice.s0[k];
                } else {
                    s = voice.a[k] * voice.s_prev[k] - voice.b[k] * voice.s_prev2[k];
                }
                voice.s_prev2[k] = voice.s_prev[k];
                voice.s_prev[k] = s;
                mono += s;
            }
            ++voice.emitted;
            if (voice.emitted >= static_cast<u64>(AudioSpec::kRing)) voice.active = false;
            left += mono * voice.gl;
            right += mono * voice.gr;
        }
        if (left < -1.0) {
            left = -1.0;
        } else if (left > 1.0) {
            left = 1.0;
        }
        if (right < -1.0) {
            right = -1.0;
        } else if (right > 1.0) {
            right = 1.0;
        }
        out_l = left;
        out_r = right;
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex);
        for (ContactVoice& voice : voices) voice.active = false;
    }

    void spawn_contact(const StreamContact& contact, f64 mu, f64 gl, f64 gr) {
        ContactVoice voice;
        for (int k = 0; k < 3; ++k) {
            const f64 omega = AudioSpec::kOmega0 * AudioSpec::kPartial[k] / mu;
            voice.a[k] = (2.0 - omega) * AudioSpec::kRho[k];
            voice.b[k] = AudioSpec::kRho[k] * AudioSpec::kRho[k];
            voice.s0[k] = AudioSpec::kAmp[k] * contact.jn;
        }
        voice.gl = gl;
        voice.gr = gr;
        // The mixer skips inactive voices; the copy below lands the fully
        // initialized ring as live (without this, spawned contacts were
        // constructed and then never sounded).
        voice.active = true;
        std::lock_guard<std::mutex> lock(mutex);
        ContactVoice* slot = nullptr;
        u64 most_emitted = 0;
        for (ContactVoice& candidate : voices) {
            if (!candidate.active) {
                slot = &candidate;
                break;
            }
            if (slot == nullptr || candidate.emitted > most_emitted) {
                slot = &candidate;
                most_emitted = candidate.emitted;
            }
        }
        *slot = voice;
    }
};

#if defined(__APPLE__)

// macOS backend: AudioQueue output at the spec-native rate; the queue
// callback pulls VoiceBank frames directly (no resampling).
struct RealtimeAudio {
    static constexpr int kVoiceCount = VoiceBank::kVoiceCount;
    static constexpr u32 kBufferFrames = 2048;
    static constexpr int kBufferCount = 4;

    VoiceBank bank;
    AudioQueueRef queue = nullptr;
    bool failed = false;
    u32 output_rate = 65536;

    ~RealtimeAudio() { stop(); }

    RealtimeAudio() = default;
    RealtimeAudio(const RealtimeAudio&) = delete;
    RealtimeAudio& operator=(const RealtimeAudio&) = delete;

    static void fill_buffer(void* user_data, AudioQueueRef queue, AudioQueueBufferRef buffer) {
        auto* self = static_cast<RealtimeAudio*>(user_data);
        float* out = static_cast<float*>(buffer->mAudioData);
        const u32 frames =
            static_cast<u32>(buffer->mAudioDataByteSize / (sizeof(float) * 2));
        std::lock_guard<std::mutex> lock(self->bank.mutex);
        for (u32 i = 0; i < frames; ++i) {
            f64 left = 0.0;
            f64 right = 0.0;
            self->bank.mix_frame(left, right);
            out[i * 2] = static_cast<float>(left);
            out[i * 2 + 1] = static_cast<float>(right);
        }
        AudioQueueEnqueueBuffer(queue, buffer, 0, nullptr);
    }

    bool start() {
        if (failed || queue != nullptr) return queue != nullptr;
        AudioStreamBasicDescription format{};
        format.mSampleRate = 65536.0;
        format.mFormatID = kAudioFormatLinearPCM;
        format.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
        format.mBytesPerPacket = 8;
        format.mFramesPerPacket = 1;
        format.mBytesPerFrame = 8;
        format.mChannelsPerFrame = 2;
        format.mBitsPerChannel = 32;
        if (AudioQueueNewOutput(&format, &RealtimeAudio::fill_buffer, this, nullptr, nullptr, 0,
                                &queue) != noErr ||
            queue == nullptr) {
            queue = nullptr;
            failed = true;
            return false;
        }
        const u32 bytes = kBufferFrames * sizeof(float) * 2;
        for (int i = 0; i < kBufferCount; ++i) {
            AudioQueueBufferRef buffer = nullptr;
            if (AudioQueueAllocateBuffer(queue, bytes, &buffer) != noErr || buffer == nullptr) {
                stop();
                failed = true;
                return false;
            }
            buffer->mAudioDataByteSize = bytes;
            fill_buffer(this, queue, buffer);
        }
        if (AudioQueueStart(queue, nullptr) != noErr) {
            stop();
            failed = true;
            return false;
        }
        return true;
    }

    void stop() {
        if (queue != nullptr) {
            AudioQueueStop(queue, true);
            AudioQueueDispose(queue, true);
            queue = nullptr;
        }
    }

    void reset() { bank.reset(); }

    void spawn_contact(const StreamContact& contact, f64 mu, f64 gl, f64 gr) {
        bank.spawn_contact(contact, mu, gl, gr);
    }
};

#elif defined(__linux__)

// Linux backend: ALSA push thread. The spec content is mixed at 65536 Hz by
// the shared VoiceBank; when the device will not run at the spec rate
// natively, each device frame is linearly interpolated between adjacent
// spec-rate samples (deterministic; viewer-path only — the offline --wav
// reference never resamples). Device format is float32 stereo, falling back
// to s16 stereo when the device rejects float; the s16 conversion uses the
// same rounding as the offline render.
struct RealtimeAudio {
    static constexpr int kVoiceCount = VoiceBank::kVoiceCount;
    static constexpr u32 kPeriodFrames = 1024;
    static constexpr int kPeriodsBuffered = 4;
    static constexpr int kMaxUnderrunRecoveries = 8;

    VoiceBank bank;
    snd_pcm_t* pcm = nullptr;
    std::thread thread;
    std::atomic<bool> running{false};
    bool use_s16 = false;
    bool failed = false;
    u32 output_rate = 65536;

    ~RealtimeAudio() { stop(); }

    RealtimeAudio() = default;
    RealtimeAudio(const RealtimeAudio&) = delete;
    RealtimeAudio& operator=(const RealtimeAudio&) = delete;

    bool configure_device() {
        snd_pcm_hw_params_t* params = nullptr;
        if (snd_pcm_hw_params_malloc(&params) < 0) return false;
        bool ok = snd_pcm_hw_params_any(pcm, params) >= 0 &&
                  snd_pcm_hw_params_set_access(pcm, params, SND_PCM_ACCESS_RW_INTERLEAVED) >= 0 &&
                  snd_pcm_hw_params_set_channels(pcm, params, 2) >= 0;
        use_s16 = false;
        if (ok && snd_pcm_hw_params_set_format(pcm, params, SND_PCM_FORMAT_FLOAT_LE) < 0) {
            use_s16 = true;
            ok = snd_pcm_hw_params_set_format(pcm, params, SND_PCM_FORMAT_S16_LE) >= 0;
        }
        unsigned rate = 65536;
        if (ok) {
            ok = snd_pcm_hw_params_set_rate_near(pcm, params, &rate, nullptr) >= 0 && rate > 0;
        }
        snd_pcm_uframes_t period = kPeriodFrames;
        if (ok) ok = snd_pcm_hw_params_set_period_size_near(pcm, params, &period, nullptr) >= 0;
        snd_pcm_uframes_t buffer = period * kPeriodsBuffered;
        if (ok) ok = snd_pcm_hw_params_set_buffer_size_near(pcm, params, &buffer) >= 0;
        if (ok) ok = snd_pcm_hw_params(pcm, params) >= 0;
        snd_pcm_hw_params_free(params);
        if (ok) output_rate = rate;
        return ok;
    }

    void run() {
        const double step = 65536.0 / static_cast<double>(output_rate);
        std::vector<float> floats(kPeriodFrames * 2);
        std::vector<short> shorts(use_s16 ? kPeriodFrames * 2 : 0);
        f64 l0 = 0.0, r0 = 0.0, l1 = 0.0, r1 = 0.0;
        {
            std::lock_guard<std::mutex> lock(bank.mutex);
            bank.mix_frame(l0, r0);
            bank.mix_frame(l1, r1);
        }
        double pos = 0.0;
        const u32 frame_bytes = use_s16 ? 4u : 8u;
        while (running.load(std::memory_order_acquire)) {
            for (u32 i = 0; i < kPeriodFrames; ++i) {
                while (pos >= 1.0) {
                    pos -= 1.0;
                    l0 = l1;
                    r0 = r1;
                    std::lock_guard<std::mutex> lock(bank.mutex);
                    bank.mix_frame(l1, r1);
                }
                floats[i * 2] = static_cast<float>(l0 + (l1 - l0) * pos);
                floats[i * 2 + 1] = static_cast<float>(r0 + (r1 - r0) * pos);
                pos += step;
            }
            const void* data = floats.data();
            if (use_s16) {
                for (u32 i = 0; i < kPeriodFrames * 2; ++i) {
                    shorts[i] = static_cast<short>(std::floor(floats[i] * 32767.0 + 0.5));
                }
                data = shorts.data();
            }
            std::size_t written = 0;
            int recoveries = 0;
            while (written < kPeriodFrames) {
                const snd_pcm_sframes_t n = snd_pcm_writei(
                    pcm, static_cast<const u8*>(data) + written * frame_bytes,
                    kPeriodFrames - written);
                if (n < 0) {
                    if (n == -EPIPE && recoveries < kMaxUnderrunRecoveries &&
                        snd_pcm_prepare(pcm) >= 0) {
                        ++recoveries;
                        continue;
                    }
                    return;
                }
                written += static_cast<std::size_t>(n);
                recoveries = 0;
            }
        }
    }

    bool start() {
        if (failed || pcm != nullptr) return pcm != nullptr;
        if (snd_pcm_open(&pcm, "default", SND_PCM_STREAM_PLAYBACK, 0) < 0 || pcm == nullptr) {
            pcm = nullptr;
            failed = true;
            return false;
        }
        if (!configure_device()) {
            snd_pcm_close(pcm);
            pcm = nullptr;
            failed = true;
            return false;
        }
        running.store(true, std::memory_order_release);
        try {
            thread = std::thread([this] { run(); });
        } catch (...) {
            running.store(false, std::memory_order_release);
            snd_pcm_close(pcm);
            pcm = nullptr;
            failed = true;
            return false;
        }
        return true;
    }

    void stop() {
        if (thread.joinable()) {
            running.store(false, std::memory_order_release);
            thread.join();
        }
        if (pcm != nullptr) {
            snd_pcm_drop(pcm);
            snd_pcm_close(pcm);
            pcm = nullptr;
        }
    }

    void reset() { bank.reset(); }

    void spawn_contact(const StreamContact& contact, f64 mu, f64 gl, f64 gr) {
        bank.spawn_contact(contact, mu, gl, gr);
    }
};

#endif

// Viewer-side stereo placement for a contact: constant-power pan from the
// contact's screen-x position plus inverse distance-squared-ish attenuation
// referenced to the visible half-height (on-screen ~1, falling off outside
// the view). Deterministic given the same stream + camera. These gains are a
// render choice like the view matrix — the spec-22 ring waveform itself
// contains no transcendentals.
void contact_gains(const Camera2D& camera, int fb_width, int fb_height, f64 cx, f64 cy,
                   f64& out_l, f64& out_r) {
    constexpr f64 kQuarterPi = 0.7853981633974483;
    const f64 ndc_x = (cx - camera.cx) * (2.0 * camera.zoom / std::max(1, fb_width));
    const f64 pan = std::clamp(ndc_x, -1.0, 1.0) * 0.7;
    const f64 angle = (pan + 1.0) * kQuarterPi;
    const f64 d = std::hypot(cx - camera.cx, cy - camera.cy);
    const f64 ref = std::max(1.0, static_cast<f64>(fb_height) / (2.0 * camera.zoom));
    const f64 att = (ref * ref) / (d * d + ref * ref);
    out_l = std::cos(angle) * att;
    out_r = std::sin(angle) * att;
}

#endif  // ONTOS_VIEW_REALTIME_AUDIO

// Contact visualization: a brief expanding ring billboard at the contact
// midpoint, fading over kFlashTicks playback ticks (no trails).
struct ContactFlash {
    float x = 0.0f, y = 0.0f;
    float half_extent = 1.0f;
    float t = 0.0f;
};

constexpr float kFlashTicks = 4.0f;
constexpr std::size_t kMaxFlashes = 64;

void fill_flash_instances(const std::vector<ContactFlash>& flashes, BodyInstance* instances) {
    for (std::size_t i = 0; i < flashes.size(); ++i) {
        const ContactFlash& flash = flashes[i];
        BodyInstance& inst = instances[i];
        const float fade = 1.0f - flash.t;
        inst.x = flash.x;
        inst.y = flash.y;
        inst.half_extent = flash.half_extent * (1.0f + 0.9f * flash.t);
        inst.shape = 2.0f;
        inst.r = 1.0f;
        inst.g = 0.82f;
        inst.b = 0.40f;
        inst.a = 0.85f * fade * fade;
    }
}

// In-window HUD: a stroked 5x7 glyph font rendered through the existing line
// pipeline plus panel/swatch quads through the body pipeline, both driven
// with a screen-space push so the overlay is fixed regardless of camera.
struct GlyphSeg {
    float x1, y1, x2, y2;
};

constexpr std::size_t kMaxHudLineVerts = 8192;
constexpr u32 kMaxHudQuads = 8;

// Segment coordinates are in font units: x in [0,4], y in [0,6], y down.
bool glyph_segments(char c, const GlyphSeg*& segs, int& count) {
    static const GlyphSeg kA[] = {{0,6,2,0},{2,0,4,6},{1,3,3,3}};
    static const GlyphSeg kB[] = {{0,0,0,6},{0,0,3,0},{3,0,4,1},{4,1,4,2},{4,2,3,3},
                                  {3,3,0,3},{3,3,4,4},{4,4,4,5},{4,5,3,6},{3,6,0,6}};
    static const GlyphSeg kC[] = {{4,1,3,0},{3,0,1,0},{1,0,0,1},{0,1,0,5},{0,5,1,6},
                                  {1,6,3,6},{3,6,4,5}};
    static const GlyphSeg kD[] = {{0,0,0,6},{0,0,2,0},{2,0,4,2},{4,2,4,4},{4,4,2,6},{2,6,0,6}};
    static const GlyphSeg kE[] = {{4,0,0,0},{0,0,0,6},{0,6,4,6},{0,3,3,3}};
    static const GlyphSeg kF[] = {{4,0,0,0},{0,0,0,6},{0,3,3,3}};
    static const GlyphSeg kG[] = {{4,1,3,0},{3,0,1,0},{1,0,0,1},{0,1,0,5},{0,5,1,6},
                                  {1,6,3,6},{3,6,4,5},{4,5,4,3},{4,3,2,3}};
    static const GlyphSeg kH[] = {{0,0,0,6},{4,0,4,6},{0,3,4,3}};
    static const GlyphSeg kI[] = {{1,0,3,0},{2,0,2,6},{1,6,3,6}};
    static const GlyphSeg kJ[] = {{1,0,3,0},{3,0,3,5},{3,5,2,6},{2,6,1,6},{1,6,0,5}};
    static const GlyphSeg kK[] = {{0,0,0,6},{4,0,0,3},{0,3,4,6}};
    static const GlyphSeg kL[] = {{0,0,0,6},{0,6,4,6}};
    static const GlyphSeg kM[] = {{0,6,0,0},{0,0,2,2},{2,2,4,0},{4,0,4,6}};
    static const GlyphSeg kN[] = {{0,6,0,0},{0,0,4,6},{4,6,4,0}};
    static const GlyphSeg kO[] = {{1,0,3,0},{3,0,4,1},{4,1,4,5},{4,5,3,6},{3,6,1,6},
                                  {1,6,0,5},{0,5,0,1},{0,1,1,0}};
    static const GlyphSeg kP[] = {{0,6,0,0},{0,0,3,0},{3,0,4,1},{4,1,4,2},{4,2,3,3},{3,3,0,3}};
    static const GlyphSeg kQ[] = {{1,0,3,0},{3,0,4,1},{4,1,4,5},{4,5,3,6},{3,6,1,6},
                                  {1,6,0,5},{0,5,0,1},{0,1,1,0},{2,4,4,6}};
    static const GlyphSeg kR[] = {{0,6,0,0},{0,0,3,0},{3,0,4,1},{4,1,4,2},{4,2,3,3},
                                  {3,3,0,3},{2,3,4,6}};
    static const GlyphSeg kS[] = {{4,1,3,0},{3,0,1,0},{1,0,0,1},{0,1,0,2},{0,2,1,3},
                                  {1,3,3,3},{3,3,4,4},{4,4,4,5},{4,5,3,6},{3,6,1,6},{1,6,0,5}};
    static const GlyphSeg kT[] = {{0,0,4,0},{2,0,2,6}};
    static const GlyphSeg kU[] = {{0,0,0,5},{0,5,1,6},{1,6,3,6},{3,6,4,5},{4,5,4,0}};
    static const GlyphSeg kV[] = {{0,0,2,6},{2,6,4,0}};
    static const GlyphSeg kW[] = {{0,0,0,6},{0,6,2,4},{2,4,4,6},{4,6,4,0}};
    static const GlyphSeg kX[] = {{0,0,4,6},{4,0,0,6}};
    static const GlyphSeg kY[] = {{0,0,2,2},{4,0,2,2},{2,2,2,6}};
    static const GlyphSeg kZ[] = {{0,0,4,0},{4,0,0,6},{0,6,4,6}};
    static const GlyphSeg k0[] = {{1,0,3,0},{3,0,4,1},{4,1,4,5},{4,5,3,6},{3,6,1,6},
                                  {1,6,0,5},{0,5,0,1},{0,1,1,0},{1,5,4,1}};
    static const GlyphSeg k1[] = {{1,1,2,0},{2,0,2,6},{1,6,3,6}};
    static const GlyphSeg k2[] = {{0,1,1,0},{1,0,3,0},{3,0,4,1},{4,1,4,2},{4,2,0,6},{0,6,4,6}};
    static const GlyphSeg k3[] = {{0,1,1,0},{1,0,3,0},{3,0,4,1},{4,1,4,5},{4,5,3,6},
                                  {3,6,1,6},{1,6,0,5},{1,3,4,3}};
    static const GlyphSeg k4[] = {{3,0,3,6},{0,3,4,3},{0,3,3,0}};
    static const GlyphSeg k5[] = {{4,0,0,0},{0,0,0,3},{0,3,3,3},{3,3,4,4},{4,4,4,5},
                                  {4,5,3,6},{3,6,1,6},{1,6,0,5}};
    static const GlyphSeg k6[] = {{4,1,3,0},{3,0,1,0},{1,0,0,1},{0,1,0,5},{0,5,1,6},
                                  {1,6,3,6},{3,6,4,5},{4,5,4,4},{4,4,3,3},{3,3,1,3},{1,3,0,4}};
    static const GlyphSeg k7[] = {{0,0,4,0},{4,0,1,6}};
    static const GlyphSeg k8[] = {{1,0,3,0},{3,0,4,1},{4,1,4,2},{4,2,3,3},{3,3,1,3},
                                  {1,3,0,2},{0,2,0,1},{0,1,1,0},{0,4,0,5},{0,5,1,6},
                                  {1,6,3,6},{3,6,4,5},{4,5,4,4},{4,4,3,3},{1,3,0,4}};
    static const GlyphSeg k9[] = {{1,0,3,0},{3,0,4,1},{4,1,4,2},{4,2,3,3},{3,3,1,3},
                                  {1,3,0,2},{0,2,0,1},{0,1,1,0},{3,3,4,4},{4,4,4,5},
                                  {4,5,3,6},{3,6,1,6},{1,6,0,5}};
    static const GlyphSeg kSlash[] = {{4,0,0,6}};
    static const GlyphSeg kMinus[] = {{1,3,3,3}};

    switch (c) {
        case 'A': segs = kA; count = 3; return true;
        case 'B': segs = kB; count = 10; return true;
        case 'C': segs = kC; count = 7; return true;
        case 'D': segs = kD; count = 6; return true;
        case 'E': segs = kE; count = 4; return true;
        case 'F': segs = kF; count = 3; return true;
        case 'G': segs = kG; count = 9; return true;
        case 'H': segs = kH; count = 3; return true;
        case 'I': segs = kI; count = 3; return true;
        case 'J': segs = kJ; count = 5; return true;
        case 'K': segs = kK; count = 3; return true;
        case 'L': segs = kL; count = 2; return true;
        case 'M': segs = kM; count = 4; return true;
        case 'N': segs = kN; count = 3; return true;
        case 'O': segs = kO; count = 8; return true;
        case 'P': segs = kP; count = 6; return true;
        case 'Q': segs = kQ; count = 9; return true;
        case 'R': segs = kR; count = 7; return true;
        case 'S': segs = kS; count = 11; return true;
        case 'T': segs = kT; count = 2; return true;
        case 'U': segs = kU; count = 5; return true;
        case 'V': segs = kV; count = 2; return true;
        case 'W': segs = kW; count = 4; return true;
        case 'X': segs = kX; count = 2; return true;
        case 'Y': segs = kY; count = 3; return true;
        case 'Z': segs = kZ; count = 3; return true;
        case '0': segs = k0; count = 9; return true;
        case '1': segs = k1; count = 3; return true;
        case '2': segs = k2; count = 6; return true;
        case '3': segs = k3; count = 8; return true;
        case '4': segs = k4; count = 3; return true;
        case '5': segs = k5; count = 8; return true;
        case '6': segs = k6; count = 11; return true;
        case '7': segs = k7; count = 2; return true;
        case '8': segs = k8; count = 15; return true;
        case '9': segs = k9; count = 13; return true;
        case '/': segs = kSlash; count = 1; return true;
        case '-': segs = kMinus; count = 1; return true;
        case ' ': segs = nullptr; count = 0; return true;
        default: return false;
    }
}

// Appends line-list vertices (pixel space, y down) for `text` starting at
// (ox, oy) with font unit `u`. Bounded by kMaxHudLineVerts total.
void append_text(std::vector<float>& verts, const char* text, float ox, float oy, float u) {
    float x = ox;
    for (const char* p = text; *p != '\0'; ++p) {
        const char c = *p >= 'a' && *p <= 'z' ? static_cast<char>(*p - 'a' + 'A') : *p;
        const GlyphSeg* segs = nullptr;
        int count = 0;
        if (glyph_segments(c, segs, count)) {
            for (int i = 0; i < count; ++i) {
                if (verts.size() / 2 + 1 >= kMaxHudLineVerts) return;
                verts.push_back(x + segs[i].x1 * u);
                verts.push_back(oy + segs[i].y1 * u);
                verts.push_back(x + segs[i].x2 * u);
                verts.push_back(oy + segs[i].y2 * u);
            }
        }
        x += 6.0f * u;
    }
}

// Builds one frame of HUD content: a translucent backing panel (written as
// explicit pixel-space corners, since the shared quad path is isotropic),
// text rows, and the region-color legend swatches. `text` receives line
// vertices; [accent_begin, accent_end) brackets the PAUSED row drawn in
// amber; `quads` receives the swatch instances.
void fill_hud(const StreamFrame& frame, std::size_t total_frames, uint32_t frames_per_tick,
              bool paused, double ema_ms, int hud_scale, std::vector<float>& text,
              uint32_t& accent_begin, uint32_t& accent_end, float* panel_verts,
              BodyInstance* quads, uint32_t& quad_count) {
    text.clear();
    accent_begin = 0;
    accent_end = 0;
    quad_count = 0;

    const float u = 2.0f * static_cast<float>(hud_scale);
    const float char_w = 6.0f * u;
    const float line_h = 9.0f * u;
    const float pad = 5.0f * u;
    const float margin = 6.0f * u;

    char l1[48];
    char l2[48];
    char l3[48];
    std::snprintf(l1, sizeof l1, "TICK %" PRIu64 "/%zu", frame.tick, total_frames);
    std::snprintf(l2, sizeof l2, "FINE %" PRIu64 "  COARSE %" PRIu64, frame.fine, frame.coarse);
    std::snprintf(l3, sizeof l3, "RATE 1/%u  %.0fMS", frames_per_tick, ema_ms);

    static const char* kLegendLabels[5] = {"R0", "R1", "R2", "R3", "EXT"};
    const u8 legend_regions[5] = {0, 1, 2, 3, 255};
    const float swatch_r = 3.5f * u;
    const float label_gap = 1.0f * u;
    const float group_gap = 3.0f * u;
    float legend_w = 0.0f;
    for (int i = 0; i < 5; ++i) {
        legend_w += 2.0f * swatch_r + label_gap + std::strlen(kLegendLabels[i]) * char_w;
        if (i < 4) legend_w += group_gap;
    }

    const float text_w =
        std::max({std::strlen(l1), std::strlen(l2), std::strlen(l3),
                  std::strlen("DIM=COARSE"), paused ? std::strlen("PAUSED") : std::size_t(0)}) *
        char_w;
    const uint32_t rows = 3 + (paused ? 1 : 0) + 2;
    const float panel_w = std::max(text_w, legend_w) + 2.0f * pad;
    const float panel_h = static_cast<float>(rows) * line_h + 2.0f * pad;
    // Corner order matches the shared quad_corners triangle strip.
    panel_verts[0] = margin;
    panel_verts[1] = margin;
    panel_verts[2] = margin + panel_w;
    panel_verts[3] = margin;
    panel_verts[4] = margin;
    panel_verts[5] = margin + panel_h;
    panel_verts[6] = margin + panel_w;
    panel_verts[7] = margin + panel_h;

    float y = margin + pad;
    const float x0 = margin + pad;
    append_text(text, l1, x0, y, u);
    y += line_h;
    append_text(text, l2, x0, y, u);
    y += line_h;
    append_text(text, l3, x0, y, u);
    y += line_h;
    if (paused) {
        accent_begin = static_cast<uint32_t>(text.size() / 2);
        append_text(text, "PAUSED", x0, y, u);
        accent_end = static_cast<uint32_t>(text.size() / 2);
        y += line_h;
    }

    const float swatch_cy = y + 3.5f * u;
    float x = x0;
    for (int i = 0; i < 5; ++i) {
        float rgb[3];
        region_fill_color(legend_regions[i], 1, rgb);
        BodyInstance& swatch = quads[quad_count++];
        swatch.x = x + swatch_r;
        swatch.y = swatch_cy;
        swatch.half_extent = swatch_r;
        swatch.shape = 1.0f;
        swatch.r = rgb[0];
        swatch.g = rgb[1];
        swatch.b = rgb[2];
        swatch.a = 0.95f;
        append_text(text, kLegendLabels[i], x + 2.0f * swatch_r + label_gap, y, u);
        x += 2.0f * swatch_r + label_gap + std::strlen(kLegendLabels[i]) * char_w + group_gap;
    }
    y += line_h;
    append_text(text, "DIM=COARSE", x0, y, u);
}

void print_usage() {
    std::fprintf(stderr,
                 "usage: ontos_view <stream-file> [--validate] [--frames N] [--wav FILE]\n"
                 "  --validate   enable Vulkan validation layers\n"
                 "  --frames N   non-interactive: render N frames (one tick per frame),\n"
                 "               print per-tick body counts, exit 0\n"
                 "  --wav FILE   render the spec-22 modal audio offline (deterministic\n"
                 "               WAV + FNV hash; the mono spec-22 reference)\n"
                 "  interactive playback also plays the contact rings through the audio\n"
                 "  device live, stereo-panned and attenuated from the camera view\n"
                 "  keys: SPACE pause  +/- rate  LEFT/RIGHT step one tick  R restart\n"
                 "        F fit view  H toggle hud  ESC quit  drag pan  wheel zoom  WASD pan\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::filesystem::path stream_path;
    bool seed_seen = false;
    bool validate = false;
    bool frames_requested = false;
    uint32_t frame_limit = 0;
    const char* wav_path = nullptr;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--validate") {
            validate = true;
        } else if (arg == "--frames" && i + 1 < argc) {
            frame_limit = static_cast<uint32_t>(std::atoi(argv[++i]));
            frames_requested = true;
        } else if (arg == "--wav" && i + 1 < argc) {
            wav_path = argv[++i];
        } else if (!arg.empty() && arg[0] != '-') {
            if (stream_path.empty()) {
                stream_path = argv[i];
            } else {
                // Optional positional seed, accepted and ignored: the viewer
                // plays back records only and never re-simulates.
                seed_seen = true;
            }
        } else {
            print_usage();
            return 1;
        }
    }
    (void)seed_seen;
    if (stream_path.empty() || (frames_requested && frame_limit == 0)) {
        print_usage();
        return 1;
    }

    GLFWwindow* window = nullptr;
    try {
        const Stream stream = parse_stream(stream_path);

        u64 total_contacts = 0;
        for (const StreamFrame& frame : stream.frames) {
            total_contacts += frame.contacts.size();
        }
        if (wav_path != nullptr) {
            const u64 digest = render_contact_audio(stream, wav_path, true);
            std::printf("audio: contacts=%" PRIu64 " wav=%s hash=%016" PRIx64 "\n",
                        total_contacts, wav_path, digest);
        }

#if ONTOS_VIEW_REALTIME_AUDIO
        const StreamFrame& final_frame = stream.frames.back();
        RealtimeAudio audio;
        bool audio_running = false;
        if (!frames_requested && total_contacts > 0) {
            audio_running = audio.start();
            if (audio_running) {
                char rate_note[64] = "";
                if (audio.output_rate != 65536u) {
                    std::snprintf(rate_note, sizeof rate_note,
                                  " -> device %u Hz linear-resampled", audio.output_rate);
                }
                std::printf("audio: realtime stereo output open (65536 Hz%s, %d voices)\n",
                            rate_note, RealtimeAudio::kVoiceCount);
            } else {
                std::fprintf(stderr, "audio: realtime output unavailable; continuing silent\n");
            }
        }
#else
        if (!frames_requested && total_contacts > 0) {
            std::fprintf(stderr,
                          "audio: realtime output requires macOS CoreAudio or Linux ALSA;"
                          " continuing silent\n");
        }
#endif

        configure_macos_moltenvk_environment();

        if (glfwInit() != GLFW_TRUE) {
            throw std::runtime_error("glfwInit failed");
        }
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        // Hidden windows cannot back a MoltenVK surface on macOS, so the
        // non-interactive mode still shows the window; it closes itself after
        // the requested frame count.
        glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
        window = glfwCreateWindow(1280, 720, "ontos_view", nullptr, nullptr);
        if (window == nullptr) {
            glfwTerminate();
            throw std::runtime_error("glfwCreateWindow failed");
        }

        WindowState state;
        glfwSetWindowUserPointer(window, &state);
        glfwSetFramebufferSizeCallback(
            window, [](GLFWwindow* w, int width, int height) {
                WindowState* s = static_cast<WindowState*>(glfwGetWindowUserPointer(w));
                s->resized = true;
                if (width > 0 && height > 0) {
                    s->fb_width = width;
                    s->fb_height = height;
                }
            });
        glfwSetScrollCallback(window, [](GLFWwindow* w, double, double yoffset) {
            WindowState* s = static_cast<WindowState*>(glfwGetWindowUserPointer(w));
            Camera2D& cam = s->camera;
            double mx = 0.0;
            double my = 0.0;
            glfwGetCursorPos(w, &mx, &my);
            int win_w = 0;
            int win_h = 0;
            glfwGetWindowSize(w, &win_w, &win_h);
            if (win_w <= 0 || win_h <= 0) return;
            // Cursor position arrives in window coordinates (points); NDC
            // spans the same normalized extent regardless of framebuffer
            // DPI, so the anchor must be computed against the window size,
            // not fb_width/fb_height (wrong anchor on retina displays).
            const double ndc_x = 2.0 * mx / win_w - 1.0;
            const double ndc_y = 1.0 - 2.0 * my / win_h;
            const double wx = cam.cx + ndc_x * s->fb_width / (2.0 * cam.zoom);
            const double wy = cam.cy + ndc_y * s->fb_height / (2.0 * cam.zoom);
            cam.zoom *= std::pow(1.15, -yoffset);
            cam.zoom = std::clamp(cam.zoom, 0.05, 40.0);
            cam.cx = wx - ndc_x * s->fb_width / (2.0 * cam.zoom);
            cam.cy = wy - ndc_y * s->fb_height / (2.0 * cam.zoom);
        });

        uint32_t available_extension_count = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &available_extension_count, nullptr);
        std::vector<VkExtensionProperties> available_extensions(available_extension_count);
        if (available_extension_count > 0) {
            vkEnumerateInstanceExtensionProperties(nullptr, &available_extension_count,
                                                   available_extensions.data());
        }
        const auto supports_extension = [&available_extensions](const char* name) {
            for (const VkExtensionProperties& extension : available_extensions) {
                if (std::strcmp(extension.extensionName, name) == 0) return true;
            }
            return false;
        };

        uint32_t glfw_extension_count = 0;
        const char** glfw_extensions =
            glfwGetRequiredInstanceExtensions(&glfw_extension_count);
        std::vector<const char*> instance_extensions;
        if (glfw_extensions != nullptr && glfw_extension_count > 0) {
            instance_extensions.assign(glfw_extensions, glfw_extensions + glfw_extension_count);
        } else {
#if defined(__APPLE__)
            // This Homebrew GLFW reports no Vulkan support; mirror
            // meridian_vk_bootstrap and drive the metal surface by hand.
            instance_extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
            if (supports_extension(VK_EXT_METAL_SURFACE_EXTENSION_NAME)) {
                instance_extensions.push_back(VK_EXT_METAL_SURFACE_EXTENSION_NAME);
            } else {
                throw std::runtime_error("required macOS metal surface extension is not available");
            }
#else
            throw std::runtime_error("GLFW did not report required Vulkan instance extensions");
#endif
        }
        // VK_KHR_get_physical_device_properties2 is core since Vulkan 1.1
        // and the requested apiVersion below is 1.2, so the extension name
        // is not requested at all (mirrors the main renderer's LS-39 fix) --
        // loaders that only expose it as an extension must not fail
        // instance creation over it.
        bool portability_enumeration = false;
        if (supports_extension(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
            instance_extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            portability_enumeration = true;
        }

        std::vector<const char*> layers;
        if (validate) {
            uint32_t layer_count = 0;
            vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
            std::vector<VkLayerProperties> available(layer_count);
            vkEnumerateInstanceLayerProperties(&layer_count, available.data());
            for (const VkLayerProperties& layer : available) {
                if (std::strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
                    layers.push_back("VK_LAYER_KHRONOS_validation");
                    instance_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
                    break;
                }
            }
            if (layers.empty()) {
                std::fprintf(stderr, "warning: validation layer not found, continuing without\n");
                validate = false;
            }
        }

        VkApplicationInfo app_info{};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "ontos_view";
        app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
        app_info.pEngineName = "light-system";
        app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
        app_info.apiVersion = VK_API_VERSION_1_2;

        VkInstanceCreateInfo instance_info{};
        instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instance_info.pApplicationInfo = &app_info;
        instance_info.enabledExtensionCount = static_cast<uint32_t>(instance_extensions.size());
        instance_info.ppEnabledExtensionNames = instance_extensions.data();
        instance_info.enabledLayerCount = static_cast<uint32_t>(layers.size());
        instance_info.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();
        // The portability enumeration flag is only valid together with the
        // VK_KHR_portability_enumeration extension; setting it when the
        // extension was not enabled fails instance creation.
        if (portability_enumeration) {
            instance_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        }

        Viewer v;
        VkResult result = vkCreateInstance(&instance_info, nullptr, &v.instance);
        if (result != VK_SUCCESS) {
            v.instance = VK_NULL_HANDLE;
            throw std::runtime_error("vkCreateInstance failed");
        }

        if (validate) {
            auto create_messenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(v.instance, "vkCreateDebugUtilsMessengerEXT"));
            if (create_messenger != nullptr) {
                VkDebugUtilsMessengerCreateInfoEXT messenger_info{};
                messenger_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
                messenger_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
                messenger_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
                messenger_info.pfnUserCallback = &debug_callback;
                create_messenger(v.instance, &messenger_info, nullptr, &v.debug_messenger);
            }
        }

        result = VK_ERROR_INITIALIZATION_FAILED;
#if defined(__APPLE__)
        {
            NSWindow* cocoa_window = glfwGetCocoaWindow(window);
            if (cocoa_window != nil) {
                NSView* cocoa_view = [cocoa_window contentView];
                if (cocoa_view != nil) {
                    [cocoa_view setWantsLayer:YES];
                    CAMetalLayer* metal_layer = [CAMetalLayer layer];
                    [cocoa_view setLayer:metal_layer];
                    VkMetalSurfaceCreateInfoEXT create_info{};
                    create_info.sType = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT;
                    create_info.pLayer = metal_layer;
                    result = vkCreateMetalSurfaceEXT(v.instance, &create_info, nullptr,
                                                     &v.surface);
                }
            }
        }
#else
        result = glfwCreateWindowSurface(v.instance, window, nullptr, &v.surface);
#endif
        if (result != VK_SUCCESS) {
            v.surface = VK_NULL_HANDLE;
            std::ostringstream message;
            message << "surface creation failed with code " << result;
            throw std::runtime_error(message.str());
        }

        uint32_t device_count = 0;
        vkEnumeratePhysicalDevices(v.instance, &device_count, nullptr);
        if (device_count == 0) throw std::runtime_error("no Vulkan physical devices");
        std::vector<VkPhysicalDevice> devices(device_count);
        vkEnumeratePhysicalDevices(v.instance, &device_count, devices.data());

        bool found = false;
        for (VkPhysicalDevice candidate : devices) {
            uint32_t family_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, nullptr);
            std::vector<VkQueueFamilyProperties> families(family_count);
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, families.data());
            for (uint32_t i = 0; i < family_count; ++i) {
                if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) continue;
                VkBool32 present_supported = VK_FALSE;
                vkGetPhysicalDeviceSurfaceSupportKHR(candidate, i, v.surface, &present_supported);
                if (present_supported != VK_TRUE) continue;

                uint32_t extension_count = 0;
                vkEnumerateDeviceExtensionProperties(candidate, nullptr, &extension_count,
                                                     nullptr);
                std::vector<VkExtensionProperties> extensions(extension_count);
                vkEnumerateDeviceExtensionProperties(candidate, nullptr, &extension_count,
                                                     extensions.data());
                bool has_swapchain = false;
                bool has_portability_subset = false;
                for (const VkExtensionProperties& extension : extensions) {
                    if (std::strcmp(extension.extensionName,
                                    VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0) {
                        has_swapchain = true;
                    } else if (std::strcmp(extension.extensionName,
                                           "VK_KHR_portability_subset") == 0) {
                        has_portability_subset = true;
                    }
                }
                if (!has_swapchain) continue;

                const float priority = 1.0f;
                VkDeviceQueueCreateInfo queue_info{};
                queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
                queue_info.queueFamilyIndex = i;
                queue_info.queueCount = 1;
                queue_info.pQueuePriorities = &priority;

                std::vector<const char*> device_extensions;
                device_extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
                if (has_portability_subset) {
                    device_extensions.push_back("VK_KHR_portability_subset");
                }

                VkDeviceCreateInfo device_info{};
                device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
                device_info.queueCreateInfoCount = 1;
                device_info.pQueueCreateInfos = &queue_info;
                device_info.enabledExtensionCount =
                    static_cast<uint32_t>(device_extensions.size());
                device_info.ppEnabledExtensionNames = device_extensions.data();

                result = vkCreateDevice(candidate, &device_info, nullptr, &v.device);
                if (result != VK_SUCCESS) {
                    v.device = VK_NULL_HANDLE;
                    continue;
                }
                v.physical_device = candidate;
                v.graphics_family = i;
                vkGetDeviceQueue(v.device, i, 0, &v.queue);
                found = true;
                break;
            }
            if (found) break;
        }
        if (!found) throw std::runtime_error("no compatible Vulkan device found");

        // Pick a depth format and the MSAA sample count supported for it.
        {
            const VkFormat depth_candidates[] = {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D24_UNORM_S8_UINT,
                                                 VK_FORMAT_D16_UNORM};
            bool depth_ok = false;
            for (VkFormat candidate : depth_candidates) {
                VkFormatProperties properties{};
                vkGetPhysicalDeviceFormatProperties(v.physical_device, candidate, &properties);
                if ((properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) !=
                    0) {
                    v.depth_format = candidate;
                    depth_ok = true;
                    break;
                }
            }
            if (!depth_ok) throw std::runtime_error("no supported depth format");

            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(v.physical_device, &properties);
            const VkSampleCountFlags color_samples = properties.limits.framebufferColorSampleCounts;
            const VkSampleCountFlags depth_samples = properties.limits.framebufferDepthSampleCounts;
            const VkSampleCountFlags shared = color_samples & depth_samples;
            v.samples = VK_SAMPLE_COUNT_1_BIT;
            for (VkSampleCountFlagBits candidate : {VK_SAMPLE_COUNT_8_BIT, VK_SAMPLE_COUNT_4_BIT,
                                                    VK_SAMPLE_COUNT_2_BIT}) {
                if ((shared & candidate) != 0) {
                    v.samples = candidate;
                    break;
                }
            }
        }

        {
            // Attachments: [0] color (MSAA when enabled, else the swapchain
            // image directly), [1] resolve target (swapchain, MSAA path only),
            // [2] depth (MSAA sample count or 1).
            VkAttachmentDescription attachments[3]{};
            attachments[0].format = v.swapchain_format;
            attachments[0].samples = v.samples;
            attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            attachments[0].storeOp = v.samples != VK_SAMPLE_COUNT_1_BIT
                                         ? VK_ATTACHMENT_STORE_OP_DONT_CARE
                                         : VK_ATTACHMENT_STORE_OP_STORE;
            attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            attachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            attachments[1].format = v.swapchain_format;
            attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
            attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            attachments[1].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

            attachments[2].format = v.depth_format;
            attachments[2].samples = v.samples;
            attachments[2].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            attachments[2].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachments[2].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachments[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachments[2].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            attachments[2].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            const uint32_t attachment_count = v.samples != VK_SAMPLE_COUNT_1_BIT ? 3 : 2;
            VkAttachmentReference color_ref{};
            color_ref.attachment = 0;
            color_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            VkAttachmentReference resolve_ref{};
            resolve_ref.attachment = 1;
            resolve_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            VkAttachmentReference depth_ref{};
            depth_ref.attachment = v.samples != VK_SAMPLE_COUNT_1_BIT ? 2 : 1;
            depth_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            VkSubpassDescription subpass{};
            subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.colorAttachmentCount = 1;
            subpass.pColorAttachments = &color_ref;
            subpass.pResolveAttachments =
                v.samples != VK_SAMPLE_COUNT_1_BIT ? &resolve_ref : nullptr;
            subpass.pDepthStencilAttachment = &depth_ref;

            VkSubpassDependency dependency{};
            dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
            dependency.dstSubpass = 0;
            dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependency.srcAccessMask = 0;
            dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

            VkRenderPassCreateInfo render_pass_info{};
            render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            render_pass_info.attachmentCount = attachment_count;
            render_pass_info.pAttachments = attachments;
            render_pass_info.subpassCount = 1;
            render_pass_info.pSubpasses = &subpass;
            render_pass_info.dependencyCount = 1;
            render_pass_info.pDependencies = &dependency;

            result = vkCreateRenderPass(v.device, &render_pass_info, nullptr, &v.render_pass);
            if (result != VK_SUCCESS) throw std::runtime_error("vkCreateRenderPass failed");
        }

        {
            VkPushConstantRange range{};
            range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
            range.offset = 0;
            range.size = sizeof(ViewPush);
            VkPipelineLayoutCreateInfo layout_info{};
            layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            layout_info.pushConstantRangeCount = 1;
            layout_info.pPushConstantRanges = &range;
            result = vkCreatePipelineLayout(v.device, &layout_info, nullptr, &v.pipeline_layout);
            if (result != VK_SUCCESS) throw std::runtime_error("vkCreatePipelineLayout failed");
        }

        v.body_pipeline =
            create_pipeline(v, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, /*depth_write=*/false);
        v.line_pipeline =
            create_pipeline(v, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, /*depth_write=*/true);
        create_swapchain(v, window);
        create_extent_resources(v);

        {
            VkCommandPoolCreateInfo pool_info{};
            pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            pool_info.queueFamilyIndex = v.graphics_family;
            result = vkCreateCommandPool(v.device, &pool_info, nullptr, &v.command_pool);
            if (result != VK_SUCCESS) throw std::runtime_error("vkCreateCommandPool failed");

            VkCommandBufferAllocateInfo allocate{};
            allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocate.commandPool = v.command_pool;
            allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocate.commandBufferCount = 1;
            result = vkAllocateCommandBuffers(v.device, &allocate, &v.command_buffer);
            if (result != VK_SUCCESS) throw std::runtime_error("vkAllocateCommandBuffers failed");

            VkSemaphoreCreateInfo semaphore_info{};
            semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            result = vkCreateSemaphore(v.device, &semaphore_info, nullptr, &v.image_available);
            if (result != VK_SUCCESS) throw std::runtime_error("vkCreateSemaphore failed");
            v.render_finished_per_image.resize(v.images.size(), VK_NULL_HANDLE);
            for (uint32_t i = 0; i < v.render_finished_per_image.size(); ++i) {
                result = vkCreateSemaphore(v.device, &semaphore_info, nullptr,
                                           &v.render_finished_per_image[i]);
                if (result != VK_SUCCESS) throw std::runtime_error("vkCreateSemaphore failed");
            }

            VkFenceCreateInfo fence_info{};
            fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            result = vkCreateFence(v.device, &fence_info, nullptr, &v.in_flight);
            if (result != VK_SUCCESS) throw std::runtime_error("vkCreateFence failed");

            const float quad_corners[4][2] = {
                {-1.0f, -1.0f}, {1.0f, -1.0f}, {-1.0f, 1.0f}, {1.0f, 1.0f}};
            create_host_buffer(v.physical_device, v.device, quad_corners, sizeof(quad_corners),
                               VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, false, v.quad_buffer);

            const float grid_lines[12][2] = {
                {0.0f, 0.0f},     {128.0f, 0.0f},    {128.0f, 0.0f},   {128.0f, 128.0f},
                {128.0f, 128.0f}, {0.0f, 128.0f},    {0.0f, 128.0f},   {0.0f, 0.0f},
                {64.0f, 0.0f},    {64.0f, 128.0f},   {0.0f, 64.0f},    {128.0f, 64.0f},
            };
            create_host_buffer(v.physical_device, v.device, grid_lines, sizeof(grid_lines),
                               VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, false, v.line_buffer);

            // Lines pass absolute world coordinates through the corner
            // attribute, so the instance scale must be the identity 1.0 (a
            // zero half_extent collapses every vertex onto in_position and
            // silently culls the whole draw -- this is why the region grid
            // never rendered before).
            BodyInstance line_instance{};
            line_instance.half_extent = 1.0f;
            line_instance.shape = 0.0f;
            line_instance.r = 0.45f;
            line_instance.g = 0.47f;
            line_instance.b = 0.52f;
            line_instance.a = 0.35f;
            create_host_buffer(v.physical_device, v.device, &line_instance, sizeof(BodyInstance),
                               VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, false, v.line_instance_buffer);

            // Faint 8-unit graph-paper grid under the major region lines,
            // skipping the axes the major grid already draws (0, 64, 128).
            std::vector<float> minor_grid;
            for (uint32_t a = 8; a < 128; a += 8) {
                if (a == 64) continue;
                minor_grid.push_back(static_cast<float>(a));
                minor_grid.push_back(0.0f);
                minor_grid.push_back(static_cast<float>(a));
                minor_grid.push_back(128.0f);
                minor_grid.push_back(0.0f);
                minor_grid.push_back(static_cast<float>(a));
                minor_grid.push_back(128.0f);
                minor_grid.push_back(static_cast<float>(a));
            }
            create_host_buffer(v.physical_device, v.device, minor_grid.data(),
                               minor_grid.size() * sizeof(float),
                               VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, false, v.minor_line_buffer);

            BodyInstance minor_line_instance{};
            minor_line_instance.half_extent = 1.0f;
            minor_line_instance.shape = 0.0f;
            minor_line_instance.r = 0.45f;
            minor_line_instance.g = 0.47f;
            minor_line_instance.b = 0.52f;
            minor_line_instance.a = 0.12f;
            create_host_buffer(v.physical_device, v.device, &minor_line_instance,
                               sizeof(BodyInstance), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, false,
                               v.minor_line_instance_buffer);

            create_host_buffer(v.physical_device, v.device, nullptr,
                               static_cast<VkDeviceSize>(kMaxHudLineVerts) * 2 * sizeof(float),
                               VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, v.hud_line_buffer);
            create_host_buffer(v.physical_device, v.device, nullptr, 8 * sizeof(float),
                               VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, v.hud_panel_buffer);
            create_host_buffer(v.physical_device, v.device, nullptr,
                               static_cast<VkDeviceSize>(kMaxHudQuads) * sizeof(BodyInstance),
                               VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, v.hud_quad_instance_buffer);
            // [0] HUD text (light gray), [1] HUD accent text (amber),
            // [2] identity instance for the panel corner-quad draw. Text
            // vertices are absolute pixel coordinates in the corner slot,
            // so the scale must be 1.0 (see line_instance above).
            BodyInstance hud_instances[3] = {};
            hud_instances[0].half_extent = 1.0f;
            hud_instances[0].shape = 0.0f;
            hud_instances[0].r = 0.87f;
            hud_instances[0].g = 0.89f;
            hud_instances[0].b = 0.92f;
            hud_instances[0].a = 0.9f;
            hud_instances[1].half_extent = 1.0f;
            hud_instances[1].shape = 0.0f;
            hud_instances[1].r = 1.0f;
            hud_instances[1].g = 0.78f;
            hud_instances[1].b = 0.35f;
            hud_instances[1].a = 0.95f;
            hud_instances[2].x = 0.0f;
            hud_instances[2].y = 0.0f;
            hud_instances[2].half_extent = 1.0f;
            hud_instances[2].shape = 0.0f;
            hud_instances[2].r = 0.02f;
            hud_instances[2].g = 0.03f;
            hud_instances[2].b = 0.05f;
            hud_instances[2].a = 0.55f;
            create_host_buffer(v.physical_device, v.device, hud_instances,
                               sizeof(hud_instances), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, false,
                               v.hud_line_instance_buffer);

            for (uint32_t slot = 0; slot < 2; ++slot) {
                create_host_buffer(v.physical_device, v.device, nullptr,
                                   static_cast<VkDeviceSize>(stream.body_count) *
                                       sizeof(BodyInstance),
                                   VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true,
                                   v.body_instance_buffers[slot]);
                create_host_buffer(v.physical_device, v.device, nullptr,
                                   static_cast<VkDeviceSize>(kMaxFlashes) * sizeof(BodyInstance),
                                   VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true,
                                   v.contact_instance_buffers[slot]);
            }
        }

        glfwGetFramebufferSize(window, &state.fb_width, &state.fb_height);
        if (state.fb_width > 0 && state.fb_height > 0) {
            state.camera.zoom = 0.9 * std::min(state.fb_width, state.fb_height) / 128.0;
        }

        uint32_t frames_per_tick = 8;
        uint32_t subframe = 0;
        bool paused = false;
        bool stepped = false;
        bool space_was_down = false;
        bool r_was_down = false;
        bool plus_was_down = false;
        bool minus_was_down = false;
        bool left_was_down = false;
        bool right_was_down = false;
        bool f_was_down = false;
        bool h_was_down = false;
        bool hud_visible = true;
        int hud_scale = 1;
        size_t tick_index = 0;
        bool tick_changed = true;
        double last_time = glfwGetTime();
        double ema_ms = 16.7;
        u64 ticks_shown = 0;
        bool printed_tick20 = false;
        double title_timer = 0.0;
        bool drag_active = false;
        double last_cursor_x = 0.0;
        double last_cursor_y = 0.0;
        double time_sum = 0.0;
        uint32_t timed_frames = 0;
        uint32_t frame_number = 0;
        std::vector<ContactFlash> flashes;
        std::vector<float> hud_text;
        uint32_t hud_accent_begin = 0;
        uint32_t hud_accent_end = 0;
        float hud_panel_verts[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        BodyInstance hud_quads[kMaxHudQuads];
        uint32_t hud_quad_count = 0;

        if (!frames_requested) {
            std::printf("keys: SPACE pause  LEFT/RIGHT step  +/- rate  R restart  F fit  H hud"
                        "  ESC quit  drag pan  wheel zoom\n");
        }

        while (frames_requested ? (frame_number < frame_limit)
                                : (glfwWindowShouldClose(window) != GLFW_TRUE)) {
            glfwPollEvents();
            if (glfwWindowShouldClose(window) == GLFW_TRUE) break;
            ++frame_number;

            if (!frames_requested) {
                int window_width = 0;
                int window_height = 0;
                glfwGetWindowSize(window, &window_width, &window_height);
                // Screen coordinates (points) vs framebuffer pixels; cursor
                // motion must be scaled by this to stay 1:1 with the world
                // on retina displays, where the two differ by 2x.
                const double content_scale =
                    window_width > 0 ? static_cast<double>(state.fb_width) / window_width : 1.0;
                hud_scale =
                    std::clamp(static_cast<int>(std::lround(content_scale)), 1, 4);

                if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
                    glfwSetWindowShouldClose(window, GLFW_TRUE);
                }
                const bool space_down = glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS;
                if (space_down && !space_was_down) paused = !paused;
                space_was_down = space_down;

                const bool r_down = glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS;
                if (r_down && !r_was_down) {
                    tick_index = 0;
                    subframe = 0;
                    tick_changed = true;
                    flashes.clear();
#if ONTOS_VIEW_REALTIME_AUDIO
                    audio.reset();
#endif
                }
                r_was_down = r_down;

                const bool left_down = glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS;
                const bool right_down = glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS;
                if ((left_down && !left_was_down) || (right_down && !right_was_down)) {
                    paused = true;
                    subframe = 0;
                    const std::size_t span = stream.frames.size();
                    tick_index =
                        (tick_index + span + (right_down ? 1 : -1)) % span;
                    tick_changed = true;
                    stepped = true;
                }
                left_was_down = left_down;
                right_was_down = right_down;

                const bool f_down = glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS;
                if (f_down && !f_was_down && state.fb_width > 0 && state.fb_height > 0) {
                    state.camera.cx = 64.0;
                    state.camera.cy = 64.0;
                    state.camera.zoom =
                        0.9 * std::min(state.fb_width, state.fb_height) / 128.0;
                }
                f_was_down = f_down;

                const bool h_down = glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS;
                if (h_down && !h_was_down) hud_visible = !hud_visible;
                h_was_down = h_down;

                const bool plus_down = glfwGetKey(window, GLFW_KEY_EQUAL) == GLFW_PRESS ||
                                       glfwGetKey(window, GLFW_KEY_KP_ADD) == GLFW_PRESS;
                if (plus_down && !plus_was_down && frames_per_tick > 1) frames_per_tick /= 2;
                plus_was_down = plus_down;

                const bool minus_down = glfwGetKey(window, GLFW_KEY_MINUS) == GLFW_PRESS ||
                                        glfwGetKey(window, GLFW_KEY_KP_SUBTRACT) == GLFW_PRESS;
                if (minus_down && !minus_was_down && frames_per_tick < 256) frames_per_tick *= 2;
                minus_was_down = minus_down;

                double cx = 0.0;
                double cy = 0.0;
                glfwGetCursorPos(window, &cx, &cy);
                const bool button_down =
                    glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
                if (button_down && drag_active) {
                    state.camera.cx -= (cx - last_cursor_x) * content_scale / state.camera.zoom;
                    state.camera.cy += (cy - last_cursor_y) * content_scale / state.camera.zoom;
                }
                drag_active = button_down;
                last_cursor_x = cx;
                last_cursor_y = cy;

                const double now = glfwGetTime();
                const double dt = std::max(now - last_time, 0.0);
                last_time = now;
                const double pan = 400.0 * dt * content_scale / state.camera.zoom;
                if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) state.camera.cy += pan;
                if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) state.camera.cy -= pan;
                if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) state.camera.cx -= pan;
                if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) state.camera.cx += pan;
            }

            if (!paused) {
                if (frames_requested ? frame_number > 1 : subframe + 1 >= frames_per_tick) {
                    subframe = 0;
                    tick_index = (tick_index + 1) % stream.frames.size();
                    tick_changed = true;
                } else {
                    ++subframe;
                }
            }

            // Flashes advance with playback, and also after a manual tick
            // step (otherwise stepped-while-paused rings never fade).
            if (!flashes.empty() && (!paused || stepped)) {
                const float ticks_per_frame =
                    frames_requested ? 1.0f : 1.0f / static_cast<float>(frames_per_tick);
                const float decay = ticks_per_frame / kFlashTicks;
                for (ContactFlash& flash : flashes) flash.t += decay;
                flashes.erase(std::remove_if(flashes.begin(), flashes.end(),
                                             [](const ContactFlash& flash) {
                                                 return flash.t >= 1.0f;
                                             }),
                              flashes.end());
            }
            stepped = false;

            const StreamFrame& frame = stream.frames[tick_index];

            if (state.resized) {
                state.resized = false;
                vkDeviceWaitIdle(v.device);
                for (VkSemaphore semaphore : v.render_finished_per_image) {
                    vkDestroySemaphore(v.device, semaphore, nullptr);
                }
                v.render_finished_per_image.clear();
                v.destroy_swapchain_views();
                create_swapchain(v, window);
                create_extent_resources(v);
                VkSemaphoreCreateInfo semaphore_info{};
                semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
                v.render_finished_per_image.resize(v.images.size(), VK_NULL_HANDLE);
                for (uint32_t i = 0; i < v.render_finished_per_image.size(); ++i) {
                    result = vkCreateSemaphore(v.device, &semaphore_info, nullptr,
                                               &v.render_finished_per_image[i]);
                    if (result != VK_SUCCESS) throw std::runtime_error("vkCreateSemaphore failed");
                }
            }

            const double frame_start = glfwGetTime();
            vkWaitForFences(v.device, 1, &v.in_flight, VK_TRUE, UINT64_MAX);

            // Fill the slot this frame renders. The slot parity alternates
            // every render frame while ticks advance only every
            // frames_per_tick frames, so filling only on tick changes left
            // the other parity slot stale (or never initialized); both the
            // body and flash instances are refilled per frame instead.
            GpuBuffer& instance_slot = v.body_instance_buffers[frame_number & 1];
            GpuBuffer& flash_slot = v.contact_instance_buffers[frame_number & 1];
            fill_body_instances(frame, static_cast<BodyInstance*>(instance_slot.mapped));
            if (!flashes.empty()) {
                fill_flash_instances(flashes, static_cast<BodyInstance*>(flash_slot.mapped));
            }
            if (tick_changed) {
                for (const StreamContact& c : frame.contacts) {
                    ContactFlash flash;
                    flash.x = static_cast<float>(c.cx);
                    flash.y = static_cast<float>(c.cy);
                    const f64 mass_a = frame.bodies[c.a].mass;
                    const f64 mass_b = contactant_mass(stream, frame, c.b, mass_a);
                    flash.half_extent =
                        std::max(body_half_extent(mass_a), body_half_extent(mass_b)) * 1.4f;
                    if (flashes.size() >= kMaxFlashes) flashes.erase(flashes.begin());
                    flashes.push_back(flash);
#if ONTOS_VIEW_REALTIME_AUDIO
                    if (audio_running) {
                        f64 gl = 1.0, gr = 1.0;
                        contact_gains(state.camera, state.fb_width, state.fb_height, c.cx, c.cy,
                                      gl, gr);
                        audio.spawn_contact(c, contact_reduced_mass(stream, final_frame, c), gl,
                                            gr);
                    }
#endif
                }
                ticks_shown += 1;
                std::printf("tick %" PRIu64 " bodies=%zu fine=%" PRIu64 " coarse=%" PRIu64 "\n",
                            frame.tick, frame.bodies.size(), frame.fine, frame.coarse);
                if (frame.tick == 20 && !printed_tick20) {
                    printed_tick20 = true;
                    for (const StreamBody& body : frame.bodies) {
                        std::printf(
                            "tick20 body=%" PRIu32 " region=%u level=%u x=%.17g y=%.17g vx=%.17g"
                            " vy=%.17g mass=%.17g\n",
                            body.id, body.region, body.level, body.x, body.y, body.vx, body.vy,
                            body.mass);
                    }
                }
                tick_changed = false;
            }

            if (!frames_requested && hud_visible) {
                fill_hud(frame, stream.frames.size(), frames_per_tick, paused, ema_ms, hud_scale,
                         hud_text, hud_accent_begin, hud_accent_end, hud_panel_verts, hud_quads,
                         hud_quad_count);
                std::memcpy(v.hud_line_buffer.mapped, hud_text.data(),
                            hud_text.size() * sizeof(float));
                std::memcpy(v.hud_panel_buffer.mapped, hud_panel_verts, sizeof hud_panel_verts);
                std::memcpy(v.hud_quad_instance_buffer.mapped, hud_quads,
                            hud_quad_count * sizeof(BodyInstance));
            }

            uint32_t image_index = 0;
            result = vkAcquireNextImageKHR(v.device, v.swapchain, UINT64_MAX, v.image_available,
                                           VK_NULL_HANDLE, &image_index);
            if (result == VK_ERROR_OUT_OF_DATE_KHR) {
                state.resized = true;
                continue;
            }
            if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
                throw std::runtime_error("vkAcquireNextImageKHR failed");
            }

            vkResetCommandBuffer(v.command_buffer, 0);
            VkCommandBufferBeginInfo begin{};
            begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            result = vkBeginCommandBuffer(v.command_buffer, &begin);
            if (result != VK_SUCCESS) throw std::runtime_error("vkBeginCommandBuffer failed");

            VkClearValue clears[3]{};
            clears[0].color = {{0.05f, 0.06f, 0.08f, 1.0f}};
            uint32_t clear_count = 0;
            if (v.samples != VK_SAMPLE_COUNT_1_BIT) {
                // [0] MSAA color clear, [1] resolve (dont-care), [2] depth clear
                clears[2].depthStencil = {1.0f, 0};
                clear_count = 3;
            } else {
                // [0] color clear, [1] depth clear
                clears[1].depthStencil = {1.0f, 0};
                clear_count = 2;
            }
            VkRenderPassBeginInfo render_begin{};
            render_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            render_begin.renderPass = v.render_pass;
            render_begin.framebuffer = v.framebuffers[image_index];
            render_begin.renderArea.extent = v.extent;
            render_begin.clearValueCount = clear_count;
            render_begin.pClearValues = clears;
            vkCmdBeginRenderPass(v.command_buffer, &render_begin, VK_SUBPASS_CONTENTS_INLINE);

            VkViewport viewport{};
            viewport.width = static_cast<float>(v.extent.width);
            viewport.height = static_cast<float>(v.extent.height);
            viewport.maxDepth = 1.0f;
            VkRect2D scissor{};
            scissor.extent = v.extent;
            vkCmdSetViewport(v.command_buffer, 0, 1, &viewport);
            vkCmdSetScissor(v.command_buffer, 0, 1, &scissor);

            const VkDeviceSize offsets[2] = {0, 0};
            vkCmdBindVertexBuffers(v.command_buffer, 0, 1, &v.minor_line_buffer.buffer,
                                   &offsets[0]);
            vkCmdBindVertexBuffers(v.command_buffer, 1, 1, &v.minor_line_instance_buffer.buffer,
                                   &offsets[1]);
            vkCmdBindPipeline(v.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, v.line_pipeline);
            const ViewPush minor_push = compute_view_push(state.camera, v.extent, 0.6f);
            vkCmdPushConstants(v.command_buffer, v.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0,
                               sizeof(ViewPush), &minor_push);
            vkCmdDraw(v.command_buffer, 56, 1, 0, 0);

            vkCmdBindVertexBuffers(v.command_buffer, 0, 1, &v.line_buffer.buffer, &offsets[0]);
            vkCmdBindVertexBuffers(v.command_buffer, 1, 1, &v.line_instance_buffer.buffer,
                                   &offsets[1]);
            vkCmdBindPipeline(v.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, v.line_pipeline);
            const ViewPush line_push = compute_view_push(state.camera, v.extent, 0.5f);
            vkCmdPushConstants(v.command_buffer, v.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0,
                               sizeof(ViewPush), &line_push);
            vkCmdDraw(v.command_buffer, 12, 1, 0, 0);

            vkCmdBindVertexBuffers(v.command_buffer, 0, 1, &v.quad_buffer.buffer, &offsets[0]);
            vkCmdBindVertexBuffers(v.command_buffer, 1, 1, &instance_slot.buffer, &offsets[1]);
            vkCmdBindPipeline(v.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, v.body_pipeline);
            const ViewPush body_push = compute_view_push(state.camera, v.extent, 0.25f);
            vkCmdPushConstants(v.command_buffer, v.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0,
                               sizeof(ViewPush), &body_push);
            vkCmdDraw(v.command_buffer, 4, static_cast<uint32_t>(frame.bodies.size()), 0, 0);

            if (!flashes.empty()) {
                vkCmdBindVertexBuffers(v.command_buffer, 1, 1, &flash_slot.buffer, &offsets[1]);
                const ViewPush flash_push = compute_view_push(state.camera, v.extent, 0.15f);
                vkCmdPushConstants(v.command_buffer, v.pipeline_layout,
                                   VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ViewPush),
                                   &flash_push);
                vkCmdDraw(v.command_buffer, 4, static_cast<uint32_t>(flashes.size()), 0, 0);
            }

            if (!frames_requested && hud_visible && hud_quad_count > 0) {
                // Panel: explicit pixel-space corners with the identity
                // instance (index 2). Swatches: isotropic disc instances.
                const ViewPush hud_quad_push = compute_screen_push(v.extent, 0.05f);
                vkCmdBindVertexBuffers(v.command_buffer, 0, 1, &v.hud_panel_buffer.buffer,
                                       &offsets[0]);
                vkCmdBindVertexBuffers(v.command_buffer, 1, 1, &v.hud_line_instance_buffer.buffer,
                                       &offsets[1]);
                vkCmdBindPipeline(v.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  v.body_pipeline);
                vkCmdPushConstants(v.command_buffer, v.pipeline_layout,
                                   VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ViewPush),
                                   &hud_quad_push);
                vkCmdDraw(v.command_buffer, 4, 1, 0, 2);

                vkCmdBindVertexBuffers(v.command_buffer, 0, 1, &v.quad_buffer.buffer,
                                       &offsets[0]);
                vkCmdBindVertexBuffers(v.command_buffer, 1, 1, &v.hud_quad_instance_buffer.buffer,
                                       &offsets[1]);
                vkCmdDraw(v.command_buffer, 4, hud_quad_count, 0, 0);

                const ViewPush hud_text_push = compute_screen_push(v.extent, 0.03f);
                vkCmdBindVertexBuffers(v.command_buffer, 0, 1, &v.hud_line_buffer.buffer,
                                       &offsets[0]);
                // The swatch draw above left the disc instances bound at
                // binding 1; the text needs the color instances back or
                // every glyph is culled by the disc falloff.
                vkCmdBindVertexBuffers(v.command_buffer, 1, 1, &v.hud_line_instance_buffer.buffer,
                                       &offsets[1]);
                vkCmdBindPipeline(v.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  v.line_pipeline);
                vkCmdPushConstants(v.command_buffer, v.pipeline_layout,
                                   VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ViewPush),
                                   &hud_text_push);
                const uint32_t hud_verts = static_cast<uint32_t>(hud_text.size() / 2);
                if (hud_accent_begin > 0) {
                    vkCmdDraw(v.command_buffer, hud_accent_begin, 1, 0, 0);
                }
                if (hud_accent_end > hud_accent_begin) {
                    vkCmdDraw(v.command_buffer, hud_accent_end - hud_accent_begin, 1,
                              hud_accent_begin, 1);
                }
                if (hud_verts > hud_accent_end) {
                    vkCmdDraw(v.command_buffer, hud_verts - hud_accent_end, 1, hud_accent_end, 0);
                }
            }

            vkCmdEndRenderPass(v.command_buffer);
            result = vkEndCommandBuffer(v.command_buffer);
            if (result != VK_SUCCESS) throw std::runtime_error("vkEndCommandBuffer failed");

            const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            VkSubmitInfo submit{};
            submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.waitSemaphoreCount = 1;
            submit.pWaitSemaphores = &v.image_available;
            submit.pWaitDstStageMask = &wait_stage;
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &v.command_buffer;
            VkSemaphore signal_semaphore =
                image_index < v.render_finished_per_image.size()
                    ? v.render_finished_per_image[image_index]
                    : v.render_finished_per_image.front();
            submit.signalSemaphoreCount = 1;
            submit.pSignalSemaphores = &signal_semaphore;
            // Reset only here, on the path that owns the submit: the
            // OUT_OF_DATE acquire branch above skips submitting entirely,
            // and a reset-but-never-signaled fence would make the next
            // vkWaitForFences block forever (swapchain recreation does not
            // repair fence state).
            vkResetFences(v.device, 1, &v.in_flight);
            result = vkQueueSubmit(v.queue, 1, &submit, v.in_flight);
            if (result != VK_SUCCESS) throw std::runtime_error("vkQueueSubmit failed");

            VkPresentInfoKHR present{};
            present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            present.waitSemaphoreCount = 1;
            present.pWaitSemaphores = &signal_semaphore;
            present.swapchainCount = 1;
            present.pSwapchains = &v.swapchain;
            present.pImageIndices = &image_index;
            result = vkQueuePresentKHR(v.queue, &present);
            if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
                state.resized = true;
            } else if (result != VK_SUCCESS) {
                throw std::runtime_error("vkQueuePresentKHR failed");
            }

            const double frame_end = glfwGetTime();
            const double ms = (frame_end - frame_start) * 1000.0;
            ema_ms = ema_ms * 0.9 + ms * 0.1;
            if (frame_number > 3) {
                time_sum += ms;
                timed_frames += 1;
            }

            if (!frames_requested) {
                title_timer += ms * 0.001;
                if (title_timer > 0.25) {
                    title_timer = 0.0;
                    char title[256];
                    std::snprintf(title, sizeof(title),
                                  "ontos_view  tick %" PRIu64 " / %zu  bodies=%zu  fine=%" PRIu64
                                  "  coarse=%" PRIu64 "  ms=%.2f  rate=1/%u%s",
                                  frame.tick, stream.frames.size(), frame.bodies.size(),
                                  frame.fine, frame.coarse, ema_ms, frames_per_tick,
                                  paused ? "  [paused]" : "");
                    glfwSetWindowTitle(window, title);
                }
            }
        }

#if ONTOS_VIEW_REALTIME_AUDIO
        audio.stop();
#endif

        const double avg_ms = timed_frames > 0 ? time_sum / timed_frames : 0.0;
        const StreamFrame& last_frame = stream.frames[tick_index];
        std::printf("OK ticks=%" PRIu64 " bodies=%zu fine=%" PRIu64 " coarse=%" PRIu64
                    " contacts=%" PRIu64 " frames=%u avg_ms=%.3f\n",
                    ticks_shown, last_frame.bodies.size(), last_frame.fine, last_frame.coarse,
                    total_contacts, timed_frames, avg_ms);

        vkDeviceWaitIdle(v.device);
        v.destroy();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 0;
    } catch (const std::exception& error) {
        if (window != nullptr) glfwDestroyWindow(window);
        glfwTerminate();
        std::fprintf(stderr, "error: %s\n", error.what());
        return 2;
    }
}
