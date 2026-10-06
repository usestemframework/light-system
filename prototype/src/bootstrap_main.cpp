#include "vk_bootstrap.h"
#include "visibility_format.h"

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string_view>

namespace {

// Checked upper bound for --threads: a sane ceiling well above any core
// count; values past it are rejected instead of trying to spawn a thread
// per requested worker.
constexpr uint32_t kMaxWorkerThreads = 1024;

// Strict unsigned parse for --threads: atoi cast to uint32_t wrapped
// negatives ("-1" -> 4294967295 threads) and accepted trailing junk.
// from_chars on uint32_t rejects leading signs; require the full string.
bool parse_thread_count(std::string_view value, uint32_t& out) {
    uint32_t parsed = 0;
    const auto* begin = value.data();
    const auto* end = value.data() + value.size();
    const auto result = std::from_chars(begin, end, parsed);
    if (result.ec != std::errc() || result.ptr != end || parsed > kMaxWorkerThreads) {
        return false;
    }
    out = parsed;
    return true;
}

// Strict positive-float parse for --exposure: strtof full-string +
// finite-and-positive check (nan/inf/0/negative rejected), mirroring the
// --threads hardening (LS-52) and manifest float rules (LS-57).
bool parse_positive_float(std::string_view value, float& out) {
    char* end = nullptr;
    const float parsed = std::strtof(value.data(), &end);
    if (end != value.data() + value.size() || !std::isfinite(parsed) || parsed <= 0.0f) {
        return false;
    }
    out = parsed;
    return true;
}

void print_usage() {
    std::cerr << "Usage: meridian_vk_bootstrap --manifest <path> [--interactive] [--screenshot <path>] [--budget <pages>] [--demand-streaming] [--error-threshold <value>] [--shadow-error-scale <value>] [--threads <count>] [--validate] [--no-texture-mips] [--tonemap] [--exposure <value>]\n"
                 "  --screenshot writes a raw PPM image (extension forced to .ppm)\n"
                 "  --error-threshold sets the LOD selection threshold (default: auto =\n"
                 "  max(0.001, 8.9x the scene's median LOD-group geometric error), so\n"
                 "  scene-scale ladders activate instead of selecting full detail)\n"
                 "  --shadow-error-scale multiplies the LOD threshold for shadow casters\n"
                 "  (default 8.0; <= 1 shares the main-pass selection)\n"
                 "  --threads sets the total worker threads for traversal and draw-list\n"
                 "  build (default: auto = min(hardware_concurrency, 8); 1 = serial;\n"
                 "  output is bit-identical at any count)\n"
                   "  --no-gpu-timers disables the per-frame GPU timestamp queries and the\n"
                   "  MERIDIAN_GPU lines (measures the unprofiled submit path)\n"
                   "  --gpu-selection also builds the (currently undispatched) cluster_select\n"
                   "  compute pipeline; creation failure is non-fatal\n"
                   "  --no-texture-mips disables the base-color texture mip chain and\n"
                   "  anisotropic filtering (single-level sampling, the historical path)\n"
                   "  --tonemap enables the ACES filmic curve + sRGB encode on the final\n"
                   "  color (default off keeps output bit-identical to the old renderer)\n"
                   "  --exposure sets the exposure multiplier applied before tone mapping\n"
                   "  (default 1.0; only observable with --tonemap)\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path manifest_path;
    std::string screenshot_path;
    uint32_t resident_budget = 0xffffffffu;
    float error_threshold = -1.0f;  // negative = auto (scene-scaled)
    float shadow_error_scale = 8.0f;
    uint32_t worker_threads = 0;
    bool validate = false;
    bool interactive = false;
    bool demand_streaming = false;
    bool enable_gpu_timers = true;
    bool gpu_selection = false;
    bool texture_mips = true;
    float exposure = 1.0f;
    bool tonemap = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--manifest" && i + 1 < argc) {
            manifest_path = argv[++i];
        } else if (arg == "--screenshot" && i + 1 < argc) {
            screenshot_path = argv[++i];
        } else if (arg == "--budget" && i + 1 < argc) {
            resident_budget = static_cast<uint32_t>(std::atoi(argv[++i]));
        } else if (arg == "--error-threshold" && i + 1 < argc) {
            error_threshold = std::atof(argv[++i]);
        } else if (arg == "--shadow-error-scale" && i + 1 < argc) {
            shadow_error_scale = std::atof(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            if (!parse_thread_count(argv[i + 1], worker_threads)) {
                std::cerr << "invalid --threads value: " << argv[i + 1]
                          << " (expected an integer in [1, " << kMaxWorkerThreads << "])\n";
                return 1;
            }
            ++i;
        } else if (arg == "--validate") {
            validate = true;
        } else if (arg == "--interactive") {
            interactive = true;
        } else if (arg == "--demand-streaming") {
            demand_streaming = true;
        } else if (arg == "--no-gpu-timers") {
            enable_gpu_timers = false;
        } else if (arg == "--gpu-selection") {
            gpu_selection = true;
        } else if (arg == "--no-texture-mips") {
            texture_mips = false;
        } else if (arg == "--tonemap") {
            tonemap = true;
        } else if (arg == "--exposure" && i + 1 < argc) {
            if (!parse_positive_float(argv[i + 1], exposure)) {
                std::cerr << "invalid --exposure value: " << argv[i + 1]
                          << " (expected a finite positive number)\n";
                return 1;
            }
            ++i;
        } else {
            print_usage();
            return 1;
        }
    }
    if (manifest_path.empty()) {
        print_usage();
        return 1;
    }

    try {
        const meridian::BuildManifest manifest = meridian::load_manifest(manifest_path);
        // Non-const: the demand-streaming runtime consumes the payload
        // vectors once the .vgeo is mmap'd (see build_vk_bootstrap_report).
        meridian::VGeoResource resource = meridian::build_resource(manifest);
        meridian::validate_resource(resource);

        meridian::VkBootstrapConfig config{};
        if (interactive) {
            config.interactive = true;
            config.visible_window = true;
            config.present_frame_count = 0xffffffffu;
        }
        config.screenshot_path = screenshot_path;
        config.resident_budget = resident_budget;
        config.debug_error_threshold = error_threshold;
        config.shadow_error_scale = shadow_error_scale;
        config.enable_validation = validate;
        config.demand_streaming = demand_streaming;
        config.worker_threads = worker_threads;
        config.enable_gpu_timers = enable_gpu_timers;
        config.enable_gpu_selection = gpu_selection;
        config.texture_mips = texture_mips;
        config.exposure = exposure;
        config.tonemap = tonemap;
        config.persisted_vgeo_path = manifest.output_path.string();
        const meridian::VkBootstrapReport report =
            meridian::build_vk_bootstrap_report(resource, config);

        std::cout << "asset_id=" << resource.asset_id << '\n';
        std::cout << "compiled_with_vulkan=" << (report.compiled_with_vulkan ? "true" : "false")
                  << '\n';
        std::cout << "instance_created=" << (report.instance_created ? "true" : "false") << '\n';
        std::cout << "window_created=" << (report.window_created ? "true" : "false") << '\n';
        std::cout << "surface_created=" << (report.surface_created ? "true" : "false") << '\n';
        std::cout << "device_created=" << (report.device_created ? "true" : "false") << '\n';
        std::cout << "swapchain_created=" << (report.swapchain_created ? "true" : "false") << '\n';
        std::cout << "scene_buffers_uploaded=" << (report.scene_buffers_uploaded ? "true" : "false")
                  << '\n';
        std::cout << "debug_pipeline_created=" << (report.debug_pipeline_created ? "true" : "false")
                  << '\n';
        std::cout << "debug_geometry_uploaded=" << (report.debug_geometry_uploaded ? "true" : "false")
                  << '\n';
        std::cout << "visibility_attachment_created="
                  << (report.visibility_attachment_created ? "true" : "false") << '\n';
        std::cout << "visibility_readback_ready="
                  << (report.visibility_readback_ready ? "true" : "false") << '\n';
        std::cout << "debug_draw_submitted=" << (report.debug_draw_submitted ? "true" : "false")
                  << '\n';
        std::cout << "present_loop_completed=" << (report.present_loop_completed ? "true" : "false")
                  << '\n';
        std::cout << "status=" << report.status << '\n';
        std::cout << "capture_status=" << report.capture_status << '\n';
        std::cout << "selected_device=" << report.selected_device << '\n';
        std::cout << "graphics_queue_family=" << report.graphics_queue_family << '\n';
        std::cout << "present_queue_family=" << report.present_queue_family << '\n';
        std::cout << "swapchain_images=" << report.swapchain_image_count << '\n';
        std::cout << "uploaded_buffer_count=" << report.uploaded_buffer_count << '\n';
        std::cout << "uploaded_buffer_bytes=" << report.uploaded_buffer_bytes << '\n';
        std::cout << "debug_selected_nodes=" << report.debug_selected_node_count << '\n';
        std::cout << "debug_rendered_clusters=" << report.debug_rendered_cluster_count << '\n';
        std::cout << "debug_rendered_lod_clusters=" << report.debug_rendered_lod_cluster_count << '\n';
        std::cout << "replay_selected_nodes=" << report.replay_selected_node_count << '\n';
        std::cout << "replay_selected_clusters=" << report.replay_selected_cluster_count << '\n';
        std::cout << "replay_selected_lod_clusters=" << report.replay_selected_lod_cluster_count << '\n';
        std::cout << "replay_selected_pages=" << report.replay_selected_page_count << '\n';
        std::cout << "replay_runtime_parity=" << (report.replay_runtime_parity ? "true" : "false")
                  << '\n';
        std::cout << "runtime_missing_pages=" << report.runtime_missing_page_count << '\n';
        std::cout << "runtime_prefetch_pages=" << report.runtime_prefetch_page_count << '\n';
        std::cout << "runtime_requested_pages=" << report.runtime_requested_page_count << '\n';
        std::cout << "runtime_loading_pages=" << report.runtime_loading_page_count << '\n';
        std::cout << "runtime_completed_pages=" << report.runtime_completed_page_count << '\n';
        std::cout << "runtime_resident_pages=" << report.runtime_resident_page_count << '\n';
        std::cout << "runtime_failed_pages=" << report.runtime_failed_page_count << '\n';
        std::cout << "visibility_valid_pixels=" << report.visibility_valid_pixels << '\n';
        std::cout << "visibility_unique_base_geometry=" << report.visibility_unique_base_geometry << '\n';
        std::cout << "visibility_unique_lod_geometry=" << report.visibility_unique_lod_geometry << '\n';
        std::cout << "visibility_invalid_ids=" << report.visibility_invalid_ids << '\n';
        std::cout << "visibility_visible_selected_base_geometry="
                  << report.visibility_visible_selected_base_geometry << '\n';
        std::cout << "visibility_visible_selected_lod_geometry="
                  << report.visibility_visible_selected_lod_geometry << '\n';
        std::cout << "visibility_invisible_selected_base_geometry="
                  << report.visibility_invisible_selected_base_geometry << '\n';
        std::cout << "visibility_invisible_selected_lod_geometry="
                  << report.visibility_invisible_selected_lod_geometry << '\n';
        std::cout << "visibility_selection_subset="
                  << (report.visibility_selection_subset ? "true" : "false") << '\n';
        std::cout << "compute_cull_visible_instances=" << report.compute_cull_visible_instances << '\n';
        std::cout << "compute_selection_draw_count=" << report.compute_selection_draw_count << '\n';
        std::cout << "compute_occlusion_surviving=" << report.compute_occlusion_surviving_draws << '\n';
        std::cout << "debug_triangles=" << report.debug_triangle_count << '\n';
        std::cout << "debug_vertices=" << report.debug_vertex_count << '\n';
        std::cout << "debug_camera_distance=" << report.debug_camera_distance << '\n';
        std::cout << "presented_frames=" << report.presented_frame_count << '\n';
        std::cout << "swapchain_width=" << report.swapchain_width << '\n';
        std::cout << "swapchain_height=" << report.swapchain_height << '\n';
        std::cout << "gpu_instances=" << report.uploadable_scene.instances.size() << '\n';
        std::cout << "gpu_nodes=" << report.uploadable_scene.hierarchy_nodes.size() << '\n';
        std::cout << "gpu_clusters=" << report.uploadable_scene.clusters.size() << '\n';
        std::cout << "gpu_lod_groups=" << report.uploadable_scene.lod_groups.size() << '\n';
        std::cout << "gpu_lod_clusters=" << report.uploadable_scene.lod_clusters.size() << '\n';
        std::cout << "gpu_pages=" << report.uploadable_scene.pages.size() << '\n';
        std::cout << "gpu_page_dependencies=" << report.uploadable_scene.page_dependencies.size()
                  << '\n';
        std::cout << "base_payload_bytes=" << report.uploadable_scene.header.base_payload_bytes
                  << '\n';
        std::cout << "lod_payload_bytes=" << report.uploadable_scene.header.lod_payload_bytes
                  << '\n';

        const meridian::VisibilityPixel example =
            meridian::encode_visibility(0, meridian::GeometryKind::base_cluster, 1, 2);
        std::cout << "visibility_words=" << report.uploadable_scene.header.visibility_format_word_count
                  << '\n';
        std::cout << "visibility_example_word0=" << example.word0 << '\n';
        std::cout << "visibility_example_word1=" << example.word1 << '\n';
        std::cout << "physical_devices=" << report.physical_devices.size() << '\n';
        std::cout << "worker_threads=" << report.worker_threads << '\n';
        for (size_t device_index = 0; device_index < report.physical_devices.size(); ++device_index) {
            std::cout << "physical_device[" << device_index << "]="
                      << report.physical_devices[device_index] << '\n';
        }

        // The report is the verdict: fail on initialization, submission, or
        // fixed-count present-loop failure. A completed fixed-frame run and
        // a clean interactive close (present_loop_completed is by design
        // false there) both stay success.
        const bool initialization_ok = report.compiled_with_vulkan && report.instance_created &&
                                       report.window_created && report.surface_created &&
                                       report.device_created && report.swapchain_created;
        const bool submission_ok = report.debug_draw_submitted;
        const bool present_loop_ok = interactive || report.present_loop_completed;
        if (!initialization_ok || !submission_ok || !present_loop_ok) {
            std::cerr << "bootstrap failed (initialization=" << initialization_ok
                      << " submission=" << submission_ok
                      << " present_loop=" << present_loop_ok
                      << "): " << report.status << '\n';
            return 4;
        }
        return 0;
    } catch (const meridian::BuilderError& error) {
        std::cerr << "Bootstrap error: " << error.what() << '\n';
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "Unexpected error: " << error.what() << '\n';
        return 3;
    }
}
