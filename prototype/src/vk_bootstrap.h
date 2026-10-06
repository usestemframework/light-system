#pragma once

#include "resource_upload.h"

#include <string>
#include <vector>

namespace meridian {

struct VkBootstrapConfig {
    bool enable_validation = false;
    bool visible_window = false;
    bool interactive = false;
    uint32_t window_width = 1280;
    uint32_t window_height = 720;
    uint32_t present_frame_count = 120;
    float debug_error_threshold = 0.001f;
    // Shadow caster LOD: the shadow pass selects casters at the main error
    // threshold multiplied by this scale (values <= 1 disable it and share
    // the main-pass selection). Depth-only silhouettes tolerate far more
    // geometric error than the shaded main pass.
    float shadow_error_scale = 8.0f;
    uint32_t resident_budget = 0xffffffffu;
    uint32_t eviction_grace_frames = 1;
    // When true, pages start in the unloaded state and the streaming
    // scheduler drives per-frame load/evict decisions. Simulated async
    // loads land in the resident state after a small number of frames.
    bool demand_streaming = false;
    uint32_t streaming_max_loads_per_frame = 32;
    uint32_t streaming_load_latency_frames = 2;
    uint32_t streaming_seed_pages = 64;
    std::string screenshot_path;
    // Persisted .vgeo for this asset (typically the manifest's output_path).
    // When it exists and its header matches the freshly built resource, the
    // demand-streaming path mmaps it directly instead of writing a temp
    // copy. Empty or stale files fall back to the temp-write path.
    std::string persisted_vgeo_path;
    // Total threads (including the submitter) for the parallel LOD
    // traversal and draw-list build. 0 = auto (min of hardware_concurrency
    // and 8). 1 forces the serial path. Output is bit-identical either way.
    uint32_t worker_threads = 0;
    // When false, the per-frame GPU timestamp queries are skipped (no query
    // pool, no MERIDIAN_GPU lines). On MoltenVK each timestamp writes a
    // counter sample, so timer-off runs measure the unprofiled submit cost.
    bool enable_gpu_timers = true;
    // Opt-in: build the (currently undispatched) cluster_select compute
    // pipeline. The live renderer builds draws on the CPU, so the default
    // is off; creation failure of the optional pipeline never aborts
    // bootstrap.
    bool enable_gpu_selection = false;
    // Base-color texture mip chain: the embedded RGBA8 texture is uploaded
    // with a full blit-generated mip chain (plus anisotropic filtering when
    // the device supports it) instead of a single level. --no-texture-mips
    // restores the single-level path bit-for-bit.
    bool texture_mips = true;
    // Opt-in tone mapping: exposure multiplier and the ACES + sRGB-encode
    // output transform. Default (exposure 1.0, tonemap off) keeps the
    // shaded output bit-identical to the historical renderer.
    float exposure = 1.0f;
    bool tonemap = false;
};

struct VkBootstrapReport {
    bool compiled_with_vulkan = false;
    bool instance_created = false;
    bool window_created = false;
    bool surface_created = false;
    bool device_created = false;
    bool swapchain_created = false;
    bool scene_buffers_uploaded = false;
    bool debug_pipeline_created = false;
    bool debug_geometry_uploaded = false;
    bool visibility_attachment_created = false;
    bool visibility_readback_ready = false;
    bool debug_draw_submitted = false;
    bool present_loop_completed = false;
    std::string status;
    // Outcome of the --screenshot request: set whenever a screenshot was
    // requested and the run reached the capture stage, on success and on
    // every skip/failure (unsupported format, missing TRANSFER_SRC, failed
    // acquisition or submission). Empty when no screenshot was requested.
    // Capture failures do not fail the run; they are reported here.
    std::string capture_status;
    std::string selected_device;
    uint32_t graphics_queue_family = 0xffffffffu;
    uint32_t present_queue_family = 0xffffffffu;
    uint32_t swapchain_image_count = 0;
    uint32_t uploaded_buffer_count = 0;
    uint64_t uploaded_buffer_bytes = 0;
    uint32_t debug_selected_node_count = 0;
    uint32_t debug_rendered_cluster_count = 0;
    uint32_t debug_rendered_lod_cluster_count = 0;
    uint32_t replay_selected_node_count = 0;
    uint32_t replay_selected_cluster_count = 0;
    uint32_t replay_selected_lod_cluster_count = 0;
    uint32_t replay_selected_page_count = 0;
    bool replay_runtime_parity = false;
    uint32_t runtime_missing_page_count = 0;
    uint32_t runtime_prefetch_page_count = 0;
    uint32_t runtime_requested_page_count = 0;
    uint32_t runtime_loading_page_count = 0;
    uint32_t runtime_completed_page_count = 0;
    uint32_t runtime_resident_page_count = 0;
    uint32_t runtime_failed_page_count = 0;
    uint32_t visibility_valid_pixels = 0;
    uint32_t visibility_unique_base_geometry = 0;
    uint32_t visibility_unique_lod_geometry = 0;
    uint32_t visibility_invalid_ids = 0;
    uint32_t visibility_visible_selected_base_geometry = 0;
    uint32_t visibility_visible_selected_lod_geometry = 0;
    uint32_t visibility_invisible_selected_base_geometry = 0;
    uint32_t visibility_invisible_selected_lod_geometry = 0;
    bool visibility_selection_subset = false;
    uint32_t compute_cull_visible_instances = 0;
    uint32_t compute_selection_draw_count = 0;
    uint32_t compute_occlusion_surviving_draws = 0;
    uint32_t debug_triangle_count = 0;
    uint32_t debug_vertex_count = 0;
    float debug_camera_distance = 0.0f;
    uint32_t presented_frame_count = 0;
    uint32_t swapchain_width = 0;
    uint32_t swapchain_height = 0;
    uint32_t worker_threads = 1;
    std::vector<std::string> physical_devices;
    UploadableScene uploadable_scene;
};

// Non-const on purpose: the demand-streaming path drops the resource's
// CPU-side geometry payload vectors after the .vgeo is serialized and
// mmap'd (the file becomes the source of truth for page bytes). A const&
// signature that mutates through const_cast is UB; this contract is honest.
VkBootstrapReport build_vk_bootstrap_report(VGeoResource& resource,
                                            const VkBootstrapConfig& config);

}  // namespace meridian
