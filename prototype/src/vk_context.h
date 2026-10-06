#pragma once

#include "math_utils.h"

#include <cstdint>
#include <vector>

#if __has_include(<vulkan/vulkan.h>)
#include <vulkan/vulkan.h>
#if defined(__APPLE__) && __has_include(<vulkan/vulkan_metal.h>)
#include <vulkan/vulkan_metal.h>
#endif
#define MERIDIAN_VK_CONTEXT_HAS_VULKAN 1
#else
#define MERIDIAN_VK_CONTEXT_HAS_VULKAN 0
#endif

#if __has_include(<GLFW/glfw3.h>)
#include <GLFW/glfw3.h>
#define MERIDIAN_VK_CONTEXT_HAS_GLFW 1
#else
#define MERIDIAN_VK_CONTEXT_HAS_GLFW 0
#endif

namespace meridian {

constexpr uint32_t kInvalidQueueFamily = 0xffffffffu;

#if MERIDIAN_VK_CONTEXT_HAS_VULKAN && MERIDIAN_VK_CONTEXT_HAS_GLFW

struct QueueFamilySelection {
    uint32_t graphics_family = kInvalidQueueFamily;
    uint32_t present_family = kInvalidQueueFamily;

    bool complete() const {
        return graphics_family != kInvalidQueueFamily && present_family != kInvalidQueueFamily;
    }
};

struct DeviceSelection {
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    QueueFamilySelection queues;
    bool enable_portability_subset = false;
    // KHR_draw_indirect_count (extension string or the Vulkan 1.2 core
    // feature) AND the core features the generated draws rely on:
    // multiDrawIndirect and drawIndirectFirstInstance. The runtime draw
    // path additionally checks max_draw_indirect_count against the draw
    // list capacities (see build_vk_bootstrap_report).
    bool has_draw_indirect_count = false;
    uint32_t max_draw_indirect_count = 0;
    // True when the device advertises the VK_KHR_draw_indirect_count
    // extension string (as opposed to only the promoted Vulkan 1.2 core
    // feature, which must then be requested via the Vulkan12Features
    // chain at device creation).
    bool has_draw_indirect_count_extension = false;
    // True when viewport/layer vertex-stage writes come from the Vulkan
    // 1.2 core shaderOutputLayer feature instead of
    // VK_EXT_shader_viewport_index_layer (devices >= 1.2).
    bool shader_output_layer_feature = false;
};

struct SwapchainContext {
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkSurfaceFormatKHR surface_format{};
    VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
    VkExtent2D extent{};
    std::vector<VkImage> images;
    std::vector<VkImageView> image_views;
    // Whether the created swapchain's images can be used as transfer
    // sources (surface supportedUsageFlags bit). The screenshot path is
    // the consumer; when false it must fail at request time instead of
    // recording a copy the usage flags never allowed.
    bool images_support_transfer_src = false;
};

struct FrameContext {
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkSemaphore image_available = VK_NULL_HANDLE;
    std::vector<VkSemaphore> render_finished_per_image;
    VkSemaphore render_finished = VK_NULL_HANDLE;
    VkFence in_flight = VK_NULL_HANDLE;
};

struct UploadedBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

struct UploadedSceneBuffers {
    UploadedBuffer header;
    UploadedBuffer instances;
    UploadedBuffer hierarchy_nodes;
    UploadedBuffer clusters;
    UploadedBuffer lod_groups;
    UploadedBuffer lod_clusters;
    UploadedBuffer node_lod_links;
    UploadedBuffer pages;
    UploadedBuffer page_dependencies;
    UploadedBuffer page_residency;
    UploadedBuffer base_payload;
    UploadedBuffer lod_payload;
};

struct DebugRenderContext {
    VkRenderPass render_pass = VK_NULL_HANDLE;
    // Same pass with the visibility attachment store-op flipped to
    // DONT_CARE; used for all frames except the capture frame that the
    // post-loop diagnostic epilogue reads back.
    VkRenderPass render_pass_transient = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuffers;
    VkImage depth_image = VK_NULL_HANDLE;
    VkDeviceMemory depth_memory = VK_NULL_HANDLE;
    VkImageView depth_view = VK_NULL_HANDLE;
    VkFormat depth_format = VK_FORMAT_UNDEFINED;
    VkImage visibility_image = VK_NULL_HANDLE;
    VkDeviceMemory visibility_memory = VK_NULL_HANDLE;
    VkImageView visibility_view = VK_NULL_HANDLE;
    VkFormat visibility_format = VK_FORMAT_R32G32_UINT;
    UploadedBuffer visibility_readback_buffer;
    UploadedBuffer frame_ubo;
    VkImage placeholder_depth_image = VK_NULL_HANDLE;
    VkDeviceMemory placeholder_depth_memory = VK_NULL_HANDLE;
    VkImageView placeholder_depth_view = VK_NULL_HANDLE;
    VkSampler placeholder_sampler = VK_NULL_HANDLE;
    // Base-color texture (binding 5): the scene's embedded RGBA8 texture,
    // or a 1x1 white placeholder when the scene is untextured.
    VkImage base_texture_image = VK_NULL_HANDLE;
    VkDeviceMemory base_texture_memory = VK_NULL_HANDLE;
    VkImageView base_texture_view = VK_NULL_HANDLE;
    VkSampler base_texture_sampler = VK_NULL_HANDLE;
    uint32_t base_texture_width = 0;
    uint32_t base_texture_height = 0;
    // Mip levels of the created base-color texture (1 = single-level
    // sampling, the historical path; > 1 = blit-generated chain).
    uint32_t base_texture_mip_levels = 1;
    bool base_texture_is_placeholder = false;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
};

struct CullPushConstants {
    float frustum_planes[6][4];
    uint32_t instance_count;
    uint32_t pad[3];
};

struct ComputeCullContext {
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    UploadedBuffer visible_instances;
    UploadedBuffer counter;
    uint32_t max_instances = 0;
};

struct SelectionPushConstants {
    float error_threshold;
    float camera_pos[3];
    uint32_t pad[4];
};

struct HzbContext {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    std::vector<VkImageView> mip_views;
    VkSampler sampler = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline depth_copy_pipeline = VK_NULL_HANDLE;
    VkPipelineLayout depth_copy_pipeline_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout depth_copy_set_layout = VK_NULL_HANDLE;
    VkDescriptorSet depth_copy_descriptor_set = VK_NULL_HANDLE;
    VkDescriptorPool depth_copy_descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> mip_descriptor_sets;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t mip_count = 0;
};

struct ComputeSelectionContext {
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    UploadedBuffer draw_list;
    UploadedBuffer draw_count;
    uint32_t max_draws = 0;
};

struct OcclusionPushConstants {
    float view_projection[16];
    uint32_t hzb_width;
    uint32_t hzb_height;
    uint32_t pad[2];
};

struct OcclusionRefineContext {
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkImageView hzb_full_view = VK_NULL_HANDLE;
    // Conservative fallback for frames where the previous frame's HZB does
    // not match the current camera/scene (see the temporal validity check in
    // the frame loop): a 1x1 image cleared to far depth. Sampling it never
    // rejects, so binding it in place of the stale HZB keeps the refine pass
    // correct (over-emits for one frame) with zero shader branches.
    VkDescriptorSet fallback_descriptor_set = VK_NULL_HANDLE;
    VkImageView fallback_hzb_view = VK_NULL_HANDLE;
    VkImage fallback_hzb_image = VK_NULL_HANDLE;
    VkDeviceMemory fallback_hzb_memory = VK_NULL_HANDLE;
    UploadedBuffer output_draws;
    UploadedBuffer output_count;
    uint32_t max_draws = 0;
};

struct ShadowContext {
    VkImage depth_image = VK_NULL_HANDLE;
    VkDeviceMemory depth_memory = VK_NULL_HANDLE;
    VkImageView depth_array_view = VK_NULL_HANDLE;
    // One layered framebuffer spanning all cascade layers; the merged draw
    // list renders every cascade in a single indirect draw via per-instance
    // gl_Layer selection.
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    VkRenderPass render_pass = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    // Merged multi-cascade draw list. Each entry carries the cluster's
    // cascade overlap mask in geometry_kind and draws one instance per
    // overlapping cascade; the vertex shader picks the light_vp and output
    // layer per instance.
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    UploadedBuffer draw_list;
    UploadedBuffer draw_count;
    uint32_t max_draws = 0;
    uint32_t resolution = 0;
    CascadeLightSetup cascades{};
    float scene_radius = 1.0f;
};

void destroy_uploaded_buffer(VkDevice device, UploadedBuffer& uploaded_buffer);
void destroy_uploaded_scene_buffers(VkDevice device, UploadedSceneBuffers& buffers);
void destroy_debug_render_context(VkDevice device, DebugRenderContext& context);
void destroy_frame_context(VkDevice device, FrameContext& frame);
void destroy_swapchain(VkDevice device, SwapchainContext& swapchain);
void destroy_compute_cull_context(VkDevice device, ComputeCullContext& context);
void destroy_hzb_context(VkDevice device, HzbContext& context);
void destroy_compute_selection_context(VkDevice device, ComputeSelectionContext& context);
void destroy_occlusion_refine_context(VkDevice device, OcclusionRefineContext& context);
void destroy_shadow_context(VkDevice device, ShadowContext& context);

// Create functions (implementations in per-subsystem .cpp files)
struct VGeoResource;
struct UploadableScene;

VkResult create_compute_cull_context(VkPhysicalDevice physical_device, VkDevice device,
                                      const UploadedSceneBuffers& scene_buffers,
                                      uint32_t instance_count,
                                      ComputeCullContext& context);

// Creates the draw-list/draw-count buffers the renderer consumes every
// frame. The cluster_select compute pipeline (11 SSBO bindings) is dead
// weight on the CPU-folded draw path, so it is only created when
// create_pipeline is explicitly requested -- and its absence or failure
// never breaks the context, because nothing dispatches it.
VkResult create_compute_selection_context(VkPhysicalDevice physical_device, VkDevice device,
                                          const UploadedSceneBuffers& scene_buffers,
                                          const ComputeCullContext& cull_context,
                                          uint32_t max_clusters,
                                          bool create_pipeline,
                                          ComputeSelectionContext& context);

VkResult create_hzb_context(VkPhysicalDevice physical_device, VkDevice device,
                            uint32_t width, uint32_t height,
                            VkImageView depth_view, VkFormat depth_format,
                            VkQueue init_queue, uint32_t init_queue_family,
                            HzbContext& context);

VkResult create_occlusion_refine_context(VkPhysicalDevice physical_device, VkDevice device,
                                          const ComputeSelectionContext& selection_ctx,
                                          const UploadedSceneBuffers& scene_buffers,
                                          const HzbContext& hzb,
                                          uint32_t max_draws,
                                          VkQueue init_queue, uint32_t init_queue_family,
                                          OcclusionRefineContext& context);

VkResult create_shadow_context(VkPhysicalDevice physical_device, VkDevice device,
                               const UploadedSceneBuffers& scene_buffers,
                               const UploadedBuffer& frame_ubo,
                               uint32_t max_draws,
                               const VGeoResource& resource,
                               uint32_t shadow_resolution,
                               ShadowContext& context);

VkResult create_depth_resources(VkPhysicalDevice physical_device, VkDevice device,
                                const VkExtent2D& extent, DebugRenderContext& context);

VkResult create_visibility_resources(VkPhysicalDevice physical_device, VkDevice device,
                                     const VkExtent2D& extent, DebugRenderContext& context);

VkResult create_debug_render_context(VkPhysicalDevice physical_device, VkDevice device,
                                      VkQueue upload_queue, uint32_t upload_queue_family,
                                      const SwapchainContext& swapchain,
                                      const UploadedSceneBuffers& scene_buffers,
                                      const struct UploadableScene& scene,
                                      const UploadedBuffer& draw_list,
                                      bool generate_texture_mips,
                                      DebugRenderContext& context);

#endif  // MERIDIAN_VK_CONTEXT_HAS_VULKAN && MERIDIAN_VK_CONTEXT_HAS_GLFW

}  // namespace meridian
