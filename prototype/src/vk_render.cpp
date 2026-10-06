#include "vk_context.h"
#include "vk_helpers.h"
#include "shader_loader.h"
#include "math_utils.h"
#include "gpu_abi.h"
#include "resource_upload.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#if __has_include(<shaderc/shaderc.hpp>)
#include <shaderc/shaderc.hpp>
#define MERIDIAN_HAS_SHADERC 1
#else
#define MERIDIAN_HAS_SHADERC 0
#endif

namespace meridian {

#if MERIDIAN_VK_CONTEXT_HAS_VULKAN && MERIDIAN_VK_CONTEXT_HAS_GLFW

VkResult create_depth_resources(VkPhysicalDevice physical_device, VkDevice device,
                                const VkExtent2D& extent, DebugRenderContext& context) {
    context.depth_format = find_depth_format(physical_device);
    if (context.depth_format == VK_FORMAT_UNDEFINED) {
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.extent.width = extent.width;
    image_info.extent.height = extent.height;
    image_info.extent.depth = 1;
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.format = context.depth_format;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult result = vkCreateImage(device, &image_info, nullptr, &context.depth_image);
    if (result != VK_SUCCESS) {
        return result;
    }

    VkMemoryRequirements memory_requirements{};
    vkGetImageMemoryRequirements(device, context.depth_image, &memory_requirements);

    VkMemoryAllocateInfo allocate_info{};
    allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate_info.allocationSize = memory_requirements.size;
    allocate_info.memoryTypeIndex =
        find_memory_type(physical_device, memory_requirements.memoryTypeBits,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (allocate_info.memoryTypeIndex == kInvalidQueueFamily) {
        return VK_ERROR_MEMORY_MAP_FAILED;
    }

    result = vkAllocateMemory(device, &allocate_info, nullptr, &context.depth_memory);
    if (result != VK_SUCCESS) {
        return result;
    }

    result = vkBindImageMemory(device, context.depth_image, context.depth_memory, 0);
    if (result != VK_SUCCESS) {
        return result;
    }

    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = context.depth_image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = context.depth_format;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;
    return vkCreateImageView(device, &view_info, nullptr, &context.depth_view);
}

VkResult create_visibility_resources(VkPhysicalDevice physical_device, VkDevice device,
                                     const VkExtent2D& extent, DebugRenderContext& context) {
    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.extent.width = extent.width;
    image_info.extent.height = extent.height;
    image_info.extent.depth = 1;
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.format = context.visibility_format;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult result = vkCreateImage(device, &image_info, nullptr, &context.visibility_image);
    if (result != VK_SUCCESS) {
        return result;
    }

    VkMemoryRequirements memory_requirements{};
    vkGetImageMemoryRequirements(device, context.visibility_image, &memory_requirements);

    VkMemoryAllocateInfo allocate_info{};
    allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate_info.allocationSize = memory_requirements.size;
    allocate_info.memoryTypeIndex =
        find_memory_type(physical_device, memory_requirements.memoryTypeBits,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (allocate_info.memoryTypeIndex == kInvalidQueueFamily) {
        return VK_ERROR_MEMORY_MAP_FAILED;
    }

    result = vkAllocateMemory(device, &allocate_info, nullptr, &context.visibility_memory);
    if (result != VK_SUCCESS) {
        return result;
    }

    result = vkBindImageMemory(device, context.visibility_image, context.visibility_memory, 0);
    if (result != VK_SUCCESS) {
        return result;
    }

    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = context.visibility_image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = context.visibility_format;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;
    result = vkCreateImageView(device, &view_info, nullptr, &context.visibility_view);
    if (result != VK_SUCCESS) {
        return result;
    }

    const VkDeviceSize readback_size =
        static_cast<VkDeviceSize>(extent.width) * static_cast<VkDeviceSize>(extent.height) * sizeof(uint32_t) * 2;
    return create_uploaded_buffer(physical_device, device, nullptr, readback_size,
                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT, context.visibility_readback_buffer);
}

// Creates the base-color texture (binding 5) from the scene's embedded
// RGBA8 payload, or a 1x1 white placeholder when the scene is untextured.
// With generate_texture_mips the image carries a full blit-generated mip
// chain (GENERAL-layout upload: vkCmdBlitImage cannot read the
// TRANSFER_DST_OPTIMAL layout, so the mip path stages through GENERAL and
// transitions once at the end) and the sampler switches to trilinear +
// anisotropic filtering when the device enables samplerAnisotropy. The
// single-level path is byte-for-byte the historical one.
VkResult create_base_texture_resources(VkPhysicalDevice physical_device, VkDevice device,
                                        VkQueue upload_queue, uint32_t upload_queue_family,
                                        const UploadableScene& scene, bool generate_texture_mips,
                                        DebugRenderContext& context) {
    const bool has_scene_texture =
        !scene.texture_payload.empty() && scene.texture_width > 0 && scene.texture_height > 0;
    context.base_texture_is_placeholder = !has_scene_texture;
    context.base_texture_width = has_scene_texture ? scene.texture_width : 1;
    context.base_texture_height = has_scene_texture ? scene.texture_height : 1;
    context.base_texture_mip_levels =
        generate_texture_mips
            ? static_cast<uint32_t>(std::floor(std::log2(
                  static_cast<float>(std::max(context.base_texture_width,
                                              context.base_texture_height))))) + 1u
            : 1u;
    const uint32_t mip_levels = context.base_texture_mip_levels;

    const uint8_t white_pixel[4] = {0xff, 0xff, 0xff, 0xff};
    const void* pixels = nullptr;
    VkDeviceSize pixel_bytes = 0;
    if (has_scene_texture) {
        pixels = scene.texture_payload.data();
        pixel_bytes = static_cast<VkDeviceSize>(scene.texture_payload.size());
    } else {
        pixels = white_pixel;
        pixel_bytes = sizeof(white_pixel);
    }

    UploadedBuffer staging{};
    VkResult result = create_uploaded_buffer(physical_device, device, pixels, pixel_bytes,
                                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT, staging);
    if (result != VK_SUCCESS) {
        return result;
    }
    const auto destroy_staging = [&]() { destroy_uploaded_buffer(device, staging); };

    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.extent = {context.base_texture_width, context.base_texture_height, 1};
    image_info.mipLevels = mip_levels;
    image_info.arrayLayers = 1;
    image_info.format = VK_FORMAT_R8G8B8A8_UNORM;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (mip_levels > 1) {
        // Blit source for the mip chain (level i-1 feeds level i).
        image_info.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    }
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    result = vkCreateImage(device, &image_info, nullptr, &context.base_texture_image);
    if (result != VK_SUCCESS) {
        destroy_staging();
        return result;
    }

    VkMemoryRequirements memory_requirements{};
    vkGetImageMemoryRequirements(device, context.base_texture_image, &memory_requirements);
    const uint32_t memory_type = find_memory_type(physical_device, memory_requirements.memoryTypeBits,
                                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (memory_type == kInvalidQueueFamily) {
        vkDestroyImage(device, context.base_texture_image, nullptr);
        context.base_texture_image = VK_NULL_HANDLE;
        destroy_staging();
        return VK_ERROR_MEMORY_MAP_FAILED;
    }
    VkMemoryAllocateInfo allocate_info{};
    allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate_info.allocationSize = memory_requirements.size;
    allocate_info.memoryTypeIndex = memory_type;
    result = vkAllocateMemory(device, &allocate_info, nullptr, &context.base_texture_memory);
    if (result != VK_SUCCESS) {
        vkDestroyImage(device, context.base_texture_image, nullptr);
        context.base_texture_image = VK_NULL_HANDLE;
        destroy_staging();
        return result;
    }
    result = vkBindImageMemory(device, context.base_texture_image, context.base_texture_memory, 0);
    if (result != VK_SUCCESS) {
        vkFreeMemory(device, context.base_texture_memory, nullptr);
        context.base_texture_memory = VK_NULL_HANDLE;
        vkDestroyImage(device, context.base_texture_image, nullptr);
        context.base_texture_image = VK_NULL_HANDLE;
        destroy_staging();
        return result;
    }

    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = upload_queue_family;
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    result = vkCreateCommandPool(device, &pool_info, nullptr, &command_pool);
    if (result != VK_SUCCESS) {
        destroy_staging();
        return result;
    }
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo command_info{};
    command_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_info.commandPool = command_pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1;
    result = vkAllocateCommandBuffers(device, &command_info, &command_buffer);
    if (result != VK_SUCCESS) {
        vkDestroyCommandPool(device, command_pool, nullptr);
        destroy_staging();
        return result;
    }

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    result = vkBeginCommandBuffer(command_buffer, &begin);
    if (result != VK_SUCCESS) {
        vkDestroyCommandPool(device, command_pool, nullptr);
        destroy_staging();
        return result;
    }

    VkBufferImageCopy copy_region{};
    copy_region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy_region.imageExtent = {context.base_texture_width, context.base_texture_height, 1};

    if (mip_levels > 1) {
        // Mip path: the whole image lives in GENERAL for upload + chain
        // generation (vkCmdBlitImage requires TRANSFER_SRC_OPTIMAL or
        // GENERAL for its source; transitioning per-mip between blits is
        // avoidable by staying in GENERAL until the single final
        // transition). Each blit reads level i-1 after a write->read
        // barrier on that level; blits target TRANSFER_DST_OPTIMAL-free
        // layouts only, so both layouts here are GENERAL.
        VkImageMemoryBarrier to_general{};
        to_general.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        to_general.srcAccessMask = 0;
        to_general.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_general.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        to_general.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        to_general.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_general.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_general.image = context.base_texture_image;
        to_general.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mip_levels, 0, 1};
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                              &to_general);

        vkCmdCopyBufferToImage(command_buffer, staging.buffer, context.base_texture_image,
                               VK_IMAGE_LAYOUT_GENERAL, 1, &copy_region);

        for (uint32_t mip = 1; mip < mip_levels; ++mip) {
            VkImageMemoryBarrier level_ready{};
            level_ready.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            level_ready.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            level_ready.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            level_ready.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            level_ready.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            level_ready.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            level_ready.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            level_ready.image = context.base_texture_image;
            level_ready.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mip - 1, 1, 0, 1};
            vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                                  &level_ready);

            VkImageBlit blit{};
            blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip - 1, 0, 1};
            blit.srcOffsets[0] = {0, 0, 0};
            blit.srcOffsets[1] = {
                static_cast<int32_t>(std::max(1u, context.base_texture_width >> (mip - 1u))),
                static_cast<int32_t>(std::max(1u, context.base_texture_height >> (mip - 1u))),
                1};
            blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, 1};
            blit.dstOffsets[0] = {0, 0, 0};
            blit.dstOffsets[1] = {
                static_cast<int32_t>(std::max(1u, context.base_texture_width >> mip)),
                static_cast<int32_t>(std::max(1u, context.base_texture_height >> mip)),
                1};
            vkCmdBlitImage(command_buffer, context.base_texture_image, VK_IMAGE_LAYOUT_GENERAL,
                           context.base_texture_image, VK_IMAGE_LAYOUT_GENERAL, 1, &blit,
                           VK_FILTER_LINEAR);
        }

        VkImageMemoryBarrier to_shader_read{};
        to_shader_read.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        to_shader_read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_shader_read.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        to_shader_read.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        to_shader_read.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        to_shader_read.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_shader_read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_shader_read.image = context.base_texture_image;
        to_shader_read.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mip_levels, 0, 1};
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                              &to_shader_read);
    } else {
        VkImageMemoryBarrier to_transfer_dst{};
        to_transfer_dst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        to_transfer_dst.srcAccessMask = 0;
        to_transfer_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_transfer_dst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        to_transfer_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_transfer_dst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_transfer_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_transfer_dst.image = context.base_texture_image;
        to_transfer_dst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                              &to_transfer_dst);

        vkCmdCopyBufferToImage(command_buffer, staging.buffer, context.base_texture_image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);

        VkImageMemoryBarrier to_shader_read{};
        to_shader_read.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        to_shader_read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_shader_read.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        to_shader_read.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_shader_read.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        to_shader_read.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_shader_read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_shader_read.image = context.base_texture_image;
        to_shader_read.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                              &to_shader_read);
    }

    result = vkEndCommandBuffer(command_buffer);
    if (result != VK_SUCCESS) {
        vkDestroyCommandPool(device, command_pool, nullptr);
        destroy_staging();
        return result;
    }

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command_buffer;
    result = vkQueueSubmit(upload_queue, 1, &submit, VK_NULL_HANDLE);
    if (result == VK_SUCCESS) {
        result = vkQueueWaitIdle(upload_queue);
    }
    vkDestroyCommandPool(device, command_pool, nullptr);
    destroy_staging();
    if (result != VK_SUCCESS) {
        return result;
    }

    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = context.base_texture_image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = VK_FORMAT_R8G8B8A8_UNORM;
    view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mip_levels, 0, 1};
    result = vkCreateImageView(device, &view_info, nullptr, &context.base_texture_view);
    if (result != VK_SUCCESS) {
        return result;
    }

    // Anisotropic filtering only pairs with a mip chain (maxLod 0 never
    // leaves level 0) and only when the device feature was enabled at
    // device creation (see the samplerAnisotropy enable-if-supported line
    // in build_vk_bootstrap_report).
    VkPhysicalDeviceFeatures supported_features{};
    vkGetPhysicalDeviceFeatures(physical_device, &supported_features);
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical_device, &properties);
    const bool anisotropic = mip_levels > 1 && supported_features.samplerAnisotropy == VK_TRUE;

    VkSamplerCreateInfo sampler_info{};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.anisotropyEnable = anisotropic ? VK_TRUE : VK_FALSE;
    sampler_info.maxAnisotropy = anisotropic
                                     ? std::min(8.0f, properties.limits.maxSamplerAnisotropy)
                                     : 1.0f;
    sampler_info.compareEnable = VK_FALSE;
    sampler_info.minLod = 0.0f;
    sampler_info.maxLod = mip_levels > 1 ? static_cast<float>(mip_levels) : 0.0f;
    return vkCreateSampler(device, &sampler_info, nullptr, &context.base_texture_sampler);
}

VkResult create_debug_render_context(VkPhysicalDevice physical_device, VkDevice device,
                                      VkQueue upload_queue, uint32_t upload_queue_family,
                                      const SwapchainContext& swapchain,
                                      const UploadedSceneBuffers& scene_buffers,
                                      const UploadableScene& scene,
                                      const UploadedBuffer& draw_list,
                                      bool generate_texture_mips,
                                      DebugRenderContext& context) {
#if !MERIDIAN_HAS_SHADERC
    (void)physical_device;
    (void)device;
    (void)upload_queue;
    (void)upload_queue_family;
    (void)swapchain;
    (void)scene_buffers;
    (void)scene;
    (void)draw_list;
    (void)generate_texture_mips;
    (void)context;
    return VK_ERROR_FEATURE_NOT_PRESENT;
#else
    VkResult result = create_base_texture_resources(physical_device, device, upload_queue,
                                                     upload_queue_family, scene,
                                                     generate_texture_mips, context);
    if (result != VK_SUCCESS) {
        return result;
    }
    result = create_depth_resources(physical_device, device, swapchain.extent, context);
    if (result != VK_SUCCESS) {
        return result;
    }
    result = create_visibility_resources(physical_device, device, swapchain.extent, context);
    if (result != VK_SUCCESS) {
        return result;
    }
    VkAttachmentDescription color_attachment{};
    color_attachment.format = swapchain.surface_format.format;
    color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentDescription depth_attachment{};
    depth_attachment.format = context.depth_format;
    depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    // Depth is sampled by the HZB build after the render pass ends (per-frame
    // drawIndirectCount path and the diagnostic epilogue both read
    // debug_render.depth_image), so DONT_CARE here leaves the HZB sourcing
    // undefined (tile-memory discard on tilers). Both pass variants inherit
    // this description; stencil is never read and stays DONT_CARE.
    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription visibility_attachment{};
    visibility_attachment.format = context.visibility_format;
    visibility_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    visibility_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    visibility_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    visibility_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    visibility_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    visibility_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    visibility_attachment.finalLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkAttachmentReference color_attachment_ref{};
    color_attachment_ref.attachment = 0;
    color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depth_attachment_ref{};
    depth_attachment_ref.attachment = 1;
    depth_attachment_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference visibility_attachment_ref{};
    visibility_attachment_ref.attachment = 2;
    visibility_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_attachment_ref;
    subpass.pDepthStencilAttachment = &depth_attachment_ref;
    subpass.colorAttachmentCount = 2;
    const VkAttachmentReference color_attachments[] = {color_attachment_ref, visibility_attachment_ref};
    subpass.pColorAttachments = color_attachments;

    VkSubpassDependency dependencies[2]{};
    // Swapchain presentation -> color/depth writes of this pass
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                   VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                   VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependencies[0].srcAccessMask = 0;
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    // Shadow-pass depth-array writes (earlier in the same command buffer) -> fragment sampling
    dependencies[1].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].dstSubpass = 0;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    const VkAttachmentDescription attachments[] = {color_attachment, depth_attachment, visibility_attachment};

    VkRenderPassCreateInfo render_pass_info{};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.attachmentCount = 3;
    render_pass_info.pAttachments = attachments;
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass;
    render_pass_info.dependencyCount = 2;
    render_pass_info.pDependencies = dependencies;
    result = vkCreateRenderPass(device, &render_pass_info, nullptr, &context.render_pass);
    if (result != VK_SUCCESS) {
        return result;
    }

    // Transient variant: identical except the visibility attachment is not
    // stored. The visibility image is only read once per run (diagnostic
    // epilogue copy after the final frame), so normal frames skip the
    // 7.3MB store-back. Store-op differences do not affect render-pass
    // compatibility, so the framebuffers created against context.render_pass
    // are used with either pass.
    VkAttachmentDescription transient_attachments[] = {color_attachment, depth_attachment,
                                                       visibility_attachment};
    transient_attachments[2].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    render_pass_info.pAttachments = transient_attachments;
    result = vkCreateRenderPass(device, &render_pass_info, nullptr,
                                &context.render_pass_transient);
    if (result != VK_SUCCESS) {
        return result;
    }

    const std::string vertex_shader_source = load_shader_source(resolve_shader_path("main_geometry.vert"));
    const std::string fragment_shader_source = load_shader_source(resolve_shader_path("main_geometry.frag"));

    const std::vector<uint32_t> vert_spirv =
        compile_glsl_to_spirv(vertex_shader_source, shaderc_vertex_shader, "pull_geometry.vert");
    const std::vector<uint32_t> frag_spirv =
        compile_glsl_to_spirv(fragment_shader_source, shaderc_fragment_shader, "pull_geometry.frag");

    VkShaderModule vert_module = create_shader_module(device, vert_spirv);
    VkShaderModule frag_module = create_shader_module(device, frag_spirv);

    VkPipelineShaderStageCreateInfo shader_stages[2] = {};
    shader_stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    shader_stages[0].module = vert_module;
    shader_stages[0].pName = "main";
    shader_stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shader_stages[1].module = frag_module;
    shader_stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vertex_input_info{};
    vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewport_state{};
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;

    VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic_state{};
    dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic_state.dynamicStateCount = 2;
    dynamic_state.pDynamicStates = dynamic_states;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth_stencil{};
    depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth_stencil.depthTestEnable = VK_TRUE;
    depth_stencil.depthWriteEnable = VK_TRUE;
    depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState color_blend_attachments[2] = {};
    color_blend_attachments[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    color_blend_attachments[1].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT;
    // The main pass writes RGBA to the color attachment but only RG to the
    // R32G32_UINT visibility attachment. VUID 00605: without independentBlend
    // (mirrored, not required, at device creation) every color attachment
    // must use an identical blend state and colorWriteMask. On such devices
    // the visibility attachment takes the color attachment's RGBA mask:
    // mask bits for channels the destination format lacks are ignored, so
    // both visibility words still land and the shaded output is unchanged.
    {
        VkPhysicalDeviceFeatures supported_features{};
        vkGetPhysicalDeviceFeatures(physical_device, &supported_features);
        if (!supported_features.independentBlend) {
            color_blend_attachments[1].colorWriteMask =
                VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        }
    }

    VkPipelineColorBlendStateCreateInfo color_blending{};
    color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blending.attachmentCount = 2;
    color_blending.pAttachments = color_blend_attachments;

    // Descriptor set layout: 0=base SSBO, 1=lod SSBO, 2=frame UBO, 3=shadow sampler, 4=draw list SSBO, 5=base-color sampler
    VkDescriptorSetLayoutBinding ds_bindings[6] = {};
    ds_bindings[0].binding = 0;
    ds_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ds_bindings[0].descriptorCount = 1;
    ds_bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    ds_bindings[1].binding = 1;
    ds_bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ds_bindings[1].descriptorCount = 1;
    ds_bindings[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    ds_bindings[2].binding = 2;
    ds_bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ds_bindings[2].descriptorCount = 1;
    ds_bindings[2].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    ds_bindings[3].binding = 3;
    ds_bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ds_bindings[3].descriptorCount = 1;
    ds_bindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    ds_bindings[4].binding = 4;
    ds_bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ds_bindings[4].descriptorCount = 1;
    ds_bindings[4].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    ds_bindings[5].binding = 5;
    ds_bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ds_bindings[5].descriptorCount = 1;
    ds_bindings[5].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo set_layout_info{};
    set_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set_layout_info.bindingCount = 6;
    set_layout_info.pBindings = ds_bindings;
    result = vkCreateDescriptorSetLayout(device, &set_layout_info, nullptr,
                                         &context.descriptor_set_layout);
    if (result != VK_SUCCESS) {
        vkDestroyShaderModule(device, frag_module, nullptr);
        vkDestroyShaderModule(device, vert_module, nullptr);
        return result;
    }

    // Create frame UBO buffer
    FrameUBO initial_ubo{};
    result = create_uploaded_buffer(physical_device, device, &initial_ubo, sizeof(FrameUBO),
                                   VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                   context.frame_ubo);
    if (result != VK_SUCCESS) {
        vkDestroyShaderModule(device, frag_module, nullptr);
        vkDestroyShaderModule(device, vert_module, nullptr);
        return result;
    }

    // Create 1x1 placeholder depth image for shadow sampler binding until real shadow map is ready
    {
        VkFormat ph_fmt = find_depth_format(physical_device);
        VkImageCreateInfo ph_img{};
        ph_img.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ph_img.imageType = VK_IMAGE_TYPE_2D;
        ph_img.format = ph_fmt;
        ph_img.extent = {1, 1, 1};
        ph_img.mipLevels = 1;
        ph_img.arrayLayers = 1;
        ph_img.samples = VK_SAMPLE_COUNT_1_BIT;
        ph_img.tiling = VK_IMAGE_TILING_OPTIMAL;
        ph_img.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        result = vkCreateImage(device, &ph_img, nullptr, &context.placeholder_depth_image);
        if (result != VK_SUCCESS) {
            return result;
        }
        VkMemoryRequirements ph_req{};
        vkGetImageMemoryRequirements(device, context.placeholder_depth_image, &ph_req);
        VkMemoryAllocateInfo ph_alloc{};
        ph_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ph_alloc.allocationSize = ph_req.size;
        ph_alloc.memoryTypeIndex = find_memory_type(physical_device, ph_req.memoryTypeBits,
                                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (ph_alloc.memoryTypeIndex == kInvalidQueueFamily) {
            vkDestroyImage(device, context.placeholder_depth_image, nullptr);
            context.placeholder_depth_image = VK_NULL_HANDLE;
            return VK_ERROR_MEMORY_MAP_FAILED;
        }
        result = vkAllocateMemory(device, &ph_alloc, nullptr, &context.placeholder_depth_memory);
        if (result != VK_SUCCESS) {
            vkDestroyImage(device, context.placeholder_depth_image, nullptr);
            context.placeholder_depth_image = VK_NULL_HANDLE;
            return result;
        }
        result = vkBindImageMemory(device, context.placeholder_depth_image,
                                   context.placeholder_depth_memory, 0);
        if (result != VK_SUCCESS) {
            vkFreeMemory(device, context.placeholder_depth_memory, nullptr);
            context.placeholder_depth_memory = VK_NULL_HANDLE;
            vkDestroyImage(device, context.placeholder_depth_image, nullptr);
            context.placeholder_depth_image = VK_NULL_HANDLE;
            return result;
        }
        VkImageViewCreateInfo ph_view{};
        ph_view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ph_view.image = context.placeholder_depth_image;
        ph_view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ph_view.format = ph_fmt;
        ph_view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        ph_view.subresourceRange.levelCount = 1;
        ph_view.subresourceRange.layerCount = 1;
        result = vkCreateImageView(device, &ph_view, nullptr, &context.placeholder_depth_view);
        if (result != VK_SUCCESS) {
            vkFreeMemory(device, context.placeholder_depth_memory, nullptr);
            context.placeholder_depth_memory = VK_NULL_HANDLE;
            vkDestroyImage(device, context.placeholder_depth_image, nullptr);
            context.placeholder_depth_image = VK_NULL_HANDLE;
            return result;
        }
        VkSamplerCreateInfo ph_samp{};
        ph_samp.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        ph_samp.magFilter = VK_FILTER_NEAREST;
        ph_samp.minFilter = VK_FILTER_NEAREST;
        ph_samp.compareEnable = VK_TRUE;
        ph_samp.compareOp = VK_COMPARE_OP_LESS;
        ph_samp.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        ph_samp.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        ph_samp.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        result = vkCreateSampler(device, &ph_samp, nullptr, &context.placeholder_sampler);
        if (result != VK_SUCCESS) {
            vkDestroyImageView(device, context.placeholder_depth_view, nullptr);
            context.placeholder_depth_view = VK_NULL_HANDLE;
            vkFreeMemory(device, context.placeholder_depth_memory, nullptr);
            context.placeholder_depth_memory = VK_NULL_HANDLE;
            vkDestroyImage(device, context.placeholder_depth_image, nullptr);
            context.placeholder_depth_image = VK_NULL_HANDLE;
            return result;
        }
    }

    VkDescriptorPoolSize pool_sizes[3] = {};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_sizes[0].descriptorCount = 3;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_sizes[1].descriptorCount = 1;
    pool_sizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_sizes[2].descriptorCount = 2;

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 3;
    pool_info.pPoolSizes = pool_sizes;
    result = vkCreateDescriptorPool(device, &pool_info, nullptr, &context.descriptor_pool);
    if (result != VK_SUCCESS) {
        vkDestroyShaderModule(device, frag_module, nullptr);
        vkDestroyShaderModule(device, vert_module, nullptr);
        return result;
    }

    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = context.descriptor_pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &context.descriptor_set_layout;
    result = vkAllocateDescriptorSets(device, &alloc_info, &context.descriptor_set);
    if (result != VK_SUCCESS) {
        vkDestroyShaderModule(device, frag_module, nullptr);
        vkDestroyShaderModule(device, vert_module, nullptr);
        return result;
    }

    VkDescriptorBufferInfo buffer_infos[4] = {};
    buffer_infos[0].buffer = scene_buffers.base_payload.buffer;
    buffer_infos[0].range = scene_buffers.base_payload.size > 0 ? scene_buffers.base_payload.size : VK_WHOLE_SIZE;
    buffer_infos[1].buffer = scene_buffers.lod_payload.buffer;
    buffer_infos[1].range = scene_buffers.lod_payload.size > 0 ? scene_buffers.lod_payload.size : VK_WHOLE_SIZE;
    buffer_infos[2].buffer = context.frame_ubo.buffer;
    buffer_infos[2].range = sizeof(FrameUBO);
    buffer_infos[3].buffer = draw_list.buffer;
    buffer_infos[3].range = draw_list.size > 0 ? draw_list.size : VK_WHOLE_SIZE;

    VkDescriptorImageInfo image_info{};
    VkWriteDescriptorSet writes[5] = {};
    uint32_t write_count = 0;
    if (scene_buffers.base_payload.buffer != VK_NULL_HANDLE) {
        writes[write_count].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[write_count].dstSet = context.descriptor_set;
        writes[write_count].dstBinding = 0;
        writes[write_count].descriptorCount = 1;
        writes[write_count].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[write_count].pBufferInfo = &buffer_infos[0];
        write_count++;
    }
    if (scene_buffers.lod_payload.buffer != VK_NULL_HANDLE) {
        writes[write_count].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[write_count].dstSet = context.descriptor_set;
        writes[write_count].dstBinding = 1;
        writes[write_count].descriptorCount = 1;
        writes[write_count].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[write_count].pBufferInfo = &buffer_infos[1];
        write_count++;
    }
    writes[write_count].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[write_count].dstSet = context.descriptor_set;
    writes[write_count].dstBinding = 2;
    writes[write_count].descriptorCount = 1;
    writes[write_count].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[write_count].pBufferInfo = &buffer_infos[2];
    write_count++;
    if (draw_list.buffer != VK_NULL_HANDLE) {
        writes[write_count].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[write_count].dstSet = context.descriptor_set;
        writes[write_count].dstBinding = 4;
        writes[write_count].descriptorCount = 1;
        writes[write_count].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[write_count].pBufferInfo = &buffer_infos[3];
        write_count++;
    }
    if (context.base_texture_view != VK_NULL_HANDLE && context.base_texture_sampler != VK_NULL_HANDLE) {
        image_info.sampler = context.base_texture_sampler;
        image_info.imageView = context.base_texture_view;
        image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        writes[write_count].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[write_count].dstSet = context.descriptor_set;
        writes[write_count].dstBinding = 5;
        writes[write_count].descriptorCount = 1;
        writes[write_count].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[write_count].pImageInfo = &image_info;
        write_count++;
    }
    vkUpdateDescriptorSets(device, write_count, writes, 0, nullptr);

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &context.descriptor_set_layout;
    result = vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &context.pipeline_layout);
    if (result != VK_SUCCESS) {
        vkDestroyShaderModule(device, frag_module, nullptr);
        vkDestroyShaderModule(device, vert_module, nullptr);
        return result;
    }

    VkGraphicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_info.stageCount = 2;
    pipeline_info.pStages = shader_stages;
    pipeline_info.pVertexInputState = &vertex_input_info;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &rasterizer;
    pipeline_info.pMultisampleState = &multisampling;
    pipeline_info.pDepthStencilState = &depth_stencil;
    pipeline_info.pColorBlendState = &color_blending;
    pipeline_info.pDynamicState = &dynamic_state;
    pipeline_info.layout = context.pipeline_layout;
    pipeline_info.renderPass = context.render_pass;
    pipeline_info.subpass = 0;
    result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr,
                                       &context.pipeline);
    vkDestroyShaderModule(device, frag_module, nullptr);
    vkDestroyShaderModule(device, vert_module, nullptr);
    if (result != VK_SUCCESS) {
        return result;
    }

    context.framebuffers.resize(swapchain.image_views.size());
    for (size_t image_index = 0; image_index < swapchain.image_views.size(); ++image_index) {
        VkImageView framebuffer_attachments[] = {swapchain.image_views[image_index], context.depth_view,
                                                 context.visibility_view};
        VkFramebufferCreateInfo framebuffer_info{};
        framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebuffer_info.renderPass = context.render_pass;
        framebuffer_info.attachmentCount = 3;
        framebuffer_info.pAttachments = framebuffer_attachments;
        framebuffer_info.width = swapchain.extent.width;
        framebuffer_info.height = swapchain.extent.height;
        framebuffer_info.layers = 1;
        result = vkCreateFramebuffer(device, &framebuffer_info, nullptr,
                                     &context.framebuffers[image_index]);
        if (result != VK_SUCCESS) {
            return result;
        }
    }

    return VK_SUCCESS;
#endif
}

#endif  // MERIDIAN_VK_CONTEXT_HAS_VULKAN && MERIDIAN_VK_CONTEXT_HAS_GLFW

}  // namespace meridian
