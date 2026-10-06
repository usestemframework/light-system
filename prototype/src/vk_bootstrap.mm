#include "vk_bootstrap.h"
#include "vk_context.h"
#include "vk_helpers.h"
#include "gpu_profiler.h"
#include "math_utils.h"
#include "async_reader.h"
#include "builder_internal.h"
#include "parallel_exec.h"
#include "runtime_model.h"
#include "shader_loader.h"
#include "streaming_scheduler.h"
#include "visibility_format.h"

#if __has_include(<vulkan/vulkan.h>)
#include <vulkan/vulkan.h>
#if defined(__APPLE__) && __has_include(<vulkan/vulkan_metal.h>)
#include <vulkan/vulkan_metal.h>
#endif
#define MERIDIAN_HAS_VULKAN 1
#else
#define MERIDIAN_HAS_VULKAN 0
#endif

#if __has_include(<GLFW/glfw3.h>)
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3native.h>
#define MERIDIAN_HAS_GLFW 1
#else
#define MERIDIAN_HAS_GLFW 0
#endif

#if MERIDIAN_HAS_VULKAN && !defined(VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR)
#define VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR 0x00000001
#endif

#if MERIDIAN_HAS_VULKAN && !defined(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)
#define VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME "VK_KHR_portability_enumeration"
#endif

#if MERIDIAN_HAS_VULKAN && !defined(VK_EXT_METAL_SURFACE_EXTENSION_NAME)
#define VK_EXT_METAL_SURFACE_EXTENSION_NAME "VK_EXT_metal_surface"
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>
#include <set>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#import <Cocoa/Cocoa.h>
#import <QuartzCore/CAMetalLayer.h>
#include <mach-o/dyld.h>
#endif

#if __has_include(<shaderc/shaderc.hpp>)
#include <shaderc/shaderc.hpp>
#define MERIDIAN_HAS_SHADERC 1
#else
#define MERIDIAN_HAS_SHADERC 0
#endif

namespace meridian {

std::filesystem::path resolve_shader_path(const char* name) {
    // Resolve relative to the executable's real location first, then fall back to CWD
    std::filesystem::path exe_dir;
#if defined(__APPLE__)
    uint32_t exe_path_size = 0;
    _NSGetExecutablePath(nullptr, &exe_path_size);
    std::string exe_path(exe_path_size, '\0');
    if (_NSGetExecutablePath(exe_path.data(), &exe_path_size) == 0) {
        std::error_code ec;
        const auto resolved = std::filesystem::canonical(exe_path, ec);
        if (!ec) {
            exe_dir = resolved.parent_path();
        }
    }
#elif defined(__linux__)
    std::error_code ec;
    const auto resolved = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (!ec) {
        exe_dir = resolved.parent_path();
    }
#endif
    std::vector<std::filesystem::path> candidates;
    if (!exe_dir.empty()) {
        candidates.push_back(exe_dir / ".." / "shaders" / name);  // build/.. = prototype root
        candidates.push_back(exe_dir / "shaders" / name);
    }
    candidates.push_back(std::filesystem::current_path() / ".." / "shaders" / name);
    candidates.push_back(std::filesystem::path("shaders") / name);
    for (const auto& p : candidates) {
        if (std::filesystem::exists(p)) return p;
    }
    throw std::runtime_error(std::string("shader not found: ") + name);
}

namespace {

#if MERIDIAN_HAS_VULKAN && MERIDIAN_HAS_GLFW

// Structs and types now provided by vk_context.h and math_utils.h

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
                // Homebrew's manifest lists a bare library filename that the
                // loader cannot dlopen outside default search paths. Write a
                // shadow manifest with an absolute path and point VK_LAYER_PATH
                // at it.
                std::ifstream input(manifest);
                std::ostringstream buffer;
                buffer << input.rdbuf();
                std::string text = buffer.str();
                const std::string from = "\"library_path\": \"libVkLayer_khronos_validation.dylib\"";
                const std::string to = "\"library_path\": \"" + std::filesystem::absolute(library).string() + "\"";
                if (text.find(from) != std::string::npos) {
                    std::error_code ec;
                    const std::filesystem::path shadow_dir =
                        std::filesystem::temp_directory_path(ec) / "meridian-vklayer";
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

bool supports_extension(const std::vector<VkExtensionProperties>& properties, const char* extension_name) {
    return std::any_of(properties.begin(), properties.end(), [&](const VkExtensionProperties& property) {
        return std::strcmp(property.extensionName, extension_name) == 0;
    });
}

}  // close anonymous namespace for shared helper definitions

uint32_t find_memory_type(VkPhysicalDevice physical_device, uint32_t type_bits,
                          VkMemoryPropertyFlags required_properties) {
    VkPhysicalDeviceMemoryProperties memory_properties{};
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

    for (uint32_t memory_index = 0; memory_index < memory_properties.memoryTypeCount; ++memory_index) {
        const bool type_supported = (type_bits & (1u << memory_index)) != 0;
        const bool properties_supported =
            (memory_properties.memoryTypes[memory_index].propertyFlags & required_properties) ==
            required_properties;
        if (type_supported && properties_supported) {
            return memory_index;
        }
    }

    return kInvalidQueueFamily;
}

VkResult create_uploaded_buffer(VkPhysicalDevice physical_device, VkDevice device, const void* data,
                                VkDeviceSize size, VkBufferUsageFlags usage,
                                UploadedBuffer& uploaded_buffer) {
    if (size == 0) {
        return VK_SUCCESS;
    }

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult result = vkCreateBuffer(device, &buffer_info, nullptr, &uploaded_buffer.buffer);
    if (result != VK_SUCCESS) {
        return result;
    }

    VkMemoryRequirements memory_requirements{};
    vkGetBufferMemoryRequirements(device, uploaded_buffer.buffer, &memory_requirements);

    VkMemoryAllocateInfo allocate_info{};
    allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate_info.allocationSize = memory_requirements.size;
    allocate_info.memoryTypeIndex =
        find_memory_type(physical_device, memory_requirements.memoryTypeBits,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (allocate_info.memoryTypeIndex == kInvalidQueueFamily) {
        return VK_ERROR_MEMORY_MAP_FAILED;
    }

    result = vkAllocateMemory(device, &allocate_info, nullptr, &uploaded_buffer.memory);
    if (result != VK_SUCCESS) {
        return result;
    }

    void* mapped = nullptr;
    result = vkMapMemory(device, uploaded_buffer.memory, 0, size, 0, &mapped);
    if (result != VK_SUCCESS) {
        return result;
    }
    if (data != nullptr) {
        std::memcpy(mapped, data, static_cast<size_t>(size));
    } else {
        std::memset(mapped, 0, static_cast<size_t>(size));
    }
    vkUnmapMemory(device, uploaded_buffer.memory);

    result = vkBindBufferMemory(device, uploaded_buffer.buffer, uploaded_buffer.memory, 0);
    if (result != VK_SUCCESS) {
        return result;
    }

    uploaded_buffer.size = size;
    return VK_SUCCESS;
}

// Uploads `data` to a DEVICE_LOCAL buffer via a one-shot HOST_VISIBLE staging
// buffer + vkCmdCopyBuffer on `queue`. Used for large, immutable assets like
// cluster geometry payload where GPU-local residency outperforms keeping the
// data CPU-mapped. Falls back to a HOST_VISIBLE buffer if the implementation
// does not surface a pure DEVICE_LOCAL memory type (no functional change,
// just skips the copy).
VkResult create_device_local_buffer_staged(VkPhysicalDevice physical_device, VkDevice device,
                                           VkQueue queue, uint32_t queue_family,
                                           const void* data, VkDeviceSize size,
                                           VkBufferUsageFlags usage,
                                           UploadedBuffer& out_buffer) {
    if (size == 0) return VK_SUCCESS;

    // If the platform has no DEVICE_LOCAL-without-HOST-VISIBLE memory type,
    // the staging dance is pure overhead -- just fall back to the existing
    // HOST_COHERENT create path.
    {
        VkBufferCreateInfo probe_info{};
        probe_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        probe_info.size = size;
        probe_info.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        probe_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkBuffer probe_buf = VK_NULL_HANDLE;
        if (vkCreateBuffer(device, &probe_info, nullptr, &probe_buf) == VK_SUCCESS) {
            VkMemoryRequirements mreq{};
            vkGetBufferMemoryRequirements(device, probe_buf, &mreq);
            vkDestroyBuffer(device, probe_buf, nullptr);
            const uint32_t dev_only = find_memory_type(physical_device, mreq.memoryTypeBits,
                                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            const uint32_t host_any = find_memory_type(physical_device, mreq.memoryTypeBits,
                                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
            if (dev_only == kInvalidQueueFamily || dev_only == host_any) {
                // Unified-memory path (typical for Apple): no staging win.
                return create_uploaded_buffer(physical_device, device, data, size, usage, out_buffer);
            }
        }
    }

    // Staging buffer (HOST_VISIBLE, TRANSFER_SRC).
    UploadedBuffer staging{};
    VkResult r = create_uploaded_buffer(physical_device, device, data, size,
                                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, staging);
    if (r != VK_SUCCESS) return r;

    // Destination buffer (DEVICE_LOCAL + usage + TRANSFER_DST).
    VkBufferCreateInfo buf_info{};
    buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_info.size = size;
    buf_info.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    r = vkCreateBuffer(device, &buf_info, nullptr, &out_buffer.buffer);
    if (r != VK_SUCCESS) { destroy_uploaded_buffer(device, staging); return r; }

    VkMemoryRequirements mreq{};
    vkGetBufferMemoryRequirements(device, out_buffer.buffer, &mreq);
    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = mreq.size;
    alloc.memoryTypeIndex = find_memory_type(physical_device, mreq.memoryTypeBits,
                                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (alloc.memoryTypeIndex == kInvalidQueueFamily) {
        destroy_uploaded_buffer(device, staging);
        vkDestroyBuffer(device, out_buffer.buffer, nullptr);
        out_buffer.buffer = VK_NULL_HANDLE;
        return VK_ERROR_MEMORY_MAP_FAILED;
    }
    r = vkAllocateMemory(device, &alloc, nullptr, &out_buffer.memory);
    if (r != VK_SUCCESS) { destroy_uploaded_buffer(device, staging); return r; }
    r = vkBindBufferMemory(device, out_buffer.buffer, out_buffer.memory, 0);
    if (r != VK_SUCCESS) { destroy_uploaded_buffer(device, staging); return r; }
    out_buffer.size = size;

    // One-shot transient command pool for the copy. Failure paths destroy
    // what this call created so the caller never sees half-built handles.
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = queue_family;
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    r = vkCreateCommandPool(device, &pool_info, nullptr, &pool);
    if (r != VK_SUCCESS) {
        destroy_uploaded_buffer(device, staging);
        destroy_uploaded_buffer(device, out_buffer);
        return r;
    }

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo cb_info{};
    cb_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cb_info.commandPool = pool;
    cb_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cb_info.commandBufferCount = 1;
    r = vkAllocateCommandBuffers(device, &cb_info, &cmd);
    if (r != VK_SUCCESS) {
        vkDestroyCommandPool(device, pool, nullptr);
        destroy_uploaded_buffer(device, staging);
        destroy_uploaded_buffer(device, out_buffer);
        return r;
    }

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    r = vkBeginCommandBuffer(cmd, &begin);
    if (r != VK_SUCCESS) {
        vkDestroyCommandPool(device, pool, nullptr);
        destroy_uploaded_buffer(device, staging);
        destroy_uploaded_buffer(device, out_buffer);
        return r;
    }

    VkBufferCopy copy{};
    copy.size = size;
    vkCmdCopyBuffer(cmd, staging.buffer, out_buffer.buffer, 1, &copy);

    r = vkEndCommandBuffer(cmd);
    if (r != VK_SUCCESS) {
        vkDestroyCommandPool(device, pool, nullptr);
        destroy_uploaded_buffer(device, staging);
        destroy_uploaded_buffer(device, out_buffer);
        return r;
    }

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    r = vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
    if (r == VK_SUCCESS) {
        r = vkQueueWaitIdle(queue);
    }
    if (r != VK_SUCCESS) {
        vkDestroyCommandPool(device, pool, nullptr);
        destroy_uploaded_buffer(device, staging);
        destroy_uploaded_buffer(device, out_buffer);
        return r;
    }

    vkDestroyCommandPool(device, pool, nullptr);
    destroy_uploaded_buffer(device, staging);
    return VK_SUCCESS;
}

namespace {  // reopen anonymous namespace for internal helpers

VkResult update_uploaded_buffer(VkDevice device, const void* data, VkDeviceSize size,
                                UploadedBuffer& uploaded_buffer) {
    if (size == 0 || uploaded_buffer.memory == VK_NULL_HANDLE) {
        return VK_SUCCESS;
    }
    if (size > uploaded_buffer.size) {
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }

    void* mapped = nullptr;
    VkResult result = vkMapMemory(device, uploaded_buffer.memory, 0, size, 0, &mapped);
    if (result != VK_SUCCESS) {
        return result;
    }
    std::memcpy(mapped, data, static_cast<size_t>(size));
    vkUnmapMemory(device, uploaded_buffer.memory);
    return VK_SUCCESS;
}

}  // close anonymous namespace for header-declared function definitions

void destroy_uploaded_buffer(VkDevice device, UploadedBuffer& uploaded_buffer) {
    if (uploaded_buffer.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, uploaded_buffer.buffer, nullptr);
    }
    if (uploaded_buffer.memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, uploaded_buffer.memory, nullptr);
    }
    uploaded_buffer = {};
}

void destroy_uploaded_scene_buffers(VkDevice device, UploadedSceneBuffers& buffers) {
    destroy_uploaded_buffer(device, buffers.lod_payload);
    destroy_uploaded_buffer(device, buffers.base_payload);
    destroy_uploaded_buffer(device, buffers.page_residency);
    destroy_uploaded_buffer(device, buffers.page_dependencies);
    destroy_uploaded_buffer(device, buffers.pages);
    destroy_uploaded_buffer(device, buffers.node_lod_links);
    destroy_uploaded_buffer(device, buffers.lod_clusters);
    destroy_uploaded_buffer(device, buffers.lod_groups);
    destroy_uploaded_buffer(device, buffers.clusters);
    destroy_uploaded_buffer(device, buffers.hierarchy_nodes);
    destroy_uploaded_buffer(device, buffers.instances);
    destroy_uploaded_buffer(device, buffers.header);
}

namespace {  // reopen anonymous namespace

// Streaming counterpart of create_uploaded_buffer: allocates the payload
// buffer at full size but never touches the bytes, so on unified-memory
// platforms the pages stay uncommitted until per-page uploads land. The
// memory type is HOST_VISIBLE|HOST_COHERENT (same search as
// create_uploaded_buffer) so upload_page_bytes can map sub-ranges.
VkResult create_empty_uploaded_buffer(VkPhysicalDevice physical_device, VkDevice device,
                                      VkDeviceSize size, VkBufferUsageFlags usage,
                                      UploadedBuffer& uploaded_buffer) {
    if (size == 0) {
        return VK_SUCCESS;
    }

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult result = vkCreateBuffer(device, &buffer_info, nullptr, &uploaded_buffer.buffer);
    if (result != VK_SUCCESS) {
        return result;
    }

    VkMemoryRequirements memory_requirements{};
    vkGetBufferMemoryRequirements(device, uploaded_buffer.buffer, &memory_requirements);

    VkMemoryAllocateInfo allocate_info{};
    allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate_info.allocationSize = memory_requirements.size;
    allocate_info.memoryTypeIndex =
        find_memory_type(physical_device, memory_requirements.memoryTypeBits,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (allocate_info.memoryTypeIndex == kInvalidQueueFamily) {
        vkDestroyBuffer(device, uploaded_buffer.buffer, nullptr);
        uploaded_buffer.buffer = VK_NULL_HANDLE;
        return VK_ERROR_MEMORY_MAP_FAILED;
    }

    result = vkAllocateMemory(device, &allocate_info, nullptr, &uploaded_buffer.memory);
    if (result != VK_SUCCESS) {
        vkDestroyBuffer(device, uploaded_buffer.buffer, nullptr);
        uploaded_buffer.buffer = VK_NULL_HANDLE;
        return result;
    }

    result = vkBindBufferMemory(device, uploaded_buffer.buffer, uploaded_buffer.memory, 0);
    if (result != VK_SUCCESS) {
        destroy_uploaded_buffer(device, uploaded_buffer);
        return result;
    }

    uploaded_buffer.size = size;
    return VK_SUCCESS;
}

// Per-page sub-buffer upload into a payload buffer. Fast path maps the
// destination range directly (HOST_COHERENT unified memory); if the memory
// is not host-mappable (discrete DEVICE_LOCAL), falls back to a one-shot
// staging buffer + vkCmdCopyBuffer, mirroring
// create_device_local_buffer_staged.
VkResult upload_page_bytes(VkPhysicalDevice physical_device, VkDevice device, VkQueue queue,
                           uint32_t queue_family, const void* data, VkDeviceSize size,
                           VkDeviceSize dst_offset, UploadedBuffer& dst) {
    if (size == 0 || dst.memory == VK_NULL_HANDLE) {
        return VK_SUCCESS;
    }
    if (dst_offset + size > dst.size) {
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }

    void* mapped = nullptr;
    if (vkMapMemory(device, dst.memory, dst_offset, size, 0, &mapped) == VK_SUCCESS) {
        std::memcpy(mapped, data, static_cast<size_t>(size));
        vkUnmapMemory(device, dst.memory);
        return VK_SUCCESS;
    }

    UploadedBuffer staging{};
    VkResult result = create_uploaded_buffer(physical_device, device, data, size,
                                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT, staging);
    if (result != VK_SUCCESS) return result;

    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = queue_family;
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    result = vkCreateCommandPool(device, &pool_info, nullptr, &pool);
    if (result != VK_SUCCESS) {
        destroy_uploaded_buffer(device, staging);
        return result;
    }

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo cb_info{};
    cb_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cb_info.commandPool = pool;
    cb_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cb_info.commandBufferCount = 1;
    result = vkAllocateCommandBuffers(device, &cb_info, &cmd);
    if (result != VK_SUCCESS) {
        vkDestroyCommandPool(device, pool, nullptr);
        destroy_uploaded_buffer(device, staging);
        return result;
    }

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    result = vkBeginCommandBuffer(cmd, &begin);
    if (result != VK_SUCCESS) {
        vkDestroyCommandPool(device, pool, nullptr);
        destroy_uploaded_buffer(device, staging);
        return result;
    }

    VkBufferCopy copy{};
    copy.srcOffset = 0;
    copy.dstOffset = dst_offset;
    copy.size = size;
    vkCmdCopyBuffer(cmd, staging.buffer, dst.buffer, 1, &copy);

    result = vkEndCommandBuffer(cmd);
    if (result != VK_SUCCESS) {
        vkDestroyCommandPool(device, pool, nullptr);
        destroy_uploaded_buffer(device, staging);
        return result;
    }

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    result = vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
    if (result == VK_SUCCESS) {
        result = vkQueueWaitIdle(queue);
    }
    if (result != VK_SUCCESS) {
        vkDestroyCommandPool(device, pool, nullptr);
        destroy_uploaded_buffer(device, staging);
        return result;
    }

    vkDestroyCommandPool(device, pool, nullptr);
    destroy_uploaded_buffer(device, staging);
    return VK_SUCCESS;
}


CameraFrameData build_camera_frame_data(const VGeoResource& resource, const VkExtent2D& extent,
                                        float& camera_distance) {
    const Vec3f center = {
        (resource.bounds.min.x + resource.bounds.max.x) * 0.5f,
        (resource.bounds.min.y + resource.bounds.max.y) * 0.5f,
        (resource.bounds.min.z + resource.bounds.max.z) * 0.5f,
    };
    const Vec3f extents = {
        resource.bounds.max.x - resource.bounds.min.x,
        resource.bounds.max.y - resource.bounds.min.y,
        resource.bounds.max.z - resource.bounds.min.z,
    };
    const float radius = std::max({extents.x, extents.y, extents.z, 1.0f});
    camera_distance = radius * 1.5f;

    const Vec3f eye = {center.x + radius * 0.4f, center.y + radius * 0.5f,
                       center.z + camera_distance};
    const float aspect_ratio =
        std::max(1.0f, static_cast<float>(extent.width)) / std::max(1.0f, static_cast<float>(extent.height));
    const Mat4f view = look_at_matrix(eye, center, {0.0f, 1.0f, 0.0f});
    const Mat4f projection = perspective_matrix(55.0f * 3.1415926535f / 180.0f, aspect_ratio,
                                                std::max(0.01f, radius * 0.01f), radius * 8.0f);

    CameraFrameData frame_data{};
    frame_data.view_projection = multiply_matrix(projection, view);
    frame_data.camera_position = eye;
    return frame_data;
}

}  // close anonymous namespace

void destroy_compute_cull_context(VkDevice device, ComputeCullContext& context) {
    destroy_uploaded_buffer(device, context.visible_instances);
    destroy_uploaded_buffer(device, context.counter);
    if (context.descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, context.descriptor_pool, nullptr);
    }
    if (context.descriptor_set_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, context.descriptor_set_layout, nullptr);
    }
    if (context.pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, context.pipeline, nullptr);
    }
    if (context.pipeline_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, context.pipeline_layout, nullptr);
    }
    context = {};
}

// create_compute_cull_context moved to vk_compute_cull.cpp

void destroy_hzb_context(VkDevice device, HzbContext& context) {
    for (VkImageView view : context.mip_views) {
        if (view != VK_NULL_HANDLE) vkDestroyImageView(device, view, nullptr);
    }
    context.mip_views.clear();
    context.mip_descriptor_sets.clear();
    if (context.sampler != VK_NULL_HANDLE) vkDestroySampler(device, context.sampler, nullptr);
    if (context.descriptor_pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, context.descriptor_pool, nullptr);
    if (context.descriptor_set_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, context.descriptor_set_layout, nullptr);
    if (context.depth_copy_set_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, context.depth_copy_set_layout, nullptr);
    if (context.depth_copy_descriptor_pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, context.depth_copy_descriptor_pool, nullptr);
    if (context.pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, context.pipeline, nullptr);
    if (context.pipeline_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, context.pipeline_layout, nullptr);
    if (context.depth_copy_pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, context.depth_copy_pipeline, nullptr);
    if (context.depth_copy_pipeline_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, context.depth_copy_pipeline_layout, nullptr);
    if (context.image != VK_NULL_HANDLE) vkDestroyImage(device, context.image, nullptr);
    if (context.memory != VK_NULL_HANDLE) vkFreeMemory(device, context.memory, nullptr);
    context = {};
}

void destroy_compute_selection_context(VkDevice device, ComputeSelectionContext& context) {
    destroy_uploaded_buffer(device, context.draw_list);
    destroy_uploaded_buffer(device, context.draw_count);
    if (context.descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, context.descriptor_pool, nullptr);
    }
    if (context.descriptor_set_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, context.descriptor_set_layout, nullptr);
    }
    if (context.pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, context.pipeline, nullptr);
    }
    if (context.pipeline_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, context.pipeline_layout, nullptr);
    }
    context = {};
}
// create_compute_selection_context moved to vk_compute_selection.cpp
// create_hzb_context moved to vk_hzb.cpp


void destroy_occlusion_refine_context(VkDevice device, OcclusionRefineContext& context) {
    destroy_uploaded_buffer(device, context.output_draws);
    destroy_uploaded_buffer(device, context.output_count);
    if (context.descriptor_pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, context.descriptor_pool, nullptr);
    if (context.descriptor_set_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, context.descriptor_set_layout, nullptr);
    if (context.hzb_full_view != VK_NULL_HANDLE) vkDestroyImageView(device, context.hzb_full_view, nullptr);
    if (context.fallback_hzb_view != VK_NULL_HANDLE) vkDestroyImageView(device, context.fallback_hzb_view, nullptr);
    if (context.fallback_hzb_image != VK_NULL_HANDLE) vkDestroyImage(device, context.fallback_hzb_image, nullptr);
    if (context.fallback_hzb_memory != VK_NULL_HANDLE) vkFreeMemory(device, context.fallback_hzb_memory, nullptr);
    if (context.pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, context.pipeline, nullptr);
    if (context.pipeline_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, context.pipeline_layout, nullptr);
    context = {};
}

// create_occlusion_refine_context moved to vk_occlusion.cpp


void destroy_shadow_context(VkDevice device, ShadowContext& context) {
    if (context.framebuffer != VK_NULL_HANDLE) vkDestroyFramebuffer(device, context.framebuffer, nullptr);
    if (context.render_pass != VK_NULL_HANDLE) vkDestroyRenderPass(device, context.render_pass, nullptr);
    if (context.depth_array_view != VK_NULL_HANDLE) vkDestroyImageView(device, context.depth_array_view, nullptr);
    if (context.sampler != VK_NULL_HANDLE) vkDestroySampler(device, context.sampler, nullptr);
    if (context.depth_image != VK_NULL_HANDLE) vkDestroyImage(device, context.depth_image, nullptr);
    if (context.depth_memory != VK_NULL_HANDLE) vkFreeMemory(device, context.depth_memory, nullptr);
    destroy_uploaded_buffer(device, context.draw_list);
    destroy_uploaded_buffer(device, context.draw_count);
    if (context.descriptor_pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, context.descriptor_pool, nullptr);
    if (context.descriptor_set_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, context.descriptor_set_layout, nullptr);
    if (context.pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, context.pipeline, nullptr);
    if (context.pipeline_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, context.pipeline_layout, nullptr);
    context = {};
}

// create_shadow_context moved to vk_shadow.cpp

namespace {  // reopen anonymous namespace for internal helpers

void update_debug_selection_report(const TraversalSelection& selection,
                                   const UploadableScene& scene,
                                   VkBootstrapReport& report) {
    report.debug_selected_node_count = static_cast<uint32_t>(selection.selected_node_indices.size());
    report.debug_rendered_cluster_count = static_cast<uint32_t>(selection.selected_cluster_indices.size());
    report.debug_rendered_lod_cluster_count = static_cast<uint32_t>(selection.selected_lod_cluster_indices.size());
    report.replay_selected_node_count = static_cast<uint32_t>(selection.selected_node_indices.size());
    report.replay_selected_cluster_count = static_cast<uint32_t>(selection.selected_cluster_indices.size());
    report.replay_selected_lod_cluster_count =
        static_cast<uint32_t>(selection.selected_lod_cluster_indices.size());
    report.replay_selected_page_count = static_cast<uint32_t>(selection.selected_page_indices.size());

    report.debug_triangle_count = 0;
    report.debug_vertex_count = 0;
    for (const uint32_t cluster_index : selection.selected_cluster_indices) {
        const GpuClusterRecord& cluster = scene.clusters[cluster_index];
        report.debug_triangle_count += cluster.local_triangle_count;
        report.debug_vertex_count += cluster.local_vertex_count;
    }
    for (const uint32_t lod_cluster_index : selection.selected_lod_cluster_indices) {
        const GpuLodClusterRecord& cluster = scene.lod_clusters[lod_cluster_index];
        report.debug_triangle_count += cluster.local_triangle_count;
        report.debug_vertex_count += cluster.local_vertex_count;
    }
}

}  // close anonymous namespace


#if MERIDIAN_HAS_SHADERC
std::vector<uint32_t> compile_glsl_to_spirv(const std::string& source, shaderc_shader_kind kind,
                                            const char* name) {
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;
    options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
    const shaderc::SpvCompilationResult result =
        compiler.CompileGlslToSpv(source, kind, name, options);
    if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
        throw BuilderError(std::string("shader compilation failed for ") + name + ": " +
                           result.GetErrorMessage());
    }
    return {result.cbegin(), result.cend()};
}
#endif

VkShaderModule create_shader_module(VkDevice device, const std::vector<uint32_t>& spirv) {
    VkShaderModuleCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = spirv.size() * sizeof(uint32_t);
    create_info.pCode = spirv.data();

    VkShaderModule shader_module = VK_NULL_HANDLE;
    const VkResult result = vkCreateShaderModule(device, &create_info, nullptr, &shader_module);
    if (result != VK_SUCCESS) {
        throw BuilderError("vkCreateShaderModule failed");
    }
    return shader_module;
}



void destroy_debug_render_context(VkDevice device, DebugRenderContext& context) {
    destroy_uploaded_buffer(device, context.visibility_readback_buffer);
    destroy_uploaded_buffer(device, context.frame_ubo);
    if (context.placeholder_depth_view != VK_NULL_HANDLE) vkDestroyImageView(device, context.placeholder_depth_view, nullptr);
    if (context.placeholder_depth_image != VK_NULL_HANDLE) vkDestroyImage(device, context.placeholder_depth_image, nullptr);
    if (context.placeholder_depth_memory != VK_NULL_HANDLE) vkFreeMemory(device, context.placeholder_depth_memory, nullptr);
    if (context.placeholder_sampler != VK_NULL_HANDLE) vkDestroySampler(device, context.placeholder_sampler, nullptr);
    if (context.base_texture_sampler != VK_NULL_HANDLE) vkDestroySampler(device, context.base_texture_sampler, nullptr);
    if (context.base_texture_view != VK_NULL_HANDLE) vkDestroyImageView(device, context.base_texture_view, nullptr);
    if (context.base_texture_image != VK_NULL_HANDLE) vkDestroyImage(device, context.base_texture_image, nullptr);
    if (context.base_texture_memory != VK_NULL_HANDLE) vkFreeMemory(device, context.base_texture_memory, nullptr);
    if (context.descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, context.descriptor_pool, nullptr);
    }
    if (context.descriptor_set_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, context.descriptor_set_layout, nullptr);
    }
    for (VkFramebuffer framebuffer : context.framebuffers) {
        vkDestroyFramebuffer(device, framebuffer, nullptr);
    }
    context.framebuffers.clear();
    if (context.depth_view != VK_NULL_HANDLE) {
        vkDestroyImageView(device, context.depth_view, nullptr);
    }
    if (context.depth_image != VK_NULL_HANDLE) {
        vkDestroyImage(device, context.depth_image, nullptr);
    }
    if (context.depth_memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, context.depth_memory, nullptr);
    }
    if (context.visibility_view != VK_NULL_HANDLE) {
        vkDestroyImageView(device, context.visibility_view, nullptr);
    }
    if (context.visibility_image != VK_NULL_HANDLE) {
        vkDestroyImage(device, context.visibility_image, nullptr);
    }
    if (context.visibility_memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, context.visibility_memory, nullptr);
    }
    if (context.pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, context.pipeline, nullptr);
    }
    if (context.pipeline_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, context.pipeline_layout, nullptr);
    }
    if (context.render_pass_transient != VK_NULL_HANDLE) {
        vkDestroyRenderPass(device, context.render_pass_transient, nullptr);
    }
    if (context.render_pass != VK_NULL_HANDLE) {
        vkDestroyRenderPass(device, context.render_pass, nullptr);
    }
    context = {};
}

namespace {  // reopen anonymous namespace

template <typename T>
VkResult upload_vector_buffer(VkPhysicalDevice physical_device, VkDevice device,
                              const std::vector<T>& values, VkBufferUsageFlags usage,
                              UploadedBuffer& uploaded_buffer);

template <typename T>
VkResult update_vector_buffer(VkDevice device, const std::vector<T>& values,
                              UploadedBuffer& uploaded_buffer);

template <typename T>
VkResult upload_vector_buffer(VkPhysicalDevice physical_device, VkDevice device,
                              const std::vector<T>& values, VkBufferUsageFlags usage,
                              UploadedBuffer& uploaded_buffer) {
    if (values.empty()) {
        return VK_SUCCESS;
    }
    return create_uploaded_buffer(physical_device, device, values.data(),
                                  static_cast<VkDeviceSize>(values.size() * sizeof(T)), usage,
                                  uploaded_buffer);
}

template <typename T>
VkResult update_vector_buffer(VkDevice device, const std::vector<T>& values,
                              UploadedBuffer& uploaded_buffer) {
    if (values.empty()) {
        return VK_SUCCESS;
    }
    return update_uploaded_buffer(device, values.data(),
                                  static_cast<VkDeviceSize>(values.size() * sizeof(T)),
                                  uploaded_buffer);
}

VkResult upload_scene_buffers(VkPhysicalDevice physical_device, VkDevice device,
                              VkQueue upload_queue, uint32_t upload_queue_family,
                              const UploadableScene& scene, UploadedSceneBuffers& buffers,
                              VkBootstrapReport& report, bool stream_payloads) {
    const VkBufferUsageFlags metadata_usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VkResult result = create_uploaded_buffer(physical_device, device, &scene.header,
                                             sizeof(scene.header), metadata_usage, buffers.header);
    if (result != VK_SUCCESS) return result;
    result = upload_vector_buffer(physical_device, device, scene.instances, metadata_usage,
                                  buffers.instances);
    if (result != VK_SUCCESS) return result;
    result = upload_vector_buffer(physical_device, device, scene.hierarchy_nodes, metadata_usage,
                                  buffers.hierarchy_nodes);
    if (result != VK_SUCCESS) return result;
    result = upload_vector_buffer(physical_device, device, scene.clusters, metadata_usage,
                                  buffers.clusters);
    if (result != VK_SUCCESS) return result;
    result = upload_vector_buffer(physical_device, device, scene.lod_groups, metadata_usage,
                                  buffers.lod_groups);
    if (result != VK_SUCCESS) return result;
    result = upload_vector_buffer(physical_device, device, scene.lod_clusters, metadata_usage,
                                  buffers.lod_clusters);
    if (result != VK_SUCCESS) return result;
    result = upload_vector_buffer(physical_device, device, scene.node_lod_links, metadata_usage,
                                  buffers.node_lod_links);
    if (result != VK_SUCCESS) return result;
    result = upload_vector_buffer(physical_device, device, scene.pages, metadata_usage,
                                  buffers.pages);
    if (result != VK_SUCCESS) return result;
    result = upload_vector_buffer(physical_device, device, scene.page_dependencies, metadata_usage,
                                  buffers.page_dependencies);
    if (result != VK_SUCCESS) return result;
    result = upload_vector_buffer(physical_device, device, scene.page_residency, metadata_usage,
                                  buffers.page_residency);
    if (result != VK_SUCCESS) return result;
    if (stream_payloads) {
        // Demand-streaming: allocate the payload buffers at full size but
        // leave them unpopulated -- page bytes arrive via per-page uploads
        // as the residency scheduler completes loads. Nothing is read from
        // these ranges until the owning page is resident, so the
        // untouched (uncommitted on unified memory) content is never
        // observed.
        if (!scene.base_payload.empty()) {
            result = create_empty_uploaded_buffer(
                physical_device, device,
                static_cast<VkDeviceSize>(scene.base_payload.size()), metadata_usage,
                buffers.base_payload);
        }
        if (result != VK_SUCCESS) return result;
        if (!scene.lod_payload.empty()) {
            result = create_empty_uploaded_buffer(
                physical_device, device,
                static_cast<VkDeviceSize>(scene.lod_payload.size()), metadata_usage,
                buffers.lod_payload);
        }
        if (result != VK_SUCCESS) return result;
        std::fprintf(stderr,
                     "MERIDIAN_STREAM: payload buffers allocated empty (base=%llu lod=%llu "
                     "bytes), pages upload on demand\n",
                     static_cast<unsigned long long>(scene.base_payload.size()),
                     static_cast<unsigned long long>(scene.lod_payload.size()));
    } else {
        // Payload buffers are large and immutable. Push them to DEVICE_LOCAL via
        // a staged copy when the platform has a dedicated device-only heap.
        if (!scene.base_payload.empty()) {
            result = create_device_local_buffer_staged(
                physical_device, device, upload_queue, upload_queue_family,
                scene.base_payload.data(),
                static_cast<VkDeviceSize>(scene.base_payload.size()),
                metadata_usage, buffers.base_payload);
        }
        if (result != VK_SUCCESS) return result;
        if (!scene.lod_payload.empty()) {
            result = create_device_local_buffer_staged(
                physical_device, device, upload_queue, upload_queue_family,
                scene.lod_payload.data(),
                static_cast<VkDeviceSize>(scene.lod_payload.size()),
                metadata_usage, buffers.lod_payload);
        }
        if (result != VK_SUCCESS) return result;
    }

    const UploadedBuffer* all_buffers[] = {
        &buffers.header,          &buffers.instances,     &buffers.hierarchy_nodes,
        &buffers.clusters,        &buffers.lod_groups,    &buffers.lod_clusters,
        &buffers.node_lod_links,  &buffers.pages,         &buffers.page_dependencies,
        &buffers.page_residency,  &buffers.base_payload,  &buffers.lod_payload,
    };

    report.uploaded_buffer_count = 0;
    report.uploaded_buffer_bytes = 0;
    for (const UploadedBuffer* buffer : all_buffers) {
        if (buffer->buffer != VK_NULL_HANDLE) {
            report.uploaded_buffer_count += 1;
            report.uploaded_buffer_bytes += static_cast<uint64_t>(buffer->size);
        }
    }
    report.scene_buffers_uploaded = true;
    return VK_SUCCESS;
}

uint32_t count_resident_pages(const ResidencyModel& model) {
    uint32_t resident_count = 0;
    for (const PageResidencyEntry& entry : model.pages) {
        if (entry.state == PageResidencyState::resident ||
            entry.state == PageResidencyState::eviction_candidate) {
            resident_count += 1;
        }
    }
    return resident_count;
}

uint32_t complete_loading_pages(ResidencyModel& model, uint32_t frame_index) {
    uint32_t completed_count = 0;
    for (PageResidencyEntry& entry : model.pages) {
        if (entry.state == PageResidencyState::loading) {
            entry.state = PageResidencyState::resident;
            entry.last_touched_frame = frame_index;
            completed_count += 1;
        }
    }
    return completed_count;
}

void snapshot_page_residency(UploadableScene& scene, const ResidencyModel& model) {
    scene.page_residency.resize(model.pages.size());
    for (uint32_t page_index = 0; page_index < model.pages.size(); ++page_index) {
        scene.page_residency[page_index].state = static_cast<uint32_t>(model.pages[page_index].state);
        scene.page_residency[page_index].last_touched_frame = model.pages[page_index].last_touched_frame;
        scene.page_residency[page_index].request_priority = model.pages[page_index].request_priority;
        scene.page_residency[page_index].flags = 0;
    }
}

void analyze_visibility_readback(VkDevice device, const SwapchainContext& swapchain,
                                 const DebugRenderContext& debug_render,
                                 const TraversalSelection& selection,
                                 VkBootstrapReport& report) {
    report.visibility_valid_pixels = 0;
    report.visibility_unique_base_geometry = 0;
    report.visibility_unique_lod_geometry = 0;
    report.visibility_invalid_ids = 0;
    report.visibility_visible_selected_base_geometry = 0;
    report.visibility_visible_selected_lod_geometry = 0;
    report.visibility_invisible_selected_base_geometry = 0;
    report.visibility_invisible_selected_lod_geometry = 0;
    report.visibility_selection_subset = false;

    if (debug_render.visibility_readback_buffer.memory == VK_NULL_HANDLE) {
        return;
    }

    void* mapped = nullptr;
    const VkResult result = vkMapMemory(device, debug_render.visibility_readback_buffer.memory, 0,
                                        debug_render.visibility_readback_buffer.size, 0, &mapped);
    if (result != VK_SUCCESS) {
        return;
    }

    const uint32_t pixel_count = swapchain.extent.width * swapchain.extent.height;
    const uint32_t* words = static_cast<const uint32_t*>(mapped);
    std::set<uint32_t> unique_base_ids;
    std::set<uint32_t> unique_lod_ids;
    for (uint32_t pixel_index = 0; pixel_index < pixel_count; ++pixel_index) {
        const VisibilityPixel pixel{words[pixel_index * 2], words[pixel_index * 2 + 1]};
        if (!visibility_valid(pixel)) {
            continue;
        }
        report.visibility_valid_pixels += 1;
        const GeometryKind kind = decode_visibility_geometry_kind(pixel);
        const uint32_t geometry_index = decode_visibility_geometry_index(pixel);
        if (kind == GeometryKind::base_cluster) {
            unique_base_ids.insert(geometry_index);
        } else {
            unique_lod_ids.insert(geometry_index);
        }
    }

    report.visibility_unique_base_geometry = static_cast<uint32_t>(unique_base_ids.size());
    report.visibility_unique_lod_geometry = static_cast<uint32_t>(unique_lod_ids.size());
    for (const uint32_t cluster_index : selection.selected_cluster_indices) {
        if (unique_base_ids.find(cluster_index) != unique_base_ids.end()) {
            report.visibility_visible_selected_base_geometry += 1;
        } else {
            report.visibility_invisible_selected_base_geometry += 1;
        }
    }
    for (const uint32_t lod_cluster_index : selection.selected_lod_cluster_indices) {
        if (unique_lod_ids.find(lod_cluster_index) != unique_lod_ids.end()) {
            report.visibility_visible_selected_lod_geometry += 1;
        } else {
            report.visibility_invisible_selected_lod_geometry += 1;
        }
    }

    bool subset_ok = true;
    for (const uint32_t geometry_index : unique_base_ids) {
        if (std::find(selection.selected_cluster_indices.begin(), selection.selected_cluster_indices.end(),
                      geometry_index) == selection.selected_cluster_indices.end()) {
            subset_ok = false;
            break;
        }
    }
    if (subset_ok) {
        for (const uint32_t geometry_index : unique_lod_ids) {
            if (std::find(selection.selected_lod_cluster_indices.begin(),
                          selection.selected_lod_cluster_indices.end(), geometry_index) ==
                selection.selected_lod_cluster_indices.end()) {
                subset_ok = false;
                break;
            }
        }
    }
    report.visibility_selection_subset = subset_ok && report.visibility_invalid_ids == 0;
    report.visibility_readback_ready = true;
    vkUnmapMemory(device, debug_render.visibility_readback_buffer.memory);
}

std::vector<const char*> collect_instance_extensions(bool& portability_enumeration) {
    portability_enumeration = false;
    uint32_t available_extension_count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &available_extension_count, nullptr);
    std::vector<VkExtensionProperties> available_extensions(available_extension_count);
    if (available_extension_count > 0) {
        vkEnumerateInstanceExtensionProperties(nullptr, &available_extension_count,
                                               available_extensions.data());
    }

    uint32_t glfw_extension_count = 0;
    const char** glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);
    std::vector<const char*> extensions;
    if (glfw_extensions != nullptr && glfw_extension_count > 0) {
        extensions.assign(glfw_extensions, glfw_extensions + glfw_extension_count);
    } else {
#if defined(__APPLE__)
        extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
        if (supports_extension(available_extensions, VK_EXT_METAL_SURFACE_EXTENSION_NAME)) {
            extensions.push_back(VK_EXT_METAL_SURFACE_EXTENSION_NAME);
        } else {
            throw BuilderError("required macOS metal surface extension is not available");
        }
#else
        throw BuilderError("GLFW did not report required Vulkan instance extensions");
#endif
    }

    // VK_KHR_get_physical_device_properties2 is core since Vulkan 1.1 and
    // the requested apiVersion below is 1.2, so the extension name is not
    // requested at all -- loaders that only expose it as an extension (and
    // nothing newer) must not fail instance creation over it.
    if (supports_extension(available_extensions, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        portability_enumeration = true;
    }
    return extensions;
}

VKAPI_ATTR VkBool32 VKAPI_CALL meridian_debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT types,
    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
    void* user_data) {
    (void)types;
    (void)user_data;
    const char* tag = (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0
                          ? "VALIDATION ERROR"
                          : "VALIDATION WARNING";
    std::cerr << tag << ": " << callback_data->pMessage << std::endl;
    return VK_FALSE;
}

std::vector<const char*> collect_validation_layers(bool enable_validation) {    std::vector<const char*> layers;
    if (!enable_validation) {
        return layers;
    }

    uint32_t layer_count = 0;
    if (vkEnumerateInstanceLayerProperties(&layer_count, nullptr) != VK_SUCCESS || layer_count == 0) {
        return layers;
    }

    std::vector<VkLayerProperties> available_layers(layer_count);
    if (vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data()) != VK_SUCCESS) {
        return layers;
    }

    for (const VkLayerProperties& layer : available_layers) {
        if (std::strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
            layers.push_back("VK_LAYER_KHRONOS_validation");
            break;
        }
    }

    return layers;
}

VkResult create_window_surface(VkInstance instance, GLFWwindow* window, VkSurfaceKHR* surface) {
#if defined(__APPLE__)
    NSWindow* cocoa_window = glfwGetCocoaWindow(window);
    if (cocoa_window == nil) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    NSView* cocoa_view = [cocoa_window contentView];
    if (cocoa_view == nil) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    [cocoa_view setWantsLayer:YES];
    CAMetalLayer* metal_layer = [CAMetalLayer layer];
    [cocoa_view setLayer:metal_layer];

    VkMetalSurfaceCreateInfoEXT create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT;
    create_info.pLayer = metal_layer;
    return vkCreateMetalSurfaceEXT(instance, &create_info, nullptr, surface);
#else
    return glfwCreateWindowSurface(instance, window, nullptr, surface);
#endif
}

QueueFamilySelection select_queue_families(VkPhysicalDevice physical_device, VkSurfaceKHR surface) {
    QueueFamilySelection selection;

    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count,
                                             queue_families.data());

    for (uint32_t family_index = 0; family_index < queue_family_count; ++family_index) {
        const VkQueueFamilyProperties& family = queue_families[family_index];
        if ((family.queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0 && selection.graphics_family == kInvalidQueueFamily) {
            selection.graphics_family = family_index;
        }

        VkBool32 supports_present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, family_index, surface, &supports_present);
        if (supports_present == VK_TRUE && selection.present_family == kInvalidQueueFamily) {
            selection.present_family = family_index;
        }

        if (selection.complete() && selection.graphics_family == selection.present_family) {
            break;
        }
    }

    return selection;
}

bool has_swapchain_support(VkPhysicalDevice physical_device, VkSurfaceKHR surface) {
    uint32_t format_count = 0;
    uint32_t present_mode_count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count, nullptr);
    vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &present_mode_count, nullptr);
    return format_count > 0 && present_mode_count > 0;
}

DeviceSelection select_device(VkInstance instance, VkSurfaceKHR surface, VkBootstrapReport& report) {
    uint32_t physical_device_count = 0;
    vkEnumeratePhysicalDevices(instance, &physical_device_count, nullptr);
    std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
    if (physical_device_count > 0) {
        vkEnumeratePhysicalDevices(instance, &physical_device_count, physical_devices.data());
    }

    DeviceSelection selection;
    for (VkPhysicalDevice physical_device : physical_devices) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physical_device, &properties);
        report.physical_devices.push_back(properties.deviceName);

        QueueFamilySelection queues = select_queue_families(physical_device, surface);
        if (!queues.complete()) {
            continue;
        }

        uint32_t extension_count = 0;
        vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr);
        std::vector<VkExtensionProperties> extensions(extension_count);
        if (extension_count > 0) {
            vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count,
                                                 extensions.data());
        }

        if (!supports_extension(extensions, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
            continue;
        }
        // The merged multi-cascade shadow pass selects the output layer per
        // instance from the vertex shader; that needs viewport/layer writes
        // from the vertex stage. On Vulkan >= 1.2 the core
        // shaderOutputLayer feature provides them; the
        // VK_EXT_shader_viewport_index_layer extension is the pre-1.2
        // fallback route.
        bool shader_output_layer_feature = false;
        if (properties.apiVersion >= VK_API_VERSION_1_2) {
            VkPhysicalDeviceFeatures2 features2{};
            features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            VkPhysicalDeviceVulkan12Features features12{};
            features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
            features2.pNext = &features12;
            vkGetPhysicalDeviceFeatures2(physical_device, &features2);
            if (features12.shaderOutputLayer != VK_TRUE) {
                continue;
            }
            shader_output_layer_feature = true;
        } else if (!supports_extension(extensions, "VK_EXT_shader_viewport_index_layer")) {
            continue;
        }
        if (!has_swapchain_support(physical_device, surface)) {
            continue;
        }

        // The indirect-count draw path generates multi-draw commands with
        // non-zero firstInstance, so the KHR extension (or its Vulkan 1.2
        // core promotion) alone is not enough: the core
        // multiDrawIndirect + drawIndirectFirstInstance features must be
        // supported too, and the draw list must fit maxDrawIndirectCount
        // (checked against the list capacities after they are known).
        VkPhysicalDeviceFeatures supported_features{};
        vkGetPhysicalDeviceFeatures(physical_device, &supported_features);
        const bool has_dic_extension =
            supports_extension(extensions, VK_KHR_DRAW_INDIRECT_COUNT_EXTENSION_NAME);
        bool draw_indirect_count_capable = has_dic_extension;
        if (!draw_indirect_count_capable && properties.apiVersion >= VK_API_VERSION_1_2) {
            VkPhysicalDeviceFeatures2 features2{};
            features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            VkPhysicalDeviceVulkan12Features features12{};
            features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
            features2.pNext = &features12;
            vkGetPhysicalDeviceFeatures2(physical_device, &features2);
            draw_indirect_count_capable = features12.drawIndirectCount == VK_TRUE;
        }
        if (draw_indirect_count_capable &&
            (supported_features.multiDrawIndirect != VK_TRUE ||
             supported_features.drawIndirectFirstInstance != VK_TRUE)) {
            draw_indirect_count_capable = false;
        }

        selection.physical_device = physical_device;
        selection.queues = queues;
        selection.enable_portability_subset =
            supports_extension(extensions, "VK_KHR_portability_subset");
        selection.has_draw_indirect_count = draw_indirect_count_capable;
        selection.has_draw_indirect_count_extension = has_dic_extension;
        selection.max_draw_indirect_count = properties.limits.maxDrawIndirectCount;
        selection.shader_output_layer_feature = shader_output_layer_feature;
        report.selected_device = properties.deviceName;
        report.graphics_queue_family = queues.graphics_family;
        report.present_queue_family = queues.present_family;
        return selection;
    }

    return selection;
}

VkSurfaceFormatKHR choose_surface_format(const std::vector<VkSurfaceFormatKHR>& formats) {
    for (const VkSurfaceFormatKHR& format : formats) {
        if (format.format == VK_FORMAT_B8G8R8A8_UNORM &&
            format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return format;
        }
    }
    return formats.front();
}

VkPresentModeKHR choose_present_mode(const std::vector<VkPresentModeKHR>& present_modes) {
    for (const VkPresentModeKHR present_mode : present_modes) {
        if (present_mode == VK_PRESENT_MODE_FIFO_KHR) {
            return present_mode;
        }
    }
    return present_modes.front();
}

// First supported composite alpha mode, preferring opaque. The spec
// guarantees at least one bit in supportedCompositeAlpha; INHERIT is the
// final fallback and is always legal.
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

VkExtent2D choose_swapchain_extent(GLFWwindow* window, const VkSurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    }

    int framebuffer_width = 0;
    int framebuffer_height = 0;
    glfwGetFramebufferSize(window, &framebuffer_width, &framebuffer_height);
    while (framebuffer_width == 0 || framebuffer_height == 0) {
        glfwWaitEvents();
        glfwGetFramebufferSize(window, &framebuffer_width, &framebuffer_height);
    }

    VkExtent2D extent{};
    extent.width = std::clamp(static_cast<uint32_t>(framebuffer_width), capabilities.minImageExtent.width,
                              capabilities.maxImageExtent.width);
    extent.height = std::clamp(static_cast<uint32_t>(framebuffer_height), capabilities.minImageExtent.height,
                               capabilities.maxImageExtent.height);
    return extent;
}

}  // close anonymous namespace

void destroy_swapchain(VkDevice device, SwapchainContext& swapchain) {
    for (VkImageView image_view : swapchain.image_views) {
        vkDestroyImageView(device, image_view, nullptr);
    }
    swapchain.image_views.clear();
    swapchain.images.clear();
    if (swapchain.swapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device, swapchain.swapchain, nullptr);
        swapchain.swapchain = VK_NULL_HANDLE;
    }
}

namespace {  // reopen anonymous namespace

VkResult create_swapchain(VkPhysicalDevice physical_device, VkDevice device, VkSurfaceKHR surface,
                           GLFWwindow* window, const QueueFamilySelection& queues,
                           SwapchainContext& swapchain) {
    VkSurfaceCapabilitiesKHR capabilities{};
    VkResult result = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface,
                                                                &capabilities);
    if (result != VK_SUCCESS) {
        return result;
    }

    uint32_t format_count = 0;
    result = vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count, nullptr);
    if (result != VK_SUCCESS) {
        return result;
    }
    std::vector<VkSurfaceFormatKHR> formats(format_count);
    if (format_count > 0) {
        result = vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count,
                                                      formats.data());
        if (result == VK_INCOMPLETE) {
            // The format set changed between the two calls; keep the
            // truncated prefix, which is still enough to select from.
            formats.resize(std::min<std::size_t>(format_count, formats.size()));
        } else if (result != VK_SUCCESS) {
            return result;
        }
    }
    // A valid surface always exposes at least one format; an empty set
    // means the surface is dead and .front() below would be UB.
    if (formats.empty()) {
        return VK_ERROR_SURFACE_LOST_KHR;
    }

    uint32_t present_mode_count = 0;
    result = vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface,
                                                       &present_mode_count, nullptr);
    if (result != VK_SUCCESS) {
        return result;
    }
    std::vector<VkPresentModeKHR> present_modes(present_mode_count);
    if (present_mode_count > 0) {
        result = vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface,
                                                           &present_mode_count,
                                                           present_modes.data());
        if (result == VK_INCOMPLETE) {
            present_modes.resize(std::min<std::size_t>(present_mode_count, present_modes.size()));
        } else if (result != VK_SUCCESS) {
            return result;
        }
    }
    if (present_modes.empty()) {
        return VK_ERROR_SURFACE_LOST_KHR;
    }

    swapchain.surface_format = choose_surface_format(formats);
    swapchain.present_mode = choose_present_mode(present_modes);
    swapchain.extent = choose_swapchain_extent(window, capabilities);

    uint32_t image_count = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 && image_count > capabilities.maxImageCount) {
        image_count = capabilities.maxImageCount;
    }

    const uint32_t queue_family_indices[] = {queues.graphics_family, queues.present_family};
    VkSwapchainCreateInfoKHR create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    create_info.surface = surface;
    create_info.minImageCount = image_count;
    create_info.imageFormat = swapchain.surface_format.format;
    create_info.imageColorSpace = swapchain.surface_format.colorSpace;
    create_info.imageExtent = swapchain.extent;
    create_info.imageArrayLayers = 1;
    // COLOR_ATTACHMENT is guaranteed to be supported for swapchains;
    // everything else must come from supportedUsageFlags. TRANSFER_SRC is
    // what the screenshot readback needs -- when the surface does not
    // allow it the swapchain is still created and the screenshot request
    // fails explicitly instead (see the capture path).
    create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if ((capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0) {
        create_info.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        swapchain.images_support_transfer_src = true;
    }
    create_info.preTransform = capabilities.currentTransform;
    create_info.compositeAlpha = choose_composite_alpha(capabilities.supportedCompositeAlpha);
    create_info.presentMode = swapchain.present_mode;
    create_info.clipped = VK_TRUE;

    if (queues.graphics_family != queues.present_family) {
        create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        create_info.queueFamilyIndexCount = 2;
        create_info.pQueueFamilyIndices = queue_family_indices;
    } else {
        create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    result = vkCreateSwapchainKHR(device, &create_info, nullptr, &swapchain.swapchain);
    if (result != VK_SUCCESS) {
        return result;
    }

    uint32_t swapchain_image_count = 0;
    result = vkGetSwapchainImagesKHR(device, swapchain.swapchain, &swapchain_image_count, nullptr);
    if (result != VK_SUCCESS) {
        destroy_swapchain(device, swapchain);
        return result;
    }
    if (swapchain_image_count == 0) {
        destroy_swapchain(device, swapchain);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    swapchain.images.resize(swapchain_image_count);
    result = vkGetSwapchainImagesKHR(device, swapchain.swapchain, &swapchain_image_count,
                                     swapchain.images.data());
    if (result == VK_INCOMPLETE) {
        swapchain.images.resize(
            std::min<std::size_t>(swapchain_image_count, swapchain.images.size()));
    } else if (result != VK_SUCCESS) {
        destroy_swapchain(device, swapchain);
        return result;
    }

    swapchain.image_views.resize(swapchain.images.size());
    for (size_t image_index = 0; image_index < swapchain.images.size(); ++image_index) {
        VkImageViewCreateInfo view_info{};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image = swapchain.images[image_index];
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = swapchain.surface_format.format;
        view_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        view_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        view_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        view_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view_info.subresourceRange.baseMipLevel = 0;
        view_info.subresourceRange.levelCount = 1;
        view_info.subresourceRange.baseArrayLayer = 0;
        view_info.subresourceRange.layerCount = 1;

        result = vkCreateImageView(device, &view_info, nullptr, &swapchain.image_views[image_index]);
        if (result != VK_SUCCESS) {
            destroy_swapchain(device, swapchain);
            return result;
        }
    }

    return VK_SUCCESS;
}

VkResult create_frame_context(VkDevice device, const QueueFamilySelection& queues, uint32_t swapchain_image_count,
                              FrameContext& frame) {
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = queues.graphics_family;
    VkResult result = vkCreateCommandPool(device, &pool_info, nullptr, &frame.command_pool);
    if (result != VK_SUCCESS) {
        return result;
    }

    VkCommandBufferAllocateInfo allocate_info{};
    allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocate_info.commandPool = frame.command_pool;
    allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate_info.commandBufferCount = 1;
    result = vkAllocateCommandBuffers(device, &allocate_info, &frame.command_buffer);
    if (result != VK_SUCCESS) {
        return result;
    }

    VkSemaphoreCreateInfo semaphore_info{};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    result = vkCreateSemaphore(device, &semaphore_info, nullptr, &frame.image_available);
    if (result != VK_SUCCESS) {
        return result;
    }
    result = vkCreateSemaphore(device, &semaphore_info, nullptr, &frame.render_finished);
    if (result != VK_SUCCESS) {
        return result;
    }

    // Per-swapchain-image signal semaphores: presenting retires a semaphore
    // only when that image is reacquired, so a single reused render_finished
    // can still be "in use by the swapchain" at the next submit.
    frame.render_finished_per_image.resize(swapchain_image_count, VK_NULL_HANDLE);
    for (uint32_t i = 0; i < swapchain_image_count; ++i) {
        result = vkCreateSemaphore(device, &semaphore_info, nullptr,
                                   &frame.render_finished_per_image[i]);
        if (result != VK_SUCCESS) {
            return result;
        }
    }

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    return vkCreateFence(device, &fence_info, nullptr, &frame.in_flight);
}


}  // close anonymous namespace for find_depth_format extraction

VkFormat find_depth_format(VkPhysicalDevice physical_device) {
    // Depth images are also sampled (HZB source, shadow-map reads), so a
    // candidate must support both attachment use and sampling; the
    // samplers are NEAREST, so SAMPLED_IMAGE suffices (no filtered bit).
    // D16 is the last-resort no-stencil fallback.
    const VkFormat candidates[] = {
        VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_D32_SFLOAT_S8_UINT,
        VK_FORMAT_D24_UNORM_S8_UINT,
        VK_FORMAT_D16_UNORM,
    };
    for (VkFormat format : candidates) {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(physical_device, format, &properties);
        const VkFormatFeatureFlags required = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                              VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        if ((properties.optimalTilingFeatures & required) == required) {
            return format;
        }
    }
    return VK_FORMAT_UNDEFINED;
}

// create_depth_resources, create_visibility_resources, create_debug_render_context moved to vk_render.cpp

void destroy_frame_context(VkDevice device, FrameContext& frame) {
    if (frame.in_flight != VK_NULL_HANDLE) {
        vkDestroyFence(device, frame.in_flight, nullptr);
    }
    if (frame.render_finished != VK_NULL_HANDLE) {
        vkDestroySemaphore(device, frame.render_finished, nullptr);
    }
    for (VkSemaphore semaphore : frame.render_finished_per_image) {
        if (semaphore != VK_NULL_HANDLE) {
            vkDestroySemaphore(device, semaphore, nullptr);
        }
    }
    frame.render_finished_per_image.clear();
    if (frame.image_available != VK_NULL_HANDLE) {
        vkDestroySemaphore(device, frame.image_available, nullptr);
    }
    if (frame.command_pool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device, frame.command_pool, nullptr);
    }
}

namespace {  // reopen anonymous namespace

// Instance-folded main-pass draws (MoltenVK has no multi-draw indirect:
// every vkCmdDrawIndirect entry becomes one Metal draw encode, so runs of
// draws are folded into single instanced draws). The vertex shader reads
// the draw entry via gl_InstanceIndex = draw.first_instance + local
// instance and collapses corners past the cluster's own count to
// degenerate triangles.
struct DrawBucket {
    uint32_t vertex_count = 0;
    uint32_t instance_count = 0;
    uint32_t first_instance = 0;
};

// Fold a globally-ordered draw list into instanced draws while PRESERVING
// the global order: only contiguous runs of equal vertex count fold into
// one instanced draw, so the interleaving of draws with different vertex
// counts is unchanged. Order matters: the depth test is
// VK_COMPARE_OP_LESS, so coplanar equal-depth overlaps resolve
// first-writer-wins -- the old quartile-bucket fold reordered the list
// bucket-major, which could make the CPU-folded fallback and the
// indirect-count path resolve a shared pixel differently. Entries keep
// their global positions and draw_first_instance is (re)assigned
// global_index * instance_stride so the vertex shader still resolves its
// entry from gl_InstanceIndex. Returns the number of runs (the vkCmdDraw
// encode count). When wasted_vertex_invocations is given it receives
// (encoded - useful) vertex-shader invocations for the fold: for the
// strided shadow list, stride slots past each entry's cascade count
// contribute whole degenerate instances (main-list runs of equal count
// waste nothing).
uint32_t fold_draws_into_buckets(std::vector<GpuDrawEntry>& draws,
                                 uint32_t instance_stride,
                                 std::vector<DrawBucket>& buckets,
                                 uint64_t* wasted_vertex_invocations = nullptr) {
    buckets.clear();
    if (draws.empty()) {
        return 0;
    }
    uint64_t useful_vertex_invocations = 0;
    for (uint32_t i = 0; i < draws.size(); ++i) {
        GpuDrawEntry& e = draws[i];
        e.draw_first_instance = i * instance_stride;
        useful_vertex_invocations +=
            static_cast<uint64_t>(e.draw_vertex_count) * e.draw_instance_count;
        if (!buckets.empty() && buckets.back().vertex_count == e.draw_vertex_count) {
            buckets.back().instance_count += instance_stride;
            continue;
        }
        DrawBucket bucket;
        bucket.vertex_count = e.draw_vertex_count;
        bucket.instance_count = instance_stride;
        bucket.first_instance = i * instance_stride;
        buckets.push_back(bucket);
    }
    if (wasted_vertex_invocations != nullptr) {
        uint64_t encoded_vertex_invocations = 0;
        for (const DrawBucket& bucket : buckets) {
            encoded_vertex_invocations += static_cast<uint64_t>(bucket.vertex_count) *
                                          bucket.instance_count;
        }
        *wasted_vertex_invocations =
            encoded_vertex_invocations > useful_vertex_invocations
                ? encoded_vertex_invocations - useful_vertex_invocations
                : 0;
    }
    return static_cast<uint32_t>(buckets.size());
}

// Per-frame GPU passes that no draw path consumes are recorded here once
// per run (see submit_diagnostic_epilogue) instead of every frame:
//   - instance cull: its output only fed the retained-but-not-dispatched
//     cluster_select compute shader; the report reads the survivor counter.
//   - occlusion refine + HZB: only the drawIndirectCount path can consume
//     the GPU-written survivor list, so on the MoltenVK fallback they are
//     dead work per frame (the fallback folds the CPU list).
//   - visibility image -> buffer copy: read back once after the present
//     loop by analyze_visibility_readback; copying 921K pixels x 8 bytes
//     every frame cost a Metal blit encode plus two barriers for nothing.
void record_instance_cull_pass(VkCommandBuffer cmd, const ComputeCullContext& compute_cull,
                               const FrustumPlanes& frustum) {
    vkCmdFillBuffer(cmd, compute_cull.counter.buffer, 0, sizeof(uint32_t), 0);

    VkMemoryBarrier fill_barrier{};
    fill_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    fill_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    fill_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &fill_barrier,
                         0, nullptr, 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_cull.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            compute_cull.pipeline_layout, 0, 1, &compute_cull.descriptor_set,
                            0, nullptr);

    CullPushConstants cull_push{};
    std::memcpy(cull_push.frustum_planes, frustum.planes, sizeof(frustum.planes));
    cull_push.instance_count = compute_cull.max_instances;
    vkCmdPushConstants(cmd, compute_cull.pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CullPushConstants), &cull_push);

    const uint32_t group_count = (compute_cull.max_instances + 63) / 64;
    vkCmdDispatch(cmd, group_count, 1, 1);

    VkMemoryBarrier compute_barrier{};
    compute_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    compute_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    compute_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &compute_barrier, 0, nullptr, 0, nullptr);
}

void record_occlusion_refine_pass(VkCommandBuffer cmd,
                                  const OcclusionRefineContext& occlusion_refine,
                                  const HzbContext& hzb,
                                  const ComputeSelectionContext& compute_selection,
                                  const CameraFrameData& camera_frame,
                                  bool temporal_hzb_valid) {
    vkCmdFillBuffer(cmd, occlusion_refine.output_count.buffer, 0, 2 * sizeof(uint32_t), 0);

    VkMemoryBarrier fill_bar{};
    fill_bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    fill_bar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    fill_bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &fill_bar,
                         0, nullptr, 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, occlusion_refine.pipeline);
    // Temporally invalid frames bind the 1x1 far-depth fallback instead of
    // the previous frame's HZB (stale camera/scene pairing): sampling max
    // depth never rejects, so the pass conservatively keeps everything.
    // Dims must match the bound image -- the shader derives its mip and
    // footprint math from them.
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            occlusion_refine.pipeline_layout, 0, 1,
                            temporal_hzb_valid ? &occlusion_refine.descriptor_set
                                               : &occlusion_refine.fallback_descriptor_set,
                            0, nullptr);

    OcclusionPushConstants occ_push{};
    std::memcpy(occ_push.view_projection, camera_frame.view_projection.m,
                sizeof(occ_push.view_projection));
    occ_push.hzb_width = temporal_hzb_valid ? hzb.width : 1;
    occ_push.hzb_height = temporal_hzb_valid ? hzb.height : 1;
    vkCmdPushConstants(cmd, occlusion_refine.pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(OcclusionPushConstants), &occ_push);

    const uint32_t occ_groups = (compute_selection.max_draws + 63) / 64;
    vkCmdDispatch(cmd, std::max(occ_groups, 1u), 1, 1);

    VkMemoryBarrier occ_bar{};
    occ_bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    occ_bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    occ_bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                            VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
                         VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                         0, 1, &occ_bar, 0, nullptr, 0, nullptr);
}

// HZB build from the main-pass depth image. depth_old_layout depends on the
// caller: the drawIndirectCount path builds the HZB every frame right after
// the main pass (DEPTH_STENCIL_ATTACHMENT_OPTIMAL), the diagnostic epilogue
// builds it once from the final frame's depth (same layout on the fallback
// path because no per-frame HZB transition ran).
void record_hzb_build_pass(VkCommandBuffer cmd, const HzbContext& hzb, VkImage depth_image,
                           VkImageLayout depth_old_layout) {
    VkImageMemoryBarrier depth_to_read{};
    depth_to_read.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    depth_to_read.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    depth_to_read.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    depth_to_read.oldLayout = depth_old_layout;
    depth_to_read.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    depth_to_read.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    depth_to_read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    depth_to_read.image = depth_image;
    depth_to_read.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    depth_to_read.subresourceRange.levelCount = 1;
    depth_to_read.subresourceRange.layerCount = 1;

    VkImageMemoryBarrier hzb_all_to_general{};
    hzb_all_to_general.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    hzb_all_to_general.srcAccessMask = 0;
    hzb_all_to_general.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    hzb_all_to_general.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    hzb_all_to_general.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    hzb_all_to_general.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    hzb_all_to_general.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    hzb_all_to_general.image = hzb.image;
    hzb_all_to_general.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    hzb_all_to_general.subresourceRange.levelCount = hzb.mip_count;
    hzb_all_to_general.subresourceRange.layerCount = 1;

    VkImageMemoryBarrier pre_barriers[2] = {depth_to_read, hzb_all_to_general};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
                         2, pre_barriers);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hzb.depth_copy_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            hzb.depth_copy_pipeline_layout, 0, 1,
                            &hzb.depth_copy_descriptor_set, 0, nullptr);
    vkCmdDispatch(cmd, (hzb.width + 7) / 8, (hzb.height + 7) / 8, 1);

    VkMemoryBarrier mip0_visibility{};
    mip0_visibility.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mip0_visibility.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mip0_visibility.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mip0_visibility, 0,
                         nullptr, 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hzb.pipeline);
    uint32_t src_w = hzb.width, src_h = hzb.height;
    for (uint32_t mip = 0; mip + 1 < hzb.mip_count; ++mip) {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                hzb.pipeline_layout, 0, 1, &hzb.mip_descriptor_sets[mip],
                                0, nullptr);

        uint32_t push_data[2] = {src_w, src_h};
        vkCmdPushConstants(cmd, hzb.pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, push_data);

        const uint32_t dst_w = std::max(src_w / 2, 1u);
        const uint32_t dst_h = std::max(src_h / 2, 1u);
        vkCmdDispatch(cmd, (dst_w + 7) / 8, (dst_h + 7) / 8, 1);

        if (mip + 2 < hzb.mip_count) {
            VkMemoryBarrier mip_visibility{};
            mip_visibility.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            mip_visibility.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mip_visibility.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mip_visibility,
                                 0, nullptr, 0, nullptr);
        }
        src_w = dst_w;
        src_h = dst_h;
    }

    VkImageMemoryBarrier hzb_to_read{};
    hzb_to_read.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    hzb_to_read.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    hzb_to_read.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    hzb_to_read.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    hzb_to_read.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    hzb_to_read.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    hzb_to_read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    hzb_to_read.image = hzb.image;
    hzb_to_read.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    hzb_to_read.subresourceRange.levelCount = hzb.mip_count;
    hzb_to_read.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
                         1, &hzb_to_read);
}

void record_visibility_copy_pass(VkCommandBuffer cmd, const DebugRenderContext& debug_render,
                                 VkExtent2D extent) {
    VkImageMemoryBarrier image_barrier{};
    image_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    image_barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    image_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    image_barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    image_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    image_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.image = debug_render.visibility_image;
    image_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    image_barrier.subresourceRange.baseMipLevel = 0;
    image_barrier.subresourceRange.levelCount = 1;
    image_barrier.subresourceRange.baseArrayLayer = 0;
    image_barrier.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                         &image_barrier);

    VkBufferImageCopy copy_region{};
    copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy_region.imageSubresource.mipLevel = 0;
    copy_region.imageSubresource.baseArrayLayer = 0;
    copy_region.imageSubresource.layerCount = 1;
    copy_region.imageExtent.width = extent.width;
    copy_region.imageExtent.height = extent.height;
    copy_region.imageExtent.depth = 1;
    vkCmdCopyImageToBuffer(cmd, debug_render.visibility_image,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           debug_render.visibility_readback_buffer.buffer, 1, &copy_region);

    VkBufferMemoryBarrier buffer_barrier{};
    buffer_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    buffer_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    buffer_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    buffer_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    buffer_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    buffer_barrier.buffer = debug_render.visibility_readback_buffer.buffer;
    buffer_barrier.size = debug_render.visibility_readback_buffer.size;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &buffer_barrier, 0,
                         nullptr);
}

VkResult record_debug_command_buffer(FrameContext& frame, const DebugRenderContext& debug_render,
                                      const ComputeCullContext& compute_cull,
                                      const ComputeSelectionContext& compute_selection,
                                      const HzbContext& hzb,
                                      const OcclusionRefineContext& occlusion_refine,
                                      const ShadowContext& shadow,
                                      const SwapchainContext& swapchain,
                                      const CameraFrameData& camera_frame,
                                      const FrustumPlanes& frustum,
                                      float error_threshold,
                                       const TraversalSelection& selection,
                                       const UploadableScene& scene,
                                        uint32_t frame_index,
                                        uint32_t image_index,
                                         bool has_draw_indirect_count,
                                         uint32_t shadow_draw_count,
                                         const std::vector<DrawBucket>& shadow_buckets,
                                         const std::vector<DrawBucket>& main_buckets,
                                        const GpuProfiler& profiler,
                                        bool capture_visibility,
                                        bool temporal_hzb_valid) {
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VkResult result = vkBeginCommandBuffer(frame.command_buffer, &begin_info);
    if (result != VK_SUCCESS) {
        return result;
    }

    // GPU profiler: reset queries and write initial timestamp
    if (profiler.query_pool != VK_NULL_HANDLE) {
        vkCmdResetQueryPool(frame.command_buffer, profiler.query_pool, 0, profiler.query_count);
        vkCmdWriteTimestamp(frame.command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            profiler.query_pool, 0); // cull start
        vkCmdWriteTimestamp(frame.command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            profiler.query_pool, 12); // total start
    }

    // Instance culling runs once in the diagnostic epilogue (see
    // submit_diagnostic_epilogue): its output only fed the retained
    // cluster_select shader, which is not dispatched, and the report reads
    // the survivor counter after the loop.

    if (profiler.query_pool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp(frame.command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            profiler.query_pool, 1); // cull end
        vkCmdWriteTimestamp(frame.command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            profiler.query_pool, 2); // sel start
    }

    // Selection runs on CPU (simulate_traversal) and its output is uploaded into
    // compute_selection.draw_list / .draw_count before command buffer recording.
    // The GPU selection compute shader is retained but not dispatched; kept for
    // future use once parallel traversal replaces serial DFS.
    if (compute_selection.draw_list.buffer != VK_NULL_HANDLE) {
        VkMemoryBarrier host_write_barrier{};
        host_write_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        host_write_barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        host_write_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                           VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        vkCmdPipelineBarrier(frame.command_buffer, VK_PIPELINE_STAGE_HOST_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                             VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                             0, 1, &host_write_barrier, 0, nullptr, 0, nullptr);
    }

    if (profiler.query_pool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp(frame.command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            profiler.query_pool, 3); // sel end
        vkCmdWriteTimestamp(frame.command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            profiler.query_pool, 4); // occ start
    }

    // Occlusion refinement pass (uses previous frame's HZB when temporally
    // valid; the 1x1 far-depth fallback otherwise; skip frame 0).
    // Only the drawIndirectCount path can consume the GPU-written survivor
    // list; on the fallback (MoltenVK) the CPU-folded list is drawn directly
    // and this pass is dead per-frame work, so it runs once in the
    // diagnostic epilogue instead.
    if (has_draw_indirect_count && frame_index > 0 &&
        occlusion_refine.pipeline != VK_NULL_HANDLE &&
        occlusion_refine.descriptor_set != VK_NULL_HANDLE) {
        record_occlusion_refine_pass(frame.command_buffer, occlusion_refine, hzb,
                                     compute_selection, camera_frame, temporal_hzb_valid);
    }

    if (profiler.query_pool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp(frame.command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            profiler.query_pool, 5); // occ end
        vkCmdWriteTimestamp(frame.command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            profiler.query_pool, 6); // shadow start
    }

    // Shadow pass: one layered render pass over all cascades. The merged
    // draw list carries one draw per caster cluster, instanced once per
    // overlapping cascade (per-cascade frustum filtering happens on the CPU
    // when the mask is built); the vertex shader picks light_vp and the
    // output layer per instance.
    if (shadow.pipeline != VK_NULL_HANDLE && shadow.descriptor_set != VK_NULL_HANDLE) {
        VkClearValue shadow_clear{};
        shadow_clear.depthStencil.depth = 1.0f;

        VkRenderPassBeginInfo shadow_rp_info{};
        shadow_rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        shadow_rp_info.renderPass = shadow.render_pass;
        shadow_rp_info.framebuffer = shadow.framebuffer;
        shadow_rp_info.renderArea.extent = {shadow.resolution, shadow.resolution};
        shadow_rp_info.clearValueCount = 1;
        shadow_rp_info.pClearValues = &shadow_clear;

        vkCmdBeginRenderPass(frame.command_buffer, &shadow_rp_info, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow.pipeline);
        vkCmdBindDescriptorSets(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                shadow.pipeline_layout, 0, 1, &shadow.descriptor_set, 0, nullptr);

        const UploadedBuffer& dl = shadow.draw_list;
        const UploadedBuffer& dc = shadow.draw_count;
        if (dl.buffer != VK_NULL_HANDLE && dc.buffer != VK_NULL_HANDLE) {
            // Draw count: on implementations without draw_indirect_count the
            // count parameter must come from the CPU (the GPU-side count
            // buffer cannot feed vkCmdDrawIndirect), so pass this frame's
            // exact list length instead of the buffer capacity -- MoltenVK
            // encodes one Metal draw per count entry and the capacity-sized
            // count dominated vkQueueSubmit. The fallback goes further and
            // folds the whole list into per-bucket instanced draws (same
            // mechanism as the main pass; the layered vertex shader skips
            // unused stride slots past each entry's cascade mask).
            if (has_draw_indirect_count) {
                vkCmdDrawIndirectCount(frame.command_buffer,
                                       dl.buffer, 0,
                                       dc.buffer, 0,
                                       std::min(shadow_draw_count, shadow.max_draws),
                                       sizeof(GpuDrawEntry));
            } else {
                for (const DrawBucket& bucket : shadow_buckets) {
                    if (bucket.instance_count == 0 || bucket.vertex_count == 0) {
                        continue;
                    }
                    vkCmdDraw(frame.command_buffer,
                              bucket.vertex_count,
                              bucket.instance_count,
                              0,
                              bucket.first_instance);
                }
            }
        }
        vkCmdEndRenderPass(frame.command_buffer);
    }

    if (profiler.query_pool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp(frame.command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            profiler.query_pool, 7); // shadow end
        vkCmdWriteTimestamp(frame.command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            profiler.query_pool, 8); // main start
    }

    VkClearValue clear_values[3] = {};
    clear_values[0].color.float32[0] = 0.05f;
    clear_values[0].color.float32[1] = 0.07f;
    clear_values[0].color.float32[2] = 0.10f;
    clear_values[0].color.float32[3] = 1.0f;
    clear_values[1].depthStencil.depth = 1.0f;
    clear_values[2].color.uint32[0] = 0u;

    VkRenderPassBeginInfo render_pass_info{};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_info.renderPass = capture_visibility && debug_render.render_pass != VK_NULL_HANDLE
        ? debug_render.render_pass
        : (debug_render.render_pass_transient != VK_NULL_HANDLE
               ? debug_render.render_pass_transient
               : debug_render.render_pass);
    render_pass_info.framebuffer = debug_render.framebuffers[image_index];
    render_pass_info.renderArea.extent = swapchain.extent;
    render_pass_info.clearValueCount = 3;
    render_pass_info.pClearValues = clear_values;

    vkCmdBeginRenderPass(frame.command_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, debug_render.pipeline);
    VkViewport dynamic_viewport{};
    dynamic_viewport.width = static_cast<float>(swapchain.extent.width);
    dynamic_viewport.height = static_cast<float>(swapchain.extent.height);
    dynamic_viewport.maxDepth = 1.0f;
    vkCmdSetViewport(frame.command_buffer, 0, 1, &dynamic_viewport);
    VkRect2D dynamic_scissor{};
    dynamic_scissor.extent = swapchain.extent;
    vkCmdSetScissor(frame.command_buffer, 0, 1, &dynamic_scissor);
    if (debug_render.descriptor_set != VK_NULL_HANDLE) {
        vkCmdBindDescriptorSets(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                debug_render.pipeline_layout, 0, 1, &debug_render.descriptor_set,
                                0, nullptr);
        if (compute_selection.draw_list.buffer != VK_NULL_HANDLE &&
            compute_selection.draw_count.buffer != VK_NULL_HANDLE) {
            if (has_draw_indirect_count) {
                // With GPU-side draw counts the occlusion-refined list can be
                // consumed directly; vkCmdDrawIndirectCount reads word 0 of
                // the count buffer (input count -- survivors keep their input
                // slot and rejected entries are zero-vertex tombstones, so
                // the draw order is input-order-defined and the extra draws
                // are no-ops).
                const bool use_occlusion_output = frame_index > 0 &&
                    occlusion_refine.output_draws.buffer != VK_NULL_HANDLE &&
                    occlusion_refine.output_count.buffer != VK_NULL_HANDLE;
                VkBuffer draw_buffer = use_occlusion_output
                    ? occlusion_refine.output_draws.buffer
                    : compute_selection.draw_list.buffer;
                VkBuffer count_buffer = use_occlusion_output
                    ? occlusion_refine.output_count.buffer
                    : compute_selection.draw_count.buffer;
                uint32_t max_draws = use_occlusion_output
                    ? occlusion_refine.max_draws
                    : compute_selection.max_draws;
                vkCmdDrawIndirectCount(frame.command_buffer,
                                       draw_buffer, 0,
                                       count_buffer, 0,
                                       max_draws,
                                       sizeof(GpuDrawEntry));
            } else {
                // No draw_indirect_count: fold the CPU-built list into one
                // instanced draw per contiguous equal-vertex-count run (one
                // Metal encode each instead of one per cluster). The fold
                // preserves the global draw order (see
                // fold_draws_into_buckets). The occlusion output cannot be
                // used here (its count is GPU-written and its tail holds
                // stale entries), and a capacity-sized indirect count would
                // encode one Metal draw per cluster slot, dominating
                // vkQueueSubmit on MoltenVK.
                for (const DrawBucket& bucket : main_buckets) {
                    if (bucket.instance_count == 0 || bucket.vertex_count == 0) {
                        continue;
                    }
                    vkCmdDraw(frame.command_buffer,
                              bucket.vertex_count,
                              bucket.instance_count,
                              0,
                              bucket.first_instance);
                }
            }
        }
    }
    vkCmdEndRenderPass(frame.command_buffer);

    if (profiler.query_pool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp(frame.command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            profiler.query_pool, 9); // main end
        vkCmdWriteTimestamp(frame.command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            profiler.query_pool, 10); // hzb start
    }

    // HZB build: only needed per frame where the occlusion-refined list is
    // consumed (drawIndirectCount path). The fallback builds it once in the
    // diagnostic epilogue for the survivor-count report.
    if (has_draw_indirect_count && hzb.pipeline != VK_NULL_HANDLE && hzb.mip_count > 1 &&
        hzb.depth_copy_pipeline != VK_NULL_HANDLE) {
        record_hzb_build_pass(frame.command_buffer, hzb, debug_render.depth_image,
                              VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    }

    // (Occlusion refinement moved before shadow/main passes above)

    if (profiler.query_pool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp(frame.command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            profiler.query_pool, 11); // hzb end
        vkCmdWriteTimestamp(frame.command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            profiler.query_pool, 13); // total end
    }

    // The visibility image -> readback-buffer copy moved to the diagnostic
    // epilogue: analyze_visibility_readback consumes it once after the
    // present loop, so the per-frame 7.3MB blit encode plus two barriers
    // and an image layout round-trip were pure submit cost.

    return vkEndCommandBuffer(frame.command_buffer);
}

// One-shot submit after the present loop that refreshes every diagnostic the
// report still needs: the instance-cull survivor counter (dropped from the
// per-frame stream -- only the not-dispatched cluster_select shader ever
// read it), the occlusion survivor count on the fallback path (HZB build +
// refine against the final frame's depth and draw list), and the visibility
// readback copy that analyze_visibility_readback consumes. Semantics note:
// the per-frame occlusion pass tested the CURRENT draw list against the
// PREVIOUS frame's HZB; the epilogue tests the final list against the final
// frame's own HZB (identical in the static-camera benchmark/validate runs).
VkResult submit_diagnostic_epilogue(VkDevice device, VkQueue queue, FrameContext& frame,
                                    const DebugRenderContext& debug_render,
                                    const ComputeCullContext& compute_cull,
                                    const ComputeSelectionContext& compute_selection,
                                    const HzbContext& hzb,
                                    const OcclusionRefineContext& occlusion_refine,
                                    const CameraFrameData& camera_frame,
                                    const FrustumPlanes& frustum,
                                    const SwapchainContext& swapchain,
                                    bool has_draw_indirect_count) {
    const bool need_cull = compute_cull.pipeline != VK_NULL_HANDLE &&
                           compute_cull.descriptor_set != VK_NULL_HANDLE;
    const bool need_occ = !has_draw_indirect_count &&
                          occlusion_refine.pipeline != VK_NULL_HANDLE &&
                          occlusion_refine.descriptor_set != VK_NULL_HANDLE &&
                          hzb.pipeline != VK_NULL_HANDLE && hzb.mip_count > 1 &&
                          hzb.depth_copy_pipeline != VK_NULL_HANDLE;
    const bool need_copy = debug_render.visibility_image != VK_NULL_HANDLE &&
                           debug_render.visibility_readback_buffer.buffer != VK_NULL_HANDLE;
    if (!need_cull && !need_occ && !need_copy) {
        return VK_SUCCESS;
    }

    vkResetCommandPool(device, frame.command_pool, 0);
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VkResult result = vkBeginCommandBuffer(frame.command_buffer, &begin);
    if (result != VK_SUCCESS) {
        return result;
    }

    if (need_cull) {
        record_instance_cull_pass(frame.command_buffer, compute_cull, frustum);
    }
    if (need_occ) {
        record_hzb_build_pass(frame.command_buffer, hzb, debug_render.depth_image,
                              VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        // Same-frame pairing: this HZB was just rebuilt from the final
        // frame's depth with the final camera, so temporal validity does not
        // apply here.
        record_occlusion_refine_pass(frame.command_buffer, occlusion_refine, hzb,
                                     compute_selection, camera_frame, true);
    }
    if (need_copy) {
        record_visibility_copy_pass(frame.command_buffer, debug_render, swapchain.extent);
    }

    result = vkEndCommandBuffer(frame.command_buffer);
    if (result != VK_SUCCESS) {
        return result;
    }

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &frame.command_buffer;
    result = vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
    if (result != VK_SUCCESS) {
        return result;
    }
    return vkQueueWaitIdle(queue);
}

#endif

}  // namespace

VkBootstrapReport build_vk_bootstrap_report(VGeoResource& resource,
                                            const VkBootstrapConfig& config) {
    VkBootstrapReport report;
    report.uploadable_scene = build_uploadable_scene(resource);
    report.compiled_with_vulkan = MERIDIAN_HAS_VULKAN != 0;

#if !(MERIDIAN_HAS_VULKAN && MERIDIAN_HAS_GLFW)
    report.status =
        "Vulkan or GLFW headers are not available in this environment; bootstrap cannot create a windowed runtime";
    return report;
#else
    configure_macos_moltenvk_environment();

    if (glfwInit() != GLFW_TRUE) {
        report.status = "glfwInit failed";
        return report;
    }

    GLFWwindow* window = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphics_queue = VK_NULL_HANDLE;
    VkQueue present_queue = VK_NULL_HANDLE;
    SwapchainContext swapchain;
    FrameContext frame;
    UploadedSceneBuffers scene_buffers;
    DebugRenderContext debug_render;
    ComputeCullContext compute_cull;
    ComputeSelectionContext compute_selection;
    HzbContext hzb;
    OcclusionRefineContext occlusion_refine;
    ShadowContext shadow;
    // Frame-stable selection/draw cache: while the camera view-projection,
    // camera position, and resident page mask are bitwise unchanged, the
    // deterministic traversal and draw-list build reproduce the previous
    // frame's outputs exactly, so the whole pipeline (main + shadow
    // traversals, chunked draw build, bucket fold, and the host->device
    // draw-list uploads) is skipped and the cached lists are reused
    // as-is. Non-interactive cameras are static for the entire run;
    // interactive cameras rest between inputs. Exact byte equality (not
    // epsilon) is the key -- same rationale as the temporal HZB validity
    // test in the frame loop: identical inputs through the same
    // deterministic float ops produce identical bytes, and the resident
    // page mask covers every scene mutation site (streaming completions,
    // failures, evictions).
    struct FrameStableCache {
        bool valid = false;
        Mat4f view_projection;
        Vec3f camera_position;
        std::vector<uint8_t> resident_pages;
        bool separate_shadow = false;
        TraversalSelection main_selection;
        TraversalSelection shadow_selection;
        std::vector<GpuDrawEntry> main_draws;
        std::vector<GpuDrawEntry> shadow_draws;
        std::vector<DrawBucket> main_buckets;
        std::vector<DrawBucket> shadow_buckets;
        uint32_t main_draw_count = 0;
        uint32_t shadow_draw_count = 0;
        uint32_t main_encode_count = 0;
        uint32_t shadow_encode_count = 0;
        uint64_t main_wasted_vs = 0;
        uint64_t shadow_wasted_vs = 0;
    };
    FrameStableCache frame_cache;
    uint32_t frame_cache_hits = 0;
    std::filesystem::path temp_vgeo_path;
    bool framebuffer_resized = false;
    bool texture_stats_reported = false;
    uint32_t gpu_draw_count = 0;
    bool has_draw_indirect_count = false;
    GpuProfiler gpu_profiler;
    // Deterministic fork-join pool for the per-frame traversals and the
    // draw-list build. Splits never change output order (ordered merges),
    // so --threads 1 and --threads N are bit-identical.
    const unsigned int resolved_worker_threads =
        config.worker_threads == 0
            ? std::min(std::thread::hardware_concurrency(), 8u)
            : config.worker_threads;
    report.worker_threads = resolved_worker_threads;
    ParallelExecutor worker_pool(resolved_worker_threads);
    ParallelExecutor* traversal_executor =
        resolved_worker_threads > 1 ? &worker_pool : nullptr;

    const auto cleanup = [&]() {
        if (!temp_vgeo_path.empty()) {
            std::error_code ec;
            std::filesystem::remove(temp_vgeo_path, ec);
        }
        if (device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
            destroy_gpu_profiler(device, gpu_profiler);
            destroy_shadow_context(device, shadow);
            destroy_occlusion_refine_context(device, occlusion_refine);
            destroy_hzb_context(device, hzb);
            destroy_compute_selection_context(device, compute_selection);
            destroy_compute_cull_context(device, compute_cull);
            destroy_debug_render_context(device, debug_render);
            destroy_uploaded_scene_buffers(device, scene_buffers);
            destroy_frame_context(device, frame);
            destroy_swapchain(device, swapchain);
            vkDestroyDevice(device, nullptr);
        }
        if (surface != VK_NULL_HANDLE && instance != VK_NULL_HANDLE) {
            vkDestroySurfaceKHR(instance, surface, nullptr);
        }
        if (debug_messenger != VK_NULL_HANDLE && instance != VK_NULL_HANDLE) {
            auto destroy_messenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
            if (destroy_messenger != nullptr) {
                destroy_messenger(instance, debug_messenger, nullptr);
            }
        }
        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
        }
        if (window != nullptr) {
            glfwDestroyWindow(window);
        }
        glfwTerminate();
    };

    try {
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_VISIBLE, config.visible_window ? GLFW_TRUE : GLFW_FALSE);
        glfwWindowHint(GLFW_RESIZABLE, config.visible_window ? GLFW_TRUE : GLFW_FALSE);

        bool portability_enumeration = false;
        std::vector<const char*> instance_extensions =
            collect_instance_extensions(portability_enumeration);
        std::vector<const char*> validation_layers = collect_validation_layers(config.enable_validation);
        if (!validation_layers.empty()) {
            instance_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        VkApplicationInfo application_info{};
        application_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        application_info.pApplicationName = "Project Meridian Bootstrap";
        application_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
        application_info.pEngineName = "Meridian";
        application_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
        application_info.apiVersion = VK_API_VERSION_1_2;

        VkInstanceCreateInfo instance_info{};
        instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instance_info.pApplicationInfo = &application_info;
        instance_info.enabledExtensionCount = static_cast<uint32_t>(instance_extensions.size());
        instance_info.ppEnabledExtensionNames = instance_extensions.data();
        instance_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
        instance_info.ppEnabledLayerNames =
            validation_layers.empty() ? nullptr : validation_layers.data();
        // The portability enumeration flag is only valid together with the
        // VK_KHR_portability_enumeration extension; setting it when the
        // extension was not enabled fails instance creation.
        if (portability_enumeration) {
            instance_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        }

        VkResult result = vkCreateInstance(&instance_info, nullptr, &instance);
        if (result != VK_SUCCESS) {
            std::ostringstream message;
            message << "vkCreateInstance failed with code " << result;
            report.status = message.str();
            cleanup();
            return report;
        }
        report.instance_created = true;

        if (!validation_layers.empty()) {
            auto create_messenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
            if (create_messenger != nullptr) {
                VkDebugUtilsMessengerCreateInfoEXT messenger_info{};
                messenger_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
                messenger_info.messageSeverity =
                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
                messenger_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
                messenger_info.pfnUserCallback = &meridian_debug_callback;
                create_messenger(instance, &messenger_info, nullptr, &debug_messenger);
            }
        }

        window = glfwCreateWindow(static_cast<int>(config.window_width),
                                  static_cast<int>(config.window_height), "Meridian Bootstrap", nullptr,
                                  nullptr);
        if (window == nullptr) {
            report.status = "glfwCreateWindow failed";
            cleanup();
            return report;
        }
        report.window_created = true;
        glfwSetWindowUserPointer(window, &framebuffer_resized);
        glfwSetFramebufferSizeCallback(
            window, [](GLFWwindow* cb_window, int, int) {
                *static_cast<bool*>(glfwGetWindowUserPointer(cb_window)) = true;
            });

        result = create_window_surface(instance, window, &surface);
        if (result != VK_SUCCESS) {
            std::ostringstream message;
            message << "create_window_surface failed with code " << result;
            report.status = message.str();
            cleanup();
            return report;
        }
        report.surface_created = true;

        const DeviceSelection selection = select_device(instance, surface, report);
        if (selection.physical_device == VK_NULL_HANDLE) {
            report.status = "no compatible physical device found for graphics, present, and swapchain support";
            cleanup();
            return report;
        }

        const float queue_priority = 1.0f;
        std::set<uint32_t> unique_families = {selection.queues.graphics_family,
                                              selection.queues.present_family};
        std::vector<VkDeviceQueueCreateInfo> queue_infos;
        queue_infos.reserve(unique_families.size());
        for (const uint32_t family_index : unique_families) {
            VkDeviceQueueCreateInfo queue_info{};
            queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queue_info.queueFamilyIndex = family_index;
            queue_info.queueCount = 1;
            queue_info.pQueuePriorities = &queue_priority;
            queue_infos.push_back(queue_info);
        }

        std::vector<const char*> device_extensions;
        device_extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
        // Viewport/layer vertex-stage writes: on >= 1.2 devices the core
        // shaderOutputLayer feature is requested through the
        // VkPhysicalDeviceVulkan12Features chain below instead of the EXT
        // extension (never both at once); only pre-1.2 devices take the
        // extension route here.
        const bool viewport_layer_via_extension = !selection.shader_output_layer_feature;
        if (viewport_layer_via_extension) {
            device_extensions.push_back("VK_EXT_shader_viewport_index_layer");
        }
        if (selection.enable_portability_subset) {
            device_extensions.push_back("VK_KHR_portability_subset");
        }
        // When the draw-indirect-count capability comes from the Vulkan
        // 1.2 core feature rather than the extension string, the feature
        // is requested through the Vulkan12Features chain below.
        bool draw_indirect_count_via_feature = false;
        if (selection.has_draw_indirect_count) {
            if (selection.has_draw_indirect_count_extension) {
                device_extensions.push_back(VK_KHR_DRAW_INDIRECT_COUNT_EXTENSION_NAME);
            } else {
                draw_indirect_count_via_feature = true;
            }
        }

        VkPhysicalDeviceFeatures supported_features{};
        vkGetPhysicalDeviceFeatures(selection.physical_device, &supported_features);
        VkPhysicalDeviceFeatures device_features{};
        device_features.multiDrawIndirect = supported_features.multiDrawIndirect;
        device_features.drawIndirectFirstInstance = supported_features.drawIndirectFirstInstance;
        device_features.independentBlend = supported_features.independentBlend;
        // Enable-if-supported (independentBlend pattern): the base-color
        // sampler requests anisotropic filtering only when the device
        // exposes the feature; vkGetPhysicalDeviceFeatures on the same
        // device gates the sampler side, so the two never disagree.
        device_features.samplerAnisotropy = supported_features.samplerAnisotropy;

        VkPhysicalDeviceVulkan12Features vulkan12_features{};
        bool chain_vulkan12_features = false;
        if (selection.shader_output_layer_feature) {
            vulkan12_features.shaderOutputLayer = VK_TRUE;
            chain_vulkan12_features = true;
        }
        if (draw_indirect_count_via_feature) {
            vulkan12_features.drawIndirectCount = VK_TRUE;
            chain_vulkan12_features = true;
        }

        VkDeviceCreateInfo device_info{};
        device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        device_info.queueCreateInfoCount = static_cast<uint32_t>(queue_infos.size());
        device_info.pQueueCreateInfos = queue_infos.data();
        device_info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
        device_info.ppEnabledExtensionNames = device_extensions.data();
        device_info.pEnabledFeatures = &device_features;
        if (chain_vulkan12_features) {
            vulkan12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
            device_info.pNext = &vulkan12_features;
        }

        result = vkCreateDevice(selection.physical_device, &device_info, nullptr, &device);
        if (result != VK_SUCCESS) {
            std::ostringstream message;
            message << "vkCreateDevice failed with code " << result;
            report.status = message.str();
            cleanup();
            return report;
        }
        report.device_created = true;
        has_draw_indirect_count = selection.has_draw_indirect_count;

        vkGetDeviceQueue(device, selection.queues.graphics_family, 0, &graphics_queue);
        vkGetDeviceQueue(device, selection.queues.present_family, 0, &present_queue);

        // The main-pass descriptors bind the whole geometry payload buffers
        // as storage buffers (vk_render.cpp writes use the full buffer
        // range), and maxStorageBufferRange (core minimum 128 MiB) caps the
        // range a device must accept. Reject an oversized payload cleanly
        // here, before any descriptor is written or rendering starts.
        // Payload splitting or 64-bit shader addressing is future work
        // (AUDIT_OPEN.md, LS-54).
        {
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(selection.physical_device, &properties);
            const uint64_t max_storage_range =
                static_cast<uint64_t>(properties.limits.maxStorageBufferRange);
            const uint64_t payload_sizes[2] = {resource.cluster_geometry_payload.size(),
                                               resource.lod_geometry_payload.size()};
            const char* payload_names[2] = {"cluster geometry payload",
                                            "LOD geometry payload"};
            for (int payload_index = 0; payload_index < 2; ++payload_index) {
                if (payload_sizes[payload_index] > max_storage_range) {
                    std::ostringstream message;
                    message << payload_names[payload_index] << " (" << payload_sizes[payload_index]
                            << " bytes) exceeds device maxStorageBufferRange ("
                            << max_storage_range
                            << " bytes); payload splitting is not implemented";
                    report.status = message.str();
                    cleanup();
                    return report;
                }
            }
        }

        // Create GPU profiler (7 timer pairs: cull, sel, occ, shadow, main, hzb, total)
        if (config.enable_gpu_timers) {
            VkResult prof_result = create_gpu_profiler(selection.physical_device, device,
                                                        selection.queues.graphics_family, 7,
                                                        gpu_profiler);
            if (prof_result == VK_SUCCESS) {
                gpu_profiler.names[0] = "cull";
                gpu_profiler.names[1] = "sel";
                gpu_profiler.names[2] = "occ";
                gpu_profiler.names[3] = "shadow";
                gpu_profiler.names[4] = "main";
                gpu_profiler.names[5] = "hzb";
                gpu_profiler.names[6] = "total";
            } else {
                std::fprintf(stderr, "MERIDIAN_GPU: timestamp queries not supported (code %d), profiling disabled\n",
                             static_cast<int>(prof_result));
            }
        }

        ResidencyModel residency_model = create_residency_model(resource);
        // Demand-streaming: start pages unloaded so the scheduler drives the
        // loads from frame 0. Default path leaves every page resident for the
        // existing "all-in-memory" benchmark behavior.
        std::vector<uint32_t> page_load_start_frame(residency_model.pages.size(), 0xffffffffu);
        // Per-page absolute file offset + size, filled when async I/O is on.
        struct PageFileRange { uint64_t offset = 0; uint32_t size = 0; };
        std::vector<PageFileRange> page_file_ranges(residency_model.pages.size());
        AsyncReader async_reader;
        bool async_io_active = false;
        StreamingScheduler streaming_scheduler;
        uint32_t stream_pages_uploaded = 0;
        uint64_t stream_bytes_uploaded = 0;
        uint32_t stream_page_failures = 0;
        if (config.demand_streaming) {
            for (PageResidencyEntry& entry : residency_model.pages) {
                entry.state = PageResidencyState::unloaded;
                entry.last_touched_frame = 0xffffffffu;
            }
            // Root-page autodetect: run a coarse traversal with all pages
            // marked resident and a very large error threshold to discover
            // which pages the hierarchy actually needs for its minimum-detail
            // render. This is the correct seed set regardless of storage
            // layout (previously we seeded the first N pages by index, which
            // happened to work only because the DFS flatten pass tends to
            // land low-detail clusters at the front of the linear layout).
            const std::vector<uint8_t> all_resident_mask(residency_model.pages.size(), 1);
            const TraversalSelection coarse =
                simulate_traversal(resource, /*error_threshold=*/1e30f, all_resident_mask);
            const uint32_t seed_cap = std::min<uint32_t>(
                config.streaming_seed_pages,
                static_cast<uint32_t>(residency_model.pages.size()));
            uint32_t seeded = 0;
            for (uint32_t p : coarse.selected_page_indices) {
                if (p >= residency_model.pages.size()) continue;
                if (residency_model.pages[p].state == PageResidencyState::resident) continue;
                residency_model.pages[p].state = PageResidencyState::resident;
                residency_model.pages[p].last_touched_frame = 0;
                if (++seeded >= seed_cap) break;
            }
            // Fall back to linear seed only if the coarse traversal produced
            // fewer pages than the seed cap (unlikely but defensive against
            // scenes where the root has zero LOD links and an empty base
            // span -- e.g. a completely uninitialised hierarchy).
            for (uint32_t p = 0; seeded < seed_cap && p < residency_model.pages.size(); ++p) {
                if (residency_model.pages[p].state == PageResidencyState::resident) continue;
                residency_model.pages[p].state = PageResidencyState::resident;
                residency_model.pages[p].last_touched_frame = 0;
                ++seeded;
            }
            StreamingConfig sc;
            sc.max_resident_pages = config.resident_budget == 0xffffffffu
                                        ? static_cast<uint32_t>(residency_model.pages.size())
                                        : config.resident_budget;
            sc.max_loads_per_frame = config.streaming_max_loads_per_frame;
            sc.eviction_grace_frames = config.eviction_grace_frames;
            streaming_scheduler = create_streaming_scheduler(resource, sc);

            // Real async disk I/O path: mmap a .vgeo and serve page loads
            // from a worker thread that copies each page's byte range out of
            // the mapping. The byte source is the manifest's persisted
            // output .vgeo when it exists and its header matches the freshly
            // built resource (no startup write at all); otherwise we
            // serialize the resource to a temp .vgeo and mmap that. If
            // anything fails we fall back to the latency-window simulation
            // (async_io_active stays false) and keep the full startup
            // payload upload.
            try {
                std::filesystem::path source_path;
                bool from_persisted = false;
                if (!config.persisted_vgeo_path.empty() &&
                    std::filesystem::path(config.persisted_vgeo_path).extension() == ".vgeo") {
                    std::error_code ec;
                    const std::filesystem::path candidate(config.persisted_vgeo_path);
                    if (std::filesystem::exists(candidate, ec)) {
                        std::ifstream header_input(candidate, std::ios::binary);
                        detail::FileHeader header{};
                        if (header_input &&
                            header_input.read(reinterpret_cast<char*>(&header), sizeof(header)) &&
                            std::equal(std::begin(header.magic), std::end(header.magic),
                                       detail::kMagic.begin()) &&
                            header.schema_version >= 1 &&
                            header.schema_version <= detail::kSchemaVersion &&
                            header.builder_version == detail::kBuilderVersion &&
                            header.total_hierarchy_nodes == resource.hierarchy_nodes.size() &&
                            header.total_clusters == resource.clusters.size() &&
                            header.total_pages == resource.pages.size() &&
                            header.total_lod_groups == resource.lod_groups.size() &&
                            header.total_lod_clusters == resource.lod_clusters.size() &&
                            header.total_cluster_geometry_bytes ==
                                resource.cluster_geometry_payload.size() &&
                            header.total_lod_geometry_bytes ==
                                resource.lod_geometry_payload.size() &&
                            // Content identity, not just shape: two builds
                            // can match every version/count/total above while
                            // their payload bytes, page layout, or metadata
                            // differ, and streaming from that file feeds the
                            // GPU bytes that do not belong to this resource.
                            header.content_fingerprint ==
                                detail::compute_content_fingerprint(resource)) {
                            source_path = candidate;
                            from_persisted = true;
                        }
                    }
                }
                if (!from_persisted) {
                    const std::string unique =
                        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
                    temp_vgeo_path = std::filesystem::temp_directory_path() /
                                     (std::string("meridian-stream-") + unique + ".vgeo");
                    write_resource(resource, temp_vgeo_path);
                    source_path = temp_vgeo_path;
                }
                // Payload region offsets come from the file's own header --
                // the layout math changed across schema versions (the base-
                // run table landed between v2 and v3), so recomputing them
                // from in-memory counts is version-fragile.
                std::ifstream header_input(source_path, std::ios::binary);
                detail::FileHeader header{};
                if (!header_input ||
                    !header_input.read(reinterpret_cast<char*>(&header), sizeof(header)) ||
                    !std::equal(std::begin(header.magic), std::end(header.magic),
                                detail::kMagic.begin())) {
                    throw BuilderError("failed to read .vgeo header back: " + source_path.string());
                }
                for (uint32_t p = 0; p < resource.pages.size(); ++p) {
                    const PageRecord& page = resource.pages[p];
                    const uint64_t base = (page.lod_cluster_count != 0)
                                              ? header.lod_geometry_payload_offset
                                              : header.cluster_geometry_payload_offset;
                    page_file_ranges[p] = {base + page.byte_offset, page.uncompressed_byte_size};
                }
                async_io_active = async_reader.open(source_path);
                if (!async_io_active) {
                    std::fprintf(stderr,
                                 "MERIDIAN_STREAM: async_reader.open failed, "
                                 "falling back to latency simulation\n");
                } else {
                    std::fprintf(stderr, "MERIDIAN_STREAM: mmap source = %s\n",
                                 from_persisted ? source_path.string().c_str()
                                                : "temp .vgeo (startup write)");
                }
            } catch (const std::exception& e) {
                std::fprintf(stderr,
                             "MERIDIAN_STREAM: streaming setup failed (%s), "
                             "falling back to latency simulation\n",
                             e.what());
                async_io_active = false;
            }
        }

        // When the mmap reader is live the payload buffers start empty and
        // stream per page; otherwise (default path, or reader-open failure)
        // keep the full startup upload so the fallback renders correctly.
        const bool stream_payloads = config.demand_streaming && async_io_active;
        result = upload_scene_buffers(selection.physical_device, device,
                                      graphics_queue, selection.queues.graphics_family,
                                      report.uploadable_scene,
                                      scene_buffers, report, stream_payloads);
        if (result != VK_SUCCESS) {
            std::ostringstream message;
            message << "upload_scene_buffers failed with code " << result;
            report.status = message.str();
            cleanup();
            return report;
        }

        if (stream_payloads) {
            // Seed pages were marked resident directly, so their bytes must
            // land in the (otherwise empty) payload buffers before frame 0.
            // Synchronous copies from the mapping -- the temp .vgeo was just
            // written, so the pages are cache-hot.
            std::vector<std::byte> seed_scratch;
            uint32_t seeded_pages = 0;
            for (uint32_t p = 0; p < residency_model.pages.size(); ++p) {
                if (residency_model.pages[p].state != PageResidencyState::resident) continue;
                const PageFileRange& range = page_file_ranges[p];
                if (range.size == 0) continue;
                seed_scratch.resize(range.size);
                if (!async_reader.read_sync(range.offset, range.size, seed_scratch.data())) {
                    // Residency is only publishable with bytes behind it:
                    // leave the seed page unloaded so the streaming path
                    // re-requests it (traversal reports it missing, the
                    // scheduler scores the demand, the async reader retries)
                    // instead of rendering a resident page whose payload
                    // slots were never uploaded.
                    residency_model.pages[p].state = PageResidencyState::unloaded;
                    stream_page_failures += 1;
                    continue;
                }
                const PageRecord& page = resource.pages[p];
                UploadedBuffer& dst = (page.lod_cluster_count != 0)
                                          ? scene_buffers.lod_payload
                                          : scene_buffers.base_payload;
                result = upload_page_bytes(selection.physical_device, device, graphics_queue,
                                           selection.queues.graphics_family,
                                           seed_scratch.data(), range.size,
                                           static_cast<VkDeviceSize>(page.byte_offset), dst);
                if (result != VK_SUCCESS) {
                    std::ostringstream message;
                    message << "seed page upload failed with code " << result;
                    report.status = message.str();
                    cleanup();
                    return report;
                }
                seeded_pages += 1;
                stream_pages_uploaded += 1;
                stream_bytes_uploaded += range.size;
            }
            std::fprintf(stderr,
                         "MERIDIAN_STREAM: seeded %u pages (%llu bytes, %u failed) "
                         "synchronously via %s\n",
                         seeded_pages, static_cast<unsigned long long>(stream_bytes_uploaded),
                         stream_page_failures,
                         async_reader.mmap_active() ? "mmap" : "pread");
            // The mmap'd .vgeo is now the source of truth for page bytes;
            // drop the CPU-side payload copies. build_vk_bootstrap_report
            // takes the resource by non-const reference exactly for this
            // consumption (see vk_bootstrap.h).
            report.uploadable_scene.base_payload.clear();
            report.uploadable_scene.base_payload.shrink_to_fit();
            report.uploadable_scene.lod_payload.clear();
            report.uploadable_scene.lod_payload.shrink_to_fit();
            resource.cluster_geometry_payload.clear();
            resource.cluster_geometry_payload.shrink_to_fit();
            resource.lod_geometry_payload.clear();
            resource.lod_geometry_payload.shrink_to_fit();
        }

        result = create_compute_cull_context(selection.physical_device, device, scene_buffers,
                                              static_cast<uint32_t>(report.uploadable_scene.instances.size()),
                                              compute_cull);
        if (result != VK_SUCCESS) {
            std::ostringstream message;
            message << "create_compute_cull_context failed with code " << result;
            report.status = message.str();
            cleanup();
            return report;
        }

        const uint32_t total_clusters =
            static_cast<uint32_t>(report.uploadable_scene.clusters.size() +
                                  report.uploadable_scene.lod_clusters.size());
        result = create_compute_selection_context(selection.physical_device, device, scene_buffers,
                                                   compute_cull, total_clusters,
                                                   config.enable_gpu_selection, compute_selection);
        if (result != VK_SUCCESS) {
            std::ostringstream message;
            message << "create_compute_selection_context failed with code " << result;
            report.status = message.str();
            cleanup();
            return report;
        }

        result = create_swapchain(selection.physical_device, device, surface, window, selection.queues,
                                  swapchain);
        if (result != VK_SUCCESS) {
            std::ostringstream message;
            message << "create_swapchain failed with code " << result;
            report.status = message.str();
            cleanup();
            return report;
        }
        report.swapchain_created = true;
        report.swapchain_image_count = static_cast<uint32_t>(swapchain.images.size());
        report.swapchain_width = swapchain.extent.width;
        report.swapchain_height = swapchain.extent.height;

        CameraFrameData camera_frame =
            build_camera_frame_data(resource, swapchain.extent, report.debug_camera_distance);

        // Interactive camera setup
        InteractiveCamera interactive_cam;
        if (config.interactive) {
            const Vec3f center = {
                (resource.bounds.min.x + resource.bounds.max.x) * 0.5f,
                (resource.bounds.min.y + resource.bounds.max.y) * 0.5f,
                (resource.bounds.min.z + resource.bounds.max.z) * 0.5f,
            };
            const float radius = std::max({
                resource.bounds.max.x - resource.bounds.min.x,
                resource.bounds.max.y - resource.bounds.min.y,
                resource.bounds.max.z - resource.bounds.min.z, 1.0f});
            interactive_cam.position = {center.x + radius * 0.55f, center.y + radius * 0.9f,
                                        center.z + radius * 2.4f};
            // Compute yaw/pitch to face the center of the model
            const Vec3f to_center = {center.x - interactive_cam.position.x,
                                     center.y - interactive_cam.position.y,
                                     center.z - interactive_cam.position.z};
            const float horiz_dist = std::sqrt(to_center.x * to_center.x + to_center.z * to_center.z);
            interactive_cam.yaw = std::atan2(to_center.x, -to_center.z);
            interactive_cam.pitch = std::atan2(to_center.y, horiz_dist);
            interactive_cam.move_speed = radius * 0.8f;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            glfwGetCursorPos(window, &interactive_cam.last_cursor_x, &interactive_cam.last_cursor_y);
            interactive_cam.cursor_captured = true;
        }
        // Two clocks, one per concern (LS-56): last_frame_time drives the
        // interactive camera integration dt; benchmark_last_frame_time is
        // the MERIDIAN_BENCHMARK frame-period sample. They must not share a
        // variable: the camera update used to reset the shared timestamp,
        // so the benchmark then measured only the camera-update sliver
        // (median_ms=0.00 at vsync frame periods), never the inter-frame
        // interval.
        double last_frame_time = glfwGetTime();
        double benchmark_last_frame_time = last_frame_time;
        uint32_t fps_frame_count = 0;
        // Draw-list upload scratch: entries past the live count are zeroed so the
        // vkCmdDrawIndirect fallback (no draw_indirect_count extension) never
        // replays stale draws with instanceCount=1 after the visible set shrinks.
        std::vector<GpuDrawEntry> draw_upload_scratch;
        std::vector<GpuDrawEntry> shadow_upload_scratch;
        uint32_t draw_list_high_water = 0;
        uint32_t shadow_list_high_water = 0;
        double fps_timer = last_frame_time;
        std::vector<double> frame_times_ms;

        snapshot_page_residency(report.uploadable_scene, residency_model);
        result = update_vector_buffer(device, report.uploadable_scene.page_residency,
                                      scene_buffers.page_residency);
        if (result != VK_SUCCESS) {
            std::ostringstream message;
            message << "initial page residency upload failed with code " << result;
            report.status = message.str();
            cleanup();
            return report;
        }

        const std::vector<uint8_t> initial_resident_pages = build_resident_page_mask(residency_model);

        // Temporal HZB validity (the refine pass binds the previous frame's
        // HZB while pushing the current camera's view-projection). The pair
        // is valid only while the view-projection is bitwise identical to
        // the previous frame's AND the scene did not change under it:
        // matrices here are products of the same inputs through the same
        // deterministic float ops, so exact equality is the correct test --
        // any real camera/resident-set change produces different bytes, and
        // an epsilon would wrongly validate matrices built from moved eyes.
        // Scene changes are coarsely tracked as the resident page mask;
        // every mutation site (streaming completions, failures, evictions)
        // lands in the mask before the next frame's draw list is built.
        // has_history additionally covers swapchain recreation, where the
        // recreated HZB carries no usable previous-frame content.
        Mat4f temporal_hzb_last_view_projection;
        uint64_t temporal_hzb_last_generation = 0;
        bool temporal_hzb_has_history = false;
        uint64_t scene_generation = 0;
        std::vector<uint8_t> previous_resident_mask = initial_resident_pages;
        // Resolve the LOD selection threshold. A negative config value means
        // auto: a fixed multiple of the scene's median LOD-group geometric
        // error, floored at the historical 0.001 default. The multiple is
        // calibrated so scenes whose LOD ladder sits near the old default
        // (e.g. the dragon scan) resolve to the floor unchanged, while
        // large, coarsely-grained scenes activate their mid-LODs instead of
        // selecting every cluster at full detail.
        const float error_threshold = [&]() {
            if (config.debug_error_threshold >= 0.0f) return config.debug_error_threshold;
            constexpr float kAutoThresholdFloor = 0.001f;
            constexpr float kAutoThresholdMedianMultiple = 8.9f;
            std::vector<float> group_errors;
            group_errors.reserve(resource.lod_groups.size());
            for (const LodGroupRecord& group : resource.lod_groups) {
                group_errors.push_back(group.geometric_error);
            }
            if (group_errors.empty()) return kAutoThresholdFloor;
            std::sort(group_errors.begin(), group_errors.end());
            const float median_error = group_errors[group_errors.size() / 2];
            if (!std::isfinite(median_error)) return kAutoThresholdFloor;
            const float threshold =
                std::max(kAutoThresholdFloor, kAutoThresholdMedianMultiple * median_error);
            std::fprintf(stderr,
                         "MERIDIAN_LOD: auto error threshold %.4f (median group error %.6f over "
                         "%zu groups)\n",
                         threshold, median_error, group_errors.size());
            return threshold;
        }();
        if (config.shadow_error_scale > 1.0f) {
            std::fprintf(stderr,
                         "MERIDIAN_SHADOW: caster LOD error threshold %.4f (scale %.2fx of "
                         "main)\n",
                         error_threshold * config.shadow_error_scale,
                         config.shadow_error_scale);
        }
        const TraversalSelection initial_selection =
            simulate_traversal(resource, error_threshold, initial_resident_pages);
        report.runtime_missing_page_count =
            static_cast<uint32_t>(initial_selection.missing_page_indices.size());
        report.runtime_prefetch_page_count =
            static_cast<uint32_t>(initial_selection.prefetch_page_indices.size());
        report.runtime_resident_page_count = count_resident_pages(residency_model);
        report.replay_runtime_parity =
            report.debug_selected_node_count == report.replay_selected_node_count &&
            report.debug_rendered_cluster_count == report.replay_selected_cluster_count &&
            report.debug_rendered_lod_cluster_count == report.replay_selected_lod_cluster_count;

        // Extent-dependent GPU resources: destroyed and recreated together on
        // swapchain out-of-date / window resize.
        const auto create_surface_resources = [&]() -> VkResult {
            VkResult r = create_debug_render_context(selection.physical_device, device,
                                                     graphics_queue, selection.queues.graphics_family,
                                                     swapchain, scene_buffers,
                                                     report.uploadable_scene,
                                                     compute_selection.draw_list,
                                                     config.texture_mips, debug_render);
            if (r != VK_SUCCESS) {
                return r;
            }
            if (!texture_stats_reported) {
                texture_stats_reported = true;
                uint32_t uv_base_clusters = 0;
                for (const GpuClusterRecord& cluster : report.uploadable_scene.clusters) {
                    if ((cluster.flags & kClusterFlagHasUv) != 0) {
                        uv_base_clusters += 1;
                    }
                }
                uint32_t uv_lod_clusters = 0;
                for (const GpuLodClusterRecord& cluster : report.uploadable_scene.lod_clusters) {
                    if ((cluster.flags & kClusterFlagHasUv) != 0) {
                        uv_lod_clusters += 1;
                    }
                }
                float uv_min[2] = {0.0f, 0.0f};
                float uv_max[2] = {0.0f, 0.0f};
                bool has_uv_range = false;
                const auto accumulate_uv_range =
                    [&](const std::vector<std::byte>& payload, uint32_t offset, uint32_t vertex_count) {
                        if (payload.empty()) {
                            return;
                        }
                        const uint32_t uv_base = offset + 8u + vertex_count * 24u;
                        if (uv_base + vertex_count * 8u > payload.size()) {
                            return;
                        }
                        for (uint32_t v = 0; v < vertex_count; ++v) {
                            float uv[2];
                            std::memcpy(uv, payload.data() + uv_base + v * 8u, sizeof(uv));
                            if (!has_uv_range) {
                                uv_min[0] = uv_max[0] = uv[0];
                                uv_min[1] = uv_max[1] = uv[1];
                                has_uv_range = true;
                            } else {
                                uv_min[0] = std::min(uv_min[0], uv[0]);
                                uv_max[0] = std::max(uv_max[0], uv[0]);
                                uv_min[1] = std::min(uv_min[1], uv[1]);
                                uv_max[1] = std::max(uv_max[1], uv[1]);
                            }
                        }
                    };
                for (const GpuClusterRecord& cluster : report.uploadable_scene.clusters) {
                    if ((cluster.flags & kClusterFlagHasUv) != 0) {
                        accumulate_uv_range(report.uploadable_scene.base_payload,
                                            cluster.payload_offset, cluster.local_vertex_count);
                    }
                }
                for (const GpuLodClusterRecord& cluster : report.uploadable_scene.lod_clusters) {
                    if ((cluster.flags & kClusterFlagHasUv) != 0) {
                        accumulate_uv_range(report.uploadable_scene.lod_payload,
                                            cluster.payload_offset, cluster.local_vertex_count);
                    }
                }
                std::fprintf(stderr,
                             "MERIDIAN_TEXTURE: %ux%u RGBA8 (%s), mips=%u, uv_clusters=%u+%u, "
                             "payload_uv_range=[%.3f,%.3f]..[%.3f,%.3f]%s\n",
                             debug_render.base_texture_width, debug_render.base_texture_height,
                             debug_render.base_texture_is_placeholder ? "placeholder" : "embedded",
                             debug_render.base_texture_mip_levels,
                             uv_base_clusters, uv_lod_clusters, uv_min[0], uv_min[1], uv_max[0],
                             uv_max[1], has_uv_range ? "" : " (none)");
            }
            r = create_hzb_context(selection.physical_device, device,
                                   swapchain.extent.width, swapchain.extent.height,
                                   debug_render.depth_view, debug_render.depth_format,
                                   graphics_queue, selection.queues.graphics_family, hzb);
            if (r != VK_SUCCESS) {
                return r;
            }
            r = create_occlusion_refine_context(selection.physical_device, device,
                                                compute_selection, scene_buffers, hzb,
                                                total_clusters, graphics_queue,
                                                selection.queues.graphics_family,
                                                occlusion_refine);
            if (r != VK_SUCCESS) {
                return r;
            }
            r = create_shadow_context(selection.physical_device, device, scene_buffers,
                                      debug_render.frame_ubo, compute_selection.max_draws,
                                      resource, 2048, shadow);
            if (r != VK_SUCCESS) {
                return r;
            }
            // Bind cascaded shadow map (2D array) to main pass descriptor set binding 3
            if (shadow.depth_array_view != VK_NULL_HANDLE && shadow.sampler != VK_NULL_HANDLE) {
                VkDescriptorImageInfo shadow_img_info{};
                shadow_img_info.sampler = shadow.sampler;
                shadow_img_info.imageView = shadow.depth_array_view;
                shadow_img_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                VkWriteDescriptorSet shadow_write{};
                shadow_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                shadow_write.dstSet = debug_render.descriptor_set;
                shadow_write.dstBinding = 3;
                shadow_write.descriptorCount = 1;
                shadow_write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                shadow_write.pImageInfo = &shadow_img_info;
                vkUpdateDescriptorSets(device, 1, &shadow_write, 0, nullptr);
            }
            return VK_SUCCESS;
        };

        result = create_surface_resources();
        if (result != VK_SUCCESS) {
            std::ostringstream message;
            message << "create_surface_resources failed with code " << result;
            report.status = message.str();
            cleanup();
            return report;
        }
        // maxDrawIndirectCount gate: every vkCmdDrawIndirectCount passes its
        // list capacity as maxDrawCount, which must not exceed the device
        // limit. Splitting is impractical here because the count buffer is
        // GPU-written (the occlusion survivor count), so when any capacity
        // exceeds the limit the run falls back to the CPU folded path.
        if (has_draw_indirect_count &&
            (selection.max_draw_indirect_count < compute_selection.max_draws ||
             selection.max_draw_indirect_count < occlusion_refine.max_draws ||
             selection.max_draw_indirect_count < shadow.max_draws)) {
            has_draw_indirect_count = false;
            std::fprintf(stderr,
                         "MERIDIAN_VK: draw list capacity %u exceeds maxDrawIndirectCount %u; "
                         "using the CPU folded draw path\n",
                         compute_selection.max_draws, selection.max_draw_indirect_count);
        }
        report.debug_pipeline_created = true;
        report.debug_geometry_uploaded = debug_render.descriptor_set != VK_NULL_HANDLE;
        report.visibility_attachment_created = debug_render.visibility_view != VK_NULL_HANDLE;

        const auto recreate_surface = [&]() -> bool {
            vkDeviceWaitIdle(device);
            destroy_frame_context(device, frame);
            destroy_shadow_context(device, shadow);
            destroy_occlusion_refine_context(device, occlusion_refine);
            destroy_hzb_context(device, hzb);
            destroy_debug_render_context(device, debug_render);
            destroy_swapchain(device, swapchain);
            // The recreated HZB carries no usable previous-frame content.
            temporal_hzb_has_history = false;
            VkResult r =
                create_swapchain(selection.physical_device, device, surface, window,
                                 selection.queues, swapchain);
            if (r != VK_SUCCESS) {
                return false;
            }
            if (create_surface_resources() != VK_SUCCESS) {
                return false;
            }
            return create_frame_context(device, selection.queues,
                                        static_cast<uint32_t>(swapchain.images.size()),
                                        frame) == VK_SUCCESS;
        };

        update_debug_selection_report(initial_selection, report.uploadable_scene, report);
        report.replay_runtime_parity =
            report.debug_selected_node_count == report.replay_selected_node_count &&
            report.debug_rendered_cluster_count == report.replay_selected_cluster_count &&
            report.debug_rendered_lod_cluster_count == report.replay_selected_lod_cluster_count;

        result = create_frame_context(device, selection.queues,
                                      static_cast<uint32_t>(swapchain.images.size()), frame);
        if (result != VK_SUCCESS) {
            std::ostringstream message;
            message << "create_frame_context failed with code " << result;
            report.status = message.str();
            cleanup();
            return report;
        }

        for (uint32_t frame_index = 0;
             config.interactive ? (glfwWindowShouldClose(window) != GLFW_TRUE)
                                 : (frame_index < config.present_frame_count);
             ++frame_index) {
            glfwPollEvents();
            if (glfwWindowShouldClose(window) == GLFW_TRUE) {
                break;
            }
            if (framebuffer_resized) {
                framebuffer_resized = false;
                if (!recreate_surface()) {
                    report.status = "swapchain recreation failed after window resize";
                    cleanup();
                    return report;
                }
                continue;
            }

            // Benchmark clock: sampled once per frame BEFORE the camera
            // update, so the sample spans the full frame period (present
            // wait -> camera -> submit -> present) regardless of what the
            // camera-integration clock does to its own timestamp.
            {
                const double frame_now = glfwGetTime();
                if (frame_index > 2) {
                    frame_times_ms.push_back((frame_now - benchmark_last_frame_time) * 1000.0);
                }
                benchmark_last_frame_time = frame_now;
            }

            // Interactive camera update
            if (config.interactive) {
                const double now = glfwGetTime();
                const float dt = static_cast<float>(now - last_frame_time);
                last_frame_time = now;

                // Mouse look
                double cx, cy;
                glfwGetCursorPos(window, &cx, &cy);
                if (interactive_cam.cursor_captured) {
                    const float dx = static_cast<float>(cx - interactive_cam.last_cursor_x);
                    const float dy = static_cast<float>(cy - interactive_cam.last_cursor_y);
                    interactive_cam.yaw += dx * interactive_cam.mouse_sensitivity;
                    interactive_cam.pitch -= dy * interactive_cam.mouse_sensitivity;
                    interactive_cam.pitch = std::max(-1.5f, std::min(1.5f, interactive_cam.pitch));
                }
                interactive_cam.last_cursor_x = cx;
                interactive_cam.last_cursor_y = cy;

                // WASD movement
                const Vec3f fwd = camera_forward(interactive_cam);
                const Vec3f rgt = camera_right(interactive_cam);
                const float spd = interactive_cam.move_speed * dt;
                if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
                    interactive_cam.position.x += fwd.x * spd;
                    interactive_cam.position.y += fwd.y * spd;
                    interactive_cam.position.z += fwd.z * spd;
                }
                if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
                    interactive_cam.position.x -= fwd.x * spd;
                    interactive_cam.position.y -= fwd.y * spd;
                    interactive_cam.position.z -= fwd.z * spd;
                }
                if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
                    interactive_cam.position.x -= rgt.x * spd;
                    interactive_cam.position.z -= rgt.z * spd;
                }
                if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
                    interactive_cam.position.x += rgt.x * spd;
                    interactive_cam.position.z += rgt.z * spd;
                }
                if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
                    interactive_cam.position.y -= spd;
                }
                if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
                    interactive_cam.position.y += spd;
                }
                if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
                    glfwSetWindowShouldClose(window, GLFW_TRUE);
                }

                // Recompute camera VP
                const Vec3f target = {interactive_cam.position.x + fwd.x,
                                      interactive_cam.position.y + fwd.y,
                                      interactive_cam.position.z + fwd.z};
                const float aspect = std::max(1.0f, static_cast<float>(swapchain.extent.width)) /
                                     std::max(1.0f, static_cast<float>(swapchain.extent.height));
                const Mat4f view = look_at_matrix(interactive_cam.position, target, {0.0f, 1.0f, 0.0f});
                const float radius = std::max({
                    resource.bounds.max.x - resource.bounds.min.x,
                    resource.bounds.max.y - resource.bounds.min.y,
                    resource.bounds.max.z - resource.bounds.min.z, 1.0f});
                const Mat4f proj = perspective_matrix(55.0f * 3.14159265f / 180.0f, aspect,
                                                      std::max(0.01f, radius * 0.01f), radius * 8.0f);
                camera_frame.view_projection = multiply_matrix(proj, view);
                camera_frame.camera_position = interactive_cam.position;

                fps_frame_count++;
                if (now - fps_timer >= 1.0) {
                    char title[256];
                    const uint32_t total_pages = static_cast<uint32_t>(residency_model.pages.size());
                    const uint32_t res_pages = count_resident_pages(residency_model);
                    std::snprintf(title, sizeof(title),
                                  "Meridian - %u FPS - %u draws - pages %u/%u",
                                  fps_frame_count, gpu_draw_count, res_pages, total_pages);
                    glfwSetWindowTitle(window, title);
                    fps_frame_count = 0;
                    fps_timer = now;
                }
            }

            // Async-load completion. Two paths:
            //   * Real async I/O (async_io_active): drain completions from
            //     the worker thread that just copied the page's byte range
            //     out of the mmap'd .vgeo. Completion time reflects
            //     actual disk latency + worker scheduling, and the
            //     completion's bytes are uploaded into the GPU payload
            //     buffer after this frame's residency step.
            //   * Simulated (fallback): a page that entered the loading state
            //     streaming_load_latency_frames ago now becomes resident.
            //   * Non-streaming default: complete_loading_pages transitions
            //     every loading page to resident instantly.
            std::vector<uint32_t> completed_this_frame;
            std::vector<AsyncReadCompletion> completed_reads;
            if (config.demand_streaming) {
                if (async_io_active) {
                    std::vector<AsyncReadCompletion> reads = async_reader.drain_completions();
                    for (auto& r : reads) {
                        if (r.page_index >= residency_model.pages.size()) continue;
                        if (!r.success) {
                            // A failed read must not strand the page in
                            // loading forever: drop it back to unloaded so
                            // next frame's traversal reports it missing and
                            // the scheduler re-requests the read (retryable
                            // failure, surfaced below and in the report).
                            if (residency_model.pages[r.page_index].state ==
                                PageResidencyState::loading) {
                                residency_model.pages[r.page_index].state =
                                    PageResidencyState::unloaded;
                                page_load_start_frame[r.page_index] = 0xffffffffu;
                                stream_page_failures += 1;
                                std::fprintf(stderr,
                                             "MERIDIAN_STREAM: page %u read failed; "
                                             "queued for retry\n",
                                             r.page_index);
                            }
                            continue;
                        }
                        if (residency_model.pages[r.page_index].state !=
                            PageResidencyState::loading) continue;
                        completed_this_frame.push_back(r.page_index);
                        completed_reads.push_back(std::move(r));
                        page_load_start_frame[r.page_index] = 0xffffffffu;
                    }
                } else {
                    for (uint32_t p = 0; p < residency_model.pages.size(); ++p) {
                        if (residency_model.pages[p].state == PageResidencyState::loading &&
                            page_load_start_frame[p] != 0xffffffffu &&
                            frame_index - page_load_start_frame[p] >=
                                config.streaming_load_latency_frames) {
                            completed_this_frame.push_back(p);
                            page_load_start_frame[p] = 0xffffffffu;
                        }
                    }
                }
                report.runtime_completed_page_count =
                    static_cast<uint32_t>(completed_this_frame.size());
            } else {
                report.runtime_completed_page_count =
                    complete_loading_pages(residency_model, frame_index);
            }

            using clock_t = std::chrono::steady_clock;
            auto t_traverse_start = clock_t::now();
            const std::vector<uint8_t> resident_pages = build_resident_page_mask(residency_model);
            if (resident_pages != previous_resident_mask) {
                // Any page becoming resident or evicted changes what the
                // next draw list renders; the HZB lags one frame behind.
                scene_generation += 1;
                previous_resident_mask = resident_pages;
            }
            // Shadow caster LOD: casters come from a second traversal at a
            // coarser error threshold. Shadow maps are depth-only and filtered
            // (2048px cascades + 8-tap PCF), so silhouette detail far below the
            // texel footprint cannot survive into the shaded result.
            // Scale <= 1 falls back to sharing the main-pass selection.
            // The main and shadow-caster traversals are independent reads of
            // (resource, resident_pages); they run concurrently on the pool,
            // and each additionally forks subtree tasks (ordered merges keep
            // the selection bit-identical to the serial path).
            TraversalSelection selection_for_frame_storage;
            TraversalSelection shadow_selection_storage;
            const bool separate_shadow_selection = config.shadow_error_scale > 1.0f;
            const bool frame_cache_hit =
                frame_cache.valid &&
                frame_cache.separate_shadow == separate_shadow_selection &&
                std::memcmp(frame_cache.view_projection.m, camera_frame.view_projection.m,
                            sizeof(frame_cache.view_projection.m)) == 0 &&
                std::memcmp(&frame_cache.camera_position, &camera_frame.camera_position,
                            sizeof(frame_cache.camera_position)) == 0 &&
                frame_cache.resident_pages == resident_pages;
            if (frame_cache_hit) {
                frame_cache_hits += 1;
            } else {
                if (separate_shadow_selection && traversal_executor != nullptr) {
                    const float shadow_threshold = error_threshold * config.shadow_error_scale;
                    std::vector<std::function<void()>> traverse_jobs;
                    traverse_jobs.push_back([&] {
                        selection_for_frame_storage =
                            simulate_traversal(resource, error_threshold, resident_pages,
                                               traversal_executor);
                    });
                    traverse_jobs.push_back([&] {
                        shadow_selection_storage =
                            simulate_traversal(resource, shadow_threshold, resident_pages,
                                               traversal_executor);
                    });
                    worker_pool.run(traverse_jobs);
                } else {
                    selection_for_frame_storage =
                        simulate_traversal(resource, error_threshold, resident_pages,
                                           traversal_executor);
                    if (separate_shadow_selection) {
                        shadow_selection_storage =
                            simulate_traversal(resource, error_threshold * config.shadow_error_scale,
                                               resident_pages, traversal_executor);
                    }
                }
                frame_cache.valid = true;
                frame_cache.view_projection = camera_frame.view_projection;
                frame_cache.camera_position = camera_frame.camera_position;
                frame_cache.resident_pages = resident_pages;
                frame_cache.separate_shadow = separate_shadow_selection;
                frame_cache.main_selection = std::move(selection_for_frame_storage);
                frame_cache.shadow_selection = std::move(shadow_selection_storage);
            }
            const TraversalSelection& selection_for_frame = frame_cache.main_selection;
            const TraversalSelection* shadow_selection =
                separate_shadow_selection ? &frame_cache.shadow_selection : &selection_for_frame;
            auto t_traverse_end = clock_t::now();
            static double acc_traverse_ms = 0.0;
            static double acc_build_ms = 0.0;
            static double acc_upload_ms = 0.0;
            static double acc_residency_ms = 0.0;
            static double acc_cmdrec_ms = 0.0;
            static double acc_submit_ms = 0.0;
            static uint32_t cpu_prof_samples = 0;
            if (!frame_cache_hit) {
                acc_traverse_ms +=
                    std::chrono::duration<double, std::milli>(t_traverse_end - t_traverse_start)
                        .count();
            }

            // Residency keeps pages for BOTH selections alive: append the
            // shadow selection's page lists to the main selection's (page
            // indices may repeat; both consumers are idempotent per page).
            TraversalSelection residency_selection_storage;
            const TraversalSelection* residency_selection = &selection_for_frame;
            if (shadow_selection != &selection_for_frame) {
                residency_selection_storage = selection_for_frame;
                residency_selection_storage.selected_page_indices.insert(
                    residency_selection_storage.selected_page_indices.end(),
                    shadow_selection->selected_page_indices.begin(),
                    shadow_selection->selected_page_indices.end());
                residency_selection_storage.missing_page_indices.insert(
                    residency_selection_storage.missing_page_indices.end(),
                    shadow_selection->missing_page_indices.begin(),
                    shadow_selection->missing_page_indices.end());
                residency_selection_storage.prefetch_page_indices.insert(
                    residency_selection_storage.prefetch_page_indices.end(),
                    shadow_selection->prefetch_page_indices.begin(),
                    shadow_selection->prefetch_page_indices.end());
                residency_selection = &residency_selection_storage;
            }

            ResidencyUpdateInput residency_input;
            residency_input.frame_index = frame_index;
            residency_input.resident_budget = config.resident_budget;
            residency_input.eviction_grace_frames = config.eviction_grace_frames;
            residency_input.selected_pages = residency_selection->selected_page_indices;
            if (config.demand_streaming) {
                // Run the scheduler against the current selection to get a
                // throttled load queue (cap max_loads_per_frame) and evict
                // queue (oldest zero-priority pages when over budget). The
                // scheduler's own ResidencyUpdateInput return value sets the
                // frame/budget fields; override missing/prefetch with the
                // throttled queue so step_residency requests only those.
                ResidencyUpdateInput sched_in = update_streaming_scheduler(
                    streaming_scheduler, residency_model, *residency_selection, frame_index);
                residency_input.missing_pages = streaming_scheduler.load_queue;
                residency_input.prefetch_pages.clear(); // load_queue already covers prefetch priority
                residency_input.completed_pages = std::move(completed_this_frame);
                // Explicit eviction: step_residency does not take an evict
                // list, so transition the scheduler-selected pages directly.
                // Evicted page ranges are dropped from the mmap page cache
                // (MADV_DONTNEED) so streaming stops holding memory for
                // geometry the GPU no longer has.
                for (uint32_t p : streaming_scheduler.evict_queue) {
                    if (p < residency_model.pages.size()) {
                        residency_model.pages[p].state = PageResidencyState::unloaded;
                        if (async_io_active && p < page_file_ranges.size() &&
                            page_file_ranges[p].size > 0) {
                            async_reader.discard_range(page_file_ranges[p].offset,
                                                       page_file_ranges[p].size);
                        }
                    }
                }
                (void)sched_in; // we use the scheduler state directly.
            } else {
                residency_input.missing_pages = residency_selection->missing_page_indices;
                residency_input.prefetch_pages = residency_selection->prefetch_page_indices;
            }
            auto t_residency_start = clock_t::now();
            const ResidencyUpdateResult residency_update =
                step_residency(residency_model, residency_input);
            if (config.demand_streaming) {
                // Any page that step_residency advanced to `loading` this
                // frame needs its load-start timestamp recorded (for the
                // simulation fallback) AND its real read submitted to the
                // worker thread (for the async I/O path). The scheduler's
                // throttle guarantees we won't spam the worker.
                for (uint32_t p : residency_update.loading_pages) {
                    if (p < page_load_start_frame.size()) {
                        page_load_start_frame[p] = frame_index;
                    }
                    if (async_io_active && p < page_file_ranges.size() &&
                        page_file_ranges[p].size > 0) {
                        AsyncReadJob job{};
                        job.offset = page_file_ranges[p].offset;
                        job.size = page_file_ranges[p].size;
                        job.page_index = p;
                        async_reader.submit(job);
                    }
                }
                if (async_io_active && !completed_reads.empty()) {
                    // step_residency has transitioned the completed pages to
                    // resident; push their bytes into the payload buffers via
                    // per-page sub-buffer uploads. Newly resident pages only
                    // enter the draw list next frame, so the GPU consumes
                    // the bytes after they land.
                    for (const AsyncReadCompletion& r : completed_reads) {
                        const PageRecord& page = resource.pages[r.page_index];
                        UploadedBuffer& dst = (page.lod_cluster_count != 0)
                                                  ? scene_buffers.lod_payload
                                                  : scene_buffers.base_payload;
                        result = upload_page_bytes(
                            selection.physical_device, device, graphics_queue,
                            selection.queues.graphics_family, r.data.data(),
                            static_cast<VkDeviceSize>(r.data.size()),
                            static_cast<VkDeviceSize>(page.byte_offset), dst);
                        if (result != VK_SUCCESS) {
                            std::ostringstream message;
                            message << "streamed page upload failed with code " << result;
                            report.status = message.str();
                            cleanup();
                            return report;
                        }
                        stream_pages_uploaded += 1;
                        stream_bytes_uploaded += r.data.size();
                    }
                }
                if (async_io_active && (frame_index % 60) == 0) {
                    std::fprintf(stderr,
                                  "MERIDIAN_STREAM: uploads=%u pages / %llu bytes, "
                                  "resident=%u/%u, pending_reads=%zu, failed_reads=%u\n",
                                  stream_pages_uploaded,
                                  static_cast<unsigned long long>(stream_bytes_uploaded),
                                  count_resident_pages(residency_model),
                                  static_cast<uint32_t>(residency_model.pages.size()),
                                  async_reader.pending_count(), stream_page_failures);
                }
            }

            snapshot_page_residency(report.uploadable_scene, residency_model);
            result = update_vector_buffer(device, report.uploadable_scene.page_residency,
                                          scene_buffers.page_residency);
            auto t_residency_end = clock_t::now();
            acc_residency_ms +=
                std::chrono::duration<double, std::milli>(t_residency_end - t_residency_start).count();
            if (result != VK_SUCCESS) {
                std::ostringstream message;
                message << "page residency update failed with code " << result;
                report.status = message.str();
                cleanup();
                return report;
            }

            // The selection report walks every selected cluster (serial);
            // its values only feed the final report, and a cache hit means
            // the selection is byte-identical to the last computed one, so
            // the stored counts are already current.
            if (!frame_cache_hit) {
                update_debug_selection_report(selection_for_frame, report.uploadable_scene, report);
            }
            report.runtime_missing_page_count =
                static_cast<uint32_t>(residency_selection->missing_page_indices.size());
            report.runtime_prefetch_page_count =
                static_cast<uint32_t>(residency_selection->prefetch_page_indices.size());
            report.runtime_requested_page_count =
                static_cast<uint32_t>(residency_update.requested_pages.size());
            report.runtime_loading_page_count =
                static_cast<uint32_t>(residency_update.loading_pages.size());
            report.runtime_resident_page_count = count_resident_pages(residency_model);
            report.runtime_failed_page_count = stream_page_failures;

            if (!frame_cache_hit) {
                report.replay_runtime_parity =
                    report.debug_selected_node_count == report.replay_selected_node_count &&
                    report.debug_rendered_cluster_count == report.replay_selected_cluster_count &&
                    report.debug_rendered_lod_cluster_count ==
                        report.replay_selected_lod_cluster_count;
            }

            auto t_fence_start = clock_t::now();
            vkWaitForFences(device, 1, &frame.in_flight, VK_TRUE, UINT64_MAX);
            auto t_fence_end = clock_t::now();
            static double acc_fence_ms = 0.0;
            static double acc_present_ms = 0.0;
            acc_fence_ms +=
                std::chrono::duration<double, std::milli>(t_fence_end - t_fence_start).count();
            if (report.presented_frame_count > 0) {
                // The full visibility analysis runs once after the present
                // loop; repeating it per frame cost several ms of CPU
                // (921K-pixel scan + set inserts) without feeding anything
                // frame-local.
                // Read back GPU draw count for debug stats (draws are consumed on GPU via indirect)
                const UploadedBuffer& readback_count_buf = compute_selection.draw_count;
                if (readback_count_buf.buffer != VK_NULL_HANDLE) {
                    void* count_mapped = nullptr;
                    if (vkMapMemory(device, readback_count_buf.memory, 0,
                                    sizeof(uint32_t), 0, &count_mapped) == VK_SUCCESS) {
                        gpu_draw_count = *static_cast<const uint32_t*>(count_mapped);
                        vkUnmapMemory(device, readback_count_buf.memory);
                    }
                }

                // GPU profiler readback: print timing every 60 frames in interactive mode,
                // or every frame in non-interactive mode
                if (gpu_profiler.query_pool != VK_NULL_HANDLE &&
                    (!config.interactive || (frame_index % 60) == 0)) {
                    auto timers = read_gpu_timers(device, gpu_profiler);
                    if (!timers.empty()) {
                        std::fprintf(stderr, "MERIDIAN_GPU:");
                        for (const auto& t : timers) {
                            std::fprintf(stderr, " %s=%.2fms", t.name.c_str(), t.ms);
                        }
                        std::fprintf(stderr, "\n");
                    }
                }
            }
            // The in-flight fence is reset only after a successful acquire,
            // immediately before the owning vkQueueSubmit (the LS-64
            // pattern, mirroring ontos_view): an OUT_OF_DATE/SUBOPTIMAL
            // acquire continues to the next frame without submitting
            // anything that would signal a reset fence, and the fence is
            // created signaled so every no-submit path leaves it signaled
            // for the next vkWaitForFences.
            vkResetCommandPool(device, frame.command_pool, 0);

            uint32_t image_index = 0;
            result = vkAcquireNextImageKHR(device, swapchain.swapchain, UINT64_MAX,
                                           frame.image_available, VK_NULL_HANDLE, &image_index);
            if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
                if (!recreate_surface()) {
                    report.status = "swapchain recreation failed after out-of-date acquire";
                    cleanup();
                    return report;
                }
                continue;
            }
            if (result != VK_SUCCESS) {
                std::ostringstream message;
                message << "vkAcquireNextImageKHR failed with code " << result;
                report.status = message.str();
                cleanup();
                return report;
            }

            // Compute cascaded shadow light view-projection matrices for the
            // current camera. Using the same camera projection parameters
            // (fov, aspect, near, far) that feed view_projection above.
            const Vec3f norm_light = normalize_vec3({0.4f, 0.7f, 0.5f});
            {
                const float aspect = std::max(1.0f,
                    static_cast<float>(swapchain.extent.width)) /
                    std::max(1.0f, static_cast<float>(swapchain.extent.height));
                const float fov = 55.0f * 3.14159265f / 180.0f;
                const float radius = std::max({
                    resource.bounds.max.x - resource.bounds.min.x,
                    resource.bounds.max.y - resource.bounds.min.y,
                    resource.bounds.max.z - resource.bounds.min.z, 1.0f});
                const float near_p = std::max(0.01f, radius * 0.01f);
                // Clamp the far plane for CSM so the cascades pack usefully
                // around the camera rather than stretching to the full
                // 8x-radius projection far, which would drown cascade 2 in
                // empty space for indoor scenes.
                const float far_p = radius * 3.0f;
                const Vec3f fwd = config.interactive
                                      ? camera_forward(interactive_cam)
                                      : normalize_vec3(subtract_vec3(
                                            {
                                                (resource.bounds.min.x + resource.bounds.max.x) * 0.5f,
                                                (resource.bounds.min.y + resource.bounds.max.y) * 0.5f,
                                                (resource.bounds.min.z + resource.bounds.max.z) * 0.5f,
                                            },
                                            camera_frame.camera_position));
                // World-space right/up derived from forward + world up.
                const Vec3f world_up = {0.0f, 1.0f, 0.0f};
                const Vec3f rgt = normalize_vec3(cross_vec3(fwd, world_up));
                const Vec3f up_corr = cross_vec3(rgt, fwd);
                shadow.cascades = compute_cascade_light_setup(
                    camera_frame.camera_position, fwd, rgt, up_corr,
                    fov, aspect, near_p, far_p, norm_light,
                    /*caster_extent=*/radius * 1.5f,
                    /*lambda=*/0.7f);
            }

            // Upload per-frame UBO (camera VP + 3 cascade light VPs + splits).
            FrameUBO frame_ubo_data{};
            std::memcpy(frame_ubo_data.view_projection, camera_frame.view_projection.m,
                        sizeof(frame_ubo_data.view_projection));
            for (uint32_t c = 0; c < kShadowCascadeCount; ++c) {
                std::memcpy(frame_ubo_data.light_vp[c], shadow.cascades.light_vp[c].m,
                            sizeof(frame_ubo_data.light_vp[c]));
                frame_ubo_data.cascade_splits[c] = shadow.cascades.splits[c];
            }
            frame_ubo_data.cascade_splits[3] = 0.0f;
            frame_ubo_data.light_dir[0] = norm_light.x;
            frame_ubo_data.light_dir[1] = norm_light.y;
            frame_ubo_data.light_dir[2] = norm_light.z;
            frame_ubo_data.light_dir[3] = 0.0f;
            frame_ubo_data.tonemap_params[0] = config.exposure;
            frame_ubo_data.tonemap_params[1] = config.tonemap ? 1.0f : 0.0f;
            frame_ubo_data.tonemap_params[2] = 0.0f;
            frame_ubo_data.tonemap_params[3] = 0.0f;
            update_uploaded_buffer(device, &frame_ubo_data, sizeof(FrameUBO),
                                   debug_render.frame_ubo);

            // Convert CPU TraversalSelection to GpuDrawEntry list and upload to the
            // same buffers the GPU selection shader used to populate. This replaces
            // the serial DFS compute dispatch (was ~18ms on 1M-tri city on M4) with
            // CPU traversal + HOST_COHERENT write (~1-3ms total).
            //
            // The main pass draws the main selection; the shadow pass draws the
            // (possibly coarser) shadow caster selection.
            //
             // Filters applied to the main list:
             //   1. Frustum AABB test (base + LOD) -- mirrors instance_cull but at
             //      cluster granularity. Assumes cluster bounds are world-space
             //      (single-instance / identity transform scenes).
             // No normal-cone backface cull on the main list: the main pass
             // rasterizes two-sided (VK_CULL_MODE_NONE), so cone-backfacing
             // clusters remain visible and must not be CPU-dropped.
             // The shadow list applies only the per-cascade ortho-frustum overlap
             // test: camera-facing culls don't apply to casters (a cluster facing
             // away from the camera can still cast a shadow into the camera's view).
            const FrustumPlanes frustum =
                extract_frustum_planes(camera_frame.view_projection);
            auto aabb_outside_frustum = [&](const float bmin[4], const float bmax[4]) -> bool {
                for (int p = 0; p < 6; ++p) {
                    const float nx = frustum.planes[p][0];
                    const float ny = frustum.planes[p][1];
                    const float nz = frustum.planes[p][2];
                    const float d  = frustum.planes[p][3];
                    const float px = nx > 0.0f ? bmax[0] : bmin[0];
                    const float py = ny > 0.0f ? bmax[1] : bmin[1];
                    const float pz = nz > 0.0f ? bmax[2] : bmin[2];
                    if (nx * px + ny * py + nz * pz + d < 0.0f) return true;
                }
                return false;
            };
            // Per-cascade frustum planes, extracted from each cascade's
            // light view-projection. Used below to filter the caster list so
            // the shadow pass only draws clusters that actually overlap the
            // cascade's volume.
            FrustumPlanes cascade_frusta[kShadowCascadeCount];
            for (uint32_t c = 0; c < kShadowCascadeCount; ++c) {
                cascade_frusta[c] = extract_frustum_planes(shadow.cascades.light_vp[c]);
            }
            auto aabb_outside_planes = [](const FrustumPlanes& fp,
                                           const float bmin[4],
                                           const float bmax[4]) -> bool {
                for (int p = 0; p < 6; ++p) {
                    const float nx = fp.planes[p][0];
                    const float ny = fp.planes[p][1];
                    const float nz = fp.planes[p][2];
                    const float d  = fp.planes[p][3];
                    const float px = nx > 0.0f ? bmax[0] : bmin[0];
                    const float py = ny > 0.0f ? bmax[1] : bmin[1];
                    const float pz = nz > 0.0f ? bmax[2] : bmin[2];
                    if (nx * px + ny * py + nz * pz + d < 0.0f) return true;
                }
                return false;
            };
            uint32_t cpu_draw_count = 0;
            uint32_t shadow_draw_count = 0;
            uint32_t main_encode_count = 0;
            uint32_t shadow_encode_count = 0;
            uint64_t main_wasted_vs = 0;
            uint64_t shadow_wasted_vs = 0;
            if (frame_cache_hit) {
                // Cached frame: the draw lists, bucket folds, and counts are
                // byte-identical to what is already in the HOST_COHERENT
                // draw-list buffers (they are host-written only), so the
                // build and the re-upload of identical bytes are skipped.
                cpu_draw_count = frame_cache.main_draw_count;
                shadow_draw_count = frame_cache.shadow_draw_count;
                main_encode_count = frame_cache.main_encode_count;
                shadow_encode_count = frame_cache.shadow_encode_count;
                main_wasted_vs = frame_cache.main_wasted_vs;
                shadow_wasted_vs = frame_cache.shadow_wasted_vs;
            } else {
            std::vector<DrawBucket> main_buckets;
            std::vector<DrawBucket> shadow_buckets;
            {
                auto t_build_start = clock_t::now();
                std::vector<GpuDrawEntry> cpu_draws;
                std::vector<GpuDrawEntry> shadow_draws;
                cpu_draws.reserve(selection_for_frame.selected_cluster_indices.size() +
                                  selection_for_frame.selected_lod_cluster_indices.size());
                 shadow_draws.reserve(shadow_selection->selected_cluster_indices.size() +
                                      shadow_selection->selected_lod_cluster_indices.size());
                // Chunked selection -> GpuDrawEntry conversion. Each chunk
                // job appends to its own output with chunk-local
                // first_instance; the ordered concat afterwards fixes
                // first_instance up with the global entry offsets, so the
                // assembled lists are byte-identical to the serial
                // single-vector path at any thread count.
                const size_t kBuildChunkIndices = 1536;
                std::vector<std::vector<GpuDrawEntry>> main_base_chunks;
                std::vector<std::vector<GpuDrawEntry>> main_lod_chunks;
                std::vector<std::vector<GpuDrawEntry>> shadow_base_chunks;
                std::vector<std::vector<GpuDrawEntry>> shadow_lod_chunks;
                std::vector<std::function<void()>> build_jobs;
                auto schedule_main_chunks = [&](const std::vector<uint32_t>& selection_list,
                                                bool is_lod_domain,
                                                std::vector<std::vector<GpuDrawEntry>>&
                                                    chunk_outputs) {
                    const size_t chunk_count = std::max<size_t>(
                        1, (selection_list.size() + kBuildChunkIndices - 1) / kBuildChunkIndices);
                    chunk_outputs.resize(chunk_count);
                    for (size_t chunk = 0; chunk < chunk_count; ++chunk) {
                        const size_t begin = chunk * kBuildChunkIndices;
                        const size_t end =
                            std::min(selection_list.size(), begin + kBuildChunkIndices);
                        build_jobs.emplace_back([&, is_lod_domain, begin, end, chunk]() {
                            std::vector<GpuDrawEntry>& out = chunk_outputs[chunk];
                            out.reserve(end - begin);
                            for (size_t i = begin; i < end; ++i) {
                                const uint32_t ci = selection_list[i];
                                // Main-pass entry has camera-frustum culls.
                                // No normal-cone backface cull here: the main
                                // pass rasterizes two-sided
                                // (VK_CULL_MODE_NONE, see vk_render.cpp), so a
                                // cone-backfacing cluster is still visible
                                // (e.g. a plane viewed from behind) and must
                                // not be CPU-dropped. This costs some extra
                                // draw entries on silhouette clusters.
                                if (is_lod_domain) {
                                    const GpuLodClusterRecord& c =
                                        report.uploadable_scene.lod_clusters[ci];
                                    if (aabb_outside_frustum(c.bounds_min.data(),
                                                             c.bounds_max.data())) {
                                        continue;
                                    }
                                    GpuDrawEntry e{};
                                    e.draw_vertex_count = c.local_triangle_count * 3u;
                                    e.draw_instance_count = 1u;
                                    e.draw_first_vertex = 0u;
                                    e.cluster_index = ci;
                                    e.geometry_kind =
                                        1u | ((c.flags & kClusterFlagHasUv) != 0
                                                  ? kGeometryKindHasUv
                                                  : 0u);
                                    e.payload_offset = c.payload_offset;
                                    e.local_vertex_count = c.local_vertex_count;
                                    e.draw_first_instance = static_cast<uint32_t>(out.size());
                                    out.push_back(e);
                                } else {
                                    const GpuClusterRecord& c =
                                        report.uploadable_scene.clusters[ci];
                                    if (aabb_outside_frustum(c.bounds_min.data(),
                                                             c.bounds_max.data())) {
                                        continue;
                                    }
                                    GpuDrawEntry e{};
                                    e.draw_vertex_count = c.local_triangle_count * 3u;
                                    e.draw_instance_count = 1u;
                                    e.draw_first_vertex = 0u;
                                    e.cluster_index = ci;
                                    e.geometry_kind =
                                        0u | ((c.flags & kClusterFlagHasUv) != 0
                                                  ? kGeometryKindHasUv
                                                  : 0u);
                                    e.payload_offset = c.payload_offset;
                                    e.local_vertex_count = c.local_vertex_count;
                                    e.draw_first_instance = static_cast<uint32_t>(out.size());
                                    out.push_back(e);
                                }
                            }
                        });
                    }
                };
                // Caster list: clusters from the shadow selection, tested only
                // against the cascade volumes. Backface culling against the
                // camera doesn't apply to shadow casters. Each surviving entry
                // carries its cascade overlap mask (geometry_kind bits) and
                // draws one instance per overlapping cascade; instance slots
                // are strided by kShadowInstanceStride so shadow.vert can
                // recover the entry index from gl_InstanceIndex.
                auto schedule_shadow_chunks = [&](const std::vector<uint32_t>& selection_list,
                                                  bool is_lod_domain,
                                                  std::vector<std::vector<GpuDrawEntry>>&
                                                      chunk_outputs) {
                    const size_t chunk_count = std::max<size_t>(
                        1, (selection_list.size() + kBuildChunkIndices - 1) / kBuildChunkIndices);
                    chunk_outputs.resize(chunk_count);
                    for (size_t chunk = 0; chunk < chunk_count; ++chunk) {
                        const size_t begin = chunk * kBuildChunkIndices;
                        const size_t end =
                            std::min(selection_list.size(), begin + kBuildChunkIndices);
                        build_jobs.emplace_back([&, is_lod_domain, begin, end, chunk]() {
                            std::vector<GpuDrawEntry>& out = chunk_outputs[chunk];
                            out.reserve(end - begin);
                            for (size_t i = begin; i < end; ++i) {
                                const uint32_t ci = selection_list[i];
                                GpuDrawEntry cascade_entry{};
                                if (is_lod_domain) {
                                    const GpuLodClusterRecord& c =
                                        report.uploadable_scene.lod_clusters[ci];
                                    cascade_entry.draw_vertex_count =
                                        c.local_triangle_count * 3u;
                                    cascade_entry.draw_instance_count = 1u;
                                    cascade_entry.draw_first_vertex = 0u;
                                    cascade_entry.cluster_index = ci;
                                    cascade_entry.geometry_kind =
                                        1u | ((c.flags & kClusterFlagHasUv) != 0
                                                  ? kGeometryKindHasUv
                                                  : 0u);
                                    cascade_entry.payload_offset = c.payload_offset;
                                    cascade_entry.local_vertex_count = c.local_vertex_count;
                                } else {
                                    const GpuClusterRecord& c =
                                        report.uploadable_scene.clusters[ci];
                                    cascade_entry.draw_vertex_count =
                                        c.local_triangle_count * 3u;
                                    cascade_entry.draw_instance_count = 1u;
                                    cascade_entry.draw_first_vertex = 0u;
                                    cascade_entry.cluster_index = ci;
                                    cascade_entry.geometry_kind =
                                        0u | ((c.flags & kClusterFlagHasUv) != 0
                                                  ? kGeometryKindHasUv
                                                  : 0u);
                                    cascade_entry.payload_offset = c.payload_offset;
                                    cascade_entry.local_vertex_count = c.local_vertex_count;
                                }
                                const float* bmin;
                                const float* bmax;
                                if (is_lod_domain) {
                                    const GpuLodClusterRecord& c =
                                        report.uploadable_scene.lod_clusters[ci];
                                    bmin = c.bounds_min.data();
                                    bmax = c.bounds_max.data();
                                } else {
                                    const GpuClusterRecord& c =
                                        report.uploadable_scene.clusters[ci];
                                    bmin = c.bounds_min.data();
                                    bmax = c.bounds_max.data();
                                }
                                uint32_t mask = 0;
                                for (uint32_t c = 0; c < kShadowCascadeCount; ++c) {
                                    if (aabb_outside_planes(cascade_frusta[c], bmin, bmax)) {
                                        continue;
                                    }
                                    mask |= 1u << c;
                                }
                                if (mask == 0) continue;
                                cascade_entry.geometry_kind |= mask << kGeometryKindCascadeMaskShift;
                                cascade_entry.draw_instance_count =
                                    (mask & 1u) + ((mask >> 1) & 1u) + ((mask >> 2) & 1u);
                                cascade_entry.draw_first_instance =
                                    static_cast<uint32_t>(out.size()) * kShadowInstanceStride;
                                out.push_back(cascade_entry);
                            }
                        });
                    }
                };
                schedule_main_chunks(selection_for_frame.selected_cluster_indices, false,
                                     main_base_chunks);
                schedule_main_chunks(selection_for_frame.selected_lod_cluster_indices, true,
                                     main_lod_chunks);
                schedule_shadow_chunks(shadow_selection->selected_cluster_indices, false,
                                       shadow_base_chunks);
                schedule_shadow_chunks(shadow_selection->selected_lod_cluster_indices, true,
                                       shadow_lod_chunks);
                worker_pool.run(build_jobs);
                uint32_t main_entry_offset = 0;
                for (const auto* chunk_list : {&main_base_chunks, &main_lod_chunks}) {
                    for (const std::vector<GpuDrawEntry>& chunk : *chunk_list) {
                        for (GpuDrawEntry e : chunk) {
                            e.draw_first_instance += main_entry_offset;
                            cpu_draws.push_back(e);
                        }
                        main_entry_offset += static_cast<uint32_t>(chunk.size());
                    }
                }
                uint32_t shadow_entry_offset = 0;
                for (const auto* chunk_list : {&shadow_base_chunks, &shadow_lod_chunks}) {
                    for (const std::vector<GpuDrawEntry>& chunk : *chunk_list) {
                        for (GpuDrawEntry e : chunk) {
                            e.draw_first_instance += shadow_entry_offset * kShadowInstanceStride;
                            shadow_draws.push_back(e);
                        }
                        shadow_entry_offset += static_cast<uint32_t>(chunk.size());
                    }
                }
                cpu_draw_count = static_cast<uint32_t>(cpu_draws.size());
                shadow_draw_count = static_cast<uint32_t>(shadow_draws.size());
                // Instance-fold bucketing: one instanced draw per contiguous
                // equal-vertex-count run for the main list (stride 1) and the
                // shadow list (stride kShadowInstanceStride so the layered
                // vertex shader recovers the entry index). The fold preserves
                // the global draw order -- with VK_COMPARE_OP_LESS,
                // coplanar equal-depth overlaps resolve first-writer-wins, so
                // reordering entries could change which path's pixels win;
                // the shaders resolve entries through gl_InstanceIndex, so no
                // other consumer cares about the run boundaries.
                main_encode_count =
                    fold_draws_into_buckets(cpu_draws, 1, main_buckets, &main_wasted_vs);
                shadow_encode_count = fold_draws_into_buckets(shadow_draws, kShadowInstanceStride,
                                                               shadow_buckets, &shadow_wasted_vs);
                auto t_build_end = clock_t::now();
                acc_build_ms +=
                    std::chrono::duration<double, std::milli>(t_build_end - t_build_start).count();
                auto t_upload_start = clock_t::now();
                if (compute_selection.draw_list.buffer != VK_NULL_HANDLE) {
                    if (cpu_draw_count > draw_list_high_water) {
                        draw_list_high_water = cpu_draw_count;
                    }
                    draw_upload_scratch.assign(draw_list_high_water, GpuDrawEntry{});
                    std::copy(cpu_draws.begin(), cpu_draws.end(), draw_upload_scratch.begin());
                    update_uploaded_buffer(device, draw_upload_scratch.data(),
                                           static_cast<VkDeviceSize>(draw_upload_scratch.size()) *
                                               sizeof(GpuDrawEntry),
                                           compute_selection.draw_list);
                }
                // Upload the merged shadow draw list + count into their
                // HOST_COHERENT buffers; the layered shadow pass reads them.
                if (shadow.draw_list.buffer != VK_NULL_HANDLE) {
                    if (shadow_draw_count > shadow_list_high_water) {
                        shadow_list_high_water = shadow_draw_count;
                    }
                    shadow_upload_scratch.assign(shadow_list_high_water, GpuDrawEntry{});
                    std::copy(shadow_draws.begin(), shadow_draws.end(),
                              shadow_upload_scratch.begin());
                    update_uploaded_buffer(device, shadow_upload_scratch.data(),
                                           static_cast<VkDeviceSize>(shadow_upload_scratch.size()) *
                                               sizeof(GpuDrawEntry),
                                           shadow.draw_list);
                }
                if (shadow.draw_count.buffer != VK_NULL_HANDLE) {
                    update_uploaded_buffer(device, &shadow_draw_count, sizeof(uint32_t),
                                           shadow.draw_count);
                }
                if (compute_selection.draw_count.buffer != VK_NULL_HANDLE) {
                    update_uploaded_buffer(device, &cpu_draw_count, sizeof(uint32_t),
                                           compute_selection.draw_count);
                }
                auto t_upload_end = clock_t::now();
                acc_upload_ms +=
                    std::chrono::duration<double, std::milli>(t_upload_end - t_upload_start).count();
                frame_cache.main_draws = std::move(cpu_draws);
                frame_cache.shadow_draws = std::move(shadow_draws);
                frame_cache.main_buckets = std::move(main_buckets);
                frame_cache.shadow_buckets = std::move(shadow_buckets);
                frame_cache.main_draw_count = cpu_draw_count;
                frame_cache.shadow_draw_count = shadow_draw_count;
                frame_cache.main_encode_count = main_encode_count;
                frame_cache.shadow_encode_count = shadow_encode_count;
                frame_cache.main_wasted_vs = main_wasted_vs;
                frame_cache.shadow_wasted_vs = shadow_wasted_vs;
            }
            }
            const std::vector<DrawBucket>& main_buckets = frame_cache.main_buckets;
            const std::vector<DrawBucket>& shadow_buckets = frame_cache.shadow_buckets;

            // frustum was already extracted above for cluster-level CPU culling
            auto t_cmdrec_start = clock_t::now();
            // Capture frame for the visibility readback: the final index of
            // a fixed-count run, or every frame while interactive (the loop
            // breaks on close before rendering the observing frame, so no
            // last-frame signal exists there).
            const bool capture_visibility =
                config.interactive || (frame_index + 1 == config.present_frame_count);
            const bool temporal_hzb_valid =
                frame_index >= 2 && temporal_hzb_has_history &&
                scene_generation == temporal_hzb_last_generation &&
                std::memcmp(camera_frame.view_projection.m,
                            temporal_hzb_last_view_projection.m,
                            sizeof(temporal_hzb_last_view_projection.m)) == 0;
            result = record_debug_command_buffer(frame, debug_render, compute_cull, compute_selection,
                                                 hzb, occlusion_refine, shadow, swapchain,
                                                 camera_frame, frustum, error_threshold,
                                                 selection_for_frame, report.uploadable_scene,
                                                 frame_index, image_index,
                                                 has_draw_indirect_count, shadow_draw_count,
                                                 shadow_buckets, main_buckets, gpu_profiler,
                                                 capture_visibility, temporal_hzb_valid);
            auto t_cmdrec_end = clock_t::now();
            acc_cmdrec_ms +=
                std::chrono::duration<double, std::milli>(t_cmdrec_end - t_cmdrec_start).count();
            if (result != VK_SUCCESS) {
                std::ostringstream message;
                message << "record_debug_command_buffer failed with code " << result;
                report.status = message.str();
                cleanup();
                return report;
            }

            const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            VkSubmitInfo submit_info{};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.waitSemaphoreCount = 1;
            submit_info.pWaitSemaphores = &frame.image_available;
            submit_info.pWaitDstStageMask = &wait_stage;
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &frame.command_buffer;
            VkSemaphore signal_semaphore =
                image_index < frame.render_finished_per_image.size()
                    ? frame.render_finished_per_image[image_index]
                    : frame.render_finished;
            submit_info.signalSemaphoreCount = 1;
            submit_info.pSignalSemaphores = &signal_semaphore;

            auto t_submit_start = clock_t::now();
            vkResetFences(device, 1, &frame.in_flight);
            result = vkQueueSubmit(graphics_queue, 1, &submit_info, frame.in_flight);
            auto t_submit_end = clock_t::now();
            acc_submit_ms +=
                std::chrono::duration<double, std::milli>(t_submit_end - t_submit_start).count();
            cpu_prof_samples++;
            if (cpu_prof_samples % 60 == 0) {
                std::fprintf(stderr,
                    "MERIDIAN_CPU: threads=%u traverse=%.2f residency=%.2f build=%.2f upload=%.2f cmdrec=%.2f submit=%.2f fence=%.2f present=%.2f draws=main:%u shadow:%u encodes main:%u shadow:%u wastedvs main:%llu shadow:%llu cache=%u/%u (ms/frame, n=%u)\n",
                    resolved_worker_threads,
                    acc_traverse_ms / cpu_prof_samples,
                    acc_residency_ms / cpu_prof_samples,
                    acc_build_ms / cpu_prof_samples,
                    acc_upload_ms / cpu_prof_samples,
                    acc_cmdrec_ms / cpu_prof_samples,
                    acc_submit_ms / cpu_prof_samples,
                    acc_fence_ms / cpu_prof_samples,
                    acc_present_ms / cpu_prof_samples,
                    cpu_draw_count,
                    shadow_draw_count,
                    main_encode_count,
                    shadow_encode_count,
                    static_cast<unsigned long long>(main_wasted_vs),
                    static_cast<unsigned long long>(shadow_wasted_vs),
                    frame_cache_hits,
                    cpu_prof_samples,
                    cpu_prof_samples);
            }
            if (result != VK_SUCCESS) {
                std::ostringstream message;
                message << "vkQueueSubmit failed with code " << result;
                report.status = message.str();
                cleanup();
                return report;
            }

            // This frame's HZB build (recorded above) is what the next
            // frame's refine pass will bind; remember the pairing inputs.
            temporal_hzb_last_view_projection = camera_frame.view_projection;
            temporal_hzb_last_generation = scene_generation;
            temporal_hzb_has_history = true;

            VkSemaphore present_wait_semaphore =
                image_index < frame.render_finished_per_image.size()
                    ? frame.render_finished_per_image[image_index]
                    : frame.render_finished;
            VkPresentInfoKHR present_info{};
            present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            present_info.waitSemaphoreCount = 1;
            present_info.pWaitSemaphores = &present_wait_semaphore;
            present_info.swapchainCount = 1;
            present_info.pSwapchains = &swapchain.swapchain;
            present_info.pImageIndices = &image_index;

            auto t_present_start = clock_t::now();
            result = vkQueuePresentKHR(present_queue, &present_info);
            auto t_present_end = clock_t::now();
            acc_present_ms +=
                std::chrono::duration<double, std::milli>(t_present_end - t_present_start).count();
            if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
                framebuffer_resized = true;
            } else if (result != VK_SUCCESS) {
                std::ostringstream message;
                message << "vkQueuePresentKHR failed with code " << result;
                report.status = message.str();
                cleanup();
                return report;
            }

            // last_submitted_selection: the frame-stable cache holds the
            // latest presented frame's selection verbatim (cache hits reuse
            // it byte-identically), so the per-frame selection copy is gone;
            // the post-loop epilogue reads frame_cache.main_selection.

            report.presented_frame_count += 1;
            report.debug_draw_submitted = true;
        }

        vkDeviceWaitIdle(device);
        // Diagnostic epilogue: refreshes the cull counter, the fallback-path
        // occlusion survivor count, and the visibility readback in one
        // post-loop submit so none of them cost per-frame submit time.
        // Runs after the device wait (the pool reset inside needs the last
        // frame's command buffer retired) and blocks once on its own queue.
        if (report.presented_frame_count > 0) {
            const FrustumPlanes epilogue_frustum =
                extract_frustum_planes(camera_frame.view_projection);
            result = submit_diagnostic_epilogue(device, graphics_queue, frame, debug_render,
                                                compute_cull, compute_selection, hzb,
                                                occlusion_refine, camera_frame,
                                                epilogue_frustum, swapchain,
                                                has_draw_indirect_count);
            if (result != VK_SUCCESS) {
                std::ostringstream message;
                message << "diagnostic epilogue submit failed with code " << result;
                report.status = message.str();
                cleanup();
                return report;
            }
        }
        // Tear down the async reader and remove the temp .vgeo before the
        // visibility readback / cleanup path runs. Closing here instead of
        // in `cleanup` keeps the reader local to the streaming block where
        // it lives.
        if (config.demand_streaming && async_io_active) {
            std::fprintf(stderr,
                         "MERIDIAN_STREAM: total page uploads=%u (%llu bytes) over %u "
                         "presented frames; payload buffers were allocated empty at startup\n",
                         stream_pages_uploaded,
                         static_cast<unsigned long long>(stream_bytes_uploaded),
                         report.presented_frame_count);
        }
        if (async_io_active) {
            async_reader.close();
            async_io_active = false;
            std::error_code ec;
            std::filesystem::remove(temp_vgeo_path, ec);
        }
        if (report.presented_frame_count > 0) {
            analyze_visibility_readback(device, swapchain, debug_render,
                                        frame_cache.main_selection, report);
        }
        report.present_loop_completed = report.presented_frame_count == config.present_frame_count;

        // Frame timing report
        if (!frame_times_ms.empty()) {
            std::sort(frame_times_ms.begin(), frame_times_ms.end());
            const double median = frame_times_ms[frame_times_ms.size() / 2];
            const double p99 = frame_times_ms[static_cast<size_t>(frame_times_ms.size() * 0.99)];
            const double avg = std::accumulate(frame_times_ms.begin(), frame_times_ms.end(), 0.0)
                               / static_cast<double>(frame_times_ms.size());
            std::fprintf(stderr, "MERIDIAN_BENCHMARK: median_ms=%.2f p99_ms=%.2f avg_ms=%.2f avg_fps=%.1f samples=%zu\n",
                         median, p99, avg, 1000.0 / avg, frame_times_ms.size());
        }
        if (compute_cull.counter.buffer != VK_NULL_HANDLE) {
            void* mapped = nullptr;
            if (vkMapMemory(device, compute_cull.counter.memory, 0, sizeof(uint32_t), 0, &mapped) == VK_SUCCESS) {
                report.compute_cull_visible_instances = *static_cast<const uint32_t*>(mapped);
                vkUnmapMemory(device, compute_cull.counter.memory);
            }
        }
        if (compute_selection.draw_count.buffer != VK_NULL_HANDLE) {
            void* mapped = nullptr;
            if (vkMapMemory(device, compute_selection.draw_count.memory, 0, sizeof(uint32_t), 0, &mapped) == VK_SUCCESS) {
                report.compute_selection_draw_count = *static_cast<const uint32_t*>(mapped);
                vkUnmapMemory(device, compute_selection.draw_count.memory);
            }
        }
        if (occlusion_refine.output_count.buffer != VK_NULL_HANDLE) {
            // [0] = draw range (input count, tombstones included), [1] = survivors.
            uint32_t occ_counts[2] = {0, 0};
            void* mapped = nullptr;
            if (vkMapMemory(device, occlusion_refine.output_count.memory, 0,
                            sizeof(occ_counts), 0, &mapped) == VK_SUCCESS) {
                std::memcpy(occ_counts, mapped, sizeof(occ_counts));
                vkUnmapMemory(device, occlusion_refine.output_count.memory);
            }
            report.compute_occlusion_surviving_draws = occ_counts[1];
        }
        // Screenshot capture (raw PPM; extension is forced to .ppm so the
        // container always matches the bytes). The target image is acquired
        // properly (dedicated semaphore + fence) -- reading a swapchain image
        // without acquiring it trips the validation layer, and post-present
        // images must be re-acquired before use.
        if (!config.screenshot_path.empty() && report.presented_frame_count > 0 &&
            !swapchain.images.empty() && frame.command_pool != VK_NULL_HANDLE) {
            if (!swapchain.images_support_transfer_src) {
                std::fprintf(stderr,
                             "screenshot requested but this surface does not support "
                             "TRANSFER_SRC swapchain usage; capture unavailable\n");
                report.capture_status = "capture failed: surface lacks TRANSFER_SRC usage";
            } else {
            std::filesystem::path screenshot_path(config.screenshot_path);
            if (screenshot_path.extension() != ".ppm") {
                screenshot_path.replace_extension(".ppm");
            }
            const uint32_t w = swapchain.extent.width;
            const uint32_t h = swapchain.extent.height;
            // Byte offsets of R and G inside each 4-byte pixel of the
            // readback, from the swapchain's actual format (B is 3 - R - G).
            // Formats outside the BGRA8/RGBA8 pairs are rejected rather
            // than written with a guessed channel order.
            uint32_t red_offset = 0;
            uint32_t green_offset = 0;
            bool swizzle_available = true;
            switch (swapchain.surface_format.format) {
                case VK_FORMAT_B8G8R8A8_UNORM:
                case VK_FORMAT_B8G8R8A8_SRGB:
                    red_offset = 2;
                    green_offset = 1;
                    break;
                case VK_FORMAT_R8G8B8A8_UNORM:
                case VK_FORMAT_R8G8B8A8_SRGB:
                    red_offset = 0;
                    green_offset = 1;
                    break;
                default:
                    std::fprintf(stderr,
                                 "screenshot requested but surface format %d has no capture "
                                 "swizzle; capture unavailable\n",
                                 static_cast<int>(swapchain.surface_format.format));
                    report.capture_status = "capture failed: surface format has no capture swizzle";
                    swizzle_available = false;
                    break;
            }
            const VkDeviceSize pixel_size = 4;
            const VkDeviceSize buf_size = w * h * pixel_size;

            VkSemaphore acquire_semaphore = VK_NULL_HANDLE;
            VkFence acquire_fence = VK_NULL_HANDLE;
            VkSemaphoreCreateInfo semaphore_info{};
            semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            VkFenceCreateInfo fence_info{};
            fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            uint32_t shot_image_index = 0;
            // An unsupported capture format is an explicit skip: no semaphore,
            // fence, acquire, or copy runs at all, so no null handles can reach
            // vkWaitForFences or the submission below.
            const bool image_acquired =
                swizzle_available &&
                (vkCreateSemaphore(device, &semaphore_info, nullptr, &acquire_semaphore) ==
                     VK_SUCCESS &&
                 vkCreateFence(device, &fence_info, nullptr, &acquire_fence) == VK_SUCCESS &&
                 vkAcquireNextImageKHR(device, swapchain.swapchain, UINT64_MAX, acquire_semaphore,
                                       acquire_fence, &shot_image_index) == VK_SUCCESS);
            if (!image_acquired) {
                if (swizzle_available) {
                    report.capture_status = "capture failed: image acquisition failed";
                }
            } else {
                const VkResult fence_result =
                    vkWaitForFences(device, 1, &acquire_fence, VK_TRUE, UINT64_MAX);
                UploadedBuffer readback{};
                if (fence_result != VK_SUCCESS) {
                    report.capture_status = "capture failed: acquire fence wait failed";
                } else if (create_uploaded_buffer(selection.physical_device, device, nullptr,
                                                  buf_size,
                                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT, readback) !=
                           VK_SUCCESS) {
                    report.capture_status = "capture failed: readback buffer allocation failed";
                } else {
                    // One-shot copy submission with every result checked
                    // (LS-21/40 pattern): on any failure the readback buffer
                    // is still destroyed, the file is not published, and the
                    // report records a non-success capture status.
                    VkResult one_shot = vkResetCommandPool(device, frame.command_pool, 0);
                    if (one_shot == VK_SUCCESS) {
                        VkCommandBufferBeginInfo begin{};
                        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                        one_shot = vkBeginCommandBuffer(frame.command_buffer, &begin);
                    }
                    if (one_shot == VK_SUCCESS) {
                        // Transition swapchain image to transfer src
                        VkImageMemoryBarrier to_src{};
                        to_src.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                        to_src.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
                        to_src.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                        to_src.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
                        to_src.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                        to_src.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                        to_src.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                        to_src.image = swapchain.images[shot_image_index];
                        to_src.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                        vkCmdPipelineBarrier(frame.command_buffer,
                                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                                             nullptr, 1, &to_src);

                        VkBufferImageCopy region{};
                        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                        region.imageExtent = {w, h, 1};
                        vkCmdCopyImageToBuffer(frame.command_buffer,
                                               swapchain.images[shot_image_index],
                                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                               readback.buffer, 1, &region);

                        one_shot = vkEndCommandBuffer(frame.command_buffer);
                    }
                    VkSubmitInfo sub{};
                    sub.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                    const VkPipelineStageFlags acquire_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
                    sub.waitSemaphoreCount = 1;
                    sub.pWaitSemaphores = &acquire_semaphore;
                    sub.pWaitDstStageMask = &acquire_stage;
                    sub.commandBufferCount = 1;
                    sub.pCommandBuffers = &frame.command_buffer;
                    if (one_shot == VK_SUCCESS) {
                        one_shot = vkQueueSubmit(graphics_queue, 1, &sub, VK_NULL_HANDLE);
                    }
                    if (one_shot == VK_SUCCESS) {
                        one_shot = vkQueueWaitIdle(graphics_queue);
                    }
                    if (one_shot != VK_SUCCESS) {
                        std::fprintf(stderr,
                                     "screenshot copy submission failed with code %d\n",
                                     static_cast<int>(one_shot));
                        report.capture_status = "capture failed: copy submission failed";
                        destroy_uploaded_buffer(device, readback);
                    } else {
                        void* mapped = nullptr;
                        if (vkMapMemory(device, readback.memory, 0, buf_size, 0, &mapped) !=
                            VK_SUCCESS) {
                            report.capture_status = "capture failed: readback map failed";
                        } else {
                            const uint8_t* pixels = static_cast<const uint8_t*>(mapped);
                            // Publish via temp + rename: the stream state is
                            // checked after every write and the close, so a
                            // mid-write ENOSPC can no longer leave a truncated
                            // PPM behind a "capture ok" report. Only a fully
                            // written file is renamed into place.
                            const std::filesystem::path screenshot_dir =
                                screenshot_path.parent_path().empty()
                                    ? std::filesystem::path(".")
                                    : screenshot_path.parent_path();
                            const std::filesystem::path temp_path =
                                screenshot_dir /
                                (screenshot_path.filename().string() + ".tmp-" +
                                 std::to_string(
                                     std::chrono::steady_clock::now().time_since_epoch().count()));
                            std::ofstream ppm(temp_path, std::ios::binary);
                            if (!ppm) {
                                report.capture_status = "capture failed: file open failed";
                            } else {
                                ppm << "P6\n" << w << " " << h << "\n255\n";
                                const uint32_t blue_offset =
                                    3 - red_offset - green_offset;
                                for (uint32_t i = 0; i < w * h; ++i) {
                                    ppm.put(static_cast<char>(pixels[i * 4 + red_offset]));
                                    ppm.put(static_cast<char>(pixels[i * 4 + green_offset]));
                                    ppm.put(static_cast<char>(pixels[i * 4 + blue_offset]));
                                }
                                ppm.flush();
                                const bool write_ok = static_cast<bool>(ppm);
                                ppm.close();
                                if (!write_ok || !ppm) {
                                    std::error_code remove_ec;
                                    std::filesystem::remove(temp_path, remove_ec);
                                    report.capture_status =
                                        "capture failed: file write failed";
                                } else if (::rename(temp_path.c_str(),
                                                    screenshot_path.c_str()) != 0) {
                                    std::error_code remove_ec;
                                    std::filesystem::remove(temp_path, remove_ec);
                                    report.capture_status =
                                        "capture failed: file publish failed";
                                } else {
                                    std::cout << "screenshot=" << screenshot_path.string()
                                              << '\n';
                                    report.capture_status = "capture ok";
                                }
                            }
                            vkUnmapMemory(device, readback.memory);
                        }
                        destroy_uploaded_buffer(device, readback);
                    }
                }
            }
            if (acquire_semaphore != VK_NULL_HANDLE) {
                vkDestroySemaphore(device, acquire_semaphore, nullptr);
            }
            if (acquire_fence != VK_NULL_HANDLE) {
                vkDestroyFence(device, acquire_fence, nullptr);
            }
            }
        }

        report.status = report.present_loop_completed ? "window, surface, device, and swapchain created successfully"
                                                     : "bootstrap completed but present loop ended early";
        cleanup();
        return report;
    } catch (const std::exception& error) {
        report.status = error.what();
        cleanup();
        return report;
    }
#endif
}

}  // namespace meridian
