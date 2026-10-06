#pragma once

#include <cmath>
#include <algorithm>
#include <array>
#include <cstdint>

#include "vgeo_builder.h"

namespace meridian {

struct Mat4f {
    float m[16] = {0.0f};
};

struct CameraFrameData {
    Mat4f view_projection;
    Vec3f camera_position = {0.0f, 0.0f, 0.0f};
};

struct InteractiveCamera {
    Vec3f position = {0.0f, 0.0f, 0.0f};
    float yaw = 0.0f;
    float pitch = 0.0f;
    float move_speed = 5.0f;
    float mouse_sensitivity = 0.003f;
    double last_cursor_x = 0.0;
    double last_cursor_y = 0.0;
    bool cursor_captured = false;
};

inline Vec3f camera_forward(const InteractiveCamera& cam) {
    return {std::sin(cam.yaw) * std::cos(cam.pitch),
            std::sin(cam.pitch),
            -std::cos(cam.yaw) * std::cos(cam.pitch)};
}

inline Vec3f camera_right(const InteractiveCamera& cam) {
    return {std::cos(cam.yaw), 0.0f, std::sin(cam.yaw)};
}

// Number of cascaded shadow map splits. Must match the array size used by
// shadow.vert and main_geometry.frag (which bind a 3-layer depth array).
constexpr uint32_t kShadowCascadeCount = 3;

struct FrameUBO {
    float view_projection[16];
    float light_vp[kShadowCascadeCount][16];
    float light_dir[4];
    // cascade_splits[i] is the view-space distance at which cascade i ends.
    // Index 0..kShadowCascadeCount-1 stores the three far splits; the last
    // slot is padding so the struct remains a multiple of 16 bytes (std140).
    float cascade_splits[4];
    // Opt-in output transform (appended; existing field offsets unchanged):
    // x = exposure multiplier, y = 1.0 when the ACES + sRGB-encode path is
    // enabled, z/w unused. Default (1.0, 0.0) reproduces the historical
    // bit-identical output.
    float tonemap_params[4];
};

struct CascadeLightSetup {
    Mat4f light_vp[kShadowCascadeCount];
    float splits[kShadowCascadeCount];
};

inline Mat4f identity_matrix() {
    Mat4f matrix;
    matrix.m[0] = 1.0f;
    matrix.m[5] = 1.0f;
    matrix.m[10] = 1.0f;
    matrix.m[15] = 1.0f;
    return matrix;
}

inline Mat4f multiply_matrix(const Mat4f& lhs, const Mat4f& rhs) {
    Mat4f result{};
    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            result.m[column * 4 + row] =
                lhs.m[0 * 4 + row] * rhs.m[column * 4 + 0] +
                lhs.m[1 * 4 + row] * rhs.m[column * 4 + 1] +
                lhs.m[2 * 4 + row] * rhs.m[column * 4 + 2] +
                lhs.m[3 * 4 + row] * rhs.m[column * 4 + 3];
        }
    }
    return result;
}

inline Vec3f subtract_vec3(const Vec3f& lhs, const Vec3f& rhs) {
    return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

inline Vec3f cross_vec3(const Vec3f& lhs, const Vec3f& rhs) {
    return {
        lhs.y * rhs.z - lhs.z * rhs.y,
        lhs.z * rhs.x - lhs.x * rhs.z,
        lhs.x * rhs.y - lhs.y * rhs.x,
    };
}

inline float dot_vec3(const Vec3f& lhs, const Vec3f& rhs) {
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

inline Vec3f normalize_vec3(const Vec3f& value) {
    const float length = std::sqrt(dot_vec3(value, value));
    if (length <= 1e-6f) {
        return {0.0f, 0.0f, 1.0f};
    }
    return {value.x / length, value.y / length, value.z / length};
}

inline Mat4f look_at_matrix(const Vec3f& eye, const Vec3f& target, const Vec3f& up) {
    const Vec3f forward = normalize_vec3(subtract_vec3(target, eye));
    const Vec3f right = normalize_vec3(cross_vec3(forward, up));
    const Vec3f corrected_up = cross_vec3(right, forward);

    Mat4f result = identity_matrix();
    result.m[0] = right.x;
    result.m[1] = corrected_up.x;
    result.m[2] = -forward.x;
    result.m[4] = right.y;
    result.m[5] = corrected_up.y;
    result.m[6] = -forward.y;
    result.m[8] = right.z;
    result.m[9] = corrected_up.z;
    result.m[10] = -forward.z;
    result.m[12] = -dot_vec3(right, eye);
    result.m[13] = -dot_vec3(corrected_up, eye);
    result.m[14] = dot_vec3(forward, eye);
    return result;
}

inline Mat4f perspective_matrix(float vertical_fov_radians, float aspect_ratio, float near_plane,
                                float far_plane) {
    const float tan_half_fov = std::tan(vertical_fov_radians * 0.5f);
    Mat4f result{};
    result.m[0] = 1.0f / (aspect_ratio * tan_half_fov);
    result.m[5] = -(1.0f / tan_half_fov);
    result.m[10] = far_plane / (near_plane - far_plane);
    result.m[11] = -1.0f;
    result.m[14] = (near_plane * far_plane) / (near_plane - far_plane);
    return result;
}

inline Mat4f ortho_matrix(float left, float right, float bottom, float top, float near_val, float far_val) {
    Mat4f result{};
    result.m[0] = 2.0f / (right - left);
    result.m[5] = 2.0f / (top - bottom);
    result.m[10] = 1.0f / (near_val - far_val);
    result.m[12] = -(right + left) / (right - left);
    result.m[13] = -(top + bottom) / (top - bottom);
    result.m[14] = near_val / (near_val - far_val);
    result.m[15] = 1.0f;
    return result;
}

struct FrustumPlanes {
    float planes[6][4];
};

inline FrustumPlanes extract_frustum_planes(const Mat4f& vp) {
    FrustumPlanes fp{};
    // Row extraction from column-major: row r = { m[0*4+r], m[1*4+r], m[2*4+r], m[3*4+r] }
    auto row = [&](int r, int c) -> float { return vp.m[c * 4 + r]; };
    // Left:   row3 + row0
    for (int i = 0; i < 4; ++i) fp.planes[0][i] = row(3, i) + row(0, i);
    // Right:  row3 - row0
    for (int i = 0; i < 4; ++i) fp.planes[1][i] = row(3, i) - row(0, i);
    // Bottom: row3 + row1
    for (int i = 0; i < 4; ++i) fp.planes[2][i] = row(3, i) + row(1, i);
    // Top:    row3 - row1
    for (int i = 0; i < 4; ++i) fp.planes[3][i] = row(3, i) - row(1, i);
    // Near:   row3 + row2
    for (int i = 0; i < 4; ++i) fp.planes[4][i] = row(3, i) + row(2, i);
    // Far:    row3 - row2
    for (int i = 0; i < 4; ++i) fp.planes[5][i] = row(3, i) - row(2, i);
    // Normalize each plane
    for (auto& plane : fp.planes) {
        const float len = std::sqrt(plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]);
        if (len > 1e-6f) {
            for (float& v : plane) v /= len;
        }
    }
    return fp;
}

// Build per-cascade light view-projection matrices and view-space split
// distances. The sub-frustum for cascade i spans [near .. splits[0]] for i=0,
// [splits[i-1] .. splits[i]] for i>0, with the last cascade ending at `far`.
// `light_dir` is the surface-to-light direction (the fragment shader's L);
// each cascade's shadow camera looks from center + light_dir*radius toward
// the cascade center.
//
// Each cascade fits a tight orthographic light projection around the 8
// world-space corners of its sub-frustum, then pushes the light near plane
// back along the light direction to catch shadow casters outside the frustum.
inline CascadeLightSetup compute_cascade_light_setup(
    const Vec3f& camera_position,
    const Vec3f& camera_forward_dir,
    const Vec3f& camera_right_dir,
    const Vec3f& camera_up_dir,
    float vertical_fov_radians,
    float aspect_ratio,
    float near_plane,
    float far_plane,
    const Vec3f& light_dir,
    float caster_extent,
    float lambda) {
    CascadeLightSetup out{};

    // Log/uniform split blend (lambda=0 uniform, lambda=1 log).
    const float clip_range = far_plane - near_plane;
    for (uint32_t i = 0; i < kShadowCascadeCount; ++i) {
        const float p = static_cast<float>(i + 1) / static_cast<float>(kShadowCascadeCount);
        const float log_split = near_plane * std::pow(far_plane / near_plane, p);
        const float uniform_split = near_plane + clip_range * p;
        out.splits[i] = lambda * log_split + (1.0f - lambda) * uniform_split;
    }

    const Vec3f light = normalize_vec3(light_dir);
    const Vec3f world_up = (std::fabs(light.y) > 0.99f) ? Vec3f{1.0f, 0.0f, 0.0f}
                                                        : Vec3f{0.0f, 1.0f, 0.0f};

    const float tan_half = std::tan(vertical_fov_radians * 0.5f);

    for (uint32_t i = 0; i < kShadowCascadeCount; ++i) {
        const float sub_near = (i == 0) ? near_plane : out.splits[i - 1];
        const float sub_far = out.splits[i];

        // Eight sub-frustum corners in world space.
        const float near_h = tan_half * sub_near;
        const float near_w = near_h * aspect_ratio;
        const float far_h = tan_half * sub_far;
        const float far_w = far_h * aspect_ratio;

        const Vec3f fc = camera_position;
        Vec3f corners[8];
        auto point = [&](float dist, float dx, float dy) {
            return Vec3f{
                fc.x + camera_forward_dir.x * dist + camera_right_dir.x * dx + camera_up_dir.x * dy,
                fc.y + camera_forward_dir.y * dist + camera_right_dir.y * dx + camera_up_dir.y * dy,
                fc.z + camera_forward_dir.z * dist + camera_right_dir.z * dx + camera_up_dir.z * dy,
            };
        };
        corners[0] = point(sub_near, -near_w, -near_h);
        corners[1] = point(sub_near,  near_w, -near_h);
        corners[2] = point(sub_near,  near_w,  near_h);
        corners[3] = point(sub_near, -near_w,  near_h);
        corners[4] = point(sub_far,  -far_w,  -far_h);
        corners[5] = point(sub_far,   far_w,  -far_h);
        corners[6] = point(sub_far,   far_w,   far_h);
        corners[7] = point(sub_far,  -far_w,   far_h);

        Vec3f center{0.0f, 0.0f, 0.0f};
        for (const Vec3f& c : corners) {
            center.x += c.x; center.y += c.y; center.z += c.z;
        }
        center.x /= 8.0f; center.y /= 8.0f; center.z /= 8.0f;

        // Use the sub-frustum bounding-sphere radius for a rotation-stable extent.
        float radius = 0.0f;
        for (const Vec3f& c : corners) {
            const Vec3f d = subtract_vec3(c, center);
            radius = std::max(radius, std::sqrt(dot_vec3(d, d)));
        }
        // Round up to avoid shimmer as camera moves.
        radius = std::ceil(radius * 16.0f) / 16.0f;

        // Convention: light_dir is the surface-to-light direction (the same
        // vector the fragment shader uses as L). The light therefore sits at
        // center + light*radius, and the shadow eye goes there, looking back
        // at the cascade center -- a light at +Y lights top faces and casts
        // shadows downward, matching the Lambert term. (The eye previously
        // sat at center - light*radius, on the opposite side of the scene
        // from the light, so depth came from the underside of the geometry.)
        const Vec3f eye{
            center.x + light.x * radius,
            center.y + light.y * radius,
            center.z + light.z * radius,
        };
        const Mat4f light_view = look_at_matrix(eye, center, world_up);
        // Orthographic fit: symmetric box around the cascade center. The
        // sphere spans view-space z in [-2*radius, 0] (eye at +light*radius
        // looking back), so far = 2*radius covers the whole cascade, and the
        // near plane pushed back to -caster_extent extends caster_extent past
        // the eye toward the light so shadow casters outside the sub-frustum
        // (between the light and the cascade) are captured.
        const Mat4f light_proj = ortho_matrix(-radius, radius, -radius, radius,
                                              -caster_extent, 2.0f * radius);
        out.light_vp[i] = multiply_matrix(light_proj, light_view);
    }
    return out;
}

} // namespace meridian
