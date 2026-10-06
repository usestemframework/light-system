#version 450

layout(set = 0, binding = 2) uniform FrameData {
    mat4 view_projection;
    mat4 light_vp[3];
    vec4 light_dir;
    vec4 cascade_splits;
    vec4 tonemap_params;
} frame;

// Plain 2D array sampler: we do the depth comparison manually because
// sampler2DArrayShadow does not round-trip reliably through SPIRV-Cross /
// MoltenVK on Apple GPUs (produces zero-cost no-op samples in practice).
layout(set = 0, binding = 3) uniform sampler2DArray shadow_map;

layout(set = 0, binding = 5) uniform sampler2D base_texture;

layout(location = 0) in vec3 frag_normal;
layout(location = 1) flat in uint frag_geometry_index;
layout(location = 2) flat in uint frag_geometry_kind;
layout(location = 3) in vec3 frag_world_pos;
layout(location = 4) flat in uint frag_local_triangle;
layout(location = 5) in vec2 frag_uv;
layout(location = 6) flat in uint frag_has_uv;

layout(location = 0) out vec4 out_color;
layout(location = 1) out uvec2 out_visibility;

void main() {
    // Per-cluster color variation using geometry index as seed.
    uint hash = frag_geometry_index * 2654435761u;
    float hue = float(hash & 0xFFu) / 255.0;
    vec3 hash_color = vec3(0.62 + hue * 0.12, 0.64 + hue * 0.08, 0.68 - hue * 0.06);
    vec3 base_color = frag_has_uv == 1u ? texture(base_texture, frag_uv).rgb : hash_color;

    // Renormalize after interpolation: the vertex stage's normalize() does
    // not survive perspective-correct interpolation, and dot(N, L) (diffuse),
    // the slope-scaled shadow bias, and the hemisphere ambient all consume
    // unit-length N.
    vec3 N = normalize(gl_FrontFacing ? frag_normal : -frag_normal);
    vec3 L = normalize(frame.light_dir.xyz);
    float ndotl = max(dot(N, L), 0.0);

    // Cascade selection: pick the first cascade whose projection contains the
    // fragment, keeping a small inset so fragments near the edge fall through
    // to the next one and the cascade seam is hidden.
    int chosen = -1;
    vec2 shadow_uv = vec2(0.0);
    float shadow_depth = 0.0;
    for (int c = 0; c < 3; c++) {
        vec4 lc = frame.light_vp[c] * vec4(frag_world_pos, 1.0);
        vec3 nd = lc.xyz / lc.w;
        vec2 uv = nd.xy * 0.5 + 0.5;
        if (uv.x >= 0.02 && uv.x <= 0.98 &&
            uv.y >= 0.02 && uv.y <= 0.98 &&
            nd.z >= 0.0 && nd.z <= 1.0) {
            chosen = c;
            shadow_uv = uv;
            shadow_depth = nd.z;
            break;
        }
    }

    float shadow = 1.0;
    if (chosen >= 0) {
        // Slope-scaled bias: grow bias as the surface grazes the light direction.
        float cos_theta = clamp(dot(N, L), 0.0, 1.0);
        float slope = sqrt(max(1.0 - cos_theta * cos_theta, 0.0)) / max(cos_theta, 1e-3);
        // Farther cascades cover more world space per texel, so the bias must
        // scale roughly with the cascade's extent.
        float cascade_scale = (chosen == 0) ? 1.0 : (chosen == 1) ? 2.0 : 4.0;
        float bias = clamp(0.0004 * slope * cascade_scale, 0.00005, 0.01);
        float shadow_ref = shadow_depth - bias;

        // 8-tap Poisson disk rotated per-pixel by interleaved-gradient noise.
        const vec2 poisson[8] = vec2[8](
            vec2(-0.94201624, -0.39906216), vec2( 0.94558609, -0.76890725),
            vec2(-0.09418410, -0.92938870), vec2( 0.34495938,  0.29387760),
            vec2(-0.91588581,  0.45771432), vec2(-0.38277543,  0.27676845),
            vec2( 0.44323325, -0.97511554), vec2( 0.53742981, -0.47373420));

        float ign = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
        float rot_ang = ign * 6.28318530718;
        float sr = sin(rot_ang);
        float cr = cos(rot_ang);
        mat2 rot = mat2(cr, -sr, sr, cr);

        vec2 texel_size = vec2(1.0) / vec2(textureSize(shadow_map, 0).xy);
        float radius = 2.5; // in texels
        shadow = 0.0;
        for (int i = 0; i < 8; i++) {
            vec2 offset = rot * poisson[i] * texel_size * radius;
            float stored = texture(shadow_map, vec3(shadow_uv + offset, float(chosen))).r;
            // Manual less-than compare: if stored depth is >= reference, we're lit.
            shadow += (stored >= shadow_ref) ? 1.0 : 0.0;
        }
        shadow *= (1.0 / 8.0);
    }

    // Hemisphere ambient (sky blue from above, ground bounce from below).
    float up = N.y * 0.5 + 0.5;
    vec3 sky_color = vec3(0.55, 0.6, 0.7);
    vec3 ground_color = vec3(0.35, 0.3, 0.28);
    vec3 ambient = base_color * mix(ground_color, sky_color, up) * 0.75;
    vec3 diffuse = base_color * ndotl * shadow * 0.8;
    vec3 color = ambient + diffuse;

    // Opt-in output transform (uniform branch; default y=0 keeps the
    // passthrough bit-identical). The swapchain is UNORM with an SRGB
    // NONLINEAR color space, so the passthrough path writes linear values
    // that clip at the display; this path applies exposure, the Narkowicz
    // ACES filmic fit, and an sRGB encode before the UNORM store.
    if (frame.tonemap_params.y > 0.5) {
        vec3 mapped = color * frame.tonemap_params.x;
        mapped = clamp((mapped * (2.51 * mapped + 0.03)) /
                       (mapped * (2.43 * mapped + 0.59) + 0.14), 0.0, 1.0);
        color = pow(mapped, vec3(1.0 / 2.2));
    }
    out_color = vec4(color, 1.0);

    // Two-word visibility encoding matching visibility_format.h:
    // word0 = instance_index (always 0 for single-instance scenes)
    // word1 = valid_bit(31) | geometry_kind(30) | geometry_index(8..29) | local_triangle(0..7)
    uint word0 = 0u;
    uint word1 = (1u << 31u) |
                 (frag_geometry_kind << 30u) |
                 ((frag_geometry_index & 0x3fffffu) << 8u) |
                 (frag_local_triangle & 0xffu);
    out_visibility = uvec2(word0, word1);
}

