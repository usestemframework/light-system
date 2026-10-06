#include "builder_internal.h"

namespace meridian {

BuildManifest load_manifest(const std::filesystem::path& manifest_path) {
    std::ifstream input(manifest_path);
    if (!input) {
        throw BuilderError("failed to open manifest: " + manifest_path.string());
    }

    BuildManifest manifest;
    bool saw_bounds_min = false;
    bool saw_bounds_max = false;
    std::string line;
    size_t line_number = 0;

    while (std::getline(input, line)) {
        ++line_number;
        const std::string trimmed = detail::trim(line);
        if (trimmed.empty() || trimmed.starts_with('#')) {
            continue;
        }

        const auto separator = trimmed.find('=');
        if (separator == std::string::npos) {
            throw BuilderError("invalid manifest line " + std::to_string(line_number));
        }

        const std::string key = detail::trim(std::string_view(trimmed).substr(0, separator));
        const std::string value = detail::trim(std::string_view(trimmed).substr(separator + 1));

        if (key == "asset_id") {
            manifest.asset_id = value;
        } else if (key == "source_asset") {
            manifest.source_asset = value;
        } else if (key == "output_path") {
            manifest.output_path = value;
        } else if (key == "bounds_min") {
            manifest.explicit_bounds.min = detail::parse_vec3(value);
            manifest.has_explicit_bounds = true;
            saw_bounds_min = true;
        } else if (key == "bounds_max") {
            manifest.explicit_bounds.max = detail::parse_vec3(value);
            manifest.has_explicit_bounds = true;
            saw_bounds_max = true;
        } else if (key == "emit_fallback") {
            manifest.emit_fallback = detail::parse_bool(value);
        } else if (key == "emit_texture") {
            manifest.emit_texture = detail::parse_bool(value);
        } else if (key == "material_slots") {
            manifest.material_slots = detail::parse_list(value);
        } else if (key == "cluster_vertex_limit") {
            manifest.cluster_vertex_limit = detail::parse_u32(value);
        } else if (key == "cluster_triangle_limit") {
            manifest.cluster_triangle_limit = detail::parse_u32(value);
        } else if (key == "page_cluster_limit") {
            manifest.page_cluster_limit = detail::parse_u32(value);
        } else if (key == "hierarchy_partition_size") {
            manifest.hierarchy_partition_size = detail::parse_u32(value);
        } else if (key == "bounds_padding") {
            manifest.bounds_padding = detail::parse_float(value);
        } else {
            throw BuilderError("unknown manifest key: " + key);
        }
    }

    if (saw_bounds_min != saw_bounds_max) {
        throw BuilderError("manifest must define both bounds_min and bounds_max together");
    }

    detail::validate_manifest(manifest);
    const std::filesystem::path manifest_directory = manifest_path.parent_path();
    if (!manifest.source_asset.empty() && manifest.source_asset.is_relative()) {
        manifest.source_asset = (manifest_directory / manifest.source_asset).lexically_normal();
    }
    if (!manifest.output_path.empty() && manifest.output_path.is_relative()) {
        manifest.output_path = (manifest_directory / manifest.output_path).lexically_normal();
    }
    return manifest;
}

VGeoResource create_stub_resource(const BuildManifest& manifest) {
    detail::validate_manifest(manifest);

    VGeoResource resource;
    resource.asset_id = manifest.asset_id;
    resource.source_asset = manifest.source_asset;
    resource.bounds = manifest.has_explicit_bounds
                          ? detail::apply_bounds_padding(manifest.explicit_bounds, manifest.bounds_padding)
                          : Bounds3f{};
    resource.has_fallback = manifest.emit_fallback;
    resource.metadata.root_hierarchy_node_index = 0;

    resource.material_sections.reserve(manifest.material_slots.size());
    for (size_t index = 0; index < manifest.material_slots.size(); ++index) {
        MaterialSection section;
        section.name = manifest.material_slots[index];
        section.fallback_section = static_cast<uint32_t>(index);
        resource.material_sections.push_back(section);
    }

    HierarchyNode root_node;
    root_node.parent_index = 0xffffffffu;
    root_node.bounds = resource.bounds;
    resource.hierarchy_nodes.push_back(root_node);
    return resource;
}

VGeoResource build_resource(const BuildManifest& manifest) {
    VGeoResource resource = create_stub_resource(manifest);
    detail::MeshData mesh = detail::load_mesh(manifest);
    // Authored normals (glTF NORMAL, transformed by the node world matrix's
    // inverse-transpose at import) are kept as-is, preserving hard edges.
    // Smooth normals are generated only when the source asset has none --
    // compute_smooth_normals overwrites any existing normal stream.
    if (mesh.normals.empty()) {
        detail::compute_smooth_normals(mesh);
    }
    mesh.emit_uv_payloads = manifest.emit_texture;
    if (manifest.emit_texture) {
        if (mesh.texcoords.empty()) {
            throw BuilderError("emit_texture requires source texcoords (glTF TEXCOORD_0)");
        }
        constexpr uint32_t kTextureSize = 256;
        resource.texture_width = kTextureSize;
        resource.texture_height = kTextureSize;
        resource.texture_payload = detail::generate_checker_texture(kTextureSize, kTextureSize);
    }
    resource.bounds = detail::resolve_resource_bounds(manifest, mesh.bounds);
    resource.source_vertex_count = static_cast<uint32_t>(mesh.positions.size());
    resource.seam_locked_vertex_count =
        static_cast<uint32_t>(std::count_if(mesh.vertex_locks.begin(), mesh.vertex_locks.end(),
                                            [](unsigned char lock) { return lock != 0; }));
    resource.source_triangle_count = 0;
    for (const detail::MeshSection& section : mesh.sections) {
        resource.source_triangle_count += static_cast<uint32_t>(section.indices.size() / 3);
    }

    const clodConfig cluster_config = detail::make_clod_config(manifest);
    std::vector<ClusterRecord> source_clusters;
    std::vector<std::byte> source_cluster_payload;
    std::vector<std::vector<unsigned int>> cluster_global_indices;

    for (const detail::MeshSection& section : mesh.sections) {
        if (section.indices.empty()) {
            continue;
        }
        detail::build_section_base_clusters(mesh, section, cluster_config, source_clusters,
                                            source_cluster_payload, cluster_global_indices);
    }
    if (source_clusters.empty()) {
        throw BuilderError("meshlet generation produced no clusters");
    }

    std::vector<detail::TempHierarchyNode> temp_hierarchy;
    temp_hierarchy.reserve(1 + source_clusters.size() * 2);
    temp_hierarchy.push_back(detail::TempHierarchyNode{});
    temp_hierarchy[0].parent_index = 0xffffffffu;
    temp_hierarchy[0].bounds = detail::make_empty_bounds();
    temp_hierarchy[0].geometric_error = 0.0f;

    std::vector<unsigned int> position_remap(mesh.positions.size());
    meshopt_generatePositionRemap(position_remap.data(),
                                  reinterpret_cast<const float*>(mesh.positions.data()),
                                  mesh.positions.size(), sizeof(Vec3f));

    for (uint32_t material_section_index = 0; material_section_index < mesh.sections.size();
         ++material_section_index) {
        std::vector<uint32_t> material_cluster_ids;
        for (uint32_t cluster_index = 0; cluster_index < source_clusters.size(); ++cluster_index) {
            if (source_clusters[cluster_index].material_section_index == material_section_index) {
                material_cluster_ids.push_back(cluster_index);
            }
        }
        if (material_cluster_ids.empty()) {
            continue;
        }

        const uint32_t section_root_temp_node_index = detail::build_temp_hierarchy(
            temp_hierarchy, mesh, cluster_global_indices, source_clusters, material_cluster_ids, 0,
            manifest.hierarchy_partition_size, position_remap);
        temp_hierarchy[0].child_indices.push_back(section_root_temp_node_index);
        temp_hierarchy[0].geometric_error =
            std::max(temp_hierarchy[0].geometric_error,
                     temp_hierarchy[section_root_temp_node_index].geometric_error);
    }
    if (temp_hierarchy[0].child_indices.empty()) {
        throw BuilderError("hierarchy construction produced no material section roots");
    }

    std::vector<uint32_t> all_cluster_ids(source_clusters.size());
    for (uint32_t cluster_index = 0; cluster_index < all_cluster_ids.size(); ++cluster_index) {
        all_cluster_ids[cluster_index] = cluster_index;
    }
    temp_hierarchy[0].bounds = detail::merge_cluster_bounds(source_clusters, all_cluster_ids);

    resource.hierarchy_nodes.clear();
    resource.hierarchy_nodes.resize(1);
    resource.clusters.clear();
    resource.clusters.reserve(source_clusters.size());
    resource.cluster_geometry_payload.clear();
    resource.cluster_geometry_payload.reserve(source_cluster_payload.size());
    std::vector<uint32_t> source_to_runtime_cluster_indices(source_clusters.size(), 0xffffffffu);
    std::vector<uint32_t> temp_to_runtime_node_indices(temp_hierarchy.size(), 0xffffffffu);
    detail::flatten_temp_hierarchy(temp_hierarchy, source_clusters, source_cluster_payload, resource,
                                   temp_to_runtime_node_indices, source_to_runtime_cluster_indices, 0,
                                   0, 0xffffffffu);

    resource.pages = detail::build_base_pages(resource.clusters, manifest.page_cluster_limit);
    for (PageRecord& page : resource.pages) {
        const uint32_t page_end = page.first_cluster_index + page.cluster_count;
        for (uint32_t cluster_index = page.first_cluster_index; cluster_index < page_end;
             ++cluster_index) {
            resource.clusters[cluster_index].page_index = page.page_index;
        }
    }

    detail::update_hierarchy_page_ranges(resource);
    resource.metadata.root_hierarchy_node_index = 0;
    detail::build_lod_metadata(resource, mesh, manifest, cluster_global_indices,
                               source_to_runtime_cluster_indices);

    // build_lod_metadata materializes lod_clusters in merged-record order
    // while the payload bytes were appended in callback order; repack before
    // paging so cluster-table order is the physical payload order
    // (build_lod_pages slices pages out of cluster order).
    detail::repack_lod_cluster_payloads(resource);

    std::vector<PageRecord> lod_pages =
        detail::build_lod_pages(resource.lod_clusters, manifest.page_cluster_limit,
                                static_cast<uint32_t>(resource.pages.size()));
    for (PageRecord& page : lod_pages) {
        const uint32_t page_end = page.first_lod_cluster_index + page.lod_cluster_count;
        for (uint32_t cluster_index = page.first_lod_cluster_index; cluster_index < page_end;
             ++cluster_index) {
            resource.lod_clusters[cluster_index].page_index = page.page_index;
        }
        resource.pages.push_back(page);
    }

    detail::build_page_dependencies(resource);
    return resource;
}

void validate_resource(const VGeoResource& resource) {
    detail::validate_resource(resource);
}

ResourceSummary read_resource_summary(const std::filesystem::path& input_path) {
    return detail::read_resource_summary(input_path);
}

TraversalSelection simulate_traversal(const VGeoResource& resource, float error_threshold,
                                      const std::vector<uint8_t>& resident_pages,
                                      ParallelExecutor* executor) {
    return detail::simulate_traversal(resource, error_threshold, resident_pages, executor);
}

void write_resource(const VGeoResource& resource, const std::filesystem::path& output_path) {
    detail::write_resource(resource, output_path);
}

void write_summary(const VGeoResource& resource, const std::filesystem::path& output_path) {
    detail::write_summary(resource, output_path);
}

}  // namespace meridian
