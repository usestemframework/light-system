#define CLUSTERLOD_IMPLEMENTATION
#include "builder_internal.h"

#include <cstdio>

namespace meridian::detail {

std::vector<unsigned int> extract_meshlet_global_indices(
    const meshopt_Meshlet& meshlet, const std::vector<unsigned int>& meshlet_vertices,
    const std::vector<unsigned char>& meshlet_triangles) {
    std::vector<unsigned int> global_indices;
    global_indices.reserve(meshlet.triangle_count * 3);
    for (size_t i = 0; i < meshlet.triangle_count * 3; ++i) {
        global_indices.push_back(
            meshlet_vertices[meshlet.vertex_offset + meshlet_triangles[meshlet.triangle_offset + i]]);
    }
    return global_indices;
}

ClusterRecord append_meshlet_payload(const MeshData& mesh, const meshopt_Meshlet& meshlet,
                                     const std::vector<unsigned int>& meshlet_vertices,
                                     const std::vector<unsigned char>& meshlet_triangles,
                                     uint32_t material_section_index,
                                     std::vector<std::byte>& payload) {
    std::vector<Vec3f> local_positions;
    std::vector<uint32_t> local_indices;
    local_positions.reserve(meshlet.vertex_count);
    local_indices.reserve(meshlet.triangle_count * 3);

    Bounds3f bounds = make_empty_bounds();
    for (size_t i = 0; i < meshlet.vertex_count; ++i) {
        const uint32_t vertex_index = meshlet_vertices[meshlet.vertex_offset + i];
        const Vec3f& position = mesh.positions[vertex_index];
        local_positions.push_back(position);
        update_bounds(bounds, position);
    }
    for (size_t i = 0; i < meshlet.triangle_count * 3; ++i) {
        local_indices.push_back(meshlet_triangles[meshlet.triangle_offset + i]);
    }

    std::vector<Vec3f> local_normals;
    local_normals.reserve(meshlet.vertex_count);
    for (size_t i = 0; i < meshlet.vertex_count; ++i) {
        const uint32_t vertex_index = meshlet_vertices[meshlet.vertex_offset + i];
        local_normals.push_back(mesh.normals[vertex_index]);
    }

    const uint32_t payload_offset = narrow_payload_u32(payload.size(), "geometry payload");
    const PayloadHeader payload_header{meshlet.vertex_count, meshlet.triangle_count};
    append_bytes(payload, payload_header);
    for (const Vec3f& position : local_positions) {
        append_bytes(payload, position);
    }
    for (const Vec3f& normal : local_normals) {
        append_bytes(payload, normal);
    }
    const bool has_uvs = mesh.emit_uv_payloads && !mesh.texcoords.empty();
    if (has_uvs) {
        for (size_t i = 0; i < meshlet.vertex_count; ++i) {
            const uint32_t vertex_index = meshlet_vertices[meshlet.vertex_offset + i];
            append_bytes(payload, mesh.texcoords[vertex_index * 2 + 0]);
            append_bytes(payload, mesh.texcoords[vertex_index * 2 + 1]);
        }
    }
    for (const uint32_t index : local_indices) {
        append_bytes(payload, index);
    }

    ClusterRecord cluster;
    cluster.owning_node_index = 0xffffffffu;
    cluster.local_vertex_count = meshlet.vertex_count;
    cluster.local_triangle_count = meshlet.triangle_count;
    cluster.geometry_payload_offset = payload_offset;
    cluster.geometry_payload_size =
        narrow_payload_u32(payload.size(), "geometry payload") - payload_offset;
    cluster.page_index = 0;
    cluster.bounds = bounds;
    const meshopt_Bounds meshlet_bounds = meshopt_computeMeshletBounds(
        &meshlet_vertices[meshlet.vertex_offset], &meshlet_triangles[meshlet.triangle_offset],
        meshlet.triangle_count, reinterpret_cast<const float*>(mesh.positions.data()),
        mesh.positions.size(), sizeof(Vec3f));
    cluster.normal_cone_axis[0] = meshlet_bounds.cone_axis[0];
    cluster.normal_cone_axis[1] = meshlet_bounds.cone_axis[1];
    cluster.normal_cone_axis[2] = meshlet_bounds.cone_axis[2];
    cluster.normal_cone_axis[3] = meshlet_bounds.cone_cutoff;
    cluster.cull_sphere[0] = meshlet_bounds.center[0];
    cluster.cull_sphere[1] = meshlet_bounds.center[1];
    cluster.cull_sphere[2] = meshlet_bounds.center[2];
    cluster.cull_sphere[3] = meshlet_bounds.radius;
    cluster.local_error = meshlet_bounds.radius;
    cluster.material_section_index = material_section_index;
    if (has_uvs) {
        cluster.flags |= kClusterFlagHasUv;
    }
    return cluster;
}

Bounds3f merge_cluster_bounds(const std::vector<ClusterRecord>& clusters,
                              const std::vector<uint32_t>& cluster_ids) {
    Bounds3f bounds = make_empty_bounds();
    for (const uint32_t cluster_id : cluster_ids) {
        update_bounds(bounds, clusters[cluster_id].bounds.min);
        update_bounds(bounds, clusters[cluster_id].bounds.max);
    }
    return bounds;
}

// Partitions use position-remapped indices so cluster adjacency matches the
// clusterlod DAG's view of the mesh (raw indices under-detect adjacency on
// position-split geometry, e.g. per-face box vertices, which scatters LOD
// group provenance and pushes group attachment toward the root).
std::vector<std::vector<uint32_t>> partition_cluster_ids(
    const MeshData& mesh, const std::vector<std::vector<unsigned int>>& cluster_global_indices,
    const std::vector<uint32_t>& cluster_ids, uint32_t target_partition_size,
    const std::vector<unsigned int>& position_remap) {
    if (cluster_ids.size() <= target_partition_size) {
        return {cluster_ids};
    }

    size_t total_index_count = 0;
    for (const uint32_t cluster_id : cluster_ids) {
        total_index_count += cluster_global_indices[cluster_id].size();
    }

    std::vector<unsigned int> flat_indices;
    std::vector<unsigned int> cluster_counts(cluster_ids.size());
    flat_indices.reserve(total_index_count);
    for (size_t i = 0; i < cluster_ids.size(); ++i) {
        const uint32_t cluster_id = cluster_ids[i];
        cluster_counts[i] = static_cast<unsigned int>(cluster_global_indices[cluster_id].size());
        for (const unsigned int index : cluster_global_indices[cluster_id]) {
            flat_indices.push_back(position_remap[index]);
        }
    }

    std::vector<unsigned int> partition_ids(cluster_ids.size());
    const size_t partition_count = meshopt_partitionClusters(
        partition_ids.data(), flat_indices.data(), flat_indices.size(), cluster_counts.data(),
        cluster_counts.size(), reinterpret_cast<const float*>(mesh.positions.data()),
        position_remap.size(), sizeof(Vec3f), target_partition_size);

    if (partition_count <= 1) {
        std::vector<std::vector<uint32_t>> fallback;
        for (size_t i = 0; i < cluster_ids.size(); i += target_partition_size) {
            const size_t end = std::min(cluster_ids.size(), i + target_partition_size);
            fallback.emplace_back(cluster_ids.begin() + static_cast<std::ptrdiff_t>(i),
                                  cluster_ids.begin() + static_cast<std::ptrdiff_t>(end));
        }
        return fallback;
    }

    std::vector<std::vector<uint32_t>> partitions(partition_count);
    for (size_t i = 0; i < cluster_ids.size(); ++i) {
        partitions[partition_ids[i]].push_back(cluster_ids[i]);
    }

    std::vector<std::vector<uint32_t>> compacted;
    compacted.reserve(partitions.size());
    for (auto& partition : partitions) {
        if (!partition.empty()) {
            compacted.push_back(std::move(partition));
        }
    }
    return compacted;
}

uint32_t build_temp_hierarchy(std::vector<TempHierarchyNode>& nodes, const MeshData& mesh,
                              const std::vector<std::vector<unsigned int>>& cluster_global_indices,
                              const std::vector<ClusterRecord>& clusters,
                              const std::vector<uint32_t>& cluster_ids, uint32_t parent_index,
                              uint32_t partition_size,
                              const std::vector<unsigned int>& position_remap) {
    const uint32_t node_index = static_cast<uint32_t>(nodes.size());
    TempHierarchyNode node;
    node.parent_index = parent_index;
    node.bounds = merge_cluster_bounds(clusters, cluster_ids);
    node.geometric_error = 0.0f;
    for (const uint32_t cluster_id : cluster_ids) {
        node.geometric_error = std::max(node.geometric_error, clusters[cluster_id].local_error);
    }
    nodes.push_back(node);

    if (cluster_ids.size() == 1) {
        nodes[node_index].leaf_cluster_index = cluster_ids[0];
        return node_index;
    }

    // District partitioning: grow the target with the cluster span so the
    // tree gains intermediate district levels (fanout ~8 per level) instead
    // of a single flat cut at partition_size. LOD groups cover progressively
    // larger base-cluster spans per DAG depth and need same-scale nodes to
    // attach to; without this, every depth>=1 group lands on the section
    // root and threshold selection collapses to a cliff.
    const uint32_t target_partition_size =
        std::max(partition_size, static_cast<uint32_t>((cluster_ids.size() + 7) / 8));

    std::vector<std::vector<uint32_t>> partitions = partition_cluster_ids(
        mesh, cluster_global_indices, cluster_ids, target_partition_size, position_remap);
    if (partitions.size() == 1 && partitions[0].size() == cluster_ids.size()) {
        partitions.clear();
        partitions.reserve(cluster_ids.size());
        for (const uint32_t cluster_id : cluster_ids) {
            partitions.push_back({cluster_id});
        }
    }

    nodes[node_index].child_indices.reserve(partitions.size());
    for (const std::vector<uint32_t>& partition : partitions) {
        nodes[node_index].child_indices.push_back(build_temp_hierarchy(
            nodes, mesh, cluster_global_indices, clusters, partition, node_index, partition_size,
            position_remap));
    }
    return node_index;
}

uint32_t append_reordered_cluster(const ClusterRecord& source_cluster,
                                  const std::vector<std::byte>& source_payload,
                                  VGeoResource& resource) {
    const uint32_t new_cluster_index = static_cast<uint32_t>(resource.clusters.size());
    ClusterRecord cluster = source_cluster;
    cluster.geometry_payload_offset =
        narrow_payload_u32(resource.cluster_geometry_payload.size(), "geometry payload");
    const auto begin = source_payload.begin() + source_cluster.geometry_payload_offset;
    const auto end = begin + source_cluster.geometry_payload_size;
    resource.cluster_geometry_payload.insert(resource.cluster_geometry_payload.end(), begin, end);
    cluster.geometry_payload_size = source_cluster.geometry_payload_size;
    resource.clusters.push_back(cluster);
    return new_cluster_index;
}

void flatten_temp_hierarchy(const std::vector<TempHierarchyNode>& temp_nodes,
                            const std::vector<ClusterRecord>& source_clusters,
                            const std::vector<std::byte>& source_payload, VGeoResource& resource,
                            std::vector<uint32_t>& temp_to_runtime_node_indices,
                            std::vector<uint32_t>& source_to_runtime_cluster_indices,
                            uint32_t temp_node_index, uint32_t node_index, uint32_t parent_index) {
    const TempHierarchyNode& temp_node = temp_nodes[temp_node_index];
    HierarchyNode node;
    temp_to_runtime_node_indices[temp_node_index] = node_index;
    node.parent_index = parent_index;
    node.bounds = temp_node.bounds;
    node.geometric_error = temp_node.geometric_error;
    node.min_resident_page = 0xffffffffu;
    node.max_resident_page = 0xffffffffu;

    const uint32_t cluster_start = static_cast<uint32_t>(resource.clusters.size());
    if (temp_node.child_indices.empty()) {
        node.first_child_index = 0;
        node.child_count = 0;
        node.first_cluster_index = cluster_start;
        node.cluster_count = 1;
        const uint32_t new_cluster_index =
            append_reordered_cluster(source_clusters[temp_node.leaf_cluster_index], source_payload, resource);
        source_to_runtime_cluster_indices[temp_node.leaf_cluster_index] = new_cluster_index;
        resource.clusters[new_cluster_index].owning_node_index = node_index;
        resource.hierarchy_nodes[node_index] = node;
        return;
    }

    node.first_child_index = static_cast<uint32_t>(resource.hierarchy_nodes.size());
    node.child_count = static_cast<uint32_t>(temp_node.child_indices.size());
    resource.hierarchy_nodes.resize(resource.hierarchy_nodes.size() + temp_node.child_indices.size());

    for (size_t child_offset = 0; child_offset < temp_node.child_indices.size(); ++child_offset) {
        const uint32_t child_index = node.first_child_index + static_cast<uint32_t>(child_offset);
        flatten_temp_hierarchy(temp_nodes, source_clusters, source_payload, resource,
                               temp_to_runtime_node_indices, source_to_runtime_cluster_indices,
                               temp_node.child_indices[child_offset], child_index, node_index);
    }

    node.first_cluster_index = cluster_start;
    node.cluster_count = static_cast<uint32_t>(resource.clusters.size()) - cluster_start;
    resource.hierarchy_nodes[node_index] = node;
}

void update_hierarchy_page_ranges(VGeoResource& resource) {
    for (HierarchyNode& node : resource.hierarchy_nodes) {
        if (node.cluster_count == 0) {
            node.min_resident_page = 0xffffffffu;
            node.max_resident_page = 0xffffffffu;
            continue;
        }

        node.min_resident_page = std::numeric_limits<uint32_t>::max();
        node.max_resident_page = 0;
        const uint32_t cluster_end = node.first_cluster_index + node.cluster_count;
        for (uint32_t cluster_index = node.first_cluster_index; cluster_index < cluster_end; ++cluster_index) {
            node.min_resident_page =
                std::min(node.min_resident_page, resource.clusters[cluster_index].page_index);
            node.max_resident_page =
                std::max(node.max_resident_page, resource.clusters[cluster_index].page_index);
        }
    }
}

std::vector<PageRecord> build_base_pages(const std::vector<ClusterRecord>& clusters,
                                         uint32_t page_cluster_limit) {
    std::vector<PageRecord> pages;
    if (clusters.empty()) {
        return pages;
    }

    const uint32_t cluster_count = static_cast<uint32_t>(clusters.size());
    for (uint32_t page_start = 0, page_index = 0; page_start < cluster_count;
         page_start += page_cluster_limit, ++page_index) {
        const uint32_t page_end = std::min(cluster_count, page_start + page_cluster_limit);
        const uint32_t first_offset = clusters[page_start].geometry_payload_offset;

        // Same packing contract as the LOD path: clusters occupy the payload
        // in cluster-table order (append_reordered_cluster guarantees it for
        // the base payload); check rather than assume so a future reorder
        // fails here instead of shipping invalid page ranges.
        uint64_t expected_offset = first_offset;
        for (uint32_t cluster_index = page_start; cluster_index < page_end; ++cluster_index) {
            const ClusterRecord& cluster = clusters[cluster_index];
            if (cluster.geometry_payload_offset != expected_offset) {
                throw BuilderError(
                    "base clusters are not packed contiguously in cluster-table order");
            }
            expected_offset += cluster.geometry_payload_size;
        }

        PageRecord page;
        page.page_index = page_index;
        page.byte_offset = first_offset;
        page.compressed_byte_size =
            narrow_payload_u32(expected_offset - first_offset, "geometry payload");
        page.uncompressed_byte_size = page.compressed_byte_size;
        page.first_cluster_index = page_start;
        page.cluster_count = page_end - page_start;
        page.first_lod_cluster_index = 0;
        page.lod_cluster_count = 0;
        page.dependency_page_start = 0;
        page.dependency_page_count = 0;
        page.flags = 0;
        pages.push_back(page);
    }
    return pages;
}

// Rebuild lod_geometry_payload in final lod_clusters order, rewriting every
// cluster's geometry_payload_offset. clodBuild appends payload bytes in
// callback-emission order while the union-find materialization in
// build_lod_metadata emits lod_clusters records in merged-record order; once
// a merge pulls non-adjacent callback groups together those orders diverge
// and the cluster offsets go stale (LS-01 regression: page ranges derived
// from [first cluster offset, last cluster end) wrapped or excluded member
// payloads). Mirrors append_reordered_cluster's contract for the base
// payload: cluster-table order is the canonical physical payload order, so
// build_lod_pages can slice pages directly out of cluster order. Call after
// the final lod_clusters order exists and before build_lod_pages.
void repack_lod_cluster_payloads(VGeoResource& resource) {
    if (resource.lod_clusters.empty()) {
        resource.lod_geometry_payload.clear();
        return;
    }

    // Records reference the callback-emission payload; keep those bytes while
    // rebuilding the payload in cluster-table order.
    std::vector<std::byte> source_payload = std::move(resource.lod_geometry_payload);
    resource.lod_geometry_payload.clear();
    resource.lod_geometry_payload.reserve(source_payload.size());

    for (LodClusterRecord& cluster : resource.lod_clusters) {
        const uint64_t source_begin = cluster.geometry_payload_offset;
        const uint64_t source_end = source_begin + cluster.geometry_payload_size;
        if (source_end > source_payload.size()) {
            throw BuilderError("lod cluster payload range exceeds staged lod geometry payload");
        }

        cluster.geometry_payload_offset =
            narrow_payload_u32(resource.lod_geometry_payload.size(), "LOD geometry payload");
        const auto begin = source_payload.begin() + static_cast<std::ptrdiff_t>(source_begin);
        const auto end = source_payload.begin() + static_cast<std::ptrdiff_t>(source_end);
        resource.lod_geometry_payload.insert(resource.lod_geometry_payload.end(), begin, end);
    }
}

std::vector<PageRecord> build_lod_pages(const std::vector<LodClusterRecord>& lod_clusters,
                                        uint32_t page_cluster_limit, uint32_t page_index_base) {
    std::vector<PageRecord> pages;
    if (lod_clusters.empty()) {
        return pages;
    }

    const uint32_t cluster_count = static_cast<uint32_t>(lod_clusters.size());
    for (uint32_t page_start = 0, local_page_index = 0; page_start < cluster_count;
         page_start += page_cluster_limit, ++local_page_index) {
        const uint32_t page_end = std::min(cluster_count, page_start + page_cluster_limit);
        const uint32_t first_offset = lod_clusters[page_start].geometry_payload_offset;

        // Contiguity is the packing contract repack_lod_cluster_payloads
        // establishes: every cluster in the page slice must occupy exactly
        // the next span of the payload. Check it here instead of deriving
        // [first offset, last end) silently -- an order regression used to
        // wrap uint32 arithmetic and fail only later in validate_resource
        // (or ship invalid pages).
        uint64_t expected_offset = first_offset;
        for (uint32_t cluster_index = page_start; cluster_index < page_end; ++cluster_index) {
            const LodClusterRecord& cluster = lod_clusters[cluster_index];
            if (cluster.geometry_payload_offset != expected_offset) {
                throw BuilderError(
                    "lod clusters are not packed contiguously in cluster-table order");
            }
            expected_offset += cluster.geometry_payload_size;
        }

        PageRecord page;
        page.page_index = page_index_base + local_page_index;
        page.byte_offset = first_offset;
        page.compressed_byte_size =
            narrow_payload_u32(expected_offset - first_offset, "LOD geometry payload");
        page.uncompressed_byte_size = page.compressed_byte_size;
        page.first_cluster_index = 0;
        page.cluster_count = 0;
        page.first_lod_cluster_index = page_start;
        page.lod_cluster_count = page_end - page_start;
        page.dependency_page_start = 0;
        page.dependency_page_count = 0;
        page.flags = kPageFlagLodPayload;
        pages.push_back(page);
    }
    return pages;
}

Bounds3f sphere_bounds_to_aabb(const clodBounds& bounds) {
    Bounds3f box;
    box.min = {bounds.center[0] - bounds.radius, bounds.center[1] - bounds.radius,
               bounds.center[2] - bounds.radius};
    box.max = {bounds.center[0] + bounds.radius, bounds.center[1] + bounds.radius,
               bounds.center[2] + bounds.radius};
    return box;
}

clodConfig make_clod_config(const BuildManifest& manifest) {
    clodConfig config = clodDefaultConfig(manifest.cluster_triangle_limit);
    config.max_vertices = manifest.cluster_vertex_limit;
    config.max_triangles = manifest.cluster_triangle_limit;
    config.min_triangles =
        std::max<size_t>(1, std::min<size_t>(config.min_triangles, config.max_triangles));
    config.partition_size = manifest.hierarchy_partition_size;
    config.optimize_bounds = true;
    config.optimize_clusters = true;
    // Must match meshopt_optimizeMeshlet (level 0) used by build_section_base_clusters:
    // provenance matching relies on clod level-0 clusters having identical index order.
    config.optimize_clusters_level = 0;
    config.simplify_permissive = false;
    config.simplify_fallback_permissive = false;
    return config;
}

std::string make_index_signature(const unsigned int* indices, size_t index_count) {
    std::string signature(index_count * sizeof(unsigned int), '\0');
    std::memcpy(&signature[0], indices, signature.size());
    return signature;
}

void build_section_base_clusters(const MeshData& mesh, const MeshSection& section,
                                 const clodConfig& config, std::vector<ClusterRecord>& source_clusters,
                                 std::vector<std::byte>& source_cluster_payload,
                                 std::vector<std::vector<unsigned int>>& cluster_global_indices) {
    const size_t max_meshlets = meshopt_buildMeshletsBound(section.indices.size(), config.max_vertices,
                                                           config.min_triangles);
    std::vector<meshopt_Meshlet> meshlets(max_meshlets);
    std::vector<unsigned int> meshlet_vertices(section.indices.size());
    std::vector<unsigned char> meshlet_triangles(section.indices.size());

    const size_t meshlet_count = config.cluster_spatial
                                     ? meshopt_buildMeshletsSpatial(
                                           meshlets.data(), meshlet_vertices.data(),
                                           meshlet_triangles.data(), section.indices.data(),
                                           section.indices.size(),
                                           reinterpret_cast<const float*>(mesh.positions.data()),
                                           mesh.positions.size(), sizeof(Vec3f), config.max_vertices,
                                           config.min_triangles, config.max_triangles,
                                           config.cluster_fill_weight)
                                     : meshopt_buildMeshletsFlex(
                                           meshlets.data(), meshlet_vertices.data(),
                                           meshlet_triangles.data(), section.indices.data(),
                                           section.indices.size(),
                                           reinterpret_cast<const float*>(mesh.positions.data()),
                                           mesh.positions.size(), sizeof(Vec3f), config.max_vertices,
                                           config.min_triangles, config.max_triangles, 0.0f,
                                           config.cluster_split_factor);

    meshlets.resize(meshlet_count);
    for (meshopt_Meshlet& meshlet : meshlets) {
        if (config.optimize_clusters) {
            meshopt_optimizeMeshlet(&meshlet_vertices[meshlet.vertex_offset],
                                    &meshlet_triangles[meshlet.triangle_offset],
                                    meshlet.triangle_count, meshlet.vertex_count);
        }
        cluster_global_indices.push_back(
            extract_meshlet_global_indices(meshlet, meshlet_vertices, meshlet_triangles));
        source_clusters.push_back(append_meshlet_payload(
            mesh, meshlet, meshlet_vertices, meshlet_triangles, section.material_section_index,
            source_cluster_payload));
    }
}

// Attach each LOD group to the deepest hierarchy node whose cluster span
// contains the full set of base clusters the group covers. Base-cluster
// coverage is stored as a flat list of (first_cluster_index, cluster_count)
// runs on each LOD group -- most groups have a single run, scenes whose
// clusterlod grouping doesn't align with the hierarchy partitioner have more.
// Subset attachment is made safe by threading a "covered" set of cluster
// ranges through the traversal: when an LOD group is selected at a node,
// its runs become the covered set for that subtree; descendants whose
// clusters fall in the covered set are skipped and base-cluster emits
// filter out already-covered clusters.
//
// Historical note: before f018cf1 this function required an exact node/group
// span match. That was semantically safe but scenes whose clusterlod groupings
// don't line up with the hierarchy partitioner (e.g. massive_city: 6230 groups
// -> 8 links) lost almost all LOD coverage and fell through to 31k base-leaf
// emits. Multi-run subset attachment + covered-set threading keeps correctness
// and recovers most of the lost coverage.
void build_node_lod_links(VGeoResource& resource,
                          const std::vector<LodGroupBuildInfo>& lod_group_infos,
                          const std::vector<uint32_t>& source_to_runtime_cluster_indices) {
    resource.node_lod_links.clear();
    resource.lod_group_base_runs.clear();

    // For each hierarchy node, the cluster span's material section (for
    // groups that cover the whole node). We only attach groups to nodes whose
    // entire span is single-material -- LOD groups are always single-material.
    std::vector<uint32_t> node_material_section(resource.hierarchy_nodes.size(), 0xffffffffu);
    std::vector<uint8_t> node_single_material(resource.hierarchy_nodes.size(), 0);
    for (uint32_t node_index = 0; node_index < resource.hierarchy_nodes.size(); ++node_index) {
        const HierarchyNode& node = resource.hierarchy_nodes[node_index];
        if (node.cluster_count == 0) {
            continue;
        }
        const uint32_t material_section_index =
            resource.clusters[node.first_cluster_index].material_section_index;
        bool single_material = true;
        for (uint32_t cluster_index = node.first_cluster_index + 1;
             cluster_index < node.first_cluster_index + node.cluster_count; ++cluster_index) {
            if (resource.clusters[cluster_index].material_section_index != material_section_index) {
                single_material = false;
                break;
            }
        }
        node_material_section[node_index] = material_section_index;
        node_single_material[node_index] = single_material ? 1u : 0u;
    }

    // cluster_owning_node[c] = the leaf hierarchy node that owns cluster c.
    std::vector<uint32_t> cluster_owning_node(resource.clusters.size(), 0xffffffffu);
    for (uint32_t node_index = 0; node_index < resource.hierarchy_nodes.size(); ++node_index) {
        const HierarchyNode& node = resource.hierarchy_nodes[node_index];
        if (node.child_count != 0 || node.cluster_count == 0) {
            continue;
        }
        for (uint32_t offset = 0; offset < node.cluster_count; ++offset) {
            cluster_owning_node[node.first_cluster_index + offset] = node_index;
        }
    }

    // Bucket candidate group -> node attachments; we resolve order and write
    // out the final link table after processing all groups.
    std::vector<std::vector<uint32_t>> links_by_node(resource.hierarchy_nodes.size());

    for (uint32_t group_index = 0; group_index < lod_group_infos.size(); ++group_index) {
        const LodGroupBuildInfo& info = lod_group_infos[group_index];
        if (info.source_cluster_ids.empty()) {
            continue;
        }

        std::vector<uint32_t> runtime_cluster_indices;
        runtime_cluster_indices.reserve(info.source_cluster_ids.size());
        for (uint32_t source_cluster_index : info.source_cluster_ids) {
            if (source_cluster_index >= source_to_runtime_cluster_indices.size()) {
                throw BuilderError("lod group provenance references invalid source cluster index");
            }
            runtime_cluster_indices.push_back(source_to_runtime_cluster_indices[source_cluster_index]);
        }

        std::sort(runtime_cluster_indices.begin(), runtime_cluster_indices.end());
        runtime_cluster_indices.erase(
            std::unique(runtime_cluster_indices.begin(), runtime_cluster_indices.end()),
            runtime_cluster_indices.end());

        // Compact consecutive cluster indices into runs of (first, count).
        std::vector<LodGroupBaseRun> runs;
        for (size_t i = 0; i < runtime_cluster_indices.size();) {
            uint32_t run_first = runtime_cluster_indices[i];
            uint32_t run_count = 1;
            while (i + run_count < runtime_cluster_indices.size() &&
                   runtime_cluster_indices[i + run_count] == run_first + run_count) {
                ++run_count;
            }
            runs.push_back(LodGroupBaseRun{run_first, run_count});
            i += run_count;
        }

        const uint32_t group_first = runtime_cluster_indices.front();
        const uint32_t group_end = runtime_cluster_indices.back() + 1;

        // Walk up from the leaf owning group_first to find the deepest
        // ancestor whose cluster span contains [group_first, group_end). The
        // group's runs must all live inside this span (they're all >= group_first
        // and < group_end by construction).
        if (group_first >= cluster_owning_node.size()) {
            continue;
        }
        uint32_t candidate = cluster_owning_node[group_first];
        uint32_t best_node = 0xffffffffu;
        while (candidate != 0xffffffffu) {
            const HierarchyNode& node = resource.hierarchy_nodes[candidate];
            const uint32_t node_first = node.first_cluster_index;
            const uint32_t node_end = node_first + node.cluster_count;
            if (node_first > group_first || node_end < group_end) {
                candidate = node.parent_index;
                continue;
            }
            if (!node_single_material[candidate] ||
                node_material_section[candidate] != info.material_section_index) {
                candidate = node.parent_index;
                continue;
            }

            best_node = candidate;
            uint32_t containing_child = 0xffffffffu;
            for (uint32_t child_offset = 0; child_offset < node.child_count; ++child_offset) {
                const HierarchyNode& child =
                    resource.hierarchy_nodes[node.first_child_index + child_offset];
                const uint32_t child_first = child.first_cluster_index;
                const uint32_t child_end = child_first + child.cluster_count;
                if (child_first <= group_first && child_end >= group_end) {
                    containing_child = node.first_child_index + child_offset;
                    break;
                }
            }
            if (containing_child != 0xffffffffu) {
                candidate = containing_child;
                continue;
            }
            break;
        }

        if (best_node == 0xffffffffu) {
            continue;
        }

        // Persist the group's base runs.
        LodGroupRecord& group_record = resource.lod_groups[group_index];
        group_record.first_base_run_index =
            static_cast<uint32_t>(resource.lod_group_base_runs.size());
        group_record.base_run_count = static_cast<uint32_t>(runs.size());
        resource.lod_group_base_runs.insert(resource.lod_group_base_runs.end(), runs.begin(),
                                            runs.end());

        links_by_node[best_node].push_back(group_index);
    }

    for (uint32_t node_index = 0; node_index < resource.hierarchy_nodes.size(); ++node_index) {
        std::vector<uint32_t>& linked = links_by_node[node_index];
        // Total order, not error alone: LOD groups tie on geometric_error
        // regularly (uniform simplification ladders), and std::sort is not
        // stable, so a keyless tie would permute the link table per
        // platform/stdlib and flip the traversal's pick. Group index (build
        // output order) is the deterministic tiebreak.
        std::sort(linked.begin(), linked.end(), [&](uint32_t lhs, uint32_t rhs) {
            const float lhs_error = resource.lod_groups[lhs].geometric_error;
            const float rhs_error = resource.lod_groups[rhs].geometric_error;
            if (lhs_error != rhs_error) {
                return lhs_error < rhs_error;
            }
            return lhs < rhs;
        });
        linked.erase(std::unique(linked.begin(), linked.end()), linked.end());

        HierarchyNode& node = resource.hierarchy_nodes[node_index];
        node.first_lod_link_index = static_cast<uint32_t>(resource.node_lod_links.size());
        node.lod_link_count = static_cast<uint32_t>(linked.size());
        for (uint32_t group_index : linked) {
            resource.node_lod_links.push_back(NodeLodLink{group_index});
        }
    }
}

LodClusterRecord append_lod_cluster_payload(const MeshData& mesh, const unsigned int* global_indices,
                                            size_t index_count, uint32_t group_index,
                                            int32_t refined_group_index,
                                            uint32_t material_section_index,
                                            std::vector<std::byte>& payload,
                                            const clodBounds& cluster_bounds,
                                            size_t vertex_count_hint) {
    std::vector<unsigned int> local_vertices(vertex_count_hint > 0 ? vertex_count_hint : index_count);
    std::vector<unsigned char> local_triangles(index_count);
    const size_t local_vertex_count =
        clodLocalIndices(local_vertices.data(), local_triangles.data(), global_indices, index_count);
    local_vertices.resize(local_vertex_count);
    local_triangles.resize(index_count);

    const uint32_t payload_offset = narrow_payload_u32(payload.size(), "LOD geometry payload");
    const PayloadHeader payload_header{static_cast<uint32_t>(local_vertices.size()),
                                       static_cast<uint32_t>(index_count / 3)};
    append_bytes(payload, payload_header);
    for (const uint32_t vertex_index : local_vertices) {
        append_bytes(payload, mesh.positions[vertex_index]);
    }
    for (const uint32_t vertex_index : local_vertices) {
        append_bytes(payload, mesh.normals[vertex_index]);
    }
    const bool has_uvs = mesh.emit_uv_payloads && !mesh.texcoords.empty();
    if (has_uvs) {
        for (const uint32_t vertex_index : local_vertices) {
            append_bytes(payload, mesh.texcoords[vertex_index * 2 + 0]);
            append_bytes(payload, mesh.texcoords[vertex_index * 2 + 1]);
        }
    }
    for (const unsigned char index : local_triangles) {
        const uint32_t widened = index;
        append_bytes(payload, widened);
    }

    const meshopt_Bounds cone_bounds = meshopt_computeClusterBounds(
        global_indices, index_count, reinterpret_cast<const float*>(mesh.positions.data()),
        mesh.positions.size(), sizeof(Vec3f));

    LodClusterRecord cluster;
    cluster.refined_group_index = refined_group_index;
    cluster.group_index = group_index;
    cluster.local_vertex_count = static_cast<uint32_t>(local_vertices.size());
    cluster.local_triangle_count = static_cast<uint32_t>(index_count / 3);
    cluster.geometry_payload_offset = payload_offset;
    cluster.geometry_payload_size =
        narrow_payload_u32(payload.size(), "LOD geometry payload") - payload_offset;
    cluster.bounds = sphere_bounds_to_aabb(cluster_bounds);
    cluster.normal_cone_axis[0] = cone_bounds.cone_axis[0];
    cluster.normal_cone_axis[1] = cone_bounds.cone_axis[1];
    cluster.normal_cone_axis[2] = cone_bounds.cone_axis[2];
    cluster.normal_cone_axis[3] = cone_bounds.cone_cutoff;
    cluster.cull_sphere[0] = cone_bounds.center[0];
    cluster.cull_sphere[1] = cone_bounds.center[1];
    cluster.cull_sphere[2] = cone_bounds.center[2];
    cluster.cull_sphere[3] = cone_bounds.radius;
    cluster.local_error = cluster_bounds.error;
    cluster.material_section_index = material_section_index;
    if (has_uvs) {
        cluster.flags |= kClusterFlagHasUv;
    }
    return cluster;
}

void build_lod_metadata(VGeoResource& resource, const MeshData& mesh, const BuildManifest& manifest,
                        const std::vector<std::vector<unsigned int>>& source_cluster_global_indices,
                        const std::vector<uint32_t>& source_to_runtime_cluster_indices) {
    const clodConfig config = make_clod_config(manifest);
    std::vector<LodGroupBuildInfo> lod_group_infos;

    for (const MeshSection& section : mesh.sections) {
        if (section.indices.empty()) {
            continue;
        }

        std::unordered_map<std::string, uint32_t> original_cluster_lookup;
        for (uint32_t source_cluster_index = 0; source_cluster_index < source_cluster_global_indices.size();
             ++source_cluster_index) {
            if (resource.clusters[source_to_runtime_cluster_indices[source_cluster_index]]
                    .material_section_index != section.material_section_index) {
                continue;
            }
            const std::vector<unsigned int>& global_indices = source_cluster_global_indices[source_cluster_index];
            original_cluster_lookup.emplace(
                make_index_signature(global_indices.data(), global_indices.size()), source_cluster_index);
        }

        // Per-clodBuild state: callback groups (clodBuild invocations) do not
        // map 1:1 onto LodGroupRecords. clod's partition() may split one
        // predecessor group's simplified clusters across several successor
        // callback groups; each successor inherits the predecessor's WHOLE
        // base provenance, and the traversal's whole-unit coverage model
        // would then suppress the siblings' base geometry after selecting
        // only one of them (hole). Groups sharing a predecessor therefore
        // merge into one record (union-find over callback group ids), so a
        // record is always the complete replacement unit its provenance
        // claims. Cluster records are staged per callback group and
        // materialized in merged-record order after clodBuild; the payload
        // bytes are appended in callback order during staging, so
        // repack_lod_cluster_payloads() re-establishes cluster-table order
        // as the physical payload order before page construction.
        struct StagedLodGroup {
            uint32_t depth = 0;
            Bounds3f bounds;
            float geometric_error = 0.0f;
            std::vector<int32_t> refined_ids;
            std::vector<LodClusterRecord> clusters;
        };
        std::vector<StagedLodGroup> staged_groups;
        std::vector<uint32_t> group_parent;
        // provenance_by_group[g] accumulates the merged record's whole
        // provenance on the record's root; reads go through find_root.
        std::vector<std::vector<uint32_t>> provenance_by_group;
        // predecessor group id -> first callback group that held children of it
        std::vector<uint32_t> children_claim_by_group;

        const auto find_root = [&](uint32_t group_id) {
            uint32_t root = group_id;
            while (group_parent[root] != root) {
                root = group_parent[root];
            }
            while (group_parent[group_id] != root) {
                const uint32_t next = group_parent[group_id];
                group_parent[group_id] = root;
                group_id = next;
            }
            return root;
        };

        // Union by smallest id: the root is the record's first callback
        // group, so record order and staging order stay deterministic.
        const auto unite_roots = [&](uint32_t lhs, uint32_t rhs) {
            if (lhs == rhs) {
                return lhs;
            }
            if (rhs < lhs) {
                std::swap(lhs, rhs);
            }
            group_parent[rhs] = lhs;
            provenance_by_group[lhs].insert(provenance_by_group[lhs].end(),
                                            provenance_by_group[rhs].begin(),
                                            provenance_by_group[rhs].end());
            return lhs;
        };

        clodMesh clod_mesh{};
        clod_mesh.indices = section.indices.data();
        clod_mesh.index_count = section.indices.size();
        clod_mesh.vertex_count = mesh.positions.size();
        clod_mesh.vertex_positions = reinterpret_cast<const float*>(mesh.positions.data());
        clod_mesh.vertex_positions_stride = sizeof(Vec3f);
        clod_mesh.vertex_lock = mesh.vertex_locks.data();
        // clod_mesh.attribute_weights must outlive the branch below: clodBuild
        // runs after the if/else closes, so a block-local array would dangle.
        static constexpr float uv_weights[2] = {1.0f, 1.0f};
        if (mesh.emit_uv_payloads && !mesh.texcoords.empty()) {
            // Attribute-aware simplification: track UVs alongside positions so
            // LOD clusters keep UVs consistent with their geometry (seam
            // vertices are additionally locked by build_vertex_locks). Only
            // affects simplification decisions; level-0 clustering is
            // attribute-blind so provenance signatures still match.
            clod_mesh.vertex_attributes = mesh.texcoords.data();
            clod_mesh.vertex_attributes_stride = 2 * sizeof(float);
            clod_mesh.attribute_weights = uv_weights;
            clod_mesh.attribute_count = 2;
        } else {
            clod_mesh.vertex_attributes = nullptr;
            clod_mesh.vertex_attributes_stride = 0;
            clod_mesh.attribute_weights = nullptr;
            clod_mesh.attribute_count = 0;
        }
        clod_mesh.attribute_protect_mask = 0;

        clodBuild(config, clod_mesh,
                  [&](clodGroup group, const clodCluster* clusters, size_t cluster_count) -> int {
                      std::vector<uint32_t> group_source_cluster_ids;
                      std::vector<int32_t> refined_group_ids;
                      for (size_t i = 0; i < cluster_count; ++i) {
                          if (clusters[i].refined == -1) {
                              const auto found = original_cluster_lookup.find(
                                  make_index_signature(clusters[i].indices, clusters[i].index_count));
                              if (found == original_cluster_lookup.end()) {
                                  throw BuilderError(
                                      "failed to resolve original cluster provenance for lod group");
                              }
                              group_source_cluster_ids.push_back(found->second);
                          } else {
                              if (clusters[i].refined < 0 ||
                                  static_cast<size_t>(clusters[i].refined) >= provenance_by_group.size()) {
                                  throw BuilderError("lod cluster refined group index is out of provenance range");
                              }
                              const std::vector<uint32_t>& refined_provenance =
                                  provenance_by_group[find_root(static_cast<uint32_t>(clusters[i].refined))];
                              group_source_cluster_ids.insert(group_source_cluster_ids.end(),
                                                              refined_provenance.begin(),
                                                              refined_provenance.end());
                              refined_group_ids.push_back(clusters[i].refined);
                          }
                      }
                      std::sort(group_source_cluster_ids.begin(), group_source_cluster_ids.end());
                      group_source_cluster_ids.erase(
                          std::unique(group_source_cluster_ids.begin(), group_source_cluster_ids.end()),
                          group_source_cluster_ids.end());
                      std::sort(refined_group_ids.begin(), refined_group_ids.end());
                      refined_group_ids.erase(
                          std::unique(refined_group_ids.begin(), refined_group_ids.end()),
                          refined_group_ids.end());

                      // Resolve the record this group belongs to: the union
                      // of records that already hold children of any of its
                      // predecessors (transitive through unite_roots when a
                      // group straddles several claimed predecessors).
                      uint32_t record_root = 0xffffffffu;
                      for (const int32_t refined : refined_group_ids) {
                          const uint32_t claim =
                              children_claim_by_group[static_cast<uint32_t>(refined)];
                          if (claim == 0xffffffffu) {
                              continue;
                          }
                          const uint32_t claim_root = find_root(claim);
                          record_root = record_root == 0xffffffffu
                                            ? claim_root
                                            : unite_roots(record_root, claim_root);
                      }
                      const uint32_t group_id = static_cast<uint32_t>(group_parent.size());
                      group_parent.push_back(record_root == 0xffffffffu ? group_id : record_root);
                      children_claim_by_group.push_back(0xffffffffu);
                      for (const int32_t refined : refined_group_ids) {
                          const uint32_t refined_index = static_cast<uint32_t>(refined);
                          if (children_claim_by_group[refined_index] == 0xffffffffu) {
                              children_claim_by_group[refined_index] = group_id;
                          }
                      }
                      provenance_by_group.push_back(group_source_cluster_ids);
                      if (record_root != 0xffffffffu) {
                          provenance_by_group[record_root].insert(
                              provenance_by_group[record_root].end(),
                              group_source_cluster_ids.begin(), group_source_cluster_ids.end());
                      }

                      StagedLodGroup& staged = staged_groups.emplace_back();
                      staged.depth = static_cast<uint32_t>(group.depth);
                      staged.bounds = sphere_bounds_to_aabb(group.simplified);
                      staged.geometric_error = group.simplified.error;
                      staged.refined_ids = std::move(refined_group_ids);

                      // group_index is 0 (and refined stays a callback-local
                      // group id) here; both are patched to final record
                      // indices when the staged records materialize below.
                      for (size_t i = 0; i < cluster_count; ++i) {
                          staged.clusters.push_back(append_lod_cluster_payload(
                              mesh, clusters[i].indices, clusters[i].index_count, 0,
                              clusters[i].refined, section.material_section_index,
                              resource.lod_geometry_payload, clusters[i].bounds,
                              clusters[i].vertex_count));
                      }

                      return static_cast<int>(group_id);
                  });

        // Materialize records: one per union-find root, in first-member
        // (callback) order; a record's members emit contiguously so its
        // [first, first + count) span covers the whole merged payload.
        std::vector<uint32_t> record_of_group(staged_groups.size(), 0xffffffffu);
        for (uint32_t g = 0; g < staged_groups.size(); ++g) {
            if (find_root(g) != g) {
                continue;
            }
            record_of_group[g] = static_cast<uint32_t>(resource.lod_groups.size());
            LodGroupRecord lod_group;
            lod_group.depth = staged_groups[g].depth;
            lod_group.material_section_index = section.material_section_index;
            lod_group.bounds = staged_groups[g].bounds;
            lod_group.geometric_error = staged_groups[g].geometric_error;
            resource.lod_groups.push_back(lod_group);
        }

        std::vector<std::vector<uint32_t>> members_of_root(staged_groups.size());
        for (uint32_t g = 0; g < staged_groups.size(); ++g) {
            members_of_root[find_root(g)].push_back(g);
        }

        uint32_t merged_group_count = 0;
        for (uint32_t root = 0; root < staged_groups.size(); ++root) {
            const std::vector<uint32_t>& members = members_of_root[root];
            if (members.empty()) {
                continue;
            }
            LodGroupRecord& lod_group = resource.lod_groups[record_of_group[root]];
            lod_group.first_lod_cluster_index =
                static_cast<uint32_t>(resource.lod_clusters.size());
            for (const uint32_t member : members) {
                record_of_group[member] = record_of_group[root];
                if (member != root) {
                    merged_group_count += 1;
                    // The record replaces the union of its members: bounds
                    // span every member, and eligibility waits for the worst
                    // member's error (selecting below it would render a
                    // member above its acceptable error).
                    update_bounds(lod_group.bounds, staged_groups[member].bounds.min);
                    update_bounds(lod_group.bounds, staged_groups[member].bounds.max);
                    lod_group.geometric_error =
                        std::max(lod_group.geometric_error, staged_groups[member].geometric_error);
                }
                for (LodClusterRecord& cluster : staged_groups[member].clusters) {
                    cluster.group_index = record_of_group[root];
                    if (cluster.refined_group_index >= 0) {
                        cluster.refined_group_index = static_cast<int32_t>(record_of_group[find_root(
                            static_cast<uint32_t>(cluster.refined_group_index))]);
                    }
                    resource.lod_clusters.push_back(cluster);
                }
                lod_group.lod_cluster_count +=
                    static_cast<uint32_t>(staged_groups[member].clusters.size());
            }

            std::vector<uint32_t> provenance = std::move(provenance_by_group[root]);
            std::sort(provenance.begin(), provenance.end());
            provenance.erase(std::unique(provenance.begin(), provenance.end()), provenance.end());
            lod_group_infos.push_back(
                LodGroupBuildInfo{section.material_section_index, std::move(provenance)});
        }

        // Invariant the coverage model depends on: no predecessor group's
        // children span more than one LodGroupRecord. The merge above
        // guarantees it for today's clusterlod (all children of a group are
        // emitted at exactly one depth, in level order); fail loudly if that
        // ever changes.
        for (uint32_t g = 0; g < staged_groups.size(); ++g) {
            for (const int32_t refined : staged_groups[g].refined_ids) {
                const uint32_t claim =
                    children_claim_by_group[static_cast<uint32_t>(refined)];
                if (claim == 0xffffffffu ||
                    record_of_group[g] != record_of_group[find_root(claim)]) {
                    throw BuilderError(
                        "lod predecessor group's children span multiple lod group records");
                }
            }
        }

        if (merged_group_count > 0) {
            std::fprintf(stderr,
                         "MERIDIAN_LOD: merged %u split successor groups into shared records "
                         "(material section %u)\n",
                         merged_group_count, section.material_section_index);
        }
    }

    build_node_lod_links(resource, lod_group_infos, source_to_runtime_cluster_indices);
}

bool collect_group_pages(const VGeoResource& resource, const LodGroupRecord& group,
                         std::vector<uint32_t>& pages) {
    pages.clear();
    for (uint32_t cluster_index = group.first_lod_cluster_index;
         cluster_index < group.first_lod_cluster_index + group.lod_cluster_count; ++cluster_index) {
        const uint32_t page_index = resource.lod_clusters[cluster_index].page_index;
        if (pages.empty() || pages.back() != page_index) {
            pages.push_back(page_index);
        }
    }
    return !pages.empty();
}

void collect_node_base_pages(const VGeoResource& resource, const HierarchyNode& node,
                             std::vector<uint32_t>& pages) {
    pages.clear();
    for (uint32_t cluster_index = node.first_cluster_index;
         cluster_index < node.first_cluster_index + node.cluster_count; ++cluster_index) {
        const uint32_t page_index = resource.clusters[cluster_index].page_index;
        if (pages.empty() || pages.back() != page_index) {
            pages.push_back(page_index);
        }
    }
}

void collect_cluster_span_pages(const VGeoResource& resource, uint32_t first_cluster_index,
                                uint32_t cluster_count, std::vector<uint32_t>& pages) {
    pages.clear();
    for (uint32_t cluster_index = first_cluster_index;
         cluster_index < first_cluster_index + cluster_count; ++cluster_index) {
        const uint32_t page_index = resource.clusters[cluster_index].page_index;
        if (pages.empty() || pages.back() != page_index) {
            pages.push_back(page_index);
        }
    }
}

void link_adjacent_page_sets(std::vector<std::vector<uint32_t>>& adjacency,
                             const std::vector<uint32_t>& lhs, const std::vector<uint32_t>& rhs) {
    for (const uint32_t lhs_page : lhs) {
        std::vector<uint32_t>& lhs_adjacency = adjacency[lhs_page];
        lhs_adjacency.insert(lhs_adjacency.end(), rhs.begin(), rhs.end());
    }
    for (const uint32_t rhs_page : rhs) {
        std::vector<uint32_t>& rhs_adjacency = adjacency[rhs_page];
        rhs_adjacency.insert(rhs_adjacency.end(), lhs.begin(), lhs.end());
    }
}

void build_page_dependencies(VGeoResource& resource) {
    std::vector<std::vector<uint32_t>> adjacency(resource.pages.size());
    std::vector<uint32_t> current_pages;
    std::vector<uint32_t> next_pages;

    for (const HierarchyNode& node : resource.hierarchy_nodes) {
        if (node.lod_link_count == 0 || node.cluster_count == 0) {
            continue;
        }

        // Each link's adjacent replacement-level dependency is between the
        // pages holding the group's covered base clusters and the pages
        // holding the group's LOD clusters.
        std::vector<uint32_t> link_base_pages;
        for (uint32_t link_offset = 0; link_offset < node.lod_link_count; ++link_offset) {
            const NodeLodLink& link =
                resource.node_lod_links[node.first_lod_link_index + link_offset];
            const LodGroupRecord& group = resource.lod_groups[link.lod_group_index];
            link_base_pages.clear();
            for (uint32_t run_offset = 0; run_offset < group.base_run_count; ++run_offset) {
                const LodGroupBaseRun& run =
                    resource.lod_group_base_runs[group.first_base_run_index + run_offset];
                for (uint32_t cluster_index = run.first_cluster_index;
                     cluster_index < run.first_cluster_index + run.cluster_count; ++cluster_index) {
                    const uint32_t page_index = resource.clusters[cluster_index].page_index;
                    if (link_base_pages.empty() || link_base_pages.back() != page_index) {
                        link_base_pages.push_back(page_index);
                    }
                }
            }
            collect_group_pages(resource, group, next_pages);
            if (!link_base_pages.empty() && !next_pages.empty()) {
                link_adjacent_page_sets(adjacency, link_base_pages, next_pages);
            }
        }
        (void)current_pages;
    }

    resource.page_dependencies.clear();
    for (uint32_t page_index = 0; page_index < resource.pages.size(); ++page_index) {
        std::vector<uint32_t>& page_adjacency = adjacency[page_index];
        std::sort(page_adjacency.begin(), page_adjacency.end());
        page_adjacency.erase(std::remove(page_adjacency.begin(), page_adjacency.end(), page_index),
                             page_adjacency.end());
        page_adjacency.erase(std::unique(page_adjacency.begin(), page_adjacency.end()),
                             page_adjacency.end());

        PageRecord& page = resource.pages[page_index];
        page.dependency_page_start = static_cast<uint32_t>(resource.page_dependencies.size());
        page.dependency_page_count = static_cast<uint32_t>(page_adjacency.size());
        resource.page_dependencies.insert(resource.page_dependencies.end(), page_adjacency.begin(),
                                          page_adjacency.end());
    }
}

}  // namespace meridian::detail
