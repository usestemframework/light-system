// Builder unit tests: LOD payload packing invariants (LS-01 follow-up).
//
// 1b8d59c union-find merged split LOD successors but reordered only the
// lod_clusters records, leaving lod_geometry_payload bytes in callback
// order; build_lod_pages() then derived page byte ranges that no longer
// held (validate_resource: "lod page payload range exceeds lod geometry
// payload" / "lod cluster payload falls outside owning page payload range"
// on every scene past demo size). These tests pin the repack contract
// directly, without fixtures:
//   - repack rewrites offsets to final cluster-table order and preserves
//     every cluster's bytes;
//   - repack is a byte-for-byte no-op when the orders already agree (the
//     no-split-manifest byte-identity property);
//   - build_lod_pages fails loudly on non-contiguous order instead of
//     emitting wrapped page ranges;
//   - the full pack pipeline (repack -> pages -> page_index assignment)
//     produces a resource that passes validate_resource, across a
//     page-cluster-limit matrix.

#include "builder_internal.h"

#include <cstdio>

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

constexpr uint32_t kClusterPayloadSize = 100;

std::vector<std::byte> slice_payload(const std::vector<std::byte>& payload, uint32_t offset,
                                     uint32_t size) {
    return std::vector<std::byte>(payload.begin() + static_cast<std::ptrdiff_t>(offset),
                                  payload.begin() + static_cast<std::ptrdiff_t>(offset + size));
}

// Minimal resource shaped like build_resource output: one material section,
// one base cluster under a root+leaf hierarchy, one base page, one LOD group
// owning lod_clusters whose payload offsets are whatever the caller staged
// (i.e. callback-emission order, possibly diverging from table order after
// union-find materialization).
meridian::VGeoResource make_test_resource(const std::vector<uint32_t>& staged_offsets) {
    meridian::VGeoResource resource;
    resource.asset_id = "lod_packing_test";
    resource.has_fallback = true;
    resource.metadata.root_hierarchy_node_index = 0;

    meridian::MaterialSection section;
    section.name = "default";
    section.fallback_section = 0;
    resource.material_sections.push_back(section);

    resource.cluster_geometry_payload.assign(8, std::byte{0x5a});

    meridian::HierarchyNode root;
    root.parent_index = 0xffffffffu;
    root.first_child_index = 1;
    root.child_count = 1;
    root.first_cluster_index = 0;
    root.cluster_count = 1;
    root.min_resident_page = 0;
    root.max_resident_page = 0;
    resource.hierarchy_nodes.push_back(root);

    meridian::HierarchyNode leaf;
    leaf.parent_index = 0;
    leaf.first_cluster_index = 0;
    leaf.cluster_count = 1;
    leaf.min_resident_page = 0;
    leaf.max_resident_page = 0;
    resource.hierarchy_nodes.push_back(leaf);

    meridian::ClusterRecord base_cluster;
    base_cluster.owning_node_index = 1;
    base_cluster.geometry_payload_offset = 0;
    base_cluster.geometry_payload_size = 8;
    base_cluster.page_index = 0;
    base_cluster.material_section_index = 0;
    resource.clusters.push_back(base_cluster);

    meridian::PageRecord base_page;
    base_page.page_index = 0;
    base_page.byte_offset = 0;
    base_page.compressed_byte_size = 8;
    base_page.uncompressed_byte_size = 8;
    base_page.first_cluster_index = 0;
    base_page.cluster_count = 1;
    resource.pages.push_back(base_page);

    meridian::LodGroupRecord lod_group;
    lod_group.material_section_index = 0;
    lod_group.first_lod_cluster_index = 0;
    lod_group.lod_cluster_count = static_cast<uint32_t>(staged_offsets.size());
    resource.lod_groups.push_back(lod_group);

    size_t payload_size = 0;
    for (const uint32_t offset : staged_offsets) {
        payload_size = std::max(payload_size, static_cast<size_t>(offset) + kClusterPayloadSize);
    }
    resource.lod_geometry_payload.assign(payload_size, std::byte{0x00});
    for (size_t i = 0; i < staged_offsets.size(); ++i) {
        meridian::LodClusterRecord lod_cluster;
        lod_cluster.group_index = 0;
        lod_cluster.geometry_payload_offset = staged_offsets[i];
        lod_cluster.geometry_payload_size = kClusterPayloadSize;
        lod_cluster.material_section_index = 0;
        resource.lod_clusters.push_back(lod_cluster);
        // Tag each cluster's payload span so byte preservation is checkable.
        const std::byte tag = static_cast<std::byte>(0x80 + i);
        std::fill_n(resource.lod_geometry_payload.begin() + static_cast<std::ptrdiff_t>(staged_offsets[i]),
                    kClusterPayloadSize, tag);
    }
    return resource;
}

// Mirrors build_resource's LOD tail: repack (optional), build pages, assign
// page indices, then validate the whole resource.
void finish_lod_packing(meridian::VGeoResource& resource, uint32_t page_cluster_limit,
                        bool repack) {
    if (repack) {
        meridian::detail::repack_lod_cluster_payloads(resource);
    }
    std::vector<meridian::PageRecord> lod_pages = meridian::detail::build_lod_pages(
        resource.lod_clusters, page_cluster_limit,
        static_cast<uint32_t>(resource.pages.size()));
    for (meridian::PageRecord& page : lod_pages) {
        const uint32_t page_end = page.first_lod_cluster_index + page.lod_cluster_count;
        for (uint32_t cluster_index = page.first_lod_cluster_index; cluster_index < page_end;
             ++cluster_index) {
            resource.lod_clusters[cluster_index].page_index = page.page_index;
        }
        resource.pages.push_back(page);
    }
}

void check_pages_contain_clusters(const meridian::VGeoResource& resource) {
    for (const meridian::PageRecord& page : resource.pages) {
        if ((page.flags & meridian::detail::kPageFlagLodPayload) == 0) {
            continue;
        }
        const uint32_t page_end = page.first_lod_cluster_index + page.lod_cluster_count;
        for (uint32_t cluster_index = page.first_lod_cluster_index; cluster_index < page_end;
             ++cluster_index) {
            const meridian::LodClusterRecord& cluster = resource.lod_clusters[cluster_index];
            check(cluster.geometry_payload_offset >= page.byte_offset,
                  "cluster offset is below owning page byte_offset");
            check(static_cast<uint64_t>(cluster.geometry_payload_offset) +
                          cluster.geometry_payload_size <=
                      page.byte_offset + page.uncompressed_byte_size,
                  "cluster payload end exceeds owning page payload end");
        }
    }
}

void check_global_adjacency(const meridian::VGeoResource& resource) {
    for (size_t i = 1; i < resource.lod_clusters.size(); ++i) {
        const meridian::LodClusterRecord& prev = resource.lod_clusters[i - 1];
        const meridian::LodClusterRecord& curr = resource.lod_clusters[i];
        check(static_cast<uint64_t>(prev.geometry_payload_offset) + prev.geometry_payload_size ==
                  curr.geometry_payload_offset,
                  "lod cluster payload order is not contiguous in cluster-table order");
    }
}

// Callback order A0, B0, C0, A1; union-find merges A0+A1 into one record, so
// final table order is A0, A1, B0, C0 while staged offsets stay A0=0,
// A1=300, B0=100, C0=200 -- the minimal failure shape from the audit.
void test_repack_permuted_offsets() {
    meridian::VGeoResource resource = make_test_resource({0, 300, 100, 200});

    std::vector<std::vector<std::byte>> bytes_before;
    for (const meridian::LodClusterRecord& cluster : resource.lod_clusters) {
        bytes_before.push_back(
            slice_payload(resource.lod_geometry_payload, cluster.geometry_payload_offset,
                          cluster.geometry_payload_size));
    }
    const size_t payload_size_before = resource.lod_geometry_payload.size();

    meridian::detail::repack_lod_cluster_payloads(resource);

    const uint32_t expected_offsets[4] = {0, 100, 200, 300};
    for (size_t i = 0; i < 4; ++i) {
        check(resource.lod_clusters[i].geometry_payload_offset == expected_offsets[i],
              "repacked offset does not follow final cluster-table order");
    }
    for (size_t i = 0; i < 4; ++i) {
        const std::vector<std::byte> after = slice_payload(
            resource.lod_geometry_payload, resource.lod_clusters[i].geometry_payload_offset,
            resource.lod_clusters[i].geometry_payload_size);
        check(after == bytes_before[i], "repack changed a cluster's payload bytes");
    }
    check(resource.lod_geometry_payload.size() == payload_size_before,
          "repack changed the total payload size");
}

void test_repack_noop_when_ordered() {
    meridian::VGeoResource resource = make_test_resource({0, 100, 200, 300});
    const std::vector<std::byte> payload_before = resource.lod_geometry_payload;
    std::vector<uint32_t> offsets_before;
    for (const meridian::LodClusterRecord& cluster : resource.lod_clusters) {
        offsets_before.push_back(cluster.geometry_payload_offset);
    }

    meridian::detail::repack_lod_cluster_payloads(resource);

    check(resource.lod_geometry_payload == payload_before,
          "repack was not a byte-for-byte no-op on already-ordered clusters");
    for (size_t i = 0; i < offsets_before.size(); ++i) {
        check(resource.lod_clusters[i].geometry_payload_offset == offsets_before[i],
              "repack rewrote offsets on already-ordered clusters");
    }
}

void test_repack_empty_clears_payload() {
    meridian::VGeoResource resource = make_test_resource({});
    resource.lod_geometry_payload.assign(64, std::byte{0x11});
    meridian::detail::repack_lod_cluster_payloads(resource);
    check(resource.lod_geometry_payload.empty(),
          "repack did not clear the payload when no lod clusters exist");
}

void test_pages_fail_loudly_on_permuted_order() {
    // Table order whose first cluster sits physically after its last:
    // the pre-fix page builder computed last_end - first_offset on uint32
    // and shipped a wrapped range; it must now throw instead.
    meridian::VGeoResource resource = make_test_resource({200, 0});
    bool threw = false;
    try {
        (void)meridian::detail::build_lod_pages(resource.lod_clusters, 2, 1);
    } catch (const meridian::BuilderError&) {
        threw = true;
    }
    check(threw, "build_lod_pages did not reject non-contiguous cluster order");
}

void test_pipeline_validates_across_page_limits() {
    for (uint32_t page_cluster_limit : {1u, 2u, 3u, 4u, 8u}) {
        meridian::VGeoResource resource = make_test_resource({0, 300, 100, 200});
        finish_lod_packing(resource, page_cluster_limit, true);
        check_pages_contain_clusters(resource);
        check_global_adjacency(resource);
        try {
            meridian::detail::validate_resource(resource);
        } catch (const meridian::BuilderError& error) {
            std::fprintf(stderr, "FAIL: validate_resource threw at page limit %u: %s\n",
                         page_cluster_limit, error.what());
            ++g_failures;
        }
    }
}

}  // namespace

int main() {
    test_repack_permuted_offsets();
    test_repack_noop_when_ordered();
    test_repack_empty_clears_payload();
    test_pages_fail_loudly_on_permuted_order();
    test_pipeline_validates_across_page_limits();

    if (g_failures != 0) {
        std::fprintf(stderr, "builder tests: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("builder tests: all passed\n");
    return 0;
}
