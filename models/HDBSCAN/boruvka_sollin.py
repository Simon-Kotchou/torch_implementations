import torch

class BoruvkaUnionFind:
    def __init__(self, size):
        self.parent = torch.arange(size)
        self.rank = torch.zeros(size, dtype=torch.long)
        self.is_component = torch.ones(size, dtype=bool)

    def find(self, x):
        mask = self.parent[x] != x
        self.parent[x][mask] = self.find(self.parent[x][mask])
        return self.parent[x]

    def union(self, x, y):
        root_x, root_y = self.find(x), self.find(y)
        mask = root_x != root_y
        x, y = x[mask], y[mask]
        root_x, root_y = root_x[mask], root_y[mask]

        self.is_component[root_x] = self.is_component[root_y] = False
        self.parent[root_x] = torch.where(self.rank[root_x] < self.rank[root_y], root_y, root_x)
        self.parent[root_y] = torch.where(self.rank[root_x] > self.rank[root_y], root_x, root_y)
        self.rank[root_x] += (self.rank[root_x] == self.rank[root_y]).long()

    def components(self):
        return self.is_component.nonzero().view(-1)

class KDTreeBoruvkaAlgorithm:
    def __init__(self, tree, min_samples=5, metric='euclidean', leaf_size=20,
                 alpha=1.0, approx_min_span_tree=False, n_jobs=4, **kwargs):
        self.tree = tree
        self.min_samples = min_samples
        self.alpha = alpha
        self.approx_min_span_tree = approx_min_span_tree
        self.n_jobs = n_jobs

        self.n_points = self.tree.data.shape[0]
        self.n_nodes = self.tree.node_data.shape[0]

        self.components = torch.arange(self.n_points)
        self.bounds = torch.full((self.n_nodes,), float('inf'))
        self.component_of_point = torch.empty(self.n_points, dtype=torch.long)
        self.component_of_node = torch.empty(self.n_nodes, dtype=torch.long)
        self.candidate_neighbor = torch.full((self.n_points,), -1, dtype=torch.long)
        self.candidate_point = torch.full((self.n_points,), -1, dtype=torch.long)
        self.candidate_distance = torch.full((self.n_points,), float('inf'))

        self.component_union_find = BoruvkaUnionFind(self.n_points)

        self.edges = torch.empty((self.n_points - 1, 3))
        self.num_edges = 0

        self._initialize_components()
        self._compute_bounds()

    @torch.jit.script_method
    def _compute_bounds(self):
        knn_dist, knn_indices = torch.jit.annotate(List[Tensor], self.tree.query(self.tree.data, k=self.min_samples + 1))
        self.core_distance = knn_dist[:, -1]
        
        mask = (knn_indices != torch.arange(self.n_points).unsqueeze(1)).any(1)
        self.candidate_point = torch.where(mask, torch.arange(self.n_points), -1)
        self.candidate_neighbor = knn_indices[torch.arange(self.n_points), (self.core_distance[knn_indices] <= self.core_distance.unsqueeze(1)).argmax(1)]
        self.candidate_distance = torch.where(mask, self.core_distance, float('inf'))

        self.update_components()

    @torch.jit.script_method
    def _initialize_components(self):
        self.component_of_point[:] = torch.arange(self.n_points)
        self.component_of_node[:] = -torch.arange(1, self.n_nodes + 1)

    @torch.jit.script_method
    def update_components(self):
        is_source = self.candidate_point != -1
        is_sink = self.candidate_neighbor != -1
        source, sink = self.candidate_point[is_source], self.candidate_neighbor[is_sink]
        
        source_component = self.component_union_find.find(source)
        sink_component = self.component_union_find.find(sink)
        
        is_same_component = source_component == sink_component
        source, sink = source[~is_same_component], sink[~is_same_component]
        self.edges[self.num_edges:self.num_edges+len(source)] = torch.stack([source, sink, self.candidate_distance[is_source][~is_same_component]], dim=1)
        self.num_edges += len(source)
        
        self.component_union_find.union(source, sink)
            
        if self.num_edges == self.n_points - 1:
            self.components = self.component_union_find.components()
            return len(self.components)

        self.component_of_point = torch.tensor([self.component_union_find.find(i) for i in range(self.n_points)])

        is_leaf = self.tree.node_data[:, 2].bool()
        leaf_start = self.tree.node_data[is_leaf, 0].long()
        leaf_end = self.tree.node_data[is_leaf, 1].long()
        leaf_component = self.component_of_point[self.tree.idx_array[leaf_start]]
        is_same_leaf_component = (self.component_of_point[self.tree.idx_array[leaf_start[:, None], leaf_end[None, :]]] == leaf_component[:, None]).all(1)
        self.component_of_node[is_leaf] = torch.where(is_same_leaf_component, leaf_component, -1)
        
        for n in range(self.n_nodes - 1, -1, -1):
            if not is_leaf[n]:
                if self.component_of_node[2 * n + 1] == self.component_of_node[2 * n + 2]:
                    self.component_of_node[n] = self.component_of_node[2 * n + 1]

        if self.approx_min_span_tree and len(self.components) == len(self.component_union_find.components()):
            self.bounds[~is_leaf] = float('inf')
        else:
            self.bounds[~is_leaf] = float('inf')
            
        return len(self.components)

    @torch.jit.script_method
    def dual_tree_traversal(self, node1, node2):
        node1_info, node2_info = self.tree.node_data[node1], self.tree.node_data[node2]
        
        if self.bounds[node1] <= self.tree.min_dist(node1_info, node2_info):
            return
        if self.component_of_node[node1] == self.component_of_node[node2] >= 0:
            return

        if node1_info[2] and node2_info[2]:  # Both nodes are leaves
            point_indices1 = self.tree.idx_array[node1_info[0]:node1_info[1]]
            point_indices2 = self.tree.idx_array[node2_info[0]:node2_info[1]]
            i, j = torch.meshgrid(point_indices1, point_indices2)
            dists = self.tree.dist(self.tree.data[i], self.tree.data[j])
            components_i = self.component_of_point[i]
            components_j = self.component_of_point[j]
            mask = components_i != components_j
            dists = dists[mask]
            if not dists.numel():
                return
            min_dist, min_idx = dists.min(0)
            min_i, min_j = i[mask][min_idx], j[mask][min_idx]
            mask = min_dist < self.candidate_distance[components_i[mask][min_idx]]
            self.candidate_distance[components_i[mask][min_idx]] = min_dist[mask]
            self.candidate_neighbor[components_i[mask][min_idx]] = min_j[mask]
            self.candidate_point[components_i[mask][min_idx]] = min_i[mask]
        elif node1_info[2] or (not node2_info[2] and node2_info[3] > node1_info[3]):
            self.dual_tree_traversal(node1, 2 * node2 + 1)
            self.dual_tree_traversal(node1, 2 * node2 + 2)
        else:
            self.dual_tree_traversal(2 * node1 + 1, node2)
            self.dual_tree_traversal(2 * node1 + 2, node2)

    @torch.jit.script_method
    def spanning_tree(self):
        while len(self.components) > 1:
            self.dual_tree_traversal(0, 0)
            self.update_components()

        return self.edges[:self.num_edges]