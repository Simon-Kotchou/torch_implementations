import torch

class BoruvkaUnionFind:
    def __init__(self, size):
        self.parent = torch.arange(size)
        self.rank = torch.zeros(size, dtype=torch.long)
        self.is_component = torch.ones(size, dtype=bool)

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
            self.is_component[root_x] = False
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
            self.is_component[root_y] = False
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
            self.is_component[root_y] = False

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

    def _compute_bounds(self):
        knn_dist, knn_indices = self.tree.query(self.tree.data, k=self.min_samples + 1)
        self.core_distance = knn_dist[:, -1]
        
        for i in range(self.n_points):
            for j in range(self.min_samples + 1):
                if knn_indices[i, j] == i:
                    continue
                if self.core_distance[knn_indices[i, j]] <= self.core_distance[i]:
                    self.candidate_point[i] = i
                    self.candidate_neighbor[i] = knn_indices[i, j]
                    self.candidate_distance[i] = self.core_distance[i]
                    break

        self.update_components()

    def _initialize_components(self):
        self.component_of_point[:] = torch.arange(self.n_points)
        self.component_of_node[:] = -torch.arange(1, self.n_nodes + 1)

    def update_components(self):
        for c in self.components:
            source, sink = self.candidate_point[c], self.candidate_neighbor[c]
            if source == -1 or sink == -1:
                continue
            if self.component_union_find.find(source) == self.component_union_find.find(sink):
                self.candidate_point[c] = self.candidate_neighbor[c] = -1
                self.candidate_distance[c] = float('inf')
                continue
            self.edges[self.num_edges] = torch.tensor([source, sink, self.candidate_distance[c]])
            self.num_edges += 1

            self.component_union_find.union(source, sink)

            if self.num_edges == self.n_points - 1:
                self.components = self.component_union_find.components()
                return len(self.components)

        self.component_of_point[:] = torch.tensor([self.component_union_find.find(i) for i in range(self.n_points)])

        for n in range(self.n_nodes - 1, -1, -1):
            if self.tree.node_data[n, 2]:  # is_leaf
                c = self.component_of_point[self.tree.idx_array[self.tree.node_data[n, 0]]]
                if (self.component_of_point[self.tree.idx_array[self.tree.node_data[n, 0]:self.tree.node_data[n, 1]]] == c).all():
                    self.component_of_node[n] = c
            else:
                if self.component_of_node[2 * n + 1] == self.component_of_node[2 * n + 2]:
                    self.component_of_node[n] = self.component_of_node[2 * n + 1]

        if self.approx_min_span_tree and len(self.components) == len(self.component_union_find.components()):
            self.bounds[:] = float('inf')
        else:
            self.bounds[:] = float('inf')

        return len(self.components)

    def dual_tree_traversal(self, node1, node2):
        node1_info, node2_info = self.tree.node_data[node1], self.tree.node_data[node2]
        
        if self.bounds[node1] <= self.tree.min_dist(node1_info, node2_info):
            return
        if self.component_of_node[node1] == self.component_of_node[node2] >= 0:
            return

        if node1_info[2] and node2_info[2]:  # Both nodes are leaves
            point_indices1 = self.tree.idx_array[node1_info[0]:node1_info[1]]
            point_indices2 = self.tree.idx_array[node2_info[0]:node2_info[1]]
            for i in point_indices1:
                for j in point_indices2:
                    if self.component_of_point[i] == self.component_of_point[j]:
                        continue
                    d = self.tree.dist(self.tree.data[i], self.tree.data[j])
                    if d < self.candidate_distance[self.component_of_point[i]]:
                        self.candidate_distance[self.component_of_point[i]] = d
                        self.candidate_neighbor[self.component_of_point[i]] = j
                        self.candidate_point[self.component_of_point[i]] = i
        elif node1_info[2] or (not node2_info[2] and node2_info[3] > node1_info[3]):
            self.dual_tree_traversal(node1, 2 * node2 + 1)
            self.dual_tree_traversal(node1, 2 * node2 + 2)
        else:
            self.dual_tree_traversal(2 * node1 + 1, node2)
            self.dual_tree_traversal(2 * node1 + 2, node2)

    def spanning_tree(self):
        while len(self.components) > 1:
            self.dual_tree_traversal(0, 0)
            self.update_components()

        return self.edges[:self.num_edges]