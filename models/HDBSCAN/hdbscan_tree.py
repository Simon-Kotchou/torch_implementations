import torch

def bfs_from_hierarchy(hierarchy, bfs_root):
    """
    Perform a breadth-first search on a tree in scipy hclust format.
    """
    dim = hierarchy.shape[0]
    max_node = 2 * dim
    num_points = max_node - dim + 1

    to_process = [bfs_root]
    result = []

    while to_process:
        result.extend(to_process)
        to_process = [x - num_points for x in to_process if x >= num_points]
        if to_process:
            to_process = hierarchy[to_process].flatten().long().tolist()

    return result


def condense_tree(hierarchy, min_cluster_size=10):
    """
    Condense a tree according to a minimum cluster size.
    """
    dim = hierarchy.shape[0]
    max_node = 2 * dim
    num_points = max_node - dim + 1

    device = hierarchy.device
    result_arr = torch.empty(0, 4, dtype=torch.double, device=device)

    node_list = bfs_from_hierarchy(hierarchy, max_node)
    relabel = torch.full((max_node + 1,), -1, dtype=torch.long, device=device)
    relabel[max_node] = num_points
    ignore = torch.zeros(len(node_list), dtype=torch.bool, device=device)

    for node in node_list:
        if ignore[node] or node < num_points:
            continue

        children = hierarchy[node - num_points]
        left, right = children[:2].long()
        lambda_value = 1.0 / children[2] if children[2] > 0 else float('inf')

        left_count = hierarchy[left - num_points][3].long() if left >= num_points else 1
        right_count = hierarchy[right - num_points][3].long() if right >= num_points else 1

        if left_count >= min_cluster_size and right_count >= min_cluster_size:
            relabel[left] = num_points
            num_points += 1
            result_arr = torch.cat([result_arr, torch.tensor([[relabel[node], relabel[left], lambda_value, left_count]], dtype=torch.double, device=device)])

            relabel[right] = num_points
            num_points += 1
            result_arr = torch.cat([result_arr, torch.tensor([[relabel[node], relabel[right], lambda_value, right_count]], dtype=torch.double, device=device)])

        elif left_count < min_cluster_size and right_count < min_cluster_size:
            for sub_node in bfs_from_hierarchy(hierarchy, left):
                if sub_node < num_points:
                    result_arr = torch.cat([result_arr, torch.tensor([[relabel[node], sub_node, lambda_value, 1]], dtype=torch.double, device=device)])
                ignore[sub_node] = True

            for sub_node in bfs_from_hierarchy(hierarchy, right):
                if sub_node < num_points:
                    result_arr = torch.cat([result_arr, torch.tensor([[relabel[node], sub_node, lambda_value, 1]], dtype=torch.double, device=device)])
                ignore[sub_node] = True

        elif left_count < min_cluster_size:
            relabel[right] = relabel[node]
            for sub_node in bfs_from_hierarchy(hierarchy, left):
                if sub_node < num_points:
                    result_arr = torch.cat([result_arr, torch.tensor([[relabel[node], sub_node, lambda_value, 1]], dtype=torch.double, device=device)])
                ignore[sub_node] = True

        else:
            relabel[left] = relabel[node]
            for sub_node in bfs_from_hierarchy(hierarchy, right):
                if sub_node < num_points:
                    result_arr = torch.cat([result_arr, torch.tensor([[relabel[node], sub_node, lambda_value, 1]], dtype=torch.double, device=device)])
                ignore[sub_node] = True

    return result_arr


def compute_stability(condensed_tree):
    """
    Compute stability scores for each cluster in the condensed tree.
    """
    _, inverse_indices = torch.unique(condensed_tree[:, 0], return_inverse=True)
    _, counts = torch.unique(inverse_indices, return_counts=True)
    cluster_sizes = counts[inverse_indices]

    lambda_vals = condensed_tree[:, 2]
    child_sizes = condensed_tree[:, 3]

    scaled_lambda_vals = lambda_vals * child_sizes / cluster_sizes[inverse_indices]
    max_lambda_vals, _ = torch.max(scaled_lambda_vals, dim=0)

    stability_scores = torch.bincount(inverse_indices, weights=scaled_lambda_vals)
    max_lambda_vals = torch.bincount(inverse_indices, weights=max_lambda_vals)
    stability_scores /= max_lambda_vals

    return dict(zip(torch.unique(condensed_tree[:, 0]).tolist(), stability_scores.tolist()))

def bfs_from_cluster_tree(tree, bfs_root):
    result = []
    to_process = torch.tensor([bfs_root], dtype=torch.long)

    while to_process.shape[0] > 0:
        result.extend(to_process.tolist())
        to_process = tree[tree[:, 0].isin(to_process), 1]

    return result

def max_lambdas(tree):
    largest_parent = tree[:, 0].max().item()
    deaths = torch.zeros(largest_parent + 1, dtype=torch.double)

    sorted_parent_data, _ = torch.sort(tree[:, [0, 2]], dim=0)
    sorted_parents = sorted_parent_data[:, 0]
    sorted_lambdas = sorted_parent_data[:, 1]

    current_parent = -1
    max_lambda = 0

    for row in range(sorted_parent_data.shape[0]):
        parent = sorted_parents[row].item()
        lambda_ = sorted_lambdas[row].item()

        if parent == current_parent:
            max_lambda = max(max_lambda, lambda_)
        elif current_parent != -1:
            deaths[current_parent] = max_lambda
            current_parent = parent
            max_lambda = lambda_
        else:
            current_parent = parent
            max_lambda = lambda_

    deaths[current_parent] = max_lambda

    return deaths


class TreeUnionFind:
    def __init__(self, size):
        self._data = torch.zeros((size, 2), dtype=torch.long)
        self._data[:, 0] = torch.arange(size)
        self.is_component = torch.ones(size, dtype=torch.bool)

    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)

        if self._data[x_root, 1] < self._data[y_root, 1]:
            self._data[x_root, 0] = y_root
        elif self._data[x_root, 1] > self._data[y_root, 1]:
            self._data[y_root, 0] = x_root
        else:
            self._data[y_root, 0] = x_root
            self._data[x_root, 1] += 1

    def find(self, x):
        if self._data[x, 0] != x:
            self._data[x, 0] = self.find(self._data[x, 0])
            self.is_component[x] = False
        return self._data[x, 0]

    def components(self):
        return self.is_component.nonzero().squeeze()


def labelling_at_cut(linkage, cut, min_cluster_size):
    root = 2 * linkage.shape[0]
    num_points = root // 2 + 1

    result = torch.full((num_points,), -1, dtype=torch.long)
    union_find = TreeUnionFind(root + 1)

    cluster = num_points
    for row in linkage:
        if row[2] < cut:
            union_find.union(row[0].item(), cluster)
            union_find.union(row[1].item(), cluster)
        cluster += 1

    cluster_size = torch.zeros(cluster, dtype=torch.long)
    for n in range(num_points):
        cluster = union_find.find(n)
        cluster_size[cluster] += 1
        result[n] = cluster

    unique_labels = result.unique()
    cluster_label_map = {-1: -1}
    cluster_label = 0

    for cluster in unique_labels:
        if cluster_size[cluster] < min_cluster_size:
            cluster_label_map[cluster.item()] = -1
        else:
            cluster_label_map[cluster.item()] = cluster_label
            cluster_label += 1

    for n in range(num_points):
        result[n] = cluster_label_map[result[n].item()]

    return result


def do_labelling(tree, clusters, cluster_label_map, allow_single_cluster, cluster_selection_epsilon, match_reference_implementation):
    child_array = tree[:, 1]
    parent_array = tree[:, 0]
    lambda_array = tree[:, 2]

    root_cluster = parent_array.min().item()
    result = torch.full((root_cluster,), -1, dtype=torch.long)
    union_find = TreeUnionFind(parent_array.max().item() + 1)

    for n in range(tree.shape[0]):
        child = child_array[n].item()
        parent = parent_array[n].item()
        if child not in clusters:
            union_find.union(parent, child)

    for n in range(root_cluster):
        cluster = union_find.find(n)
        if cluster < root_cluster:
            result[n] = -1
        elif cluster == root_cluster:
            if len(clusters) == 1 and allow_single_cluster:
                if cluster_selection_epsilon != 0.0:
                    if lambda_array[child_array == n][0] >= 1 / cluster_selection_epsilon:
                        result[n] = cluster_label_map[cluster]
                    else:
                        result[n] = -1
                elif lambda_array[child_array == n][0] >= lambda_array[parent_array == cluster].max():
                    result[n] = cluster_label_map[cluster]
                else:
                    result[n] = -1
            else:
                result[n] = -1
        else:
            if match_reference_implementation:
                point_lambda = lambda_array[child_array == n][0].item()
                cluster_lambda = lambda_array[child_array == cluster][0].item()
                if point_lambda > cluster_lambda:
                    result[n] = cluster_label_map[cluster]
                else:
                    result[n] = -1
            else:
                result[n] = cluster_label_map[cluster]

    return result


def get_probabilities(tree, cluster_map, labels):
    child_array = tree[:, 1]
    parent_array = tree[:, 0]
    lambda_array = tree[:, 2]

    result = torch.zeros(labels.shape[0], dtype=torch.double)
    deaths = max_lambdas(tree)
    root_cluster = parent_array.min().item()

    for n in range(tree.shape[0]):
        point = child_array[n].item()
        if point >= root_cluster:
            continue

        cluster_num = labels[point].item()

        if cluster_num == -1:
            continue

        cluster = cluster_map[cluster_num]
        max_lambda = deaths[cluster].item()
        if max_lambda == 0.0 or not torch.isfinite(lambda_array[n]):
            result[point] = 1.0
        else:
            lambda_ = min(lambda_array[n].item(), max_lambda)
            result[point] = lambda_ / max_lambda

    return result


def labelling_at_cut(linkage, cut, min_cluster_size):
    """
    Given a single linkage tree and a cut value, return the cluster labels at that cut.
    """
    device = linkage.device
    num_points = linkage.shape[0] + 1
    result_arr = torch.full((num_points,), -1, dtype=torch.long, device=device)

    cluster = num_points
    for row in linkage:
        if row[2] < cut:
            result_arr[row[:2].long()] = cluster
        cluster += 1

    unique_labels, counts = torch.unique(result_arr, return_counts=True)
    for label, count in zip(unique_labels, counts):
        if count < min_cluster_size:
            result_arr[result_arr == label] = -1

    return result_arr


def outlier_scores(tree):
    """
    Generate GLOSH outlier scores from a condensed tree.
    """
    num_points = tree[:, 0].min().long()
    result = torch.zeros(num_points, dtype=torch.double)

    deaths = torch.zeros(tree[:, 1].max().long() + 1, dtype=torch.double)
    deaths[tree[:, 1].long()] = tree[:, 2]

    parents = tree[:, 0].long()
    lambdas = tree[:, 2]

    for n in range(tree.shape[0] - 1, -1, -1):
        parent = parents[n]
        if parent >= num_points:
            deaths[parent] = max(deaths[parent], deaths[tree[n, 1].long()])

    for n in range(tree.shape[0]):
        point = tree[n, 1].long()
        if point < num_points:
            max_lambda = deaths[parents[n]]
            if torch.isfinite(max_lambda) and max_lambda > 0:
                result[point] = (max_lambda - lambdas[n]) / max_lambda

    return result

def recurse_leaf_dfs(cluster_tree, current_node):
    children = cluster_tree[cluster_tree[:, 0] == current_node, 1]

    if children.numel() == 0:
        return [current_node]
    else:
        return sum([recurse_leaf_dfs(cluster_tree, child.item()) for child in children], [])


def get_cluster_tree_leaves(cluster_tree):
    if cluster_tree.shape[0] == 0:
        return []

    root = cluster_tree[:, 0].min().item()
    return recurse_leaf_dfs(cluster_tree, root)


def traverse_upwards(cluster_tree, cluster_selection_epsilon, leaf, allow_single_cluster):
    root = cluster_tree[:, 0].min().item()
    parent = cluster_tree[cluster_tree[:, 1] == leaf, 0].item()

    if parent == root:
        if allow_single_cluster:
            return parent
        else:
            return leaf  # return node closest to root

    parent_eps = 1 / cluster_tree[cluster_tree[:, 1] == parent, 2].item()

    if parent_eps > cluster_selection_epsilon:
        return parent
    else:
        return traverse_upwards(cluster_tree, cluster_selection_epsilon, parent, allow_single_cluster)


def epsilon_search(leaves, cluster_tree, cluster_selection_epsilon, allow_single_cluster):
    selected_clusters = []
    processed = []

    for leaf in leaves:
        eps = 1 / cluster_tree[cluster_tree[:, 1] == leaf, 2].item()

        if eps < cluster_selection_epsilon:
            if leaf not in processed:
                epsilon_child = traverse_upwards(cluster_tree, cluster_selection_epsilon, leaf, allow_single_cluster)
                selected_clusters.append(epsilon_child)

                for sub_node in bfs_from_cluster_tree(cluster_tree, epsilon_child):
                    if sub_node != epsilon_child:
                        processed.append(sub_node.item())
        else:
            selected_clusters.append(leaf)

    return set(selected_clusters)


def get_clusters(tree, stability, cluster_selection_method='eom', allow_single_cluster=False, match_reference_implementation=False):
    """
    Given a tree and stability dict, produce the cluster labels and probabilities for a flat clustering.
    """
    device = tree.device
    node_list = sorted(stability.keys(), reverse=True)
    if not allow_single_cluster:
        node_list = node_list[:-1]

    cluster_tree = tree[tree[:, 3] > 1]
    is_cluster = {cluster: True for cluster in node_list}
    num_points = tree[tree[:, 3] == 1, 1].max().long() + 1
    max_lambda = tree[:, 2].max()

    for node in node_list:
        if cluster_selection_method == 'eom':
            subtree_stability = stability[node]
            if subtree_stability > stability[node]:
                is_cluster[node] = False
                stability[node] = subtree_stability
            else:
                for sub_node in bfs_from_hierarchy(cluster_tree, node):
                    if sub_node != node:
                        is_cluster[sub_node] = False
        elif cluster_selection_method == 'leaf':
            if node not in set(get_cluster_tree_leaves(cluster_tree)):
                is_cluster[node] = False
        else:
            raise ValueError('Invalid cluster selection method')

    clusters = set(c for c in is_cluster if is_cluster[c])
    cluster_map = {c: n for n, c in enumerate(sorted(clusters))}
    reverse_cluster_map = {n: c for c, n in cluster_map.items()}

    result_arr = torch.full((num_points,), -1, dtype=torch.long, device=device)
    root_cluster = tree[:, 0].min().long()

    for n in range(tree.shape[0]):
        point = tree[n, 1].long()
        if point < num_points:
            cluster = tree[n, 0].long()
            if cluster not in clusters:
                cluster = root_cluster
            if match_reference_implementation:
                if tree[n, 2] > tree[cluster - num_points, 2]:
                    result_arr[point] = cluster_map[cluster]
            else:
                result_arr[point] = cluster_map[cluster]

    labels = result_arr
    probs = torch.zeros(num_points, dtype=torch.double, device=device)

    for n in range(tree.shape[0]):
        point = tree[n, 1].long()
        if point < num_points:
            cluster = reverse_cluster_map[labels[point].item()]
            if max_lambda > 0 and torch.isfinite(tree[n, 2]):
                probs[point] = min(tree[n, 2], stability[cluster]) / max_lambda

    return labels, probs, list(clusters)


def get_stability_scores(labels, clusters, stability, max_lambda):
    """
    Compute stability scores for each cluster.
    """
    scores = torch.zeros(len(clusters), dtype=torch.double)

    for n, c in enumerate(clusters):
        cluster_size = (labels == n).sum()
        if torch.isfinite(max_lambda) and max_lambda > 0 and cluster_size > 0:
            scores[n] = stability[c] / (cluster_size * max_lambda)
        else:
            scores[n] = 1.0

    return scores