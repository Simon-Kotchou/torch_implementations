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