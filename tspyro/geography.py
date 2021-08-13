import itertools
import operator

import numpy as np
import torch


def radians_center_weighted(x, y, z, weights):
    total_weight = np.sum(weights)
    weighted_avg_x = np.sum(weights * np.array(x)) / total_weight
    weighted_avg_y = np.sum(weights * np.array(y)) / total_weight
    weighted_avg_z = np.sum(weights * np.array(z)) / total_weight
    central_longitude = np.arctan2(weighted_avg_y, weighted_avg_x)
    central_square_root = np.sqrt(
        weighted_avg_x * weighted_avg_x + weighted_avg_y * weighted_avg_y
    )
    central_latitude = np.arctan2(weighted_avg_z, central_square_root)
    return central_latitude, central_longitude


def vectorized_weighted_geographic_center(lat_arr, long_arr, weights):
    lat_arr = np.radians(lat_arr)
    long_arr = np.radians(long_arr)
    x = np.cos(lat_arr) * np.cos(long_arr)
    y = np.cos(lat_arr) * np.sin(long_arr)
    z = np.sin(lat_arr)
    if len(weights.shape) > 1:
        total_weights = np.sum(weights, axis=1)
    else:
        total_weights = np.sum(weights)
    weighted_avg_x = np.sum(weights * x, axis=1) / total_weights
    weighted_avg_y = np.sum(weights * y, axis=1) / total_weights
    weighted_avg_z = np.sum(weights * z, axis=1) / total_weights
    central_longitude = np.arctan2(weighted_avg_y, weighted_avg_x)
    central_sqrt = np.sqrt((weighted_avg_x ** 2) + (weighted_avg_y ** 2))
    central_latitude = np.arctan2(weighted_avg_z, central_sqrt)
    return np.degrees(central_latitude), np.degrees(central_longitude)


def weighted_geographic_center(lat_list, long_list, weights):
    x = list()
    y = list()
    z = list()
    if len(lat_list) == 1 and len(long_list) == 1:
        return (lat_list[0], long_list[0])
    lat_radians = np.radians(lat_list)
    long_radians = np.radians(long_list)
    x = np.cos(lat_radians) * np.cos(long_radians)
    y = np.cos(lat_radians) * np.sin(long_radians)
    z = np.sin(lat_radians)
    weights = np.array(weights)
    central_latitude, central_longitude = radians_center_weighted(x, y, z, weights)
    return (np.degrees(central_latitude), np.degrees(central_longitude))


def get_parent_age(self, edge):
    times = self.ts.tables.nodes.time[:]
    return times[operator.attrgetter("parent")(edge)]


def edges_by_parent_age_asc(self):
    return itertools.groupby(self.ts.edges(), self.get_parent_age)


def edges_by_parent_asc(ts):
    """
    Return an itertools.groupby object of edges grouped by parent in ascending order
    of the time of the parent. Since tree sequence properties guarantee that edges
    are listed in nondecreasing order of parent time
    (https://tskit.readthedocs.io/en/latest/data-model.html#edge-requirements)
    we can simply use the standard edge order
    """
    return itertools.groupby(ts.edges(), operator.attrgetter("parent"))


def parents_in_epoch(parents):
    return itertools.groupby(parents, operator.attrgetter("parent"))


def edge_span(edge):
    return edge.right - edge.left


def average_edges(parent_edges, locations):
    parent = parent_edges[0]
    edges = parent_edges[1]

    child_spanfracs = list()
    child_lats = list()
    child_longs = list()

    for edge in edges:
        child_spanfracs.append(edge_span(edge))
        child_lats.append(locations[edge.child][0])
        child_longs.append(locations[edge.child][1])
    val = weighted_geographic_center(
        child_lats, child_longs, np.ones_like(len(child_lats))
    )
    return parent, val


def get_ancestral_geography(ts, sample_locations, show_progress=False):
    """
    Use dynamic programming to find approximate posterior to sample from
    """
    locations = np.zeros((ts.num_nodes, 2))
    locations[ts.samples()] = sample_locations
    fixed_nodes = set(ts.samples())

    # Iterate through the nodes via groupby on parent node
    for parent_edges in edges_by_parent_asc(ts):
        if parent_edges[0] not in fixed_nodes:
            parent, val = average_edges(parent_edges, locations)
            locations[parent] = val
    return torch.tensor(locations[ts.num_samples :])  # noqa
