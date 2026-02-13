# imports
import csv
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# use convolved shape features around a neighbourhood
# find the average curvature around a neighbourhood of points
# contains most images from report

def make_connected_graph(file_path):
    """
    Read graph from csv file and make a connected graph

    input:
    file_path: path to csv file of neighbour information

    returns:
    graph
    """
    graph = {}
    with open(file_path, newline='') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            parts = row[0].split()
            node_id = int(parts[0])
            num_neighbors = int(parts[1])
            neighbors = list(map(int, parts[2:2 + num_neighbors]))
            distances = list(map(float, parts[2 + num_neighbors:2 + 2 * num_neighbors]))

            # Add the node and its neighbors to the graph
            if node_id not in graph:
                graph[node_id] = {}
            for neighbor, distance in zip(neighbors, distances):
                if neighbor not in graph[node_id]:
                    graph[node_id][neighbor] = distance
                if neighbor not in graph:
                    graph[neighbor] = {}
                if node_id not in graph[neighbor]:
                    graph[neighbor][node_id] = distance

    # Convert the nested dictionary to a list of tuples for each node
    for node in graph:
        graph[node] = list(graph[node].items())

    return graph

def average_local_curvature(graph, curvature_df):
    """"
    convolve curvature by taking average of neighbourhood (box blur method)

    input:
    graph: connected graph dictionary made by make_connected_graph
    curvature_df: dataframe containing curvature information per cell

    returns: average curvature per cell as a dictionary
    """
    curvature_dict = curvature_df.set_index('id')[['curv1', 'curv2']].to_dict('index')
    average_curvature = {}

    for node, neighbors in graph.items():
        total_curv1 = 0
        total_curv2 = 0
        neighbor_count = 0

        for neighbor, _ in neighbors:
            if neighbor in curvature_dict:
                total_curv1 += curvature_dict[neighbor]['curv1']
                total_curv2 += curvature_dict[neighbor]['curv2']
                neighbor_count += 1

        # normalise over all neighbours
        if neighbor_count > 0:
            average_curvature[node] = {
                'avg_curv1': total_curv1 / neighbor_count,
                'avg_curv2': total_curv2 / neighbor_count
            }
        else: # if no neighbours
            average_curvature[node] = {
                'avg_curv1': None,
                'avg_curv2': None
            }

    average_curvature_df = pd.DataFrame.from_dict(average_curvature, orient='index').reset_index()
    average_curvature_df.columns = ['id', 'avg_curv1', 'avg_curv2']

    return average_curvature_df

def edge_kernel(graph, curvature_df):
    """"
    convolve curvature using edge filter

    input:
    graph: connected graph dictionary made by make_connected_graph
    curvature_df: dataframe containing curvature information per cell

    returns: convolved edge kernel curvature
    """
    curvature_dict = curvature_df.set_index('id')[['cmax', 'cmin']].to_dict('index')
    local_curvature = {}

    for node, neighbors in graph.items():
        num_neighbors = len(neighbors)
        central_value = num_neighbors  # The center value is the number of neighbors

        total_cmax = 0
        total_cmin = 0

        # central node has highest contribution
        if node in curvature_dict:
            total_cmax += curvature_dict[node]['cmax'] * central_value
            total_cmin += curvature_dict[node]['cmin'] * central_value

        # Neighbors contribution
        for neighbor, distance in neighbors:
            if neighbor in curvature_dict:
                total_cmax += curvature_dict[neighbor]['cmax'] * -1
                total_cmin += curvature_dict[neighbor]['cmin'] * -1

        local_curvature[node] = {
            'weighted_cmax': total_cmax,
            'weighted_cmin': total_cmin
        }

    average_curvature_df = pd.DataFrame.from_dict(local_curvature, orient='index').reset_index()
    average_curvature_df.columns = ['id', 'avg_cmax', 'avg_cmin']

    return average_curvature_df

def gaussian_weight(distance, sigma):
    """
    gaussian function needed to calculate weight for gaussian kernel

    inputs:
    distance: distance between two neighbors
    sigma: int, specifying degree of blur
    """
    norm = 1/ (2*np.pi * sigma **2)
    exp = np.exp(- (distance ** 2) / (2 * sigma ** 2))
    return exp*norm

def gauss_kernel(graph, curvature_df, sigma=7):
    """"
    convolve curvature through gaussian kernel. weight of each cell calculated by continous gaussian function.

    input:
    graph: connected graph dictionary made by make_connected_graph
    curvature_df: dataframe containing curvature information per cell
    sigma: int, specifying degree of blur for weight calculation

    returns: average curvature per cell as a dictionary
    """
    curvature_dict = curvature_df.set_index('id')[['cmax', 'cmin']].to_dict('index')
    local_curvature = {}

    for node, neighbors in graph.items():
        total_cmax = 0
        total_cmin = 0
        weight_sum = 0

        # Central node contribution
        if node in curvature_dict:
            weight = gaussian_weight(0, sigma)  # distance is 0 for the central node
            total_cmax += curvature_dict[node]['cmax'] * weight
            total_cmin += curvature_dict[node]['cmin'] * weight
            weight_sum += weight

        # Neighbors contribution
        for neighbor, distance in neighbors:
            if neighbor in curvature_dict:
                weight = gaussian_weight(distance, sigma)
                total_cmax += curvature_dict[neighbor]['cmax'] * weight
                total_cmin += curvature_dict[neighbor]['cmin'] * weight
                weight_sum += weight

        local_curvature[node] = {
            'weighted_cmax': total_cmax / weight_sum if weight_sum != 0 else None,
            'weighted_cmin': total_cmin / weight_sum if weight_sum != 0 else None
        }

    average_curvature_df = pd.DataFrame.from_dict(local_curvature, orient='index').reset_index()
    average_curvature_df.columns = ['id', 'avg_cmax', 'avg_cmin']

    return average_curvature_df

def neighbourhood_curvature_features(df, convolved_shape):
    """
    Add neighbourhood features using convolved curvature features.

    inputs:
    df (pd.DataFrame): df containing shape information - particularly vol, distance to center
    convolved_shape (pd.DataFrame): dataframe with new cmin and cmax values
    features_of_interest (list): features to take from original dataframe

    Rreturns
    dataframe with all shape features from new curvature, with volume and distance to center and id
    """
    # new shape features
    convolved_shape['mean_curvature'] = (convolved_shape['avg_cmin'] + convolved_shape['avg_cmax']) / 2
    convolved_shape['deviator'] = (convolved_shape['avg_cmax'] - convolved_shape['avg_cmin']) / 2
    convolved_shape['gaussian'] = convolved_shape['avg_cmax'] * convolved_shape['avg_cmin']

    # features to inherit from original df
    features_of_interest = ["id", "vol", "distance_to_center"]
    selected_features_df = df[features_of_interest]
    merged_df = pd.merge(selected_features_df, convolved_shape, on='id', how='left')

    return merged_df
