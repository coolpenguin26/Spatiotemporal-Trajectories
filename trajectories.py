#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 12:28:24 2024

@author: juliannanowaczek
"""

# imports
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from matplotlib.lines import Line2D

from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colorbar import ColorbarBase
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from mk_lins import get_inverted_linss, sequence_linss
from cluster_maps import shape_cluster_no, gene_cluster_no, shape_cluster_cell_type, gene_cluster_cell_type, gene_cluster_transitions, shape_cluster_transitions


# Initialise global variables

# use functions from mk_lins to generate lineage dictionary
all_dictionary = get_inverted_linss()
filtered_132 = pd.read_csv("data/FM1/csv/clustered_on_all/t132_clustered.csv")
gene_dict,shape_dict = sequence_linss(all_dictionary, dataframe = filtered_132, by_cluster=True)

# specify timesteps
times = ("96","104","112","120","128","132")
gene_features = ['AG','AHP6','ANT','AP1','AP2','AP3','AS1','CLV3','CUC1_2_3','ETTIN','FIL','LFY','MP','PHB_PHV','PI','PUCHI','REV','SEP1','SEP2','SEP3','STM','SUP','SVP','WUS']
gene_features_subset = ['AG','CLV3','CUC1_2_3','LFY']

# dictionary indicating what index of the lineage correlates to which timestep (lineage generated in reverse time order)
lineage_index = {"132":0,"128":1,"120":2,"112":3,"104":4,"96":5}

colours = {0: cm.Purples(np.linspace(0, 1, len(times))),
            1: cm.Greens(np.linspace(0, 1, len(times))),
            2: cm.Oranges(np.linspace(0, 1, len(times))),
            3: cm.Reds(np.linspace(0, 1, len(times))),
            4: cm.Greys(np.linspace(0, 1, len(times))),
            5: cm.Blues(np.linspace(0, 1, len(times)))}



# TRAJECTORY AND CORRELATION FUNCTIONS

def cell_level_trajectory(space_dict, shape = False, gene = False, avg_method="mean"):
    """
    Uses cluster assignment at final step to trace back the ancestors of the cells in each cluster
    Collects cell data and computes averages of data for ancestors at each timestep, then stores in output dictionary

    Parameters
    ----------
    space_dict (dict): gene or space dictionary - ancestor lists of each cell id, separated by cluster assignment in final timestep
    shape (boolean): true if working with shape space
    gene (boolean): true if working with gene space
    avg_method : method to take average values, mean or median

    Returns
    -------
    cluster_pc_values : dictonary of principal component and gene expression data of each cell at each timepoint, separated by the final cluster they evolve into
    avg_values : dictionary of time-average PC and gene expression data of cells that evolve into each final cluster

    """
    
    #initialize dictionary to store data of cells within each cluster
    cluster_pc_values = {
        cluster: {t: [] for t in times} 
        for cluster in space_dict.keys()
    }

    #initialize dictionary to store the average of data of cells within each cluster
    avg_values = {
        cluster: {gene: [] for gene in ['PC1','PC2','shape'] + gene_features
        } for cluster in space_dict.keys()
    }
    
    for timestep in times:
        # Load the CSV file for the current timestep
        csv_file = f"data/FM1/csv/clustered_on_all/t{timestep}_clustered.csv"
        df = pd.read_csv(csv_file)
    
        # Iterate over each cluster
        for cluster, lineage_dict in space_dict.items():
            for cell_id, lineage in lineage_dict.items(): #lineage is reverse list of ancestors
            
                # fetch row in dataframe that corresponds to current cell_id's ancestor at this timestep
                row = df.loc[df['id'] == lineage[lineage_index[timestep]]].iloc[0]
                # store PC values for this ancestor cell
                if shape:
                    cell_data = {'cell_id': int(row['id']), 'PC1': float(row['shape_PC1_value']), 'PC2': float(row['shape_PC2_value'])}
                elif gene:
                    cell_data = {'cell_id': int(row['id']), 'PC1': float(row['gene_PC1_value']), 'PC2': float(row['gene_PC2_value'])}
                
                # store gene expression levels for this ancestor cell
                for gene in gene_features:
                    cell_data[gene] = float(row[gene])
                    
                # commit all cell data to the dictionary   
                cluster_pc_values[cluster][timestep].append(cell_data)
    
            # Extract values for averaging
            PC1_values = [point['PC1'] for point in cluster_pc_values[cluster][timestep]]
            PC2_values = [point['PC2'] for point in cluster_pc_values[cluster][timestep]]

            # Compute mean or median
            avg_func = np.mean if avg_method == "mean" else np.median
            avg_PC1, avg_PC2 = avg_func(PC1_values), avg_func(PC2_values)
            
            # Average shape measure
            avg_shape = abs(avg_PC1) + abs(avg_PC2)

            # Store average values for this cluster in dictionary
            avg_values[cluster]['PC1'].append(avg_PC1)
            avg_values[cluster]['PC2'].append(avg_PC2)
            avg_values[cluster]['shape'].append(avg_shape)
            
            # Repeat average process for genes
            for gene in gene_features:
                gene_values = [point[gene] for point in cluster_pc_values[cluster][timestep]]
                avg_values[cluster][gene].append(avg_func(gene_values))
           

    return cluster_pc_values, avg_values 
    

def cell_type_level_trajectory(cluster_no,space,backfill):
    """
    Uses cluster assignment at each timestep to compute a trajectory
    

    Parameters
    ----------
    shape (boolean): true if working with shape space
    gene (boolean): true if working with gene space

    Returns
    -------
    cluster_means (dict): contains the mean PC1 and PC2 value of each cluster at each timepoint

    """


    # Initialize a dictionary to store the means for each cluster at each timepoint
    cluster_means = {t: {cluster: {'PC1': [], 'PC2': []} for cluster in range(cluster_no[str(t)])} for t in cluster_no}

    cluster_assignments = {
    cluster: {t: [] for t in times}
    for cluster in range(cluster_no["132"])
    }

    plt.figure(figsize=(8, 6),dpi=300)

    # Process data for each timepoint
    for timestep in cluster_no.keys():
        csv_file = f"data/FM1/csv/clustered_on_all/t{timestep}_clustered.csv"
        df = pd.read_csv(csv_file)

        for cluster in range(0, cluster_no[timestep]):
            # Select cells for the current cluster at this timepoint
            cluster_cells = df[df[f'{space}_cluster_gmm'] == cluster]
            
            cluster_means[timestep][cluster]['PC1'] = float(cluster_cells[f'{space}_PC1_value'].mean())
            cluster_means[timestep][cluster]['PC2'] = float(cluster_cells[f'{space}_PC2_value'].mean())
            
            cluster_assignments[cluster][timestep] = [{'cell_id': int(cid)} for cid in cluster_cells['id']]

    if backfill:
        if space == "shape":
            cluster_assignments = backfill_assignments(cluster_assignments, shape_cluster_transitions)
        else:
            cluster_assignments = backfill_assignments(cluster_assignments, gene_cluster_transitions)


    return cluster_assignments, cluster_means


def backfill_assignments(cluster_assignments, cluster_transitions):
    
    for parent, child in cluster_transitions.items():
        if len(child) > 1:
            
            last_backfill = parent[0]
            parent_cluster = parent[1]
            child_cluster = child[1][1]
            
            #print(f"time:{last_backfill},parent:{parent_cluster},child:{child_cluster}")
            
            for time in times:
                if int(time) <= int(last_backfill):
                    cluster_assignments[child_cluster][time] = cluster_assignments[parent_cluster][time].copy()
            
    return cluster_assignments
    

def compute_triple_correlation(avg_values):
    """
    Correlation of PCs, shape measure, and gene expression between different cell-types
    Returns difference, normalised difference, and fold difference between the three pairwise shape regions at each time step.

    Parameters
    ----------
    avg_values (dict): Averages of cell data across timestep and clusters - points for trajectory

    Returns
    -------
    raw_gene_corr (dict): absolute difference in gene expression values of pairwise regions at each timestep for each gene
    raw_shape_corr (dict): absolute difference in overall shape measure of pairwise regions at each timestep
    normalised_gene_corr (dict): raw dictionary, each value normalised by largest difference for that gene for that region pair
    normalised_shape_corr (dict): raw dictionary, each value normalised by largest shape measure difference for that region pair
    fold_gene_corr (dict): fold difference in gene expression values of pairwise regions at each timestep for each gene
    """

    # Initialise dictionaries
    raw_gene_corr = {gene: {"boundary_meristem": [], "sepal_meristem": [], "boundary_sepal": []} for gene in gene_features}
    raw_shape_corr = {"boundary_meristem": [], "sepal_meristem": [], "boundary_sepal": []}
    
    normalised_gene_corr = {gene: {"boundary_meristem": [], "sepal_meristem": [], "boundary_sepal": []} for gene in gene_features}
    normalised_shape_corr = {"boundary_meristem": [], "sepal_meristem": [], "boundary_sepal": []}
    
    fold_gene_corr = {gene: {"boundary_meristem": [], "sepal_meristem": [], "boundary_sepal": []} for gene in gene_features}
    fold_shape_corr = {"boundary_meristem": [], "sepal_meristem": [], "boundary_sepal": []}

    # Loop over each timestep
    for i in range(len(times)):  
        for gene in gene_features:
            meristem = avg_values[0][gene][i]  # Cluster 0 = Meristem
            sepal = avg_values[1][gene][i]  # Cluster 1 = Boundary
            boundary = avg_values[2][gene][i]  # Cluster 2 = Sepal

            # Store raw absolute differences
            raw_gene_corr[gene]["boundary_meristem"].append(abs(boundary - meristem))
            raw_gene_corr[gene]["sepal_meristem"].append(abs(sepal - meristem))
            raw_gene_corr[gene]["boundary_sepal"].append(abs(boundary - sepal))
            
            # Store fold differences
            fold_gene_corr[gene]["boundary_meristem"].append(np.log(abs(boundary/meristem)))
            fold_gene_corr[gene]["sepal_meristem"].append(np.log(abs(sepal/meristem)))
            fold_gene_corr[gene]["boundary_sepal"].append(np.log(abs(boundary/sepal)))

        # Compute absolute differences for shape feature
        raw_shape_corr["boundary_meristem"].append(abs(avg_values[1]["shape"][i] - avg_values[0]["shape"][i]))
        raw_shape_corr["sepal_meristem"].append(abs(avg_values[2]["shape"][i] - avg_values[0]["shape"][i]))
        raw_shape_corr["boundary_sepal"].append(abs(avg_values[1]["shape"][i] - avg_values[2]["shape"][i]))
        
        fold_shape_corr["boundary_meristem"].append(np.log(abs(avg_values[1]["shape"][i] / avg_values[0]["shape"][i])))
        fold_shape_corr["sepal_meristem"].append(np.log(abs(avg_values[2]["shape"][i] / avg_values[0]["shape"][i])))
        fold_shape_corr["boundary_sepal"].append(np.log(abs(avg_values[1]["shape"][i] / avg_values[2]["shape"][i])))

    # Normalize by largest difference over all timesteps
    for gene in gene_features:
        comparison = ["boundary_meristem", "sepal_meristem", "boundary_sepal"]
        all_diffs = [val for pair in comparison for val in raw_gene_corr[gene][pair]]
        max_diff = max(all_diffs)
            
        for pair in comparison:
            if max_diff == 0:
                normalised_gene_corr[gene][pair] = [0] * len(times)  # Avoid division by zero
            else:
                normalised_gene_corr[gene][pair] = [val / max_diff for val in raw_gene_corr[gene][pair]]

    for pair in comparison:
        all_diffs = [val for pair in comparison for val in raw_shape_corr[pair]]
        max_diff = max(all_diffs)  # Compute cumulative sum
        if max_diff == 0:
            normalised_shape_corr[pair] = [0] * len(times)  # Avoid division by zero
        else:
            normalised_shape_corr[pair] = [val / max_diff for val in raw_shape_corr[pair]]

    return raw_gene_corr, raw_shape_corr, normalised_gene_corr, normalised_shape_corr, fold_gene_corr, fold_shape_corr


def compute_lag_correlation(gene_corr,shape_corr):
    """
    Given the correlation between features in pairwise regions at each timestep
    Returns the difference in correlations between two subsequent timesteps i.e. rate of change of differentiation

    Parameters
    ----------
    gene_corr (dict): absolute difference in gene expression values of pairwise regions at each timestep for each gene
    shape_corr (dict): absolute difference in overall shape measure of pairwise regions at each timestep

    Returns
    -------
    lag_gene_corr (dict): change between subsequent timesteps of absolute difference in gene expression values of pairwise regions for each gene
    lag_shape_corr (dict): change between subsequent timesteps of absolute difference in overall shape measure of pairwise regions

    """
    
    # Initialise dictionaries
    lag_gene_corr = {gene: {"boundary_meristem": [], "sepal_meristem": [], "boundary_sepal": []} for gene in gene_features}
    lag_shape_corr = {"boundary_meristem": [], "sepal_meristem": [], "boundary_sepal": []}  # Store shape differences

    #np.diff calculates the difference of consecutive elements along array
    for gene in gene_features:
        lag_gene_corr[gene]["boundary_meristem"] = np.insert(np.diff(gene_corr[gene]["boundary_meristem"]),0,0)
        lag_gene_corr[gene]["sepal_meristem"] = np.insert(np.diff(gene_corr[gene]["sepal_meristem"]),0,0)
        lag_gene_corr[gene]["boundary_sepal"] = np.insert(np.diff(gene_corr[gene]["boundary_sepal"]),0,0)
            
    lag_shape_corr["boundary_meristem"] = np.insert(np.diff(shape_corr["boundary_meristem"]), 0, 0)
    lag_shape_corr["sepal_meristem"] = np.insert(np.diff(shape_corr["sepal_meristem"]), 0, 0)
    lag_shape_corr["boundary_sepal"] = np.insert(np.diff(shape_corr["boundary_sepal"]), 0, 0)
    
    # Replace NaN and Inf values with 0
    lag_gene_corr = {gene: {comp: np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0) for comp, arr in comps.items()} for gene, comps in lag_gene_corr.items()}
    lag_shape_corr = {comp: np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0) for comp, arr in lag_shape_corr.items()}
        
    return lag_gene_corr, lag_shape_corr
        



# PLOTTING/RESULT FUNCTIONS


def plot_trajectories(cluster_pc_values, avg_values, cluster_cell_type,xlim,ylim,space="shape"):
    """
    Produces a figure of separate trajectory plots for each final cluster
    Plots the trajectory by connecting average PCs across time, also plots PC values of all cells contributing to the averages

    Parameters
    ----------
    cluster_pc_values (dict): All cell data across timestep and clusters
    avg_values (dict): Averages of cell data across timestep and clusters - points for trajectory
    cluster_cell_type (dict): description of cell-type associated with each cluster number
    xlim x axes limits
    ylim: y axes limits

    Returns
    -------
    Outputs plot

    """
    
    # Setup separate axes for each final cluster
    if space == 'shape':
        fig, axes = plt.subplots(1, 3, figsize=(6, 2), sharey=True)
    elif space == 'gene':
        fig, axes = plt.subplots(2, 3, figsize=(6, 4), sharey=True)
        axes = axes.flatten()
    
    
    for ax, (cluster, timestep_data) in zip(axes, sorted(cluster_pc_values.items())):
        
        avg_PC1_values = avg_values[cluster]['PC1']
        avg_PC2_values = avg_values[cluster]['PC2']
        
        # At each timestep, plot PC values of all ancestors and the average values in this cluster
        for i, timestep in enumerate(times):
            
            PC1_values = [point['PC1'] for point in timestep_data[timestep]]
            PC2_values = [point['PC2'] for point in timestep_data[timestep]]

            # Plot all points as transparent dots
            ax.scatter(PC1_values, PC2_values, label=f'{timestep}h', color=colours[cluster][i], marker='^', s=0.1)
            
            # Add the legend handle/label once per timestep
            #if cluster == 0:  # Only collect the legend handles/labels from the first subplot
            #    handles.append(scatter)
            #    labels.append(f'{timestep}h')
            
            # Plot averages as opaque dots
            ax.scatter(avg_PC1_values[i], avg_PC2_values[i], color=colours[cluster][i], marker='o', s=20, edgecolor='black',zorder=10) 

        # Plot a line connecting the averages - 'trajectory'        
        ax.plot(avg_PC1_values, avg_PC2_values, color='black')
        
        # Figure setup
        ax.set_title(f'{cluster_cell_type[cluster].capitalize()}', fontsize=12)
        ax.set_xlabel('PC1', fontsize=12)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    
    for i, ax in enumerate(axes):
        if i % 3 == 0:  # first column of each row
            ax.set_ylabel('PC2', fontsize=12)
            
        if i // 3 == (len(axes) // 3) - 1:  # last row
            ax.set_xlabel('PC1', fontsize=12)
        else:
            ax.set_xlabel('')
            ax.tick_params(labelbottom=False)  # hide tick labels
            
            
            
    # Add a shared y-axis label
    plt.subplots_adjust(hspace=0.3)
    if space == "shape":
        header = 1.10
    elif space == "gene":
        header = 1
    fig.suptitle(f"{space.capitalize()} Trajectories of Each Region - Final Fate Ancestors Method",y=header)
        
    
    # COLOR BARS 
    
    cluster_order = sorted(cluster_pc_values)
    
    ncols = 3
    n_clusters = len(cluster_order)
    nrows = math.ceil(n_clusters / ncols)
    
    bar_width = 0.06
    bar_spacing = 0.02
    

    for row in range(nrows):
        start = row * ncols
        end = min(start + ncols, n_clusters)
        clusters_in_row = cluster_order[start:end]
    
        last_ax_in_row = axes[end - 1]
    
        bar_axes = [
            inset_axes(last_ax_in_row,
                       width=f"{bar_width*100}%", height="80%",
                       loc='center left',
                       bbox_to_anchor=(1.05 + i * (bar_width + bar_spacing), 0.0, 1, 1),
                       bbox_transform=last_ax_in_row.transAxes,
                       borderpad=0)
            for i in range(len(clusters_in_row))
        ]
    
        for i, (bar_ax, cluster) in enumerate(zip(bar_axes, clusters_in_row)):
            cmap = ListedColormap(colours[cluster])
            n_colors = len(colours[cluster])
            norm = BoundaryNorm(range(n_colors + 1), cmap.N)
    
            cb = ColorbarBase(bar_ax, cmap=cmap, norm=norm, orientation='vertical')
            cb.ax.collections[0].set_edgecolor("face")
            cb.ax.collections[0].set_linewidth(0)
            cb.ax.set_yticks([])
            cb.ax.set_ylabel("")
            cb.ax.tick_params(which='both', length=0)
    
            if (cluster+1) %3 == 0:  # label last colorbar of each row
                tick_positions = [j + 0.5 for j in range(n_colors)]
                cb.set_ticks(tick_positions)
                cb.set_ticklabels([f"{t}h" for t in times])
                cb.ax.tick_params(which='major', length=4)
                cb.ax.set_ylim(0, n_colors)
            
        
def plot_trajectories_avg(avg_values, cluster_cell_type,xlim,ylim,space="shape"):
    """
    Produces one plot of trajectories for all final clusters
    Plots the trajectory by connecting average PCs of contributing cells across time

    Parameters
    ----------
    avg_values (dict): Averages of cell data across timestep and clusters - points for trajectory
    cluster_cell_type (dict): description of cell-type associated with each cluster number
    xlim x axes limits
    ylim: y axes limits

    Returns
    -------
    Outputs plot

    """
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    
    for cluster, data in avg_values.items():
        avg_PC1_values = data['PC1']
        avg_PC2_values = data['PC2']
        color_map = colours[cluster]

        # Plot the trajectory line for the cluster
        ax.plot(avg_PC1_values, avg_PC2_values, color=color_map[-3], label=f"{cluster_cell_type[cluster].capitalize()}", lw=2)

        # Plot individual average points with progression color
        for i, (pc1, pc2) in enumerate(zip(avg_PC1_values, avg_PC2_values)):
            ax.scatter(pc1, pc2, color=color_map[i], s=100, edgecolor='black')

    # Labels and limits
    ax.set_title(f"{space.capitalize()} Trajectories - Final Fate Ancestors Method", fontsize=12)
    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    #ax.legend(title="Cell Type", fontsize=12,loc='lower right')
    
    custom_legend = [
    Line2D([0], [0],
           color=colours[i][-3],
           label=cluster_cell_type[i].capitalize(),
           linewidth=2)  # line width (optional)
    for i in sorted(avg_values)
    ]
    
    # Add the custom legend
    plt.legend(handles=custom_legend, title="Cell Type", loc='lower right', fontsize=8,title_fontsize=8)
    
    plt.tight_layout()
    plt.show()


def plot_cell_type_trajectories(cluster_means,cluster_no,cluster_cell_type,cluster_transitions,xlim,ylim,space="shape"):
    """

    """
    

    plt.figure(figsize=(6, 4),dpi=300)

    # Process data for each timepoint
    for timestep in cluster_no.keys():
        for cluster in range(0, cluster_no[timestep]):
            # Calculate the mean for each cluster at this timepoint
            mean_PC1 = cluster_means[timestep][cluster]['PC1']
            mean_PC2 = cluster_means[timestep][cluster]['PC2']
            
            # Plot the mean for this cluster at the given timepoint
            plt.scatter(mean_PC1, mean_PC2, color=colours[cluster][times.index(timestep)], label=f'Timepoint {timestep}, Cluster {cluster}',
                        s=100, edgecolor='black')


    # Draw lines based on manually defined cluster transitions
    for (parent_time, parent_cluster), children in cluster_transitions.items():
        parent_PC1 = cluster_means[parent_time][parent_cluster]['PC1']
        parent_PC2 = cluster_means[parent_time][parent_cluster]['PC2']
        
        for (child_time, child_cluster) in children:
            if child_cluster in cluster_means[child_time]:
                child_PC1 = cluster_means[child_time][child_cluster]['PC1']
                child_PC2 = cluster_means[child_time][child_cluster]['PC2']
                
                # Use the color scheme of the child cluster
                child_color = colours[child_cluster][-3]

                # Draw line from parent to child
                #plt.plot([parent_PC1, child_PC1], [parent_PC2, child_PC2],color=child_color, linestyle='--', linewidth=2, alpha=0.8)
                
                # Draw arrow from parent to child
                plt.annotate(
                    '',  # No text
                    xy=(child_PC1, child_PC2),
                    xytext=(parent_PC1, parent_PC2),
                    arrowprops=dict(arrowstyle="->, head_length=0.8, head_width=0.4", 
                    color=child_color, lw=2, linestyle = '--', alpha=0.8),
                )
    
    
    custom_legend = [
    Line2D([0], [0],
           color=colours[i][-3],
           label=cluster_cell_type[i].capitalize(),
           linewidth=2)  # line width (optional)
    for i in range(cluster_no["132"])
    ]   
    
    # Add the custom legend
    plt.legend(handles=custom_legend, title="Cell Type", loc='lower right', fontsize=8,title_fontsize=8)

    # Add titles and labels
    plt.title(f"{space.capitalize()} Trajectories - Cell Type Level Branching Method", fontsize=12)
    plt.xlabel('PC1', fontsize=12)
    plt.ylabel('PC2', fontsize=12)
    plt.xlim(xlim)
    plt.ylim(ylim)

    # Display the plot with tight layout
    plt.tight_layout()
    plt.show()


def plot_combined_trajectories(avg_values, cluster_means, cluster_cell_type, xlim, ylim, points=True):
    """
    Plots average trajectory from cell_level_trajectory and overlays cluster means from cell_type_level_trajectory.

    Parameters
    ----------
    avg_values (dict): Averaged cell data across time and clusters from cell_level_trajectory
    cluster_means (dict): Cluster mean PC1, PC2 values across timepoints from cell_type_level_trajectory
    cluster_cell_type (dict): Mapping of cluster index to cell type
    xlim, ylim: Axes limits for consistent plotting
    """

    times_numeric = list(map(int, times))
    colours = cm.viridis(np.linspace(0, 1, len(times)))

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot avg trajectory for each final cluster
    for cluster in sorted(avg_values.keys()):
        avg_PC1 = avg_values[cluster]['PC1']
        avg_PC2 = avg_values[cluster]['PC2']
        ax.plot(avg_PC1, avg_PC2, color='black', label=f'Avg {cluster_cell_type[cluster]}')

    if points:
        # Overlay cluster_means at each time
        for i, t in enumerate(times):
            t_means = cluster_means[t]
            for cluster_id, coords in t_means.items():
                ax.scatter(coords['PC1'], coords['PC2'], color=colours[i], edgecolor='black', s=100, label=f'{t}h Cluster {cluster_id}' if i == 0 else None)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Combined Average Trajectories and Cluster Means")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()
    

def timeseries_plots(cluster_pc_values, avg_values, cluster_cell_type, plot_gene_features = gene_features):
    """
    For each shape cluster, plots PC1, PC2, shape measure over time on same axis
    Also plots on separate axes for each gene expression over time


    Parameters
    ----------
    cluster_pc_values (dict): All cell data across timestep and clusters
    avg_values (dict): Averages of cell data across timestep and clusters - points for trajectory
    cluster_cell_type (dict): description of cell-type associated with each cluster number

    Returns
    -------
    Outputs plot
    
    """

    # Setup plot
    fig, axes = plt.subplots(len(plot_gene_features) + 1, len(cluster_pc_values), figsize=(6*len(cluster_pc_values), 3*(len(plot_gene_features)+1)), sharex=True, sharey='row')
    numeric_times = list(map(int, times))
    
    # Loops over clusters
    for cluster_idx, cluster in enumerate(sorted(cluster_pc_values.keys())):
        # Plots PC1, PC2, and shape on same axis
        axes[0, cluster_idx].plot(numeric_times, avg_values[cluster]["PC1"], label="PC1", color="blue")
        axes[0, cluster_idx].plot(numeric_times, avg_values[cluster]["PC2"], label="PC2", color="orange")
        axes[0, cluster_idx].plot(numeric_times, avg_values[cluster]["shape"], label="Shape measure", color="pink")
        axes[0, cluster_idx].set_title(f'Cluster {cluster} - {cluster_cell_type[cluster_idx]}', fontsize=14)
        axes[0, cluster_idx].grid(True)
        axes[0, cluster_idx].legend()
        
        # New plot for each gene
        for gene_idx, gene in enumerate(plot_gene_features):
            axes[gene_idx + 1, cluster_idx].plot(numeric_times, avg_values[cluster][gene], label=gene, color=np.random.rand(3,))
            axes[gene_idx + 1, cluster_idx].set_title(f'{gene} Expression - Cluster {cluster} - {cluster_cell_type[cluster_idx]}', fontsize=14)
            axes[gene_idx + 1, cluster_idx].grid(True)
    
    # Figure settings
    fig.supylabel('Time (h)', fontsize=12)
    fig.suptitle('PC1, PC2, and Gene Expression over Time', fontsize=16, y=0.93)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()
    
    
def timeseries_correlate(gene_corr,shape_corr,cluster_pc_values, avg_values, cluster_cell_type,plot_gene_features = gene_features):
    """
    For each shape cluster, plots PC1, PC2, shape measure over time on same axis
    Also plots on separate axes for each gene expression over time
    Then also plots the correlation between pairwise regions for each of these parameters

    Parameters
    ----------
    gene_corr (dict): desired correlation in gene expression values of pairwise regions for each gene
    shape_corr (dict): desired correlation in overall shape measure of pairwise regions
    cluster_pc_values (dict): All cell data across timestep and clusters
    avg_values (dict): Averages of cell data across timestep and clusters - points for trajectory
    cluster_cell_type (dict): description of cell-type associated with each cluster number

    Returns
    -------
    Outputs plot

    """

    fig, axes = plt.subplots(len(plot_gene_features) + 1, 6, figsize=(10, 1.5*(len(plot_gene_features)+1)), sharex=True, sharey='row')
    numeric_times = list(map(int, times))
    
    for cluster_idx, cluster in enumerate(sorted(cluster_pc_values.keys())):
        # First three columns: PC1, PC2, Shape, and gene expression
        #axes[0, cluster_idx].plot(numeric_times, avg_values[cluster]["PC1"], label="PC1", color="blue")
        #axes[0, cluster_idx].plot(numeric_times, avg_values[cluster]["PC2"], label="PC2", color="orange")
        axes[0, cluster_idx].plot(numeric_times, avg_values[cluster]["shape"], label="Shape measure", color="blue")
        axes[0, cluster_idx].set_title(f'Shape - {cluster_cell_type[cluster_idx].capitalize()}', fontsize=10)
        axes[0, cluster_idx].grid(True)
        #axes[0, cluster_idx].legend()
        
        for gene_idx, gene in enumerate(plot_gene_features):
            # Gene expression per cluster
            axes[gene_idx + 1, cluster_idx].plot(numeric_times, avg_values[cluster][gene], label=gene, color='blue')
            axes[gene_idx + 1, cluster_idx].set_title(f'{gene} - {cluster_cell_type[cluster_idx].capitalize()}', fontsize=10)
            axes[gene_idx + 1, cluster_idx].grid(True)

            # Fourth column: Boundary-Meristem Correlation
            axes[gene_idx + 1, 3].plot(numeric_times, gene_corr[gene]["boundary_meristem"], label=f'{gene} Corr', color='orange')
            axes[gene_idx + 1, 3].set_title(f'{gene} (B-M)', fontsize=10)
            axes[gene_idx + 1, 3].grid(True)
            
            # Fifth column: Sepal-Meristem Correlation
            axes[gene_idx + 1, 4].plot(numeric_times, gene_corr[gene]["sepal_meristem"], label=f'{gene} Corr', color='orange')
            axes[gene_idx + 1, 4].set_title(f'{gene} (S-M)', fontsize=10)
            axes[gene_idx + 1, 4].grid(True)

            # Sixth column: Boundary-Sepal Correlation
            axes[gene_idx + 1, 5].plot(numeric_times, gene_corr[gene]["boundary_sepal"], label=f'{gene} Corr', color='orange')
            axes[gene_idx + 1, 5].set_title(f'{gene} (B-S)', fontsize=10)
            axes[gene_idx + 1, 5].grid(True)


    # Plot shape comparisons in the first row (final 3 columns)
    axes[0, 3].plot(numeric_times, shape_corr["boundary_meristem"], label="Shape Corr", color='orange')
    axes[0, 3].set_title('Shape (B-M)', fontsize=10)
    axes[0, 3].grid(True)

    axes[0, 4].plot(numeric_times, shape_corr["sepal_meristem"], label="Shape Corr", color='orange')
    axes[0, 4].set_title('Shape (S-M)', fontsize=10)
    axes[0, 4].grid(True)

    axes[0, 5].plot(numeric_times, shape_corr["boundary_sepal"], label="Shape Corr", color='orange')
    axes[0, 5].set_title('Shape (B-S)', fontsize=10)
    axes[0, 5].grid(True)

    '''
    # Set different y-limits for first row and all other rows
    for ax in axes[0, :]:  
        ax.set_ylim(-3, 3)  # First row

    for ax_row in axes[1:]:  # All other rows
        for ax in ax_row:
            ax.set_ylim(0, 1)
    '''
        
    # Further figure settings
    fig.suptitle('Timseries Shape Measure, Gene Expression, and Pairwise Region Correlation', fontsize=12, y=0.96)
        
    for row_idx in range(len(plot_gene_features) + 1):
        if row_idx == 0:
            axes[row_idx, 0].set_ylabel('|PC1| + |PC2|', fontsize=10)
            axes[row_idx, 0].tick_params(axis='y', labelsize=8)
        else:
            axes[row_idx, 0].set_ylabel('Expression', fontsize=10)
            axes[row_idx, 0].tick_params(axis='y', labelsize=8)
            
    # Format y-axis to 1 decimal place
    def format_1dp(x, _):
        return f"{x:.1f}"
    
    for row in axes:
        for ax in row:
            ax.yaxis.set_major_formatter(FuncFormatter(format_1dp))
        
    for col_idx in range(6):
        axes[-1, col_idx].set_xlabel("Time [h]", fontsize=10)
        numeric_times = list(map(int, times))
        axes[-1, col_idx].set_xticks(numeric_times)
        axes[-1, col_idx].set_xticklabels(times, fontsize=8, rotation=90)
        
        
    fig.text(
        0.85, 0.98,
        "Pairwise Regions:\n"
        "B-M: Boundary–Meristem\n"
        "S-M: Sepal–Meristem\n"
        "B-S: Boundary–Sepal",
        ha='left', va='top',
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="orange")
    )   
        
    
    custom_lines = [
        Line2D([0], [0], color='blue', lw=2, label='Expression'),
        Line2D([0], [0], color='orange', lw=2, label='Correlation')
    ]
    
    # Add to figure
    fig.legend(
        handles=custom_lines,
        loc='upper right',
        bbox_to_anchor=(0.65, 0.92),  # Adjust Y position as needed
        fontsize=9,
        ncol=2,
        frameon=True
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


def plot_difference_table(correlation, times, gene_features, title):
    """
    Alternate visualisation of the correlation/differences between pairwise regions across time
    Generates a table where rows are gene features and columns are timesteps
    Three stacked tables for each pariwise region
    Colored dots represent the size of the difference between gene expression at given time between two given regions


    Parameters
    ----------
    correlation (dict): Correlations in gene expression for the pairwise regions
    times (list): Column headings
    gene_features (list): Row headings
    title (text): Describes which type of correlation is being displayed

    Returns
    -------
    Outputs plot

    """
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(len(times) * 2, len(gene_features) * 0.7))
    num_genes = len(gene_features)
    num_times = len(times)
    comparisons = ["boundary_meristem", "sepal_meristem", "boundary_sepal"]
    num_comparisons = len(comparisons)
    
    # Create a grid for positioning
    x_positions = np.arange(num_comparisons * num_times)
    y_positions = np.arange(num_genes)
    X, Y = np.meshgrid(x_positions, y_positions)
    
    # Flatten the X, Y grids
    X = X.flatten()
    Y = Y.flatten()
    
    # Extract color values based on the expression difference
    colors = []
    for gene in gene_features:
        for comparison in comparisons:
            colors.extend(correlation[gene][comparison])
    
    # Normalize colors for colormap
    norm = plt.Normalize(min(colors), max(colors))
    #cmap = plt.cm.viridis
    cmap = plt.cm.Oranges
    
    # Scatter plot for dots
    scatter = ax.scatter(X, Y, c=colors, cmap=cmap, norm=norm, edgecolors='k', s=100)
    
    # Formatting
    ax.set_yticks(np.arange(num_genes))
    ax.set_yticklabels(gene_features, fontsize=10)
    ax.invert_yaxis()
    
    
    # Set minor xticks for individual time points
    ax.set_xticks(np.arange(num_comparisons * num_times))
    ax.set_xticklabels([str(t+"h") for t in times] * num_comparisons, fontsize=8, rotation=90)

    # Move x-axis ticks to the top
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    
    # Add vertical separator lines
    ax.axvline(num_times - 0.5, color='black', linestyle='--', linewidth=1.5)
    ax.axvline(2 * num_times - 0.5, color='black', linestyle='--', linewidth=1.5)

    # Add region subheadings
    midpoints = [num_times / 2 - 0.5, 3 * num_times / 2 - 0.5, 5 * num_times / 2 - 0.5]
    for mid, label in zip(midpoints, comparisons):
        ax.text(mid, - 2.4, label.replace('_', ' - ').title(), ha='center', fontsize=12, fontweight='bold')

    # Add colorbar with modified width
    cbar = plt.colorbar(scatter, ax=ax, orientation='vertical')
    cbar.set_label("Expression Difference", fontsize=10)
    
    # Title
    ax.set_title(title, fontsize=12, pad=80)



    
    plt.show()


def gene_contribution(correlation):
    """
    Produces a table for each pairwise region that ranks, by size, gene difference at each timestep
    i.e. ranks their contribution to the differentiation

    Parameters
    ----------
    correlation (dict): The correlation dictionary to analyse and output
    
    Returns
    -------
    Outputs table to console

    """
    
    col_width = 14  # consistent width for all columns

    # Table for each pair
    for comparison in ["boundary_meristem", "sepal_meristem", "boundary_sepal"]:
        print(f"\n=== {comparison.replace('_', ' ').title()} ===") #title format

        # Collect ranked gene lists for each time
        ranked_by_time = []
        for t in range(len(times)):
            gene_values = []
            for gene in correlation:
                val = correlation[gene][comparison][t]
                gene_values.append((gene, val))
            # Sort genes in descending order of their value for this timepoint
            gene_values.sort(key=lambda x: x[1], reverse=True)
            ranked_by_time.append(gene_values)

        # Numver of rows
        max_genes = len(correlation)

        # Header format
        header = f"| {'Rankk':^{col_width}} |" + "".join([f" t={t} ".center(col_width, ' ') + "|" for t in range(len(times))])
        print(header)
        print("|" + "-" * (len(header) - 2) + "|") #horizontal separator lines

        # Rows
        for rank in range(max_genes):
            row = f"| {str(rank+1):^{col_width}} |"
            for t in range(len(times)):
                gene, val = ranked_by_time[t][rank] #fetch gene and its value and rank at this timepoint
                cell = f"{gene}({val:.2f})"
                row += f"{cell:^{col_width}}|"
            print(row)
    
    
    
# MAIN

def main(plot=True, backfill=True):
    
    # Cell-level trajectory computation
    shape_cell_values, shape_cell_avg = cell_level_trajectory(space_dict = shape_dict,shape=True,avg_method="mean")
    gene_cell_values, gene_cell_avg = cell_level_trajectory(space_dict = gene_dict,gene=True,avg_method="mean")
    
    # Cell-type-level trajectory computation
    shape_cell_type_values, shape_cell_type_avg = cell_type_level_trajectory(shape_cluster_no,space='shape',backfill=backfill)
    gene_cell_type_values, gene_cell_type_avg = cell_type_level_trajectory(gene_cluster_no,space='gene',backfill=backfill)
    
    
    if plot:
        # Plot trajectories for shape space
        #plot_trajectories(shape_cell_values, shape_cell_avg, shape_cluster_cell_type,xlim=(-6,6),ylim=(-4,4))
        #plot_trajectories_avg(shape_cell_avg, shape_cluster_cell_type,xlim=(-3,3.5),ylim=(-1.5,1))
        #plot_cell_type_trajectories(shape_cell_type_avg,shape_cluster_no,shape_cluster_cell_type,shape_cluster_transitions, xlim=(-3,3.5),ylim=(-1.5,1))
        
        # Plot trajectories for gene space
        #plot_trajectories(gene_cell_values, gene_cell_avg, gene_cluster_cell_type,xlim=(-10,10),ylim=(-10,10),space="gene")
        #plot_trajectories_avg(gene_cell_avg, gene_cluster_cell_type,xlim=(-6,6),ylim=(-7,6),space="gene")
        #plot_cell_type_trajectories(gene_cell_type_avg,gene_cluster_no, gene_cluster_cell_type,gene_cluster_transitions,xlim=(-6,6),ylim=(-7,6),space="gene")
    
        # Plot combined trajectories - not updates
        #plot_combined_trajectories(shape_avg_values, shape_cluster_means, shape_cluster_cell_type, xlim=(-4, 4), ylim=(-4, 2))
        #plot_combined_trajectories(gene_avg_values, gene_cluster_means, shape_cluster_cell_type, xlim=(-4, 4), ylim=(-4, 2), points=False)
    
        # Determine correlation of features between pairwise regions
        gene_corr, shape_corr, normalised_gene_corr, normalised_shape_corr, fold_gene_corr,fold_shape_corr = compute_triple_correlation(shape_cell_avg)
        lag_gene_corr, lag_shape_corr = compute_lag_correlation(normalised_gene_corr,normalised_shape_corr)
        
        # Plot evolution of features and their correlation
        #timeseries_plots(shape_cell_values, shape_cell_avg, shape_cluster_cell_type)
        #timeseries_correlate(fold_gene_corr,fold_shape_corr,shape_cell_values, shape_cell_avg, shape_cluster_cell_type)
        
        timeseries_correlate(gene_corr,shape_corr,shape_cell_values, shape_cell_avg, shape_cluster_cell_type,plot_gene_features=gene_features_subset)
        
        # Plot tables visualising the differences
        #plot_difference_table(gene_corr, times, gene_features, title = "Gene Expression Absolute Difference Across Time and Regions")
        #plot_difference_table(normalised_gene_corr, times, gene_features, title = "Gene Expression Normalised Absolute Difference Across Time and Regions")
        plot_difference_table(fold_gene_corr, times, gene_features, title = "Gene Expression Fold Difference Across Time and Regions")
        #plot_difference_table(lag_gene_corr, times[1:], gene_features, title = "Gene Expression Absolute Difference Change (t2-t1) Across Time and Regions")
    
        # Ouput to console tables ranking gene contributions
        #gene_contribution(normalised_gene_corr)
        #gene_contribution(gene_corr)
    
    return shape_cell_values, gene_cell_values, shape_cell_type_values, gene_cell_type_values


if __name__ == "__main__":
    _,_,_,_ = main(plot=True)







