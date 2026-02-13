#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 10:02:45 2025

@author: juliannanowaczek
"""

# imports
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from cluster_maps import gene_cluster_cell_type, shape_cluster_cell_type

from trajectories import main
shape_cell_values, gene_cell_values, shape_cell_type_values, gene_cell_type_values = main(plot=False,backfill=False)

#from latent_space import flower_by_feature

# specify timesteps
times = ("96","104","112","120","128","132")


def trajectory_overlap(trajectory_1_dict, trajectory_2_dict, final_fate = True):
    
    # intialise dictionary
    clusters_1 = len(trajectory_1_dict)
    clusters_2 = len(trajectory_2_dict)
    trajectory_best_map = {time:{} for time in times}
    trajectory_component_map = {time:{cluster:{} for cluster in range(clusters_1)} for time in times}
    
    # iterate cluster assignments
    for cluster_x in range(clusters_1):
        print('--')
        for time in times:
            
            #store all ids at this time that belong to cluster x
            cluster_x_ids = [entry["cell_id"] for entry in trajectory_1_dict[cluster_x][time]]
            #print(len(cluster_x_ids))
            
            overlaps = []
            
            for cluster_y in range(clusters_2):
                #store all ids at this time that belong to cluster y
                cluster_y_ids = [entry["cell_id"] for entry in trajectory_2_dict[cluster_y][time]]
            
                # count overlap between cluster x and cluster y
                overlap_count = len(set(cluster_x_ids) & set(cluster_y_ids))
                overlaps.append(overlap_count)
                
            overlap_percentage = np.round(np.array(overlaps)/sum(overlaps),2)
            for cluster_y in range(clusters_2):
                trajectory_component_map[time][cluster_x][cluster_y] = overlap_percentage[cluster_y]
        
            # store cluster y that had highest cell_id count overlap with x at this timestep
            trajectory_best_map[time][cluster_x] = overlaps.index(max(overlaps))
            
    return trajectory_best_map, trajectory_component_map


def plot_stacked_bars(data_dict,text,legend=True):
    times = list(data_dict.keys())
    num_times = len(times)

    # Dynamically get sorted shape cluster indices from the first time entry
    shape_clusters = sorted(data_dict[times[0]].keys())

    # Dynamically get sorted gene cluster indices from the first shape in the first time
    example_shape = shape_clusters[0]
    
    inner_clusters = sorted(data_dict[times[0]][example_shape].keys())
    num_inner_clusters = len(inner_clusters)

    colors = {0: cm.Purples(0.6),
            1: cm.Greens(0.65),
            2: cm.Oranges(0.6),
            3: cm.Reds(0.75),
            4: cm.Greys(0.55),
            5: cm.Blues(0.6)}

    fig, axs = plt.subplots(nrows=len(shape_clusters), ncols=1, figsize=(6, 3 * len(shape_clusters)), sharex=True)

    # Handle case where axs is not a list (if only 1 shape cluster)
    if len(shape_clusters) == 1:
        axs = [axs]
        
    if num_inner_clusters == 6:
        inner_cluster_cell_type = gene_cluster_cell_type
        outer_cluster_cell_type = shape_cluster_cell_type
        legend_title = "Gene Trajectory"
    elif num_inner_clusters == 3:
        inner_cluster_cell_type = shape_cluster_cell_type
        outer_cluster_cell_type = gene_cluster_cell_type
        legend_title = "Shape Trajectory"

    for i, shape in enumerate(shape_clusters):
        ax = axs[i]
        bottom = np.zeros(num_times)

        for inner_idx, inner in enumerate(inner_clusters):
            heights = [data_dict[time][shape].get(inner, 0) for time in times]
            ax.bar(times, heights, bottom=bottom, color=colors[inner_idx], label=f'{inner_cluster_cell_type[inner].title()}' if i == 0 else "")
            bottom = np.add(bottom, heights)

        ax.set_ylim(0, 1)
        ax.set_ylabel('Match Fraction')
        ax.set_title(f'{outer_cluster_cell_type[shape].title()} Trajectory')

    axs[-1].set_xticks(times)
    axs[-1].set_xticklabels(times, rotation=45)
    axs[-1].set_xlabel('Time [h]')


    if legend:
        # Only add legend to the top subplot
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, title=legend_title, bbox_to_anchor=(1.02, 0.93), loc='upper left')

    plt.suptitle(f'Percentage of Cell Overlap Between {text} Trajectories Across Time',fontsize=12,y=0.99)
    plt.tight_layout()
    plt.show()



shape_to_gene, shape_to_gene_percentage = trajectory_overlap(shape_cell_values, gene_cell_values)
gene_to_shape, gene_to_shape_percentage = trajectory_overlap(gene_cell_values, shape_cell_values)

#print(shape_to_gene)
#plot_stacked_bars(shape_to_gene_percentage,"Final Fate",legend=False)
#plot_stacked_bars(gene_to_shape_percentage,"Final Fate",legend=False)


shape_to_gene_2, shape_to_gene_percentage_2 = trajectory_overlap(shape_cell_type_values, gene_cell_type_values)
gene_to_shape_2, gene_to_shape_percentage = trajectory_overlap(gene_cell_type_values, shape_cell_values)

#print(shape_to_gene_2)
plot_stacked_bars(shape_to_gene_percentage_2, "Cell-type Branching")
#plot_stacked_bars(gene_to_shape_percentage, "Cell-type Branching")



