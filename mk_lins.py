"""
@author: juliannanowaczek
Code based on A.Zardilis
"""

# imports
import common.lin as lin
import common.edict as edict

import pandas as pd
import numpy as np


def get_inverted_linss():
    """
    Uses common package to get inverted cell lineages. 
    How cell id changes between subsequent time steps.

    Returns
    -------
    iLinss (dictionary) : key = (t1,t2), value = lineage dict 
        with key = id @ t2, value =  id @ t1

    """
    lins = lin.readLins1("./data/FM1/tracking_data/")
    linss = {ll[1]: ll[0] for ll in lins}

    iLinss = {}
    for k, v in linss.items():
        t1, t2 = k
        iLinss[(t2, t1)] = edict.invertListD(v)


    return iLinss


def sequence_linss(all_dictionary, path_id=None, dataframe=None, by_cluster = True):
    """
    Uses cell id mapping between time steps to follow ancestors through all time steps
    Follow ancestors across all time steps of cell ids in final timestep.

    Parameters
    ----------
    all_dictionary (dict): dictionary of the format from get_inverted_linss containing all cell id mappings between time steps
    path_id (string): path for csv of data in final time step
    dataframe: can directly input exisiting dataframe instead of loading from path
    by_cluster (boolean): whether the gene and shape dictionaries should be split by cluster assignment
    
    Returns
    -------
    gene_dict (dict): key = final gene cluster assignment, value = dictionary
        with key = final cell id, value = list of ancestors
    shape_dict (dict): key = final shape cluster assignment, value = dictionary
        with key = final cell id, value = list of ancestors
    
    or
    
    lineage_dict (dict): key = final cell id, value = list of ancestors
    """
    
    #load in data from final timestep
    if path_id:
        df = pd.read_csv(path_id)
    else: 
        df = dataframe

    #create the tracing dictionaries
    gene_dict = {}
    shape_dict = {}
    lineage_dict = {}
    
    #loop over every cell in final timestep
    for cell_id in df['id']:
        #start tracing from 132h
        trace = [cell_id]
        current_id = cell_id
        for key in [(132, 128), (128, 120), (120, 112), (112, 104), (104, 96)]:
            #get the next cell ID from all_dictionary
            if key in all_dictionary and current_id in all_dictionary[key]:
                current_id = all_dictionary[key][current_id]
                trace.append(current_id)
            else:
                #break if there's no further lineage
                trace.append(None)
                #print("INCOMPLETE")
                break
    
        if by_cluster:
            #separate lineages by final gene cluster assignment
            gene_cluster = df.loc[df['id'] == cell_id, 'gene_cluster_gmm'].values[0]
            if isinstance(gene_cluster, np.int64):
               gene_cluster = int(gene_cluster)
            #add the traced lineage to the corresponding gene cluster
            if gene_cluster not in gene_dict:
                gene_dict[gene_cluster] = {}
            gene_dict[gene_cluster][cell_id] = trace
            
            
            #separate lineages by final shape cluster assignment
            shape_cluster = df.loc[df['id'] == cell_id, 'shape_cluster_gmm'].values[0]
            if isinstance(shape_cluster, np.int64):
                shape_cluster = int(shape_cluster)
            if shape_cluster not in shape_dict:
                shape_dict[shape_cluster] = {}
            shape_dict[shape_cluster][cell_id] = trace
            
        else:
            lineage_dict[cell_id] = trace
            
        
    if by_cluster:
        return gene_dict, shape_dict
    else:
        return lineage_dict
    
    

# Use functions

#all_dictionary = get_inverted_linss()
#gene_dict,shape_dict = sequence_linss(all_dictionary,by_cluster = True,path_id="./data/FM1/csv/clustered_on_all/t132_clustered.csv")
#lineage_dict = sequence_linss(all_dictionary,by_cluster = False,path_id="./data/FM1/csv/t132.csv")