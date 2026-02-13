"""
@author: juliannanowaczek
Code based on A.Budnikova
"""


# imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples
from collections import Counter

from mk_lins import get_inverted_linss, sequence_linss
from trajectories import main as trajectory_fn
from cluster_maps import shape_cluster_no, gene_cluster_no, gene_mapping_reduced, gene_mapping_all, shape_mapping_reduced, shape_mapping_all


# Initialise global variables

# gene features - remove ATML since this is just a marker for L1
gene_columns = ['AG','AHP6','ANT','AP1','AP2','AP3','AS1','CLV3','CUC1_2_3','ETTIN','FIL','LFY','MP','PHB_PHV','PI','PUCHI','REV','SEP1','SEP2','SEP3','STM','SUP','SVP','WUS']

# shape features
shape_columns = ['vol','curv1','curv2','mean_curv','gauss_curv','dev_curv','dist_to_center']

# colors for categorical plots
cmap_colors = ['purple','green','orange','red','grey','blue', 'pink','cyan','brown','yellow','magenta']
monochrome = ListedColormap(["grey"])


# DATA SETUP FUNCTIONA

def add_dist(data):
    """
    computes distance to center and appends to dataframe

    input: 
    data - dataframe containing coordinates
    
    returns: 
    data - data frame with additional dist_to_center feature
    """
    
    center = data[['x', 'y', 'z']].mean()
    # distance to center
    data['dist_to_center'] = np.sqrt((data['x'] - center['x']) ** 2 +
                                         (data['y'] - center['y']) ** 2 +
                                         (data['z'] - center['z']) ** 2)

    return data


def scale_features(data, features):
    """
    scale features using StandardScaler()
    rescaling the features s.t they have a mean of 0 and a standard deviation of 1 (Gaussian)
    n.b. Kmeans sensitive to scale

    input:
    data - dataframe to be scaled
    features (list) - column names (features) to be scaled

    returns:
    scaled_df_full - scaled dataframe, preserving non-scaled columns e.g. id
    """
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[features])
    scaled_df = pd.DataFrame(scaled_data, columns=features) #give column names
    
    scaled_df_full = data.copy()
    scaled_df_full[features] = scaled_df

    return scaled_df_full


def ATML_filter(plot=False):
    """
    reads the csv of raw data for timestep
    keeps only cells that express ATML1
    
    input: 
    time - the (final) time step of interest (in h)
    
    returns:
    filtered_df - the csv as a dataframe containing only filtered rows
    """
    
    string = "data/FM1/csv/original_t132.csv"
    
    #read data from csv
    df = pd.read_csv(string)
    
    if plot:
    #plot histogram of ATML to determine cutoff
        plt.figure(figsize=(6, 4), dpi=300)
        plt.hist(df['ATML1'], bins=20, color='blue', alpha=0.7, edgecolor='black')
        plt.xlabel("ATML Expression",fontsize=16)
        plt.ylabel("Frequency",fontsize=16)
        plt.title("Histogram of Cell ATML1 Epression (t=132h)",fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.show()
    
    
    #filter ATML1 expression (>0.5 based on bimodality)
    #filtered_df = df[df["ATML1"]>0.5]

    #return filtered_df


def dataframe_setup(time,lineages,timestep={"132":0, "128":1, "120":2, "112":3, "104":4, "96":5}):
    """
    reads the csv of raw data for time step
    separates into two df for gene and shape features
    scales the data

    input: 
    time (int or str) - the time step of interest (in h)
    lineages (dict) - ancestor paths of cells
    timestep (dict) - order of timesteps in lineage paths
    
    returns:
    filtered_df - dataframe preserving only cells that contribute to final flower (non-scaled)
    scaled_gene_df - scaled dataframe of gene features
    scaled_shape_df - scaled dataframe of shape features
    """
    
    time = str(time)
    string = f"data/FM1/csv/t{time}.csv"
    
    #read data from csv
    df = pd.read_csv(string)
    
    #add distance to centre shape feature
    df = add_dist(df)
    
    #reorder the df
    features = gene_columns + shape_columns
    list1 = ["id","x","y","z"]
    column_order = list1 + features
    df = df[column_order]
    
    #scale the features
    scaled_df = scale_features(df,features)
    
    index = timestep[time]
    
    valid_ids = []
    #collate ids from this timestep that are traced back from final flower
    for cell_id in lineages:
        paths = lineages[cell_id]
        valid_ids.append(paths[index])
    
    #print(sorted(valid_ids))
    #print(f"timestep: {time}, number of valid ids: {len(valid_ids)}")
        
    scaled_df = scaled_df[scaled_df["id"].isin(valid_ids)]
    #create dfs of gene features and shape features
    scaled_gene_df = scaled_df[gene_columns]
    scaled_shape_df = scaled_df[shape_columns]

    #preserve same cells in the unscaled df
    filtered_df = df[df["id"].isin(valid_ids)]

    return filtered_df, scaled_gene_df, scaled_shape_df



# PCA AND CLUSTERING FUNCTIONS

def apply_pca(scaled_data, n_components=2):
    """
    Dimensionality reduction using PCA.

    input:
    scaled_data - DataFrame or array-like, the scaled data to which PCA will be applied
    n_components - int, the number of principal components to keep (default is 2)

    returns:
    pca_df - DataFrame containing the transformed principal components
    weights - 2D array of the component loadings
    eigenvalues - 1D array of the component variance
    """

    pca = PCA(n_components=n_components).fit(scaled_data)
    pca_transformed = pca.transform(scaled_data)
    
    pca_df = pd.DataFrame(pca_transformed, columns=[f'PC{i + 1}' for i in range(n_components)],index=scaled_data.index)
    
    weights = pca.components_
    #eigenvalues = pca.explained_variance_
    var_explained = pca.explained_variance_ratio_

    return pca_df, weights, var_explained


def cluster_optimise(time,data,space,method,cluster_range=range(1,9),random_state=22):
    """
    Identifies ideal number of clusters at a timestep by taking BIC and silouhette score.
    Also produces plot of BIC & Silhouette score for a range of cluster number
    Plot error bars for silhouette since this is the mean of all datapoints for that cluster no

    Parameters
    ----------
    - time: timestep for title
    - data: original scaled dataframe to cluster on all features or reduced df
    - space: reduced space working in e.g gene, shape
    - method: name of the dimensionality reduction method
    - cluster_range: range of cluster numbers to test
    - random_state: int or None, random state for reproducibility

    Returns
    -------
    None. Creates plot evaluating ideal number of clusters

    """
    
    #aic_scores = []
    bic_scores = []
    silhouette_gmm_scores = []
    silhouette_gmm_std = []
    silhouette_kmeans_scores = []
    silhouette_kmeans_std = []
    
    for n_clusters in cluster_range:
        gmm_model = GaussianMixture(n_components=n_clusters, random_state=random_state)
        gmm_model.fit(data)
        gmm_cluster_labels = gmm_model.fit_predict(data)
        
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=random_state)
        kmeans_model.fit(data)
        kmeans_cluster_labels = kmeans_model.fit_predict(data)
        
        
        #aic_scores.append(model.aic(data))
        #BIC Score only relevant for probabilistc model
        bic_scores.append(gmm_model.bic(data))
        
        if n_clusters > 1:
            sil_samples_gmm = silhouette_samples(data, gmm_cluster_labels)
            silhouette_gmm = sil_samples_gmm.mean()
            sil_std_gmm = sil_samples_gmm.std()
            
            sil_samples_kmeans = silhouette_samples(data, kmeans_cluster_labels)
            silhouette_kmeans = sil_samples_kmeans.mean()
            sil_std_kmeans = sil_samples_kmeans.std()
        else:
            silhouette_gmm = np.nan
            sil_std_gmm = np.nan
            
            silhouette_kmeans = np.nan
            sil_std_kmeans = np.nan
            
        silhouette_gmm_scores.append(silhouette_gmm)
        silhouette_gmm_std.append(sil_std_gmm)
        
        silhouette_kmeans_scores.append(silhouette_kmeans)
        silhouette_kmeans_std.append(sil_std_kmeans)
        
    
    #best_clusters = cluster_range[np.argmin(bic_scores)]
   
    fig, ax1 = plt.subplots(figsize=(5, 3),dpi=300)


    ax1.set_xlabel("Number of Clusters",fontsize=10)
    ax1.set_xticks(cluster_range)
    
    # BIC on left y-axis (inverted)
    ax1.set_ylabel("BIC Score",fontsize=10)
    line1, = ax1.plot(cluster_range, bic_scores, label="BIC - GMM", marker='o', color="blue")
    ax1.tick_params(axis='y',labelsize=10)
    ax1.invert_yaxis() #inverted because lower BIC is better score
    
    # Silhouette Score on right y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("Silhouette Score (SS)",fontsize=10)
    line2 = ax2.errorbar(cluster_range, silhouette_gmm_scores, yerr=silhouette_gmm_std, 
                         marker='s',markersize=8, color="orange",linewidth=3, elinewidth=4,capsize=5)
    line2[0].set_label("SS - GMM")
    line3 = ax2.errorbar(cluster_range, silhouette_kmeans_scores, yerr=silhouette_kmeans_std, 
                         marker='^', markersize=6,color="green",linewidth=1.8, elinewidth=2,capsize=3)
    line3[0].set_label("SS - KMeans")
    ax2.tick_params(axis='y',labelsize=10)
    
    # Collect handles and labels manually
    handles = [line1, line2[0], line3[0]]
    labels = [line.get_label() for line in handles]
    
    # Manually add the legend
    plt.legend(handles, labels, loc='center left', bbox_to_anchor=(-0.3, -0.22), borderaxespad=0,fontsize=7)
    
    # Title and layout
    plt.title(f"{space.capitalize()} Clustering on {method.capitalize()} Features\nEvaluation Metrics (time = {time}h)",fontsize=12,pad=10)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.3)
    plt.show()

     
def apply_clustering(data, mapping = False, n_clusters=3, method='GMM', random_state=22):
    """
    Cluster PCA transformed data using either KMeans or GMM.

    inputs:
    - data: original scaled dataframe to cluster on all features not just reduced
    - n_clusters: int, number of clusters (default is 3)
    - method: str, clustering method to use ('KMeans' or 'GMM')
    - random_state: int or None, random state for reproducibility

    returns:
    - clustered_df: DataFrame containing the PC components and cluster labels
    """

    # check that method input correctly
    if method not in ['KMeans', 'GMM','DBSCAN']:
        raise ValueError("Method must be either 'KMeans' or 'GMM' or 'DBSCAN'.")

    # perform clustering
    if method == 'KMeans':
        model = KMeans(n_clusters=n_clusters, random_state=random_state)
    elif method == 'GMM':
        model = GaussianMixture(n_components=n_clusters, random_state=random_state)
    elif method == 'DBSCAN':
        model = DBSCAN(eps=0.5, min_samples=5)
        
    cluster_labels = model.fit_predict(data)
    #n_found_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    
    #remap cluster labels
    if mapping != False:
        cluster_labels = np.vectorize(mapping.get)(cluster_labels)

    # create cluster labels for dataframe
    clustered_df = data.copy()
    clustered_df['cluster'] = cluster_labels
 
    return clustered_df #, n_found_clusters


def final_cluster_assignment(time,df):
    """
    Fetches the final shape and gene cluster assignment that a cell id evolves into.
    Cell id descendants are one-to-many, this function takes the majority final cluster evolution.

    Parameters
    ----------
    time : Current time
    df : dataframe for current time

    Returns
    -------
    Two new columns for dataframe with shape and gene final cluster assignement

    """
    shape_cluster_pc_values, gene_cluster_pc_values,_,_ = trajectory_fn(plot=False)
    
    shape_map = {}
    for cluster_num in shape_cluster_pc_values.keys():
        cell_list = shape_cluster_pc_values[cluster_num][time]
        for entry in cell_list:
            cell_id = entry["cell_id"]
            if cell_id not in shape_map:
                shape_map[cell_id] = []
            shape_map[cell_id].append(cluster_num)

    gene_map = {}
    for cluster_num in gene_cluster_pc_values.keys():
        cell_list = gene_cluster_pc_values[cluster_num][time]
        for entry in cell_list:
            cell_id = entry["cell_id"]
            if cell_id not in gene_map:
                gene_map[cell_id] = []
            gene_map[cell_id].append(cluster_num)
            
    #for key in sorted(shape_map.keys()):
    #    print(key, shape_map[key])
    
    #pick the majority clusters
    for cell_id in shape_map:
        most_common_cluster = Counter(shape_map[cell_id]).most_common(1)[0][0]
        shape_map[cell_id] = most_common_cluster

    for cell_id in gene_map:
        most_common_cluster = Counter(gene_map[cell_id]).most_common(1)[0][0]
        gene_map[cell_id] = most_common_cluster
        
    return df["id"].map(shape_map), df["id"].map(gene_map)
    

# PLOTTING FUNCTIONS

def plot_clusters(plot_data, c, title, cmap, xlim=None, ylim=None, bar=True,size=10):
    """
    Plot clusters on a 2D scatter plot using the first two principal components (PC1 and PC2).

    input:
    plot_data: dataframe containing the data to be plotted, specifically 'PC1' and 'PC2' columns
    c: array-like, the cluster labels for each data point
    title: str, title of the plot
    cmap

    returns:
    None (displays the plot)
    """
    
    n_clusters = len(c.unique())
    norm = BoundaryNorm(boundaries=np.arange(-0.5, n_clusters + 0.5, 1), ncolors=n_clusters)
    
    plt.figure(figsize=(2.6, 2.5), dpi=300)
    scatter_plot = plt.scatter(plot_data['PC1'], plot_data['PC2'], c=c, cmap=cmap, norm=norm, edgecolors='k',linewidths=0.3, s=size)
    
    if bar:
        plt.colorbar(scatter_plot, ticks=range(n_clusters), label='Cluster')
    
    plt.title(title,fontsize=12)
    plt.xlabel('PC1',fontsize=8)
    plt.ylabel('PC2',fontsize=8)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    
    
def flower_by_feature(dataframe, c, title, cmap, save=False, bar=True, continuous_feature=False,size=50):
    """
    plot flower with the different feature represented
    used for cluster assignment and PCA value

    inputs:
    dataframe: dataframe for plotting contains coordinates
    c: output of feature
    title: str, title of the plot
    feature: str, feature to represent for labelling use
    save: boolean, path for saving optional
    cmap
    
    returns:
    None (displays the plot)
    """
    
    if bar:
        plt.figure(figsize=(2.8, 2.5), dpi=300)
    else:
        plt.figure(figsize=(3, 3), dpi=300)
    
    
    if continuous_feature:
        scatter_plot = plt.scatter(dataframe['x'], dataframe['y'], c=c, cmap=cmap, s=size)
        if bar:
            plt.colorbar(scatter_plot, label='PC1')
    else:
        n_clusters = len(np.unique(c))
        norm = BoundaryNorm(boundaries=np.arange(-0.5, n_clusters + 0.5, 1), ncolors=n_clusters)
        scatter_plot = plt.scatter(dataframe['x'], dataframe['y'], c=c, cmap=cmap, norm=norm, s=size)
        if bar:
            plt.colorbar(scatter_plot, ticks=range(n_clusters), label='Cluster')

    
    plt.title(title,fontsize=12)
    plt.grid(False)
    plt.axis('off')
    plt.tight_layout()

    if save:
        plt.savefig(title + '.png')

    plt.show()  


def plot_pca_feature_weights(loadings, feature_titles, title):
    """
    Plot heat map of PCA feature weights for each principal component.

    input:
    loadings: 2D array-like, PCA loadings (weights)
    feature_titles: list of str, labels for feature
    title: str, title of the entire plot

    returns:
    None (displays the plot)
    """
    
    loadings = loadings[:2, :]
    
    pc_titles = [f"PC{n+1}" for n in range(loadings.shape[0])]
    
    n_components = loadings.shape[0]
    n_features = loadings.shape[1]
    
    
    plt.figure(figsize=(n_features*0.5,n_components))
    sns.heatmap(loadings, annot=True, cmap='Blues', xticklabels=feature_titles, yticklabels=pc_titles, fmt=".2f", annot_kws={"size":7},cbar=False)   

    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.show()
    
    
def plot_pca_variance(plot_data, title):  
    """
    Plot barchart of the variance explained by each principal component
    
    Parameters
    ----------
    plot_data : array of eigenvaues of each PC
    title : str, title of the entire plot

    Returns
    -------
    None (displays the plot)

    """
    
    pc_titles = [f"PC{n+1}" for n in range(plot_data.shape[0])]
    
    plt.figure(figsize=(10,6))
    bars = plt.bar(range(len(plot_data)), plot_data, tick_label = pc_titles)
    
    #Add percentages over bars
    for i,bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height+0.01, f"{round(height*100)}%",ha='center',va='bottom',fontsize=30)

    # Add labels and title
    plt.ylabel("Variance Explained",fontsize=36)
    plt.xlabel("Principal Component",fontsize=36)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.title(title,fontsize=36,pad=15)
    
    max_height = max(plot_data)
    plt.ylim(0, max_height * 1.2)
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    

# EXECUTE FUNCTION

def go(time, shape_cluster_no, gene_cluster_no, reduced = True, save_data = False):
    """
    Calls functions to setup data and generate plots
    
    input: 
    time - time step for which to read data and make plots
    shape_cluster_no (dict) - cluster no for shape space at each timepoint
    gene_cluster_no (dict) - cluster no for gene space at each timepoint
    reduced (Boolean) - cluster on reduced features (PCs) as opposed to all features
    save_data (Boolean) - export the dataframe as csv
    
    returns: 
    None (displays plots)
    
    clustered PCA scatter plots AND flower with clusters represented for
        gene features  (GMM)
        shape features (KMeans)
        shape features (GMM)
    """

    #get dictionary of all cell id mappings between timepoints
    all_dictionary = get_inverted_linss()
    
    #create dictionary of individual cell lineages across timepoints
    #filtered_132 = ATML_filter() #csv files changed s.t t=132 pre-filtered
    filtered_132 = pd.read_csv("data/FM1/csv/t132.csv")
    lineage_dict = sequence_linss(all_dictionary, dataframe = filtered_132, by_cluster=False)
    
    #setup the dataframes for t=xh
    df,gene,shape = dataframe_setup(time,lineage_dict)
    
    ####

    # Flower segmentation by base shape features
    #flower_by_feature(df,df['vol'],title='Volume',cmap='viridis',continuous_feature=True, bar=False)
    #flower_by_feature(df,df['curv1'],title='Maximum Curvature',cmap='viridis',continuous_feature=True, bar=False)
    #flower_by_feature(df,df['curv2'],title='Minimum Curvature',cmap='viridis',continuous_feature=True, bar=False)
    #flower_by_feature(df,df['mean_curv'],title='Mean Curvature',cmap='viridis',continuous_feature=True, bar=False)
    #flower_by_feature(df,df['gauss_curv'],title='Gaussian Curvature',cmap='viridis',continuous_feature=True, bar=False)
    #flower_by_feature(df,df['dev_curv'],title='Deviatoric Curvature',cmap='viridis',continuous_feature=True, bar=False)
    #flower_by_feature(df,df['dist_to_center'],title='Distance to Centre',cmap='viridis',continuous_feature=True, bar=True)

    
    ####
    
    
    # PCA
    pca_genes, weights_genes, variance_genes = apply_pca(gene, n_components=5)
    df['gene_PC1_value'] = pca_genes['PC1']
    df['gene_PC2_value'] = pca_genes['PC2'] 
    
    pca_shape, weights_shape, variance_shape = apply_pca(shape, n_components=5)
    df['shape_PC1_value'] = pca_shape['PC1']
    df['shape_PC2_value'] = pca_shape['PC2']
    
    pca_genes = pca_genes[['PC1', 'PC2']]
    pca_shape = pca_shape[['PC1', 'PC2']]
    
    #produce plots for ideal cluster number using BIC/Silhouette score
    #on all features
    #cluster_optimise(time,gene,space="gene",method="all")
    #cluster_optimise(time,shape,space="shape",method="all")
    #on reduced features
    #cluster_optimise(time,pca_genes,space="gene",method="reduced")
    #cluster_optimise(time,pca_shape,space="shape",method="reduced")
    
    #fetch the number of clusters that should be used for this timestep
    shape_no = shape_cluster_no[time]
    gene_no = gene_cluster_no[time]
    
    
    #clustering
    if reduced:
        # cluster on the reduced features i.e. PC1 and PC2
        clustered_genes_gmm = apply_clustering(pca_genes,mapping=gene_mapping_reduced[time], n_clusters=gene_no, method='GMM')
        clustered_genes_kmeans = apply_clustering(pca_genes,mapping=gene_mapping_reduced[time],n_clusters=gene_no, method='KMeans')
        #clustered_genes_dbscan,n_found_clusters = apply_clustering(pca_genes, n_clusters=gene_no, method='DBSCAN')

        clustered_shape_kmeans = apply_clustering(pca_shape, mapping=shape_mapping_reduced[time], n_clusters=shape_no, method='KMeans')
        clustered_shape_gmm = apply_clustering(pca_shape, mapping=shape_mapping_reduced[time], n_clusters=shape_no, method='GMM')
        #clustered_shape_dbscan,n_found_clusters = apply_clustering(pca_shape, n_clusters=shape_no, method='DBSCAN')
    else:
        #cluster on all features
        clustered_genes_gmm = apply_clustering(gene,mapping=gene_mapping_all[time], n_clusters=gene_no, method='GMM')
        clustered_genes_kmeans = apply_clustering(gene, mapping=gene_mapping_all[time],n_clusters=gene_no, method='KMeans')
        #clustered_genes_dbscan,n_found_clusters = apply_clustering(gene, n_clusters=gene_no, method='DBSCAN')
        
        clustered_shape_kmeans = apply_clustering(shape, mapping=shape_mapping_all[time], n_clusters=shape_no, method='KMeans')
        clustered_shape_gmm = apply_clustering(shape, mapping=shape_mapping_all[time], n_clusters=shape_no, method='GMM')
        #clustered_shape_dbscan,n_found_clusters = apply_clustering(shape, n_clusters=shape_no, method='DBSCAN')
    
    df['gene_cluster_gmm'] = clustered_genes_gmm['cluster']
    df['gene_cluster_kmeans'] = clustered_genes_kmeans['cluster']
    #df['gene_cluster_dbscan'] = clustered_genes_dbscan['cluster']

    df['shape_cluster_gmm'] = clustered_shape_gmm['cluster']
    df['shape_cluster_kmeans'] = clustered_shape_kmeans['cluster']
    #df['shape_cluster_dbscan'] = clustered_shape_dbscan['cluster']
        
    df["shape_cluster_final"], df["gene_cluster_final"] = final_cluster_assignment(time,df)


    ####
    
    #plotting
    
    # for plot titles
    text = f" (t = {time}h)"
    
    #fetch color map with appropriate number of colors for cluster number
    shape_cmap = ListedColormap(cmap_colors[:shape_no])
    gene_cmap = ListedColormap(cmap_colors[:gene_no])
    
    # plot PCA feature weights and PC variance for shapes
    #plot_pca_feature_weights(weights_shape, shape_columns, 'Shape PCA Feature Weights'+text)
    #plot_pca_variance(variance_shape, 'Variance of Shape\nPrincipal Components'+text)

    # plot PCA feature weights and PC variance for genes
    #plot_pca_feature_weights(weights_genes, gene_columns, 'Gene PCA Feature Weights'+text)
    #plot_pca_variance(variance_genes, 'Variance of Gene\nPrincipal Components'+text)

    

    # Reduced space plot (PC1xPC2) for genes (KMeans clustered)
    #plot_clusters(pca_genes, df['gene_cluster_kmeans'], 'Gene Space\n- KMeans'+text, cmap=gene_cmap, xlim=(-8,8), ylim=(-8,8))
    # Flower segmentation by cluster assigment
    #flower_by_feature(df, df['gene_cluster_kmeans'], title="Segmentation on Gene Expression\n- KMeans"+text,cmap=gene_cmap)
    
    # Reduced space plot (PC1xPC2) for genes (GMM clustered)
    #plot_clusters(pca_genes, df['gene_cluster_gmm'], 'Gene Space\n- GMM'+text, cmap=gene_cmap, xlim=(-8,8), ylim=(-8,8))
    # Flower segmentation by cluster assigment
    #flower_by_feature(df, df['gene_cluster_gmm'], title="Segmentation on Gene Expression\n- GMM"+text,cmap=gene_cmap)
    
    # Reduced space plot (PC1xPC2) for genes (DBScan clustered)
    #plot_clusters(pca_genes, df['gene_cluster_dbscan'], 'Gene Space\n- DBSCAN'+text, cmap=ListedColormap(cmap_colors[:n_found_clusters+1]), xlim=(-8,8), ylim=(-8,8))
    # Flower segmentation by cluster assigment
    #flower_by_feature(df, df['gene_cluster_dbscan'], title="Segmentation on Gene Expression\n- DBSCAN"+text,cmap=ListedColormap(cmap_colors[:n_found_clusters+1]))
    
    # Flower segmentation by gene PC1 value
    #flower_by_feature(df,df['gene_PC1_value'],title='Spatial PC1 of Gene Features'+text,cmap='viridis',continuous_feature=True)

    # Flower segmentation by final cluster assignment
    #flower_by_feature(df,df['gene_cluster_final'],title='Final Gene Cluster\nAssignment'+text,cmap=ListedColormap(cmap_colors[:7]))



    # Reduced space plot (PC1xPC2) for shape (KMeans clustered)
    #plot_clusters(pca_shape, df['shape_cluster_kmeans'], 'Shape space\n- KMeans'+text,cmap=shape_cmap,xlim = (-6,6), ylim=(-4,4))
    # Flower segmentation by cluster assigment
    #flower_by_feature(df, df['shape_cluster_kmeans'], title="Segmentation on Morphology\n- KMeans"+text,cmap=shape_cmap)
    
    # Reduced space plot (PC1xPC2) for shape (GMM clustered)
    #plot_clusters(pca_shape, df['shape_cluster_gmm'], 'Shape space\n- GMM'+text,cmap=shape_cmap, xlim = (-6,6), ylim=(-4,4))
    # Flower segmentation by cluster assignment
    #flower_by_feature(df, df['shape_cluster_gmm'], title="Segmentation on Morphology\n- GMM"+text,cmap=shape_cmap)
    
    # Reduced space plot (PC1xPC2) for shape(DBScan clustered)
    #plot_clusters(pca_shape, df['shape_cluster_dbscan'], 'Shape Space\n- DBSCAN'+text, cmap=ListedColormap(cmap_colors[:n_found_clusters+1]), xlim=(-6,6), ylim=(-4,4))
    # Flower segmentation by cluster assigment
    #flower_by_feature(df, df['shape_cluster_dbscan'], title="Segmentation on Morphology\n- DBSCAN"+text,cmap=ListedColormap(cmap_colors[:n_found_clusters+1]))
    
    # Flower segmentation by shape PC1 value
    #flower_by_feature(df,df['shape_PC1_value'],title='Spatial PC1 of Shape Features'+text,cmap='viridis',continuous_feature=True)
    
    # Flower segmentation by final cluster assignment
    #flower_by_feature(df,df['shape_cluster_final'],title='Final Shape Cluster\nAssignment'+text,cmap=ListedColormap(cmap_colors[:3]))
    
    
    # save dataframe
    if save_data:
        df.to_csv(f"t{time}_clustered.csv",index=False)
    

    return weights_genes, weights_shape

# MAIN

def main():
    """
    Runs the program producing plots for multiple time steps
    """
    #all_times = ("0","10","18","24","32","40","48","57","64","72","81","88","96","104","112","120","128","132")
    #for time in all_times: #plot cell coordinates for all times
    #    time = str(time)
    #    string = f"data/FM1/csv/t{time}.csv"
    #    
    #    df = pd.read_csv(string)
    #    
    #    text = f" (t = {time}h)"
    #    flower_by_feature(df, df['vol'], title=""+text,cmap=monochrome,bar=False,size=20)
    
    #ATML_filter(plot=True)
    
    times = ("96","104","112","120","128","132")
    gene_weights_list = []
    shape_weights_list = []
    
    #run for all times
    for time in times:
       gw,sw = go(time,shape_cluster_no,gene_cluster_no,reduced=True,save_data=False)
       gene_weights_list.append(gw)
       shape_weights_list.append(sw)
    
    # run for single time step
    #go("132",shape_cluster_no,gene_cluster_no,reduced=False,save_data=False)
    
    
    
    
    gene_weights_array = np.array(gene_weights_list)  # shape (6, n_PCs, n_features)
    shape_weights_array = np.array(shape_weights_list)
    
    # Average over the time dimension (axis=0)
    avg_gene_weights = np.mean(gene_weights_array, axis=0)  # shape (n_PCs, n_features)
    avg_shape_weights = np.mean(shape_weights_array, axis=0)
    
    # plot PCA feature weights and PC variance for shapes
    #plot_pca_feature_weights(avg_shape_weights, shape_columns, 'Shape PCA Feature Weights')

    # plot PCA feature weights and PC variance for genes
    #plot_pca_feature_weights(avg_gene_weights, gene_columns, 'Gene PCA Feature Weights')
        

if __name__ == "__main__":
    main()

