#!/usr/bin/env python
# coding: utf-8

# In[54]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv  # For an initial graph convolution layer.
import numpy as np                   
import matplotlib.pyplot as plt
from torch_geometric.data import Data  # Data structure for graph data.
from torch_geometric.loader import DataLoader
import umap.umap_ as umap
#from torch_scatter import scatter_mean
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GATConv
from torch_geometric.utils import k_hop_subgraph, subgraph
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.cluster import KMeans
#import math
from torchinfo import summary
from torchviz import make_dot
import torch.fx as fx
import os
import optuna

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Setting device to:", device)
#elif torch.backends.mps.is_available:
 #   device = torch.device("mps")  # Instead of "cuda" or "cpu"
  #  print("Setting device to:", device)
else:
    device = torch.device("cpu")
    print("Setting device to:", device)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# **Chromosome Tested**: Chromosome 3

# In[28]:


"""
TODO: Change the path to the Chromosome 3.
"""
path_chr3 = '/scratch/ma8308/Deep_Learning/HiC/imr90/hic_matrix/chr3.npz'  # Load the Hi-C matrix.


# # Creating the Hi-C Matrix

# ## Hi-C Matrix Normalization

# Given a raw matrix $M$, I am going to normalize it according to the equation (1) in the [paper I am following](https://pubmed.ncbi.nlm.nih.gov/37287135/). First we log scale the data, and then we min-max normalize it as follows:
# 
# 1. **Log-transforming the raw matrix**: $M_{\log_{10}} = \log_{10}(1 + M)$
# 
# 2. **Min-max normalization of the log-transformed matrix**: $M_{\text{log}_{10}\_\text{normalized}} = \frac{M_{\log_{10}} - \min_{i,j}(M_{\log_{10}})}{\max_{i,j}(M_{\log_{10}}) - \min_{i,j}(M_{\log_{10}})}$
# 

# In[29]:


def normalize_matrix(matrix):
    # Step 1: Log transform
    matrix_log10 = np.log10(1 + matrix)
    
    # Step 2: Min-max normalization
    min_val = np.min(matrix_log10)
    max_val = np.max(matrix_log10)
    matrix_log10_normalized = (matrix_log10 - min_val) / (max_val - min_val)
    
    return matrix_log10_normalized


# In[30]:


def load_hic_data(path):
    data = np.load(path)  # Load the Hi-C matrix from the NPZ file.
    matrix_size = len(data["0"])  # Assuming all diagonal keys have the same length
    hic_matrix = np.zeros((matrix_size, matrix_size))  # Initialize an empty matrix

    # Fill the matrix using diagonal keys
    for key in data.files:
        diag_values = data[key]
        diag_index = int(key)  # Convert key to integer

        # Upper diagonals
        if diag_index >= 0:
            np.fill_diagonal(hic_matrix[:, diag_index:], diag_values)
        # Lower diagonals (symmetric)
        if diag_index < 0:
            np.fill_diagonal(hic_matrix[-diag_index:, :], diag_values)

    # Ensure symmetry
    hic_matrix = hic_matrix + hic_matrix.T - np.diag(np.diag(hic_matrix))

    hic_matrix_normalized = normalize_matrix(hic_matrix)

    return hic_matrix_normalized


# In[31]:


chr3_hic_matrix = load_hic_data(path_chr3)  # Load and normalize the Hi-C matrix.


# ## Creating the Training Mask
# 
# Here, I am zeroing out a specific window in the Hi-C matrix. This window will not become a part of the graph data because the graph selects for only the positions that have a normalized interaction frequency of $> 0.3$.
# 
# The training mask itself becomes the test set that can be used to measure the performance of the model at the end.

# In[32]:


def mask_defined_window(hic_matrix, start_position, mask_size):
    
    # Mask the window
    hic_matrix[start_position:start_position+mask_size, start_position:start_position+mask_size] = 0

    print(f"Masked matrix \n"
          f"Masked Rows: {start_position}-{start_position + mask_size}"
          f"\nMasked Columns: {start_position}-{start_position + mask_size}")
    
    return hic_matrix
#----------------------------------------------------------------------------------------
start_position = 250  # Starting position of the mask
mask_size = 150  # Size of the mask

chr3_matrix_masked = mask_defined_window(chr3_hic_matrix.copy(), 
                                         start_position=start_position, 
                                         mask_size=mask_size)


# In[96]:


fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=500)

# Top-left: Full 5000x5000 normalized matrix
im1 = axes[0, 0].imshow(chr3_hic_matrix[:5000, :5000], cmap='Reds', interpolation='nearest', origin='lower')
axes[0, 0].set_title("Normalized Hi-C (5000x5000)")
axes[0, 0].set_xlabel("Genomic Position")
axes[0, 0].set_ylabel("Genomic Position")
fig.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04, label="Normalized Contact Frequency")

# Top-right: Zoom-in on 500x500 region
im2 = axes[0, 1].imshow(chr3_hic_matrix[:500, :500], cmap='Reds', interpolation='nearest', origin='lower')
axes[0, 1].set_title("Normalized Hi-C (500x500)")
axes[0, 1].set_xlabel("Genomic Position")
axes[0, 1].set_ylabel("Genomic Position")
fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04, label="Normalized Contact Frequency")

# Bottom-left: Masked version of same 500x500 region
im3 = axes[1, 0].imshow(chr3_matrix_masked[:500, :500], cmap='Reds', interpolation='nearest', origin='lower')
axes[1, 0].set_title("Masked Hi-C (500x500)")
axes[1, 0].set_xlabel("Genomic Position")
axes[1, 0].set_ylabel("Genomic Position")
fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04, label="Normalized Contact Frequency")

# Bottom-right: Just the masked region
im4 = axes[1, 1].imshow(chr3_hic_matrix[start_position:start_position+mask_size, start_position:start_position+mask_size], 
                        cmap='Reds', interpolation='nearest', origin='lower')
axes[1, 1].set_title(f"Masked Region ({start_position}:{start_position+mask_size}x{start_position}:{start_position+mask_size})")
axes[1, 1].set_xlabel("Genomic Position")
axes[1, 1].set_ylabel("Genomic Position")
fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04, label="Normalized Contact Frequency")

plt.tight_layout()
plt.show()


# # Generating the Graph Data

# ## Creating the Graph Dataset 
# 
# I am doing the following to conver the Hi-C matrix into a graph:
# 
# 1. **Normalizing genomic locations**: I am normalizing genmoic locations and converting them to a value between 0-1, which is directly proportional to the genomic distance. This helps in efficient processing by using a smaller number, and it also helps avoid the model placing a higher weight on larger values (more distant loci), just because they are bigger numbers. 
# 
# 2. **Getting the upper triange**: I am using `triu_indices` to get the upper triangle of the matrix and then removing any cells that are zeroes - this also helps in improving efficiency because the Hi-C matrix is symmetrical so it's the same thing either way. 
# 
# 3. **Stacking is necessary for PyTorch Geometric**: PyTorch Geometric expects the edge index to be of the shape `[2, num_edges]`.
# 
# 4. **Calculating the distance between genomic loci**: Again, I am calculating a normalized distance between the nodes (i, j) and I am adding them to the `edge_features` variable. Since it's a bidirectional graph, I am doing $\times 2$ so that I get the same thing for nodes (j, i) too because they are both equivalent. 

# In[34]:


def hic_matrix_to_graph(hic_matrix, chromosome):
    matrix_size = hic_matrix.shape[0]

    # 1. Node features (normalized position)
    node_positions = np.array([i / matrix_size for i in range(matrix_size)]).reshape(-1, 1)

    # 2. Edge indices and features (undirected by design)
    rows, cols = np.triu_indices(matrix_size, k=0)
    non_zero_cells = hic_matrix[rows, cols] > 0.3  # Threshold for non-zero contact frequency
    rows, cols = rows[non_zero_cells], cols[non_zero_cells]

    # Create bidirectional edges
    edge_sources = np.concatenate([rows, cols])
    edge_targets = np.concatenate([cols, rows])

    edge_index = torch.tensor(np.stack([edge_sources, edge_targets]), dtype=torch.long).to(device)

    # Edge features: [distance, interaction_frequency]
    edge_features = []
    node_edge_distance_sum = np.zeros(matrix_size)
    node_edge_count = np.zeros(matrix_size)

    for i, j in zip(rows, cols):
        distance = abs(i - j) / matrix_size
        interaction = hic_matrix[i, j]
        edge_features.extend([interaction] * 2)

        # Sum distances for node i and node j
        node_edge_distance_sum[i] += distance
        node_edge_distance_sum[j] += distance
        node_edge_count[i] += 1
        node_edge_count[j] += 1

    # Avoid divide by zero
    avg_node_distances = node_edge_distance_sum / np.maximum(node_edge_count, 1)
    avg_node_distances = avg_node_distances.reshape(-1, 1)

    chromosome_number = np.full((matrix_size, 1), chromosome)

    # Combine normalized position and average edge distance as node features
    node_features = np.concatenate([node_positions, avg_node_distances, chromosome_number], axis=1)
    node_features = torch.tensor(node_features, dtype=torch.float32).to(device)

    # Final edge attributes
    edge_attr = torch.tensor(edge_features, dtype=torch.float32).unsqueeze(1).to(device)

    # 3. Create PyG Data object
    data = Data(x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr)

    return data.to(device)


# In[ ]:


graph_data = hic_matrix_to_graph(chr3_hic_matrix[:500, :500], chromosome=3)

print("Graph summary:\n",
      f"- Nodes: {graph_data.num_nodes}\n",
      f"- Edges: {graph_data.num_edges}\n",
      f"- Node features: {graph_data.x.shape}\n",
      f"- Edge features: {graph_data.edge_attr.shape}\n")

"""
TODO: Change the input size of the training graph data.
"""

train_graph_data = hic_matrix_to_graph(chr3_hic_matrix[:250, :250], chromosome=3)

print("Graph summary:\n",
      f"- Nodes: {train_graph_data.num_nodes}\n",
      f"- Edges: {train_graph_data.num_edges}\n",
      f"- Node features: {train_graph_data.x.shape}\n",
      f"- Edge features: {train_graph_data.edge_attr.shape}\n",
      f"- Number of Masked Edges: {graph_data.num_edges - train_graph_data.num_edges}\n")

test_graph_data = hic_matrix_to_graph(chr3_hic_matrix[start_position:start_position+mask_size, start_position:start_position+mask_size],
                                      chromosome=3)

print("Graph summary:\n",
      f"- Nodes: {test_graph_data.num_nodes}\n",
      f"- Edges: {test_graph_data.num_edges}\n",
      f"- Node features: {test_graph_data.x.shape}\n",
      f"- Edge features: {test_graph_data.edge_attr.shape}")


# ## Subgraph Extraction
# 
# In this chunk of code, I am doing the following:
# 
# 1. Extracting two subgraphs centered around two different nodes $(u, v)$
# 
# 2. Joining the subgraphs by an edge $(e_{u,v})$. To do this, I sample the nodes $u$ & $v$ from the edge index of the graph object to ensure that we only get the nodes that have an edge present between them. This way, we extract the same triplet that the paper did $(u, e_{u,v}, v)$.
# 
# 3. Finally, I join the two graphs together while keeping only the unique nodes. This essentially replicates the paper's strategy of keeping only unique nodes. 

# In[107]:


def subgraph_sampling_for_contrastive_learning(graph_data, u, v, hops=2, verbose=False):
    
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(u, num_hops=hops, edge_index=graph_data.edge_index, relabel_nodes=False)
    subset_2, edge_index_2, mapping_2, edge_mask_2 = k_hop_subgraph(v, num_hops=hops, edge_index=graph_data.edge_index, relabel_nodes=False)

    combined_subset = torch.cat([subset, subset_2]).unique()

    node_mapping = {orig: idx for idx, orig in enumerate(combined_subset.tolist())}
    
    edge_index_combined, edge_attr_combined = subgraph(combined_subset,
                                                       graph_data.edge_index,
                                                       edge_attr=graph_data.edge_attr,
                                                       relabel_nodes=True,
                                                       num_nodes=graph_data.num_nodes)

    final_data = Data(x=graph_data.x[combined_subset],
                      edge_index=edge_index_combined,
                      edge_attr=edge_attr_combined,
                      original_nodes=combined_subset,
                      u_orig=u,  # Track original u
                      v_orig=v)   # Track original v

    if verbose:
        print("Subset:", len(subset), "Subset 2:", len(subset_2))
        print("Combined subset:", len(combined_subset))
        print("Edge index combined:", edge_index_combined)
        print("Edge attr combined:", edge_attr_combined)
        print("\n=== Final Merged Subgraph ===")
        print(f"Original nodes: {len(combined_subset.tolist())}")
        print(f"Total edges: {edge_index_combined.shape[1]}")
        print("Edge list sample:", edge_index_combined.t()[:5].tolist())

    return final_data


# ## Graph Dataloaders

# In[108]:


class SubgraphDataset(torch.utils.data.Dataset):
    def __init__(self, graph_data, num_hops, verbose=False):
        self.data = graph_data
        self.verbose = verbose
        self.num_hops = num_hops
        # list of target edges
        self.pairs = [(u.item(), v.item())
                      for u, v in graph_data.edge_index.t()]
        self.edge_attrs = graph_data.edge_attr  # Store original edge attributes

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        u, v = self.pairs[idx]
        
        #u_tensor = torch.tensor(u, device=self.data.edge_index.device)
        #v_tensor = torch.tensor(v, device=self.data.edge_index.device)
        
        subgraph_data = subgraph_sampling_for_contrastive_learning(self.data, u, v, hops=self.num_hops, verbose=self.verbose).to(device)
        subgraph_data.real_interaction = self.edge_attrs[idx].to(device)
        return subgraph_data


# In[109]:


train_data = SubgraphDataset(graph_data=train_graph_data, num_hops=2, verbose=False)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)  # Use torch_geometric.loader.DataLoader

# 4. Test one batch
for batch in train_loader:
    print("✅ Batch loaded!")
    print("Batch is of type:", type(batch))
    print("Edge index shape:", batch.edge_index.shape)
    print("Node features shape:", batch.x.shape)
    print("Edge attributes shape:", batch.edge_attr.shape)
    print("Original node indices:", batch.original_nodes[:5])
    print("Original u:", batch.real_interaction)
    break


# # Setting up the Graph Attention Network

# ## Node Embeddings Generation using Graph Attention

# This is the equation of attention coefficients of a node pair $\alpha_{i,j}$. It tells us how much importance to place on the pair's interaction when updating the embeddings for a node. 
# 
# $$
# \alpha_{i,j} =
# \frac{
# \exp\left(\mathrm{LeakyReLU}\left(
# \mathbf{a}^{\top}_{s} \mathbf{\Theta}_{s}\mathbf{x}_i
# + \mathbf{a}^{\top}_{t} \mathbf{\Theta}_{t}\mathbf{x}_j
# \right)\right)}
# {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
# \exp\left(\mathrm{LeakyReLU}\left(
# \mathbf{a}^{\top}_{s} \mathbf{\Theta}_{s}\mathbf{x}_i
# + \mathbf{a}^{\top}_{t}\mathbf{\Theta}_{t}\mathbf{x}_k
# \right)\right)}
# $$

# In[40]:


class Nodes_GraphAttentionNetwork(torch.nn.Module):

    #----------------------------------------------------------------------------        
    def __init__(self, in_features, out_features, hidden_dims, num_heads, dropout, self_loops, bias = False):
        
        super(Nodes_GraphAttentionNetwork, self).__init__()

        self.attention_GCV_first = GATConv(in_channels = in_features, 
                                           out_channels = hidden_dims, 
                                           heads=num_heads, 
                                           dropout=dropout,
                                           bias = bias,
                                           add_self_loops = self_loops)
                
        self.attention_GCV_middle = GATConv(in_channels = hidden_dims*num_heads, 
                                          out_channels = hidden_dims, 
                                          heads=num_heads, 
                                          dropout=dropout,
                                          bias = bias,
                                          add_self_loops = self_loops)
        
        self.attention_GCV_last = GATConv(in_channels = hidden_dims*num_heads, 
                                          out_channels = out_features, 
                                          heads=num_heads, 
                                          dropout=dropout,
                                          bias = bias,
                                          add_self_loops = self_loops)
        
        self.activation = nn.LeakyReLU(negative_slope=0.5)
        
        #for conv in [self.attention_GCV_first, self.attention_GCV_middle, self.attention_GCV_last]:
         #   torch.nn.init.xavier_normal_(conv.lin_src.weight)
          #  if conv.lin_dst is not None:
           #     torch.nn.init.xavier_normal_(conv.lin_dst.weight)

    #----------------------------------------------------------------------------

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.attention_GCV_first(x, edge_index)
        x = self.activation(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.attention_GCV_middle(x, edge_index)
        x = self.activation(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.attention_GCV_last(x, edge_index)
        x = self.activation(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        return x  


# ## Edge Embeddings Generation using an MLP and Node Embeddings

# Uses the same parameters as the fully-connected layer above with the following new additions:
# 
# 1. `num_layers`: This parameter specifies how many layers to use for the MLP. 
# 
# 2. `hidden_dim`: If the parameter `num_layers` is set to anything >1, this parameter will specify the input AND output dimensions of all the layers that come between the first and the last layer. 
# 
# 3. `last_activation`: This is the activation function that will be used for the last layer. 
# 
# 4. `mid_activation`: If the parameter `num_layers` is set to anything >1, this is the activation function that will be used for all the hidden layers. 
# 
# 5. `last_batch_norm`: The 1-dimensional batch normalization to be applied to the last layer. 
# 
# 6. `mid_batch_norm`: If the parameter `num_layers` is set to anything >1, this is the 1-dimensional batch normalization applied to all the hidden layers. 
# 
# *Copied from the [paper](https://github.com/lihan97/sslHiC/blob/main/src/eegnn/layers/mlp.py).*

# In[41]:


#---------------------------------------------------------------------------------------------------------
# Multi-layer perceptron defined below uses the fully-connected layers above.
#---------------------------------------------------------------------------------------------------------

class Edge_MLP(nn.Module):
    """
    A simple multi-layer perceptron built with a series of fully-connected layers.
    """
    def __init__(self, input_dim, output_dim, dropout=0., activation=nn.ReLU(), bias = False):

        super(Edge_MLP, self).__init__()

        self.FCLayer = nn.Linear(in_features=input_dim, out_features=output_dim, bias=bias)
        self.FCLayer_Last = nn.Linear(in_features=output_dim, out_features=output_dim, bias=bias) 
        self.activation = activation   
        self.batch_norm = nn.BatchNorm1d(output_dim)
        if dropout:
            self.dropout = nn.Dropout(p=dropout)


    def forward(self, x):

        x = self.FCLayer(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.batch_norm(x)

        x = self.FCLayer_Last(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.batch_norm(x)

        return x


# ## NT-Xent Loss Function
# 
# The equation for the normalized temperature-scaled cross-entropy loss is as follows:
# 
# $$ 
# \mathbb{l}_{i,j} = -\log\frac{\exp\left(\text{sim}\left(\mathbf{z}_{i}, \mathbf{z}_{j}\right)/\tau\right)}{\sum^{2N}_{k=1}\mathcal{1}_{[k\neq{i}]}\exp\left(\text{sim}\left(\mathbf{z}_{i}, \mathbf{z}_{k}\right)/\tau\right)}
# $$
# 
# NT-Xent Loss function taken from [Lightning AI's Collab Notebook](https://colab.research.google.com/drive/1UK8BD3xvTpuSj75blz8hThfXksh19YbA?usp=sharing#scrollTo=GBNm6bbDT9J3).

# In[42]:


def nt_xent_loss(subgraph_embeddings, u_idx, v_idx, temperature):
    """
    Contrastive loss where:
    - Positive pair: (u, v)
    - Negative pairs: All other node pairs in subgraph
    """
    # Normalize embeddings
    embeddings = F.normalize(subgraph_embeddings, dim=1)
    
    # Calculate similarity matrix
    sim_matrix = torch.mm(subgraph_embeddings, subgraph_embeddings.t()) / temperature
    
    # Positive similarity (u-v)
    pos_sim = sim_matrix[u_idx, v_idx]
    
    # Create mask for negative pairs
    num_nodes = embeddings.size(0)
    neg_mask = torch.ones(num_nodes, num_nodes, dtype=torch.bool)
    
    neg_mask[u_idx, v_idx] = False #This part is masking our target pair (positive pair)
    neg_mask[v_idx, u_idx] = False
    neg_mask[range(num_nodes), range(num_nodes)] = False #This part masks the diagnoal 
    
    # Gather negative similarities
    neg_sim = sim_matrix[neg_mask]
    
    # Calculate loss
    numerator = torch.exp(pos_sim)
    denominator = torch.exp(neg_sim).sum()
    
    return -torch.log(numerator / denominator)


# ## Combined Embedding and Interaction Frequency Training Loop

# $$
# Loss_{weighted} = {\alpha} ({NT-Xent}) + \frac{1}{\alpha}MSE
# $$

# In[43]:


def self_supervised_training_step(subgraph_batch, edge_model, node_model, temperature=0.1, alpha=0.5):
    
    subgraph_batch = subgraph_batch.to(device)
    
    u_embeddings = []
    v_embeddings = []
    
    node_model.train()
    edge_model.train()

    optimizer_node.zero_grad()
    optimizer_edge.zero_grad()

    # Get node embeddings
    h = node_model(subgraph_batch)

    batch_size = subgraph_batch.batch.max().item() + 1  # Number of subgraphs in the batch
    total_node_loss = 0.0
    total_edge_loss = 0.0

    for i in range(batch_size):
        mask = (subgraph_batch.batch == i)
        subgraph_embeddings = h[mask]
        original_nodes = subgraph_batch.original_nodes[mask].tolist()

        try:
            u_pos = original_nodes.index(subgraph_batch.u_orig[i].item())
            v_pos = original_nodes.index(subgraph_batch.v_orig[i].item())
        except ValueError:
            continue  # Skip if nodes not found

        real_interaction_frequency = subgraph_batch.real_interaction[i]  # Get the edge attribute for the current subgraph
    
        h_u = subgraph_embeddings[u_pos]
        h_v = subgraph_embeddings[v_pos]

        stacked_embeddings = torch.stack([h_u.detach(), h_v.detach()]) # 🔥 Prevents the gradients (of h_u, h_v) from flowing into the MSE loss

        predicted_interaction_frequency = edge_model(stacked_embeddings)

        #mean_squared_error = F.mse_loss(predicted_interaction_frequency, real_interaction_frequency)  

        u_embeddings.append(h_u)
        v_embeddings.append(h_v)

        nt_xent_error = nt_xent_loss(subgraph_embeddings, u_pos, v_pos, temperature)

            # Calculate scaled losses
        scaled_nt_xent = alpha * nt_xent_error
        #scaled_mse = (1/alpha) * mean_squared_error
    
        total_node_loss += scaled_nt_xent
        #total_edge_loss += scaled_mse

    average_node_loss = total_node_loss/batch_size  # Average loss over the batch
    #average_edge_loss = total_edge_loss/batch_size  # Average edge loss over the batch
    total_average_loss = (total_node_loss + total_edge_loss)/batch_size
    
    average_node_loss.backward()

    optimizer_node.step()
    #optimizer_edge.step()

    return average_node_loss.item(), total_average_loss.item(), #average_edge_loss.item(), 


# ## Link Prediction

# In[44]:


def link_prediction_test_step(data_test, node_model, edge_model):

  """
    Predicts edge attributes on the test graph (constructed from test nodes only).
    
    Steps:
      - Computes node embeddings for all test nodes.
      - For each edge in the test graph, uses the predictor to map the node embeddings to a predicted edge attribute.
      - Compares predictions to the ground truth using the provided criterion (e.g. MSE).
    """
  node_model.eval()
  edge_model.eval()

  with torch.no_grad():
    
    node_emb = node_model(data_test)  # Get node embeddings
    u = data_test.edge_index[0]
    v = data_test.edge_index[1]

    # Option 2: Use consecutive edges as "views" (better)
    paired_embeddings = []
    for i in range(0, len(u)-1):  # Step by 2
        h_u1 = node_emb[u[i]]
        h_v1 = node_emb[v[i]]
        paired_embeddings.extend([h_u1, h_v1])
    
    both_node_emb_test = torch.stack(paired_embeddings)  # [num_pairs*2, 8]
      
    pred_edge_attr = edge_model(both_node_emb_test)  # Predicted attribute: shape [num_edges, out_attr_dim]
    
    return pred_edge_attr


# # Model Training

# In[45]:


def train_and_plot_simple(epochs, train_loader, node_model, edge_model, temperature=0.5, alpha=0.5, save_model=True):
    total_losses = []
    interaction_losses = []
    embedding_losses = []

    for epoch in range(epochs):
        epoch_total = 0.0
        epoch_interaction = 0.0
        epoch_embedding = 0.0
        num_batches = 0
        
        for batch in train_loader:
            
            batch = batch.to(device)

            node_loss, total_loss_average = self_supervised_training_step(subgraph_batch=batch,
                                                                                             node_model=node_model,
                                                                                             edge_model = edge_model,
                                                                                             temperature=temperature)
            
            # Accumulate losses
            epoch_total += total_loss_average
            #epoch_interaction += edge_loss
            epoch_embedding += node_loss
            num_batches += 1
        
        # Calculate epoch averages
        avg_total = epoch_total / num_batches
        #avg_interaction = epoch_interaction / num_batches
        avg_embedding = epoch_embedding / num_batches
        
        # Store for plotting
        total_losses.append(avg_total)
        #interaction_losses.append(avg_interaction)
        embedding_losses.append(avg_embedding)
        
        print(f"Epoch {epoch+1}/{epochs}",
            f"| Combined Loss: {avg_total:.4f}",
            #f"Interaction Loss: {avg_interaction:.4f}",
            f"| Embedding Loss: {avg_embedding:.4f}")
        
        if save_model:
            if (epoch + 1) % 10 == 0:
                model_save_path = os.path.join(f"node_model_epoch{epoch+1}_no_GPU.pt")
                torch.save(node_model.state_dict(), model_save_path)
                print(f"Saved node_model at {model_save_path}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

    # Plot Total Loss
    axes[0].plot(total_losses, label='Total Loss', color='purple')
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(False)

    # Plot Interaction Loss
    #axes[1].plot(interaction_losses, label='Interaction Loss', color='red')
    #axes[1].set_title('Interaction Loss')
    #axes[1].set_xlabel('Epoch')
    #axes[1].set_ylabel('Loss')
    #axes[1].legend()
    #axes[1].grid(False)

    # Plot Embedding Loss
    axes[2].plot(embedding_losses, label='Embedding Loss', color='green')
    axes[2].set_title('Embedding Loss')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].legend()
    axes[2].grid(False)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    if save_model:
        if (epoch + 1) % 10 == 0:
            plot_save_path = os.path.join(f"loss_plot_epoch_no_GPU_{epoch+1}.png")
            plt.savefig(plot_save_path)
            plt.close()
            print(f"Saved loss plot at {plot_save_path}")

    return total_losses, interaction_losses, embedding_losses

# ## Hyperparameter Optimization

node_model = Nodes_GraphAttentionNetwork(in_features=3, 
                                  out_features=8, 
                                  hidden_dims=64, 
                                  num_heads=1, 
                                  dropout=0.3,
                                  self_loops=True).to(device)

edge_model = Edge_MLP(input_dim=8,
                      output_dim=1,
                      dropout=0.3).to(device)

optimizer_node = torch.optim.Adam(node_model.parameters(), lr=0.01, weight_decay=1e-5)
optimizer_edge = torch.optim.Adam(edge_model.parameters(), lr=0.01, weight_decay=1e-5)
# In[ ]:


def objective(trial):

    # 2.1 Suggest hyperparameters to try
    hidden_dim   = trial.suggest_categorical('hidden_dim',    [16, 32, 64, 128])
    #num_heads    = trial.suggest_categorical('num_heads',      [1, 2, 4])
    dropout_rate = trial.suggest_float      ('dropout_rate',  0.1, 0.8)
    lr           = trial.suggest_float ('lr',          1e-4, 1e-1, log=True)
    wd           = trial.suggest_float ('weight_decay',1e-6, 1e-2, log=True)
    batch_size   = trial.suggest_categorical('batch_size',   [4, 8, 16, 32, 64, 128, 256])
    temperature  = trial.suggest_float      ('temperature',  1e-3, 4.8)

    # 2.2 Build models with these hyperparameters
    node_model = Nodes_GraphAttentionNetwork(in_features=3,
                                             out_features=8,                # keep this fixed or tune as well
                                             hidden_dims=hidden_dim,
                                             num_heads=1,
                                             dropout=dropout_rate,
                                             self_loops=True).to(device)

    edge_model = Edge_MLP(input_dim=8,                   # must match out_features of node model
                          output_dim=1,
                          dropout=dropout_rate).to(device)

    optimizer_node = torch.optim.Adam(node_model.parameters(),
                                      lr=lr,
                                      weight_decay=wd)
    
    optimizer_edge = torch.optim.Adam(edge_model.parameters(),
                                      lr=lr,
                                      weight_decay=wd)

    # 2.3 Split your dataset into train & val
    dataset = SubgraphDataset(graph_data=train_graph_data, num_hops=2)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_val_loss = 0.0
    count = 0

    # 2.4 Train for a small number of epochs
    for epoch in range(5):  # just 5 epochs per trial to save time
        for batch in train_loader:
            batch = batch.to(device)
            node_loss, _ = self_supervised_training_step(subgraph_batch=batch,
                                                         node_model=node_model,
                                                         edge_model=edge_model,
                                                         temperature=temperature,  # you could also tune this
                                                         alpha=1.0)
            total_val_loss += node_loss
            count += 1
    
    # 2.6 Return average validation loss
    avg_val_loss = total_val_loss / max(1, count)
    return avg_val_loss


# In[ ]:


print("Starting hyperparameter tuning with Optuna...")

study = optuna.create_study(study_name="node_model_hyperparameter_tuning", direction='minimize')
study.optimize(objective, n_trials=50)  

print("🔍 Best hyperparameters found:")
for key, val in study.best_params.items():
    print(f"  • {key}: {val:.4g}")

print("Best loss:", study.best_value)


# ## Node (Self-supervised) & Edge (Supervised) Model Training 

# In[ ]:


learning_rate = study.best_params['lr']
hidden_dims = study.best_params['hidden_dim']
dropout = study.best_params['dropout_rate']
weight_decay = study.best_params['weight_decay']
temperature = study.best_params['temperature']  
batch_size = study.best_params['batch_size']    # if you want to reuse batch_size


# In[ ]:


"""
TODO: Change the size of the graph data for training after hyperparameter optimization.
"""

train_graph_data = hic_matrix_to_graph(chr3_hic_matrix[:500, :500], chromosome=3)
train_data = SubgraphDataset(graph_data=train_graph_data, num_hops=2, verbose=False)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)  # Use torch_geometric.loader.DataLoader

# 4. Test one batch
for batch in train_loader:
    print("✅ Batch loaded!")
    print("Batch is of type:", type(batch))
    print("Edge index shape:", batch.edge_index.shape)
    print("Node features shape:", batch.x.shape)
    print("Edge attributes shape:", batch.edge_attr.shape)
    print("Original node indices:", batch.original_nodes[:5])
    print("Original u:", batch.real_interaction)
    break


# In[ ]:


epochs = 10
out_features = 8

node_model = Nodes_GraphAttentionNetwork(in_features=3, 
                                  out_features=out_features, 
                                  hidden_dims=hidden_dims, 
                                  num_heads=1, 
                                  dropout=dropout,
                                  self_loops=True).to(device)

edge_model = Edge_MLP(input_dim=out_features,
                      output_dim=1,
                      dropout=dropout).to(device)

optimizer_node = torch.optim.Adam(node_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
optimizer_edge = torch.optim.Adam(edge_model.parameters(), lr=learning_rate, weight_decay=weight_decay)


# In[ ]:


total_losses, interaction_losses, embedding_losses = train_and_plot_simple(epochs=epochs,
                                                                           train_loader=train_loader,
                                                                           node_model=node_model.to(device),
                                                                           edge_model=edge_model.to(device),
                                                                           temperature=temperature,
                                                                           alpha=1)

