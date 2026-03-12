import os
import pandas as pd
import numpy as np
import scanpy as sc
import torch
import networkx as nx
from scNET.MultyGraphModel import scNET
from scNET.Utils import save_model, save_obj
import torch
from torch_geometric.utils import  convert
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from scNET.KNNDataset import KNNDataset, CellDataset
from torch.utils.data import DataLoader
import warnings
import gc 
import scNET.Utils as ut
import pkg_resources
from tqdm import tqdm
import warnings
import random 

INTER_DIM = 250
EMBEDDING_DIM = 75
NETWORK_CUTOFF = 0.5
MAX_CELLS_BATCH_SIZE = 4000
MAX_CELLS_FOR_SPLITING = 10000
DE_GENES_NUM = 3000
EXPRESSION_CUTOFF = 0.0
NUM_LAYERS = 3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
warnings.filterwarnings('ignore')
save_path_emb = "/path/to/scNET/Embedding"
save_path_models = "/path/to/scNET/Models"
save_path_knn = "/path/to/scNET/KNNs"

os.makedirs(save_path_emb, exist_ok=True)
os.makedirs(save_path_models, exist_ok=True)
os.makedirs(save_path_knn, exist_ok=True)

def build_network(obj, net, biogrid_flag = False, human_flag = False, remove_non_expressed_genes=True):
    """
    Build a gene-gene network from the provided interaction information.
    Args:
      obj (anndata.AnnData): Single-cell data object (AnnData) containing gene expression data.
      net (pandas.DataFrame): DataFrame containing gene interactions (Source, Target, and Conn columns).
      biogrid_flag (bool, optional): If True, columns for net are set to ["Source", "Target"] only.
      human_flag (bool, optional): If True, keeps gene names unchanged; otherwise adjusts gene name casing.
    Returns:
      tuple:
        pandas.DataFrame: Filtered interaction DataFrame for valid genes.
        networkx.Graph: Graph representation of the gene network.
        pandas.DataFrame: Node-level gene expression features.
    """
    if not biogrid_flag:
        net.columns = ["Source","Target","Conn"]
        net = net.loc[net.Conn >= NETWORK_CUTOFF]
    
    else:
         net.columns = ["Source","Target"]
    
    if not human_flag:
        net["Source"] = net["Source"].apply(lambda x: x[0] + x[1:].lower()).astype(str)
        net["Target"] = net["Target"].apply(lambda x: x[0] + x[1:].lower()).astype(str)

         
    genes = list(pd.concat([net.Source, net.Target]).drop_duplicates())
    genes =  obj.var[obj.var.index.isin(genes)].index
    node_feature = sc.get.obs_df(obj.raw.to_adata(),list(genes)).T
    if remove_non_expressed_genes:
      node_feature["non_zero"] = node_feature.apply(lambda x: x.astype(bool).sum(), axis=1)
      node_feature = node_feature.loc[node_feature.non_zero > node_feature.shape[1] * EXPRESSION_CUTOFF]
      node_feature.drop("non_zero",axis=1,inplace=True)

    net = net.loc[net.Source != net.Target]
    net = net.loc[net.Source.isin(node_feature.index)]
    net = net.loc[net.Target.isin(node_feature.index)]

    gp = nx.from_pandas_edgelist(net, "Source", "Target")

    node_feature = node_feature.loc[list(gp.nodes)]


    return net, gp, node_feature

def test_recon(model,x, data, knn_edge_index):
    """
    Evaluate model reconstruction performance on test edges.
    Args:
      model (torch.nn.Module): Trained scNET model.
      x (torch.Tensor): Input features for the nodes.
      data (torch_geometric.data.Data): Graph data object containing positive and negative edges.
      knn_edge_index (torch.Tensor): k-NN graph edges for the rows.
    Returns:
      float: AUC score of edge reconstruction.
    """
    model.eval()
    with torch.no_grad():
        embbed_rows, _, _ = model(x, knn_edge_index, data.train_pos_edge_index)
    return model.test(embbed_rows, data.test_pos_edge_index, data.test_neg_edge_index)

def pre_processing(adata, n_neighbors): 
    print("Filtering cells", flush=True)
    sc.pp.filter_cells(adata, min_genes=200)

    print("Filtering genes", flush=True)
    sc.pp.filter_genes(adata, min_cells=3)

    print("Setting raw", flush=True)
    adata.raw = adata.copy()  # ❗ avoid copy

    print("Running PCA", flush=True)
    sc.pp.pca(adata, n_comps=50)

    print("Computing neighbors", flush=True)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=15)

    return adata

def crate_knn_batch(knn,idxs,k=15):
  """
  Create a mini-batch of the k-NN graph for the given subset of indices.
  Args:
    knn (scipy.sparse.csr_matrix): Sparse adjacency matrix representing the k-NN graph.
    idxs (list[int]): List of indices used to subset the k-NN graph.
    k (int, optional): Number of nearest neighbors (used for reference if needed).
  Returns:
    torch.Tensor: Edge index for the sub-batch of the k-NN graph.
  """
  idxs = idxs.cpu().numpy()
  adjacency_matrix = torch.tensor(knn[idxs][:,idxs].toarray())
  row_indices, col_indices = torch.nonzero(adjacency_matrix, as_tuple=True)
  knn_edge_index = torch.stack((row_indices, col_indices))
  knn_edge_index = torch.unique(knn_edge_index, dim=1)
  return knn_edge_index.to(device)

def train(data, loader, highly_variable_index,number_of_batches=5 ,
          max_epoch = 500, rduce_interavel = 30,model_name="", cell_flag=False, reshuffle_cells=True):
    """
      Train the scNET model using mini-batches of the k-NN graph or cells.
      Args:
        data (torch_geometric.data.Data): Graph data including edge information.
        loader (torch.utils.data.DataLoader): DataLoader for batches of edges or cells.
        highly_variable_index (pandas.Series or np.ndarray): Boolean mask for highly variable genes.
        number_of_batches (int, optional): Number of mini-batches.
        max_epoch (int, optional): Maximum number of training epochs.
        rduce_interavel (int, optional): Interval at which the model attempts graph reduction.
        model_name (str, optional): Custom string identifier for saving the model and outputs.
        cell_flag (bool, optional): If True, performs mini-batch training by cells rather than by edges.
      Returns:
        scNET: Trained scNET model instance.
      Build a k-NN graph from precomputed distances in the AnnData object.
      Args:
        obj (anndata.AnnData): Single-cell data object with 'distances' stored in obsp.
      Returns:
        tuple:
          torch.Tensor: Edge index of the k-NN graph.
          pandas.Series: Boolean mask for highly variable genes.
      Create a mini-batch DataLoader for k-NN edges.
      Args:
        edge_index (torch.Tensor): All edges of the k-NN graph.
        batch_size (int): Number of edges per mini-batch.
      Returns:
        torch.utils.data.DataLoader: DataLoader object for batching edges.
    """
    x_full = data.x.clone()
    
    if cell_flag:
      model = scNET(x_full.shape[0], x_full.shape[1]//number_of_batches,
                                INTER_DIM, EMBEDDING_DIM, INTER_DIM, EMBEDDING_DIM, lambda_rows = 1, lambda_cols=1,num_layers=NUM_LAYERS).to(device)
      print(x_full.shape[0], x_full.shape[1]//number_of_batches)
    else:
      model = scNET(x_full.shape[0], x_full.shape[1], INTER_DIM, EMBEDDING_DIM, INTER_DIM, EMBEDDING_DIM, 
                                lambda_rows = 1, lambda_cols=1, num_layers=NUM_LAYERS).to(device)
      x = x_full.clone()
      x = ((x.T - (x.mean(axis=1)))/ (x.std(axis=1)+ 0.00001)).T
      

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    best_auc = 0.5 
    concat_flag = False

    pbar = tqdm(range(max_epoch), desc="Training", total=max_epoch)
    for epoch in pbar:

        total_row_loss = 0
        total_col_loss = 0
        num_loss_batches = 0
        col_emb_lst = []
        row_emb_lst = []
        imput_lst = []
        out_features_lst = []
        concat_flag = False 

        for _,batch in enumerate(loader):
            model.train()
           
            if cell_flag:
              x = batch[0].T
              #print(x.shape)
              x = ((x.T - (x.mean(axis=1)))/ (x.std(axis=1)+ 0.00001)).T
              knn_edge_index = crate_knn_batch(loader.dataset.knn, batch[1])

              if reshuffle_cells:
                #print("Reshuffling cells")
                knn_edge_index = knn_edge_index.cpu()
              #  Get expression values of ancor gene for all cells (get the row for ancor gene)
                cell_scores = loader.dataset.ordering[batch[1]]  # Shape: [num_cells]
                cell_sort_idx = torch.argsort(cell_scores, descending=True)
                # Reorder cells (columns) in x
                x = x[:, cell_sort_idx]
                
                # Update knn_edge_index to reflect the cell reordering
                # knn_edge_index has shape [2, num_edges] where each column represents an edge
                # Row 0: source node indices, Row 1: target node indices
                # Create inverse mapping: old_index -> new_index
                # cell_sort_idx[i] = old_index of cell at new position i
                # We need: old_index -> new_index, which is the inverse
                inverse_mapping = torch.zeros_like(cell_sort_idx)
                inverse_mapping[cell_sort_idx] = torch.arange(len(cell_sort_idx), device=cell_sort_idx.device)
                # Remap both source and target node indices in the edge index
                knn_edge_index = inverse_mapping[knn_edge_index]  # Shape [2, num_edges], remaps both rows
                knn_edge_index = knn_edge_index.to(device)
           
            else:
              knn_edge_index = batch.T.to(device)

            if cell_flag or knn_edge_index.shape[1] == loader.dataset.edge_index.shape[0] // number_of_batches :
                
                loss, row_loss, col_loss = model.calculate_loss(x.clone().to(device), knn_edge_index.to(device),
                                                                data.train_pos_edge_index,highly_variable_index)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_row_loss += row_loss.item()
                total_col_loss += col_loss.item()
                num_loss_batches += 1

                with torch.no_grad():
                  if cell_flag:
                    row_embed, col_embed, out_features = model(x.clone().to(device), knn_edge_index, data.train_pos_edge_index)
                    imput = model.encoder(x.to(device), knn_edge_index, data.train_pos_edge_index)
                    col_emb_lst.append(col_embed.cpu())
                    row_emb_lst.append(row_embed.cpu())
                    imput_lst.append(imput.T.cpu())
                    out_features_lst.append(out_features.cpu())
                  else:
                    row_embed, col_embed, out_features = model(x.to(device),knn_edge_index.to(device), data.train_pos_edge_index)

            else:
              concat_flag = True
            
            gc.collect()
            torch.cuda.empty_cache()
        
        # Update progress bar with losses
        avg_row_loss = total_row_loss / num_loss_batches if num_loss_batches > 0 else 0
        avg_col_loss = total_col_loss / num_loss_batches if num_loss_batches > 0 else 0
        pbar.set_postfix({'Row Loss': f'{avg_row_loss:.4f}', 'Col Loss': f'{avg_col_loss:.4f}'})

        if not cell_flag:
          new_knn_edge_index, _ = model.cols_encoder.reduce_network()   

          if concat_flag:
              new_knn_edge_index = torch.concat([new_knn_edge_index,knn_edge_index], axis=-1)
              knn_edge_index = new_knn_edge_index

          if (epoch+1) % rduce_interavel == 0:
              #print(new_knn_edge_index.shape[1] / loader.dataset.edge_index.shape[0])
              loader = mini_batch_knn(new_knn_edge_index, new_knn_edge_index.shape[1] // number_of_batches)
 


        if epoch%10 == 0:
          if not cell_flag:
            knn_edge_index = list(loader)[0].T.to(device)

          auc, ap = test_recon(model, x.to(device), data, knn_edge_index)
          
          if auc > best_auc:
            best_auc = auc

          if cell_flag:
            st = torch.stack(row_emb_lst)
            row_embed = st.mean(dim=0)
            save_obj(
        torch.concat(col_emb_lst).cpu().detach().numpy(),
        os.path.join(save_path_emb, f"col_embedding_{model_name}")
    )

            save_obj(
        row_embed.cpu().detach().numpy(),
        os.path.join(save_path_emb, f"row_embedding_{model_name}")
    )

            save_obj(
        torch.concat(out_features_lst).cpu().detach().numpy(),
        os.path.join(save_path_emb, f"out_features_{model_name}")
    )
          else:
            save_obj(
        new_knn_edge_index.cpu(),
        os.path.join(save_path_knn, f"best_new_knn_graph_{model_name}")
    )

            save_obj(
        col_embed.cpu().detach().numpy(),
        os.path.join(save_path_emb, f"col_embedding_{model_name}")
    )

            save_obj(
        row_embed.cpu().detach().numpy(),
        os.path.join(save_path_emb, f"row_embedding_{model_name}")
    )

            save_obj(
        out_features.cpu().detach().numpy(),
        os.path.join(save_path_emb, f"out_features_{model_name}")
    )

    print(f"Best Network AUC: {best_auc}")
   # if cell_flag:
   #   save_obj(loader, "knn_loader"+model_name)
   # else:
   #   save_obj(new_knn_edge_index.cpu(), "new_knn_graph_"+model_name)

    return model

def build_knn_graph(obj):
    graph = obj.obsp["distances"].toarray()
    graph = (graph > 0).astype(int)
    graph = nx.from_numpy_array(np.matrix(graph))
    ppi_geo = convert.from_networkx(graph)
    edge_index = ppi_geo.edge_index
    sc.pp.highly_variable_genes(obj, n_top_genes=DE_GENES_NUM)
    return edge_index, obj.var.highly_variable

def mini_batch_knn(edge_index, batch_size):
    """
    Create a mini-batch DataLoader for cells and their corresponding edges.
    Args:
      x (torch.Tensor): Matrix of gene expression features.
      edge_index (scipy.sparse.spmatrix): Distance or similarity matrix for cells.
      batch_size (int): Number of cells per mini-batch.
    Returns:
      torch.utils.data.DataLoader: DataLoader object for batching cells.
    Convert a NetworkX graph to a PyTorch Geometric edge index.
    Args:
      G (networkx.Graph): Input NetworkX graph.
      mapping (dict, optional): Dictionary mapping original node IDs to new indices.
    Returns:
      tuple:
        torch.Tensor: PyTorch Geometric edge index.
        dict: Mapping of graph nodes to tensor indices.
    """
    knn_dataset = KNNDataset(edge_index)
    knn_loader = DataLoader(knn_dataset,batch_size=batch_size, shuffle=True, drop_last=False)
    return knn_loader

def mini_batch_cells(x,edge_index, batch_size, ordering=None):
    cell_dataset = CellDataset(x, edge_index, ordering) 
    if ordering is not None:
      cell_loader = DataLoader(cell_dataset,batch_size=batch_size, shuffle=True, drop_last=True)
    else:
      cell_loader = DataLoader(cell_dataset,batch_size=batch_size, shuffle=False, drop_last=True)
    return cell_loader

def nx_to_pyg_edge_index(G, mapping=None):
    G = G.to_directed() if not nx.is_directed(G) else G
    if mapping is None:  
       mapping = dict(zip(G.nodes(), range(G.number_of_nodes())))
    edge_index = torch.empty((2, G.number_of_edges()), dtype=torch.long).to(device)
    for i, (src, dst) in enumerate(G.edges()):
        edge_index[0, i] = mapping[src]
        edge_index[1, i] = mapping[dst]
    return edge_index, mapping

def calculate_cell_embeddings(obj, model=None, model_name=None, 
                               biogrid_flag=False, human_flag=False, 
                               n_neighbors=25, batch_size=1000, split_cells=False, ordering=None, bbknn_flag=False):
    """
    Calculate cell embeddings using a pre-trained or untrained scNET model without training.
    
    Args:
      obj (anndata.AnnData): Single-cell data object (AnnData) containing gene expression data.
      model (scNET, optional): Pre-trained scNET model instance. If None, model will be loaded from model_path or created.
      model_path (str, optional): Path to saved model file. Used if model is None.
      model_name (str, optional): Model name identifier for loading saved embeddings/networks if needed.
      biogrid_flag (bool, optional): If True, use BioGRID-formatted data for network building.
      human_flag (bool, optional): Controls gene name casing in the network.
      n_neighbors (int, optional): Number of neighbors for building the adjacency graph.
      number_of_batches (int, optional): Number of mini-batches (used for model initialization if needed).
      split_cells (bool, optional): If True, process by cells instead of edges.
      ordering (array-like, optional): Cell ordering scores (e.g., pseudotime). If None, will use obj.obs["dpt_pseudotime"].values.
        When provided and split_cells is True, cells are reordered by these scores before processing.
    
    Returns:
      numpy.ndarray: Cell embeddings of shape (n_cells, embedding_dim).
    """
    #obj = sc.pp.subsample(obj, n_obs=obj.obs.shape[0], random_state=42, copy=True)
    # Set model to eval mode
    print("\n========== START calculate_cell_embeddings ==========", flush=True)
    if model is not None:
        model.eval()

    # Ensure raw data exists
    if obj.raw is None:
        obj.raw = obj
        print("raw assigned without copy", flush=True)
    if "distances" not in obj.obsp and not bbknn_flag:
        print("Computing neighbors...", flush=True)
        sc.pp.neighbors(obj, n_neighbors=n_neighbors, n_pcs=15)
    
    if obj.obs.shape[0] > MAX_CELLS_FOR_SPLITING:
        split_cells = True
        print("split_cells=True due to large dataset", flush=True)

    print("Building PPI network...", flush=True)
    if not biogrid_flag:
        net = pd.read_csv(pkg_resources.resource_filename(__name__, r"Data/format_h_sapiens.csv"))[["g1_symbol","g2_symbol","conn"]].drop_duplicates()
        net, ppi, node_feature = build_network(obj, net, human_flag=human_flag, remove_non_expressed_genes=False)
    else:
        net = pd.read_table(pkg_resources.resource_filename(__name__, r"Data/BIOGRID.tab.txt"))[["OFFICIAL_SYMBOL_A","OFFICIAL_SYMBOL_B"]].drop_duplicates()
        net, ppi, node_feature = build_network(obj, net, biogrid_flag, human_flag, remove_non_expressed_genes=False)
    print("After build_network", flush=True)

    ppi_edge_index, _ = nx_to_pyg_edge_index(ppi)
    ppi_edge_index = ppi_edge_index.to(device)
    print("After ppi_edge_index", flush=True)
    obj = obj[:, node_feature.index]
    print("Subset genes done", flush=True)

    print("Preparing feature matrix...", flush=True)
    if split_cells:
        sc.pp.highly_variable_genes(obj, n_top_genes=DE_GENES_NUM)
        highly_variable_index = obj.var.highly_variable
        if highly_variable_index.sum() < 1000 or highly_variable_index.sum() > 15000:
            obj.var["std"] = sc.get.obs_df(obj.raw.to_adata(), list(obj.var.index)).std()
            highly_variable_index = obj.var["std"] >= obj.var["std"].sort_values(ascending=False)[3500]
    else:
        knn_edge_index, highly_variable_index = build_knn_graph(obj)

    print("finish",flush=True)
    x = node_feature.values
    x = torch.tensor(x, dtype=torch.float32).to(device)
    x = ((x.T - (x.mean(axis=1))) / (x.std(axis=1) + 0.00001)).T

    if ordering is None:
      if "dpt_pseudotime" in obj.obs.columns:
        ordering = obj.obs["dpt_pseudotime"].values
      else:
        ut.order_cells(obj)
        ordering = obj.obs["dpt_pseudotime"].values
    
    data = Data(x, ppi_edge_index)
    data = train_test_split_edges(data, test_ratio=0.2, val_ratio=0)
    
    if model is None:
        print("Loading model...", flush=True)
        if model_name is not None:
            if split_cells:
                model = scNET(x.shape[0], batch_size,
                            INTER_DIM, EMBEDDING_DIM, INTER_DIM, EMBEDDING_DIM, 
                            lambda_rows=1, lambda_cols=1, num_layers=NUM_LAYERS).to(device)
            else:
                model = scNET(x.shape[0], x.shape[1], INTER_DIM, EMBEDDING_DIM, 
                            INTER_DIM, EMBEDDING_DIM, lambda_rows=1, lambda_cols=1, 
                            num_layers=NUM_LAYERS).to(device)

            model_path = os.path.join(save_path_models, f"scNET_{model_name}.pt")

            print("loading model from:", model_path)
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            # Create untrained model
            if split_cells:
                model = scNET(x.shape[0], x.shape[1] // number_of_batches,
                            INTER_DIM, EMBEDDING_DIM, INTER_DIM, EMBEDDING_DIM, 
                            lambda_rows=1, lambda_cols=1, num_layers=NUM_LAYERS).to(device)
            else:
                model = scNET(x.shape[0], x.shape[1], INTER_DIM, EMBEDDING_DIM, 
                            INTER_DIM, EMBEDDING_DIM, lambda_rows=1, lambda_cols=1, 
                            num_layers=NUM_LAYERS).to(device)
    
    model.eval()
    print("Starting inference...", flush=True)

    # Calculate embeddings
    with torch.no_grad():
        if split_cells:
            number_of_batches = x.shape[1] // batch_size
            # Process in batches for cell-based approach
            if batch_size > MAX_CELLS_BATCH_SIZE:
                number_of_batches = x.shape[1] // MAX_CELLS_BATCH_SIZE
                batch_size = x.shape[1] // number_of_batches
            
            col_emb_lst = []
            out_features_lst = []
            n_cells = x.shape[1]
            
            # Process cells in batches
            for i in range(number_of_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_cells)
                
                if start_idx >= n_cells:
                    break
                
                # Get batch of cells
                x_batch = x[:, start_idx:end_idx]
                cell_indices = torch.arange(start_idx, end_idx, dtype=torch.long)
                
                # Create k-NN graph for this batch
                knn_edge_index = crate_knn_batch(obj.obsp["distances"], cell_indices, k=n_neighbors)
                
                # Apply ordering if provided
                restore_order = None
                if ordering is not None:
                    knn_edge_index = knn_edge_index.cpu()
                    # Get ordering scores for this batch
                    cell_scores = torch.tensor(ordering[cell_indices.cpu().numpy().tolist()], dtype=torch.float32)
                    cell_sort_idx = torch.argsort(cell_scores, descending=True)
                    # Store the inverse mapping to restore original order later
                    # cell_sort_idx maps: original_idx -> sorted_idx
                    # We need: sorted_idx -> original_idx
                    restore_order = torch.argsort(cell_sort_idx)
                    
                    # Reorder cells (columns) in x_batch
                    x_batch = x_batch[:, cell_sort_idx]
                    
                    # Update knn_edge_index to reflect the cell reordering
                    # Create inverse mapping: old_index -> new_index
                    inverse_mapping = torch.zeros_like(cell_sort_idx)
                    inverse_mapping[cell_sort_idx] = torch.arange(len(cell_sort_idx), device=cell_sort_idx.device)
                    # Remap both source and target node indices in the edge index
                    knn_edge_index = inverse_mapping[knn_edge_index]
                    knn_edge_index = knn_edge_index.to(device)
                    x_batch = x_batch.to(device)
                
                # Get embeddings
                row_embed, col_embed, out_features = model(x_batch, knn_edge_index, data.train_pos_edge_index)
                
                # Restore original order immediately after getting embeddings
                if restore_order is not None:
                    col_embed = col_embed[restore_order]
                    out_features = out_features[restore_order]
                
                col_emb_lst.append(col_embed.cpu()) 
                out_features_lst.append(out_features.cpu())
            
            # Concatenate all embeddings (already in original order)
            cell_embeddings = torch.concat(col_emb_lst, dim=0).numpy()
            out_features = torch.concat(out_features_lst, dim=0).numpy()
        else:
            knn_edge_index, _ = build_knn_graph(obj)
            knn_edge_index = knn_edge_index.to(device)
            row_embed, col_embed, out_features = model(x, knn_edge_index, data.train_pos_edge_index)
            cell_embeddings = col_embed.cpu().numpy()
            out_features = out_features.cpu().numpy()
    print("========== END calculate_cell_embeddings ==========\n", flush=True)
    return cell_embeddings, out_features


def run_scNET(obj,pre_processing_flag = True ,biogrid_flag = False,
          human_flag=False,number_of_batches=5,split_cells = False, n_neighbors=25,
          max_epoch=150, model_name="", save_model_flag = False, bbknn_flag = False,
          subsample_size=None):
  
    """
    Main function to load data, build networks, and run the scNET training pipeline.
    Args:
      obj (AnnData, optional): AnnData obj.
      pre_processing_flag (bool, optional): If True, perform pre-processing steps.
      biogrid_flag (bool, optional): If True, use BioGRID-formatted data for network building.
      human_flag (bool, optional): Controls gene name casing in the network.
      number_of_batches (int, optional): Number of mini-batches for the training.
      split_cells (bool, optional): If True, split by cells instead of edges during training.
      n_neighbors (int, optional): Number of neighbors for building the adjacency graph.
      max_epoch (int, optional): Max number of epochs for model training.
      model_name (str, optional): Identifier for saving the model outputs.
      save_model_flag (bool, optional): If True, save the trained model.
      bbknn_flag (bool, optional): If True, use BBKNN for building the adjacency graph.
    Returns:
      scNET: A trained scNET model.
    """
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
  
    print('start Preprocess',flush=True)
    if pre_processing_flag:
       obj = pre_processing(obj,n_neighbors)

    else:
      if obj.raw is None:
        obj.raw = obj.copy()
      sc.pp.log1p(obj)
      obj.X = obj.raw.X
      if not bbknn_flag:
        print(f"Building k-NN graph with {n_neighbors} neighbors")
        sc.pp.neighbors(obj, n_neighbors=n_neighbors, n_pcs=15)
    print('subsampling')
    print('subsampling',flush=True)
    if subsample_size is not None:

      print("Stratified subsampling to preserve all cell types...",flush=True)

      group_key = "cell_ontology_class"
      target_cells = subsample_size
      random_state = 42
      np.random.seed(random_state)

      total_cells = obj.n_obs
      sampled_indices = []

      groups = obj.obs[group_key].unique()

      for g in groups:
        idx = np.where(obj.obs[group_key] == g)[0]

        # proportional sampling
        proportion = len(idx) / total_cells
        n_sample = max(50, int(proportion * target_cells))   # minimum per type

        if len(idx) <= n_sample:
            sampled_indices.extend(idx)
        else:
            sampled_indices.extend(
                np.random.choice(idx, n_sample, replace=False)
            )

      sampled_indices = np.array(sampled_indices)

    # trim if overshoot
      if len(sampled_indices) > target_cells:
          sampled_indices = np.random.choice(
            sampled_indices, target_cells, replace=False
        )

    # ---- SUBSAMPLE OBJECT ----
      obj = obj[sampled_indices].copy()

      print("Subsample size:", obj.n_obs)
      print("Cell type distribution:",flush=True)
      print(obj.obs[group_key].value_counts())
    # Save subsampled dataset
      save_path = "/workspace/sivakami/scNET/Data/25k_cell.h5ad"

      obj.write(save_path)

      print("Subsampled dataset saved to:", save_path)
  
    if obj.obs.shape[0] > MAX_CELLS_FOR_SPLITING:
       split_cells = True
    
    if split_cells:
       batch_size = obj.obs.shape[0] // number_of_batches
       if batch_size > MAX_CELLS_BATCH_SIZE:
          number_of_batches = obj.obs.shape[0] // MAX_CELLS_BATCH_SIZE
          
    if not biogrid_flag:
      print(pkg_resources.resource_filename(__name__,r"Data/format_h_sapiens.csv"))

      net = pd.read_csv(pkg_resources.resource_filename(__name__,r"Data/format_h_sapiens.csv"))[["g1_symbol","g2_symbol","conn"]].drop_duplicates()
      net, ppi, node_feature = build_network(obj, net,human_flag=human_flag)
      print(f"N genes: {node_feature.shape}")

    else:
      print(pkg_resources.resource_filename(__name__,r"Data/BIOGRID.tab.txt"))
      net = pd.read_table(pkg_resources.resource_filename(__name__,r"Data/BIOGRID.tab.txt"))[["OFFICIAL_SYMBOL_A","OFFICIAL_SYMBOL_B"]].drop_duplicates()
      net, ppi, node_feature  = build_network(obj, net, biogrid_flag,human_flag)
      print(f"N genes: {node_feature.shape}")

    ppi_edge_index, _ = nx_to_pyg_edge_index(ppi)
    ppi_edge_index = ppi_edge_index.to(device)

    if split_cells:
      obj = obj[:,node_feature.index]
      sc.pp.highly_variable_genes(obj,n_top_genes=DE_GENES_NUM)
      highly_variable_index =  obj.var.highly_variable 
      if highly_variable_index.sum() < 1000 or highly_variable_index.sum() > 15000:
        print("Highly variable genes are not in the range of 1000-5000, using std to select highly variable genes",flush=True)
        obj.var["std"] = sc.get.obs_df(obj.raw.to_adata(),list(obj.var.index)).std()
        highly_variable_index = obj.var["std"]  >= obj.var["std"].sort_values(ascending=False)[3500]
      
      print(f"Highly variable genes: {highly_variable_index.sum()}")
      print("CD4 is in the highly variable genes:", "CD4" in obj[:,highly_variable_index].var.index)

  
    else:
      obj = obj[:,node_feature.index]
      knn_edge_index, highly_variable_index = build_knn_graph(obj)    
      loader = mini_batch_knn(knn_edge_index, knn_edge_index.shape[1] // number_of_batches)
  
    highly_variable_index = highly_variable_index[node_feature.index]
    #node_feature.to_csv(pkg_resources.resource_filename(__name__,r"Embedding/node_features_" + model_name))
    node_feature.to_pickle(
    os.path.join(save_path_emb, f"node_features_{model_name}.pkl")
)

    x = node_feature.values

    x = torch.tensor(x, dtype=torch.float32).cpu()
    if split_cells: 
      if subsample_size is not None:
        ordering = np.arange(obj.n_obs)
        loader = mini_batch_cells(
          x,
          obj.obsp["distances"],
          x.shape[1] // number_of_batches,
          ordering=ordering
      )
      else:
        loader = mini_batch_cells(x, obj.obsp["distances"], x.shape[1] // number_of_batches)

    data = Data(x,ppi_edge_index)
    data = train_test_split_edges(data,test_ratio=0.2, val_ratio=0)
    model = train(data, loader, highly_variable_index, number_of_batches=number_of_batches, max_epoch=max_epoch, 
                    rduce_interavel=30,model_name=model_name, cell_flag=split_cells, reshuffle_cells=subsample_size is not None)
    
    if save_model_flag and subsample_size is not None:
      save_model(
    os.path.join(save_path_models, f"scNET_{model_name}.pt"),
    model
)

    #cell_embeddings = calculate_cell_embeddings(obj, model, model_name=model_name, biogrid_flag=biogrid_flag, human_flag=human_flag, n_neighbors=n_neighbors, number_of_batches=number_of_batches, split_cells=split_cells)
    #save_obj(cell_embeddings, pkg_resources.resource_filename(__name__, r"Embedding/cell_embeddings_" + model_name))

    return model

