import os
import torch, scipy, json
from torch import nn
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from torch_geometric import seed_everything
import torch.nn.functional as F
import torch_geometric.transforms as T
import networkx as nx
import numpy as np

class GCNEncoder(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        
        hidden=2*out_channels
        layers = nn.ModuleList()
        layers.append(GCNConv(in_channels=in_channels, out_channels=hidden, bias=args.bias))
        
        for i in range(args.num_layers - 2):
            layers.append(GCNConv(in_channels=hidden, out_channels=hidden, cached=True, num_workers=2, bias=args.bias))
        
        layers.append(GCNConv(in_channels=hidden, out_channels=out_channels, cached=True, num_workers=2, bias=args.bias))
        
        self.layers = layers
        self.dropout = args.dropout
        self.activation = args.act

    def forward(self, x, edge_index):
        
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            if self.activation == 'relu':
                x = x.relu()
            elif self.activation == 'tanh':
                x = x.tanh()

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
                
        x = self.layers[-1](x, edge_index) # last layer before latent space does not have activation
        
        return x
        
def runGCN(coo, n_features, args, encoder=GCNEncoder):
    
    def train():
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_label_index[:, train_data.edge_label == 1])
        loss = model.recon_loss(z, train_data.edge_label_index[:, train_data.edge_label == 1],
                                train_data.edge_label_index[:, train_data.edge_label == 0])
        loss.backward()
        optimizer.step()
        return float(loss)

    def test():
        model.eval()
        with torch.no_grad():
            z = model.encode(train_data.x, train_data.edge_label_index[:, train_data.edge_label == 1])
        return model.test(z, train_data.edge_label_index[:, train_data.edge_label == 1],
                          train_data.edge_label_index[:, train_data.edge_label == 0])
    
    # Setup
    device = torch.device(args.device)
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    
    transform = T.Compose([
                T.ToDevice(device),
                T.RandomLinkSplit(num_val=0.0, num_test=0.0,
                                  is_undirected=args.symmetrize_adj, add_negative_train_samples=True)])

    data = Data(x=torch.eye(n_features),
                edge_index=torch.sparse_coo_tensor(i, v, torch.Size(shape)).to_dense().nonzero().t().contiguous())
        
    train_data, val_data, test_data = transform(data)
    del(data)
    # parameters
    out_channels = args.dim
    num_features = n_features

    # model
    model = GAE(encoder(args, num_features, out_channels))

    # move to GPU (if available)
    model = model.to(device)
    
    # inizialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_auroc = 0
    count = 0
    best_model = model
    for epoch in range(1, args.epochs + 1):
        train_loss = train()
        train_auroc, train_auprc = test()
        if (train_auroc > best_auroc):
            best_auroc = train_auroc
            best_model = model
            count = 0
        else:
            count += 1
        if (count >= args.patience) & (count >= args.min_epochs):
            break
    
    del (train_data)
    del (val_data)
    del (test_data)
    gae_latent = best_model.encode(torch.eye(n_features).to(device),
                                   torch.sparse_coo_tensor(i, v, torch.Size(shape)).to_dense().nonzero().t().contiguous().to(device))
    
    return (gae_latent.detach().cpu().numpy())

def run_gae(data, args):

    seed_everything(args.seed)

    if args.symmetrize_adj == 0:
        G = nx.from_pandas_edgelist(data, source='source_genesymbol', target='target_genesymbol')
    else:
        G = nx.from_pandas_edgelist(data, source='source_genesymbol', target='target_genesymbol', create_using=nx.DiGraph)
        
    A = scipy.sparse.coo_matrix(nx.adjacency_matrix(G))
    embedding = runGCN(A, n_features=G.number_of_nodes(), args=args)
    
    if not os.path.exists(f'results/{args.model}/{args.dataset}'):
        os.makedirs(f'results/{args.model}/{args.dataset}')
        
    np.savez_compressed(f'results/{args.model}/{args.dataset}/{args.save_as}_results.npz',
                        embedding=embedding,
                        config=vars(args),
                        names=G.nodes)
