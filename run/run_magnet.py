import pandas as pd
import scipy
import networkx as nx
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from torch_geometric_signed_directed.utils import link_class_split, in_out_degree
from torch_geometric_signed_directed.data import DirectedData
from torch_geometric_signed_directed.nn.directed import complex_relu_layer
from torch_geometric_signed_directed.nn.directed import MagNetConv
from torch_geometric.utils import train_test_split_edges

device = torch.device('cpu')

class MagNet_link_prediction(nn.Module):
    r"""The MagNet model for link prediction from the
    `MagNet: A Neural Network for Directed Graphs. <https://arxiv.org/pdf/2102.11391.pdf>`_ paper.

    Args:
        num_features (int): Size of each input sample.
        hidden (int, optional): Number of hidden channels.  Default: 2.
        K (int, optional): Order of the Chebyshev polynomial.  Default: 2.
        q (float, optional): Initial value of the phase parameter, 0 <= q <= 0.25. Default: 0.25.
        label_dim (int, optional): Number of output classes.  Default: 2.
        activation (str, optional): whether to use activation function or not. (default: :obj:`complexrelu`)
        trainable_q (bool, optional): whether to set q to be trainable or not. (default: :obj:`False`)
        layer (int, optional): Number of MagNetConv layers. Default: 2.
        dropout (float, optional): Dropout value. (default: :obj:`0.5`)
        normalization (str, optional): The normalization scheme for the magnetic
            Laplacian (default: :obj:`sym`):
            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A} Hadamard \exp(i \Theta^{(q)})`
            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2} Hadamard \exp(i \Theta^{(q)})`
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the __norm__ matrix on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
    """

    def __init__(self, num_features: int, hidden: int = 2, q: float = 0.25, K: int = 2, label_dim: int = 2,
                 activation: str = 'complexrelu', trainable_q: bool = False, layer: int = 2, dropout: float = 0.5, normalization: str = 'sym', cached: bool = False, bias: bool = True):
        super(MagNet_link_prediction, self).__init__()

        chebs = nn.ModuleList()
        chebs.append(MagNetConv(in_channels=num_features, out_channels=hidden, K=K,
                                q=q, trainable_q=trainable_q, normalization=normalization, cached=cached,
                                bias=bias))
        self.normalization = normalization
        self.activation = activation
        if self.activation == 'complexrelu':
            self.complex_relu = complex_relu_layer()

        for _ in range(1, layer):
            chebs.append(MagNetConv(in_channels=hidden, out_channels=hidden, K=K,
                                    q=q, trainable_q=trainable_q, normalization=normalization, cached=cached))

        self.Chebs = chebs
        self.linear = nn.Linear(hidden*4, label_dim)
        self.dropout = dropout

    def reset_parameters(self):
        for cheb in self.Chebs:
            cheb.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, real: torch.FloatTensor, imag: torch.FloatTensor, edge_index: torch.LongTensor,
                query_edges: torch.LongTensor, edge_weight: None) -> torch.FloatTensor:
        """
        Making a forward pass of the MagNet node classification model.

        Arg types:
            * real, imag (PyTorch Float Tensor) - Node features.
            * edge_index (PyTorch Long Tensor) - Edge indices.
            * query_edges (PyTorch Long Tensor) - Edge indices for querying labels.
            * edge_weight (PyTorch Float Tensor, optional) - Edge weights corresponding to edge indices.
        Return types:
            * log_prob (PyTorch Float Tensor) - Logarithmic class probabilities for all nodes, with shape (num_nodes, num_classes).
        """
        for cheb in self.Chebs:
            real, imag = cheb(real, imag, edge_index, edge_weight)
            if self.activation:
                real, imag = self.complex_relu(real, imag)
                
        embedding = torch.cat((real, imag), dim=1)
        
        x = torch.cat((real[query_edges[:, 0]], real[query_edges[:, 1]],
                      imag[query_edges[:, 0]], imag[query_edges[:, 1]]), dim=-1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return embedding, x
    
def train(model, optimizer, X_real, X_img, y, edge_index, edge_weight, query_edges):
    model.train()
    embedding, out = model(X_real, X_img, edge_index=edge_index,
                    query_edges=query_edges,
                    edge_weight=edge_weight)
    
    criterion = torch.nn.NLLLoss()
    loss = criterion(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_acc = accuracy_score(y.cpu(),
    out.max(dim=1)[1].cpu())
    return loss.detach().item(), train_acc, embedding

def test(model, X_real, X_img, y, edge_index, edge_weight, query_edges):
    model.eval()
    with torch.no_grad():
        embedding, out = model(X_real, X_img, edge_index=edge_index,
                    query_edges=query_edges,
                    edge_weight=edge_weight)
    test_acc = accuracy_score(y.cpu(),
    out.max(dim=1)[1].cpu())
    return test_acc, embedding

def run_magnet(data, args):
    
    G = nx.from_pandas_edgelist(data, source='source_genesymbol', target='target_genesymbol', create_using=nx.DiGraph)
    coo = scipy.sparse.coo_matrix(nx.adjacency_matrix(G))
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    edge_index = torch.sparse_coo_tensor(i, v, torch.Size(shape)).to_dense().nonzero().t().contiguous()
    data = DirectedData(x=torch.Tensor(np.eye(G.number_of_nodes())), edge_index=edge_index)
    
    link_data = link_class_split(data, prob_val=args.val_prop, prob_test=args.test_prop, task=args.task,
                                 device=device, seed=args.split_seed, splits=1)
    
    model = MagNet_link_prediction(q=args.q, K=1, num_features=data.x.shape[0],
                                   hidden=int(args.dim / 2), label_dim=args.num_classes, bias=bool(args.bias), dropout=args.dropout,
                                   activation=args.act, layer=args.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    edge_index = link_data[0]['graph']
    edge_weight = link_data[0]['weights']
    query_edges = link_data[0]['train']['edges']
    y = link_data[0]['train']['label']
    
    query_val_edges = link_data[0]['val']['edges']
    y_val = link_data[0]['val']['label']
    
    X_real = torch.eye(data.x.shape[0]).to(device)
    X_img = X_real.clone()
    
    best_val_acc = 0
    count = 0
    best_model = model
    for epoch in range(args.epochs):
        train_loss, train_acc, train_embedding = train(model, optimizer, X_real, X_img, y, edge_index, edge_weight, query_edges)
        val_acc, val_embedding = test(model, X_real, X_img, y_val, edge_index, edge_weight, query_val_edges)
        if (val_acc > best_val_acc):
            best_val_acc = val_acc.copy()
            best_model = model
            count = 0
        else:
            count += 1
        if (count >= args.patience) & (count >= args.min_epochs):
            break
            
    X_real = torch.eye(data.x.shape[0]).to(device)
    X_img = X_real.clone()

    embedding, out = best_model(X_real, X_img, edge_index=data.edge_index,
                           query_edges = data.edge_index.T, edge_weight=data.edge_weight)
    
    if not os.path.exists(f'results/{args.model}/'):
        os.makedirs(f'results/{args.model}')
    
    np.save(f'results/{args.model}/{args.save_as}_{args.dataset}_embedding.npy', embedding.detach().numpy())
    with open(f'results/{args.model}/{args.save_as}_{args.dataset}_config.json', 'w') as f:
        json.dump(vars(args), f)
