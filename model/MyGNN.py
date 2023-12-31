import torch.nn as nn
import torch.nn.functional as F
import torch as th
import dgl
from dgl.nn.pytorch import GATConv


def extract_metapaths(category, canonical_etypes, self_loop=False):
    meta_paths_dict = {}
    for etype in canonical_etypes:
        if etype[0] in category:
            for dst_e in canonical_etypes:
                if etype[0] == dst_e[2] and etype[2] == dst_e[0]:
                    if self_loop:
                        mp_name = 'mp' + str(len(meta_paths_dict))
                        meta_paths_dict[mp_name] = [etype, dst_e]
                    else:
                        if etype[0] != etype[2]:
                            mp_name = 'mp' + str(len(meta_paths_dict))
                            meta_paths_dict[mp_name] = [etype, dst_e]
    return meta_paths_dict

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z, nty=None):
        if len(z) == 0:
            return None
        z = th.stack(z, dim=1)
        print(z)
        w = self.project(z).mean(0)                    # (M, 1)
        beta = th.softmax(w, dim=0)
        print(beta)# (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)
        #print(beta)
        #[0.6160],
        #[0.2694],
        #[0.1146]
        return (beta * z).sum(1)                       # (N, D * K)

class MetapathConv(nn.Module):
    def __init__(self, meta_paths_dict, mods, macro_func, **kargs):
        super(MetapathConv, self).__init__()
        # One GAT layer for each meta path based adjacency matrix
        self.mods = mods
        self.meta_paths_dict = meta_paths_dict
        self.SemanticConv = macro_func

    def forward(self, g_list, h_dict):
        outputs = {g.dsttypes[0]: [] for s, g in g_list.items()}
        for mp, meta_path in self.meta_paths_dict.items():
            new_g = g_list[mp]
            h = h_dict[new_g.srctypes[0]]
            outputs[new_g.dsttypes[0]].append(self.mods[mp](new_g, h).flatten(1))
        #semantic_embeddings = th.stack(semantic_embeddings, dim=1)  # (N, M, D * K)
        # Aggregate the results for each destination node type
        rsts = {}
        for ntype, ntype_outputs in outputs.items():
            if len(ntype_outputs) != 0:
                rsts[ntype] = self.SemanticConv(ntype_outputs) # (N, D * K)
        return rsts

class GNN(nn.Module):
    """
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Attributes
    ------------
    _cached_graph : dgl.DGLHeteroGraph
        a cached graph
    _cached_coalesced_graph : list
        _cached_coalesced_graph list generated by *dgl.metapath_reachable_graph()*
    """

    def __init__(self, meta_paths_dict, in_size, out_size, layer_num_heads, dropout):
        super(GNN, self).__init__()
        self.meta_paths_dict = meta_paths_dict
        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        mods = nn.ModuleDict({mp: GATConv(in_size, out_size, layer_num_heads,
                                          dropout, dropout, activation=F.elu,
                                          allow_zero_in_degree=True) for mp in meta_paths_dict})
        self.model = MetapathConv(meta_paths_dict, mods, semantic_attention)
        self._cached_graph = None
        self._cached_coalesced_graph = {}
        self.linear=nn.Linear(in_features=out_size * layer_num_heads,out_features=out_size)
        self.Dense1 = nn.Linear(in_features=out_size, out_features=8, bias=False)
        self.Dense2 = nn.Linear(in_features=8, out_features=1, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, h):
        r"""
        Parameters
        -----------
        g : DGLHeteroGraph
            The heterogeneous graph
        h : tensor
            The input features

        Returns
        --------
        h : tensor
            The output features
        """
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for mp, mp_value in self.meta_paths_dict.items():
                self._cached_coalesced_graph[mp] = dgl.metapath_reachable_graph(
                    g, mp_value)
                #print(mp,self._cached_coalesced_graph[mp])
        #print(h)
        h = self.model(self._cached_coalesced_graph, h)
        x = h['domain']
        x = self.linear(x)
        x = self.Dense1(x)
        x = self.Dense2(x)
        y_pried = self.sigmoid(x)
        y_pried = y_pried.squeeze(-1)
        return y_pried

    def get_embed(self, g, h):
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for mp, mp_value in self.meta_paths_dict.items():
                self._cached_coalesced_graph[mp] = dgl.metapath_reachable_graph(
                    g, mp_value)
                #print(mp,self._cached_coalesced_graph[mp])
        #print(h)
        h = self.model(self._cached_coalesced_graph, h)
        h['domain'] =self.linear(h['domain'])
        return h

