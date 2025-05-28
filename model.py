import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing, GINConv, GraphNorm
from torch_geometric.utils import add_self_loops
from torch.nn import Linear


class MLP(nn.Module):
    def __init__(self, dim_in=256, dim_hidden=128, dim_pred=1, num_layer=1, p_drop=0.1):
        super(MLP, self).__init__()
        layers = []
        in_features = dim_in
        for i in range(num_layer):
            layers.append(nn.Linear(in_features, dim_hidden))
            layers.append(nn.LayerNorm(dim_hidden))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p_drop))
            in_features = dim_hidden
        layers.append(nn.Linear(dim_hidden, dim_pred))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)

    def reset_parameters(self):
        for m in self.fc:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

class GraphNormWithRate(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
    
        self.rate_scale = nn.Linear(1, hidden_dim)
        self.rate_bias = nn.Linear(1, hidden_dim)
        self.eps = 1e-5
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        self.rate_scale.reset_parameters()
        self.rate_bias.reset_parameters()

    def forward(self, x, rate_b=None):
        # x: [num_nodes, hidden_dim]
        mean = x.mean(dim=0, keepdim=True)
        var = x.var(dim=0, unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        if rate_b is not None:
        
            rate_b = rate_b.to(x.device).view(1, 1)
            dynamic_scale = self.rate_scale(rate_b)  # [1, hidden_dim]
            dynamic_bias = self.rate_bias(rate_b)    # [1, hidden_dim]
            gamma = self.weight + dynamic_scale
            beta = self.bias + dynamic_bias
        else:
            gamma = self.weight
            beta = self.bias

        return x_norm * gamma + beta
    
# GraphSAGEConv
class StableEfficientEdgeSAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__(aggr='mean')
        self.lin = Linear(in_channels, out_channels)
        self.skip = Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        self.norm = GraphNormWithRate(out_channels)
        self.dropout = dropout
        self.act = nn.GELU()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if isinstance(self.skip, Linear):
            self.skip.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, x, edge_index, edge_sign, rate_b):
        x_trans = self.lin(x)
        edge_index, edge_sign = add_self_loops(edge_index, edge_attr=edge_sign,
                                               fill_value=1.0, num_nodes=x.size(0))
        m = self.propagate(edge_index, x=x_trans, sign=edge_sign, size=None)
        h = m + self.skip(x)
        h = self.norm(h, rate_b=rate_b) 
        h = self.act(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def message(self, x_j, sign):
        return sign.view(-1, 1) * x_j

class FuncGNN(nn.Module):
    def __init__(self,
                 args,
                 node_num: int,
                 device: torch.device,
                 in_dim: int = 64,
                 out_dim: int = 64,
                 layer_num: int = 3,  
                 dropout: float = 0.1,

                 **kwargs):
        super().__init__(**kwargs)
        self.args = args
        self.node_num = node_num
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.dropout = dropout


        self.x = None
        self.layers = nn.ModuleList()

        layer_num *= 2
    
        for i in range(layer_num):
            if i % 2 == 0:
                self.layers.append(StableEfficientEdgeSAGEConv(
                    in_dim if i == 0 else out_dim, out_dim, dropout))
            else:
                mlp = nn.Sequential(
                    Linear(out_dim, out_dim),
                    nn.GELU(),
                    Linear(out_dim, out_dim)
                )
                gin_conv = GINConv(mlp)
                self.layers.append(gin_conv)

        # Dense 
        self.fuse_linear = Linear(layer_num * out_dim, out_dim)
        self.readout_prob = MLP(out_dim, 128, 1, num_layer=2, p_drop=0.1)
        self.reset_parameters() 

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        if hasattr(self.fuse_linear, 'reset_parameters'):
            self.fuse_linear.reset_parameters()
        if hasattr(self.readout_prob, 'reset_parameters'):
            self.readout_prob.reset_parameters()

    def get_x_edge_index(self, init_emb, edge_index_s):
        """
        Args:
            init_emb: [num_nodes, in_dim]
            edge_index_s:  [E, 3]
        """
        edge_index = edge_index_s[:, :2].t().contiguous()
        edge_sign = edge_index_s[:, 2].float().to(self.device)

        if init_emb is None:
            init_emb = torch.randn(self.node_num, self.in_dim, device=self.device)

        self.x = init_emb
        self.edge_index = edge_index
        self.edge_sign = edge_sign

    def forward(self, init_emb, edge_index_s, rate_b) -> Tensor:
        self.get_x_edge_index(init_emb, edge_index_s)
        x = self.x
        outputs = []
        for layer in self.layers:
            if isinstance(layer, StableEfficientEdgeSAGEConv):
                x = layer(x, self.edge_index, self.edge_sign, rate_b)
            else:
                x = layer(x, self.edge_index)
            outputs.append(x)
        x_dense = torch.cat(outputs, dim=1)
        x_final = self.fuse_linear(x_dense)
        prob = self.readout_prob(x_final)
        prob = torch.sigmoid(prob)
        return x_final, prob