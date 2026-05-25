import torch
import torch.nn as nn
import torch.nn.functional as F

class TongueGNN(nn.Module):
    def __init__(self, d_r=512, d_g=768, num_layers=3, num_heads=8, dropout=0.1):
        super().__init__()
        self.d_r = d_r
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.global_proj = nn.Linear(d_g, d_r)
        self.node_proj = nn.Sequential(
            nn.Linear(d_r, d_r), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_r, d_r)
        )
        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(MultiHeadGATLayer(in_features=d_r, out_features=d_r // num_heads, num_heads=num_heads, dropout=dropout))
            self.norm_layers.append(nn.LayerNorm(d_r))

        adj = torch.tensor([
            [1, 1, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]
        ], dtype=torch.bool)
        self.register_buffer('adj_matrix_with_loop', adj | torch.eye(6, dtype=torch.bool))

    def forward(self, r_local, r_global):
        B = r_local.size(0)
        r_global_aligned = self.global_proj(r_global)
        r_local = self.node_proj(r_local)
        r_global_aligned = self.node_proj(r_global_aligned)
        H = torch.cat([r_local, r_global_aligned], dim=1)
        adj = self.adj_matrix_with_loop.unsqueeze(0).expand(B, -1, -1)

        for i in range(self.num_layers):
            H_new = self.gat_layers[i](H, adj)
            H = self.norm_layers[i](H + 0.5 * H_new)  # 缩放残差防过平滑
        return H

class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.dropout = dropout
        self.W = nn.Linear(in_features, num_heads * out_features, bias=False)
        self.a = nn.Parameter(torch.zeros(2 * out_features))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

    def forward(self, h, adj):
        B, N, _ = h.shape
        Wh = self.W(h).view(B, N, self.num_heads, self.out_features).transpose(1, 2)
        Wh_i = Wh.unsqueeze(3).expand(-1, -1, N, N, self.out_features)
        Wh_j = Wh.unsqueeze(2).expand(-1, -1, N, N, self.out_features)
        e = self.leaky_relu(torch.einsum("bhnjd,d->bhnj", torch.cat([Wh_i, Wh_j], dim=-1), self.a))
        e = e.masked_fill(~adj.unsqueeze(1), float('-inf'))
        attention = F.softmax(e, dim=-1)
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh).transpose(1, 2).contiguous()
        return F.elu(h_prime.view(B, N, self.num_heads * self.out_features))