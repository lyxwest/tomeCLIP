from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
# from torch_geometric.nn.pool.select.topk import topk
# from torch_geometric.nn.pool.connect.filter_edges import filter_adj
from torch.nn import Parameter
import torch
import math

class SAGPool(torch.nn.Module):
    def __init__(self,in_channels,ratio=0.8,Conv=GCNConv,non_linearity=torch.tanh):
        super(SAGPool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels,1)
        self.non_linearity = non_linearity
        
    def forward(self, x, edge_index_list, edge_attr=None, batch=None):
        # edge_index_list = edge_index_list.to(x.device)
        if batch is None:
            batch_list = []
            for b in range(0, x.size(0)):
                batch = edge_index_list[b].new_zeros(x.size(1))
                batch_list.append(batch)

        x_pool = torch.empty((x.size(0), math.ceil(x.size(1) * self.ratio), x.size(-1))).to(x.device)
        # TODO
        # DGL、pyg
        for b in range(0, x.size(0)):
            score = self.score_layer(x[b],edge_index_list[b]).squeeze()
            perma = topk(score, self.ratio, batch_list[b])
            perm, _= torch.sort(perma)
            x_pool[b] = x[b][perm]# * self.non_linearity(score[perm]).view(-1, 1)  # 会改变第二维的顺序（点的顺序）
            # x_pool[b] = torch.cat([x[b], x[b][-1] + torch.zeros(1, x[b][-1].shape[-1], dtype=x.dtype, device=x.device)], dim=0)  # 会改变第二维的顺序（点的顺序）
            # batch_list[b] = batch_list[b][perm]
            
            edge_index_list[b], _ = filter_adj(
                edge_index_list[b], None, perm, num_nodes=score.size(0))
            
        return x_pool, edge_index_list, None, batch_list#, perm
    
    
# class SAGPool(torch.nn.Module):
#     def __init__(self,in_channels,ratio=0.8,Conv=GCNConv,non_linearity=torch.tanh):
#         super(SAGPool,self).__init__()
#         self.in_channels = in_channels
#         self.ratio = ratio
#         self.score_layer = Conv(in_channels,1)
#         self.non_linearity = non_linearity
        
#     def forward(self, x, edge_index, edge_attr=None, batch=None):
#         edge_index = edge_index.to(x.device)
#         if batch is None:
#             batch = edge_index.new_zeros(x.size(0))

#         # print("edge_index,", edge_index.shape)
#         # print("1111", x.shape)      # 1111 torch.Size([35518, 128])
#         score = self.score_layer(x,edge_index).squeeze()
#         # print("3333", score.shape)  # 2222 torch.Size([35518])
#         perm = topk(score, self.ratio, batch)
#         print("3333", x[perm].shape, self.non_linearity(score[perm]).view(-1, 1).shape)
#         x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
#         # print("4444", x.shape)
#         batch = batch[perm]
#         edge_index, edge_attr = filter_adj(
#             edge_index, edge_attr, perm, num_nodes=score.size(0))

#         return x, edge_index, edge_attr, batch, perm