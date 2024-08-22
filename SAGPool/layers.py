from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
# from torch_geometric.nn.pool.select.topk import topk
# from torch_geometric.nn.pool.connect.filter_edges import filter_adj
from torch.nn import Parameter
import torch
import math

import datetime



class SAGPool(torch.nn.Module):
    def __init__(self,in_channels,fix_pool=None,ratio=0.8,Conv=GCNConv,non_linearity=torch.relu):
        super(SAGPool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.fix_pool = fix_pool
        self.score_layer = Conv(in_channels,1)
        self.non_linearity = non_linearity
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        edge_index = edge_index.to(x.device)
        batch = batch.to(x.device)
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
            
            
        self.ratio = (x.size(0) - self.fix_pool) / x.size(0)
        # if self.fix_pool != None:
        #     x_pool = torch.empty((x.size(0), x.size(1) - self.fix_pool, x.size(-1))).to(x.device)
        # else:
        #     x_pool = torch.empty((x.size(0), math.ceil(x.size(1) * self.ratio), x.size(-1))).to(x.device)

        # print(edge_index.shape, x.shape, batch.shape)
        score = self.score_layer(x,edge_index).squeeze()
        # score = torch.ones((x.shape[0]), device=x.device)
        # print(score.shape, x.shape, edge_index.shape)

        perm = topk(score, self.ratio, batch)
        # print("3333", x[perm].shape, self.non_linearity(score[perm]).view(-1, 1).shape)
        x = x[perm] # * self.non_linearity(score[perm]).view(-1, 1)



        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))
        
        
        return x, edge_index, edge_attr, batch # , perm            

        # TODO
        # DGL、pyg
        for b in range(0, x.size(0)):
            score = self.score_layer(x[b],edge_index_list[b]).squeeze()
            perma = topk(score, self.ratio, batch_list[b])
            perm, _= torch.sort(perma)
            x_pool[b] = x[b][perm] # * self.non_linearity(score[perm]).view(-1, 1)  # 会改变第二维的顺序（点的顺序）

            
            edge_index_list[b], _ = filter_adj(
                edge_index_list[b], None, perm, num_nodes=score.size(0))
        return x_pool, edge_index_list, None, batch_list
    
# class SAGPool(torch.nn.Module):
#     def __init__(self,in_channels,ratio=0.8,Conv=GCNConv,non_linearity=torch.tanh):
#         super(SAGPool,self).__init__()
#         self.in_channels = in_channels
#         self.ratio = ratio
#         self.score_layer = Conv(in_channels,1)
#         self.non_linearity = non_linearity
        
#     def forward(self, x, edge_index, edge_attr=None, batch=None):
#         edge_index = edge_index.to(x.device)
#         batch = batch.to(x.device)
#         if batch is None:
#             batch = edge_index.new_zeros(x.size(0))

#         # print("edge_index,", edge_index.shape)
#         # print("1111", x.shape, batch.shape)
#         score = self.score_layer(x,edge_index).squeeze()
#         # print("3333", score.shape, batch.shape)
#         perm = topk(score, self.ratio, batch)
#         # print("3333", x[perm].shape, self.non_linearity(score[perm]).view(-1, 1).shape)
#         x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)

#         batch = batch[perm]
#         edge_index, edge_attr = filter_adj(
#             edge_index, edge_attr, perm, num_nodes=score.size(0))

#         return x, edge_index, edge_attr, batch # , perm