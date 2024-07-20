import torch
import numpy as np
import scipy.sparse as sp
from typing import Dict, Any, Tuple, Optional
from torch import nn

# from ..utils import isinstance_str, init_generator
from timm.models.vision_transformer import Attention

from ..utils import parse_r
from ..merge import merge_wavg, merge_source, bipartite_soft_matching #bipartite_soft_matching_random2d

from SAGPool.layers import SAGPool

class ToMeResidualAttentionBlock(nn.Module):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """
    # def __init__(self):
    #     super(ToMeResidualAttentionBlock, self).__init__()
    #     self.edge_index = self.create_adj(16, 16, 3, 8)

    def pool(self, x, edge_index_list, edge_attr=None, batch=None):
        pool = SAGPool(1024, ratio=0.99).to(x.device)
        return pool(x, edge_index_list, edge_attr, batch)
    
    # copy from timm.py,不加则 ls_12报错无定义
    def ls_1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else x

    def ls_2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else x
    # end for copy

    def forward(self, 
                # q_x: torch.Tensor, edge_index_list: list,
                data: tuple,
                attn_mask: Optional[torch.Tensor] = None):
        
        q_x, edge_index_list = data
        
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        
        # x_attn.shape: torch.Size([64, 257, 1024])
        # metric.shape: torch.Size([64, 257, 64])
        x_attn, metric = self.attn(self.ln_1(q_x), attn_size)
        x = q_x + self.ls_1(x_attn)

        # original tome        
        # r = self._tome_info["r"].pop(0) 
        # if r > 0:
        #     # Apply ToMe here
        #     merge, _ = bipartite_soft_matching(
        #         metric,
        #         r,
        #         self._tome_info["class_token"],
        #         self._tome_info["distill_token"],
        #     )
        #     if self._tome_info["trace_source"]:
        #         self._tome_info["source"] = merge_source(
        #             merge, x, self._tome_info["source"]
        #         )
        #     # x.shape: torch.Size([64, 257, 1024])
        #     x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])
        #     # x.shape: torch.Size([64, 241, 1024])
        #     # self._tome_info["size"]: 1
        # 
        # cls token 不进入 pooling
        #
        cls = torch.unsqueeze(x[:, 0, :], 1)
        pooled, edge_index_list, _,  _ = self.pool(x[:, 1:, :], edge_index_list, None, None)
        x = torch.cat([cls, pooled], dim=1)
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return (x, edge_index_list)


class ToMeAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, 
        size: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
        need_weights = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)
        # q k v shape: torch.Size([64, 16, 257, 64])
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn.shape: torch.Size([64, 16, 257, 257])
        # Apply proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Return k as well here
        return x, k.mean(1)
    
def convert_attention_block(
    src: nn.MultiheadAttention, dst: ToMeAttention
) -> Tuple[ToMeAttention, torch.device]:
    src_state_dict = src.state_dict()
    dst_state_dict = dst.state_dict()
    src_to_dst_keys = [
        ("in_proj_weight", "qkv.weight"),
        ("in_proj_bias", "qkv.bias"),
        ("out_proj.weight", "proj.weight"),
        ("out_proj.bias", "proj.bias"),
    ]

    for src_key, dst_key in src_to_dst_keys:
        dst_state_dict[dst_key] = src_state_dict[src_key]
    dst.load_state_dict(dst_state_dict)
    src_device = src_state_dict["in_proj_weight"].device
    return dst.to(src_device), src_device


def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self._tome_info["r"] = parse_r(len(self.transformer.resblocks), self.r) #self._tome_info["ratio"]
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            # to patches - whether to use dual patchnorm - https://arxiv.org/abs/2302.01327v1
            # if self.input_patchnorm is not None: # 不注释掉 报错 'ToMeVisionTransformer' object has no attribute 'patchnorm'
            #     # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
            #     x = x.reshape(x.shape[0], x.shape[1], self.grid_size[0], self.patch_size[0], self.grid_size[1], self.patch_size[1])
            #     x = x.permute(0, 2, 4, 1, 3, 5)
            #     x = x.reshape(x.shape[0], self.grid_size[0] * self.grid_size[1], -1)
            #     x = self.patchnorm_pre_ln(x)
            #     x = self.conv1(x)
            # else:
            # (conv1): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14), bias=False)
            # print("x:", x.shape) # x: torch.Size([64, 3, 224, 224])
            x = self.conv1(x)  # shape = [bs*n_seg, demension, 16, 16]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [bs*n_seg, demension, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, 256, width]

            # class embeddings and positional embeddings
            x = torch.cat(
                [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            x = x + self.positional_embedding.to(x.dtype)

            # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
            # x = self.patch_dropout(x)     # 不注释掉 报错'ToMeVisionTransformer' object has no attribute 'patch_dropout'
            x = self.ln_pre(x)
            # print(self.edge_index_list[0].shape)
            edge_index_list = adj_to_edge(create_adj(16,16,3,8))
            x, edge_index_list = self.transformer((x, edge_index_list))

            # 不注释会报错
            # if hasattr(self, "attnpool"):
            #     x = self.attnpool(x)
            #     x = self.ln_post(x)
            #     pooled, tokens = self._global_pool(x)
            # else:
            #     pooled, tokens = self._global_pool(x)
            #     pooled = self.ln_post(pooled)
            # x.shape: torch.Size([64, 2, 1024])
            if self.proj is not None:
                # pooled = pooled @ self.proj
                # print(x.shape)      # torch.Size([1, 65, 1024])
                x = self.ln_post(x[:, 0, :])
                # print(x.shape)
                pooled = x @ self.proj
                # print(pooled.shape)

            # 不注释会报错
            # if self.output_tokens:
            #     return pooled, tokens
            
            return pooled

    return ToMeVisionTransformer


def create_adj( H, W, C, neibour):
    """
    功能：
        根据featuremap的高和宽建立对应的空域邻接矩阵,
    输入：
        h featuremap的高度
        w featuremap的宽
        C featuremap的通道数 
        neibour  4或8决定空域adj的邻居数   2 决定计算channel的adj
    """
    h = H
    w = W
    n = h*w
    x = [] #保存节点
    y = [] #保存对应的邻居节点
    #判断是生成8邻居还是4邻居
    if neibour==8:
        l =np.reshape(np.arange(n),(h,w))
        # print(l)
        # print(((l[:,2])+w)[:1])
        #print(l[:,2])
        for i in range(h): 
            #邻界条件需要考虑，故掐头去尾先做中间再两边
            r = l[i,:]
            #左邻
            x = np.append(x,r[1:]).astype(int) #0没有上一个邻居
            y = np.append(y,(r-1)[1:]).astype(int)
            #每个元素的右邻居
            x = np.append(x,r[:-1]).astype(int) #最后一个没有下一个邻居
            y = np.append(y,(r+1)[:-1]).astype(int) 
            if i >0:
                #上邻
                x = np.append(x,r).astype(int) #最后一个没有下一个邻居
                y = np.append(y,(r-w)).astype(int) 
                #左上
                x = np.append(x,r[1:]).astype(int) #最后一个没有下一个邻居
                y = np.append(y,(r-w-1)[1:]).astype(int) 
                #右上
                x = np.append(x,r[:-1]).astype(int) #最后一个没有下一个邻居
                y = np.append(y,(r-w+1)[:-1]).astype(int) 
            if i <h-1:
                #下邻
                x = np.append(x,r).astype(int) #最后一个没有下一个邻居
                y = np.append(y,(r+w)).astype(int) 
                #左下
                x = np.append(x,r[1:]).astype(int) #最后一个没有下一个邻居
                y = np.append(y,(r+w-1)[1:]).astype(int) 
                #右上
                x = np.append(x,r[:-1]).astype(int) #最后一个没有下一个邻居
                y = np.append(y,(r+w+1)[:-1]).astype(int)                           
    elif neibour==4:       #4邻居
        l =np.reshape(np.arange(n),(h,w))
        for i in range(h): 
            v = l[i,:]
            x = np.append(x,v[1:]).astype(int) #0没有上一个邻居
            y = np.append(y,(v-1)[1:]).astype(int)
            #每个元素的右邻居
            x = np.append(x,v[:-1]).astype(int) #最后一个没有下一个邻居
            y = np.append(y,(v+1)[:-1]).astype(int) 

        for i in range(w):
            p = l[:,i]
            #上邻
            x = np.append(x,p[1:]).astype(int) #0没有上一个邻居
            y = np.append(y,(p-w)[1:]).astype(int)
            #下邻  
            x = np.append(x,p[:-1]).astype(int) #0没有上一个邻居
            y = np.append(y,(p+w)[:-1]).astype(int)
    elif neibour==2:       #4邻居
        n = C
        l =np.arange(n)

        #每个元素的上一个邻居
        x = np.append(x,l[1:]).astype(int) #0没有上一个邻居
        y = np.append(y,(l-1)[1:]).astype(int)
        #每个元素的下一个邻居
        x = np.append(x,l[:-1]).astype(int) #最后一个没有下一个邻居
        y = np.append(y,(l+1)[:-1]).astype(int) 
    adj = np.array((x,y)).T  #生成的两列合并得到节点及其邻居的矩阵
    #print(adj)
    #使用sp.coo_matrix() 和 np.ones() 共同生成临界矩阵，右边的

    adj = sp.coo_matrix((np.ones(adj.shape[0]), (adj[:, 0], adj[:, 1])),shape=(n, n),dtype=np.float32)

    # build symmetric adjacency matrix 堆成矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #adj = np.hstack((x,y)).rehsape(-1,2) 这样reshape得到的是临近两个一组变化成两列，不符合条件
    #adj =normalize( adj + sp.eye(adj.shape[0]))
    adj = np.array(adj.todense())
    '''      保存adj的数据查看是什么形状      	'''
    # np.save('./adj.txt',x) 
    adj = torch.tensor(adj)#.cuda()

    #adj = torch.FloatTensor(x).cuda()
    return adj


def adj_to_edge(adj):
    h, w = adj.shape
    l = []
    for i in range(0, h):
        for j in range(0, i):
            if adj[i,j] == 1:
                # 重复的边
                l.append((i, j))
                
    edge_tensor = torch.stack([torch.tensor(x) for x in zip(*l)], dim=0).cuda()
    edge_index_list = [edge_tensor.clone() for _ in range(64)]
    return edge_index_list            


# def make_batch(edge_index_list):
#     batch = edge_index_list[0].new_zeros(257)
#     batch = batch.unsqueeze(0)
#     batch = batch.expand(257, -1).cuda()
    
    


def patch_openclip(
    model, ratio, trace_source: bool = False, prop_attn: bool = True
):
    """
    Applies ToMe the OpenCLIP model.
    """
    vision_model = model.visual
    ToMeVisionTransformer = make_tome_class(vision_model.__class__)

    vision_model.__class__ = ToMeVisionTransformer
    vision_model.r = ratio
    vision_model._tome_info = {
        "ratio": ratio,
        "r": vision_model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": True,
        "distill_token": False,
    }

    if hasattr(vision_model, "dist_token") and vision_model.dist_token is not None:
        vision_model._tome_info["distill_token"] = True


    for i, resblock in enumerate(vision_model.transformer.resblocks):     
        resblock.__class__ = ToMeResidualAttentionBlock
        resblock._tome_info = vision_model._tome_info
        attn = ToMeAttention(resblock.attn.embed_dim, resblock.attn.num_heads, qkv_bias=True)
        _, device = convert_attention_block(resblock.attn, attn)
        attn = attn.to(device)
        resblock.attn = attn