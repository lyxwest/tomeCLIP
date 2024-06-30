import numpy as np
import torch

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import scipy.sparse as sp
# import torch.utils.model_zoo as model_zoo

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
    ''''      保存adj的数据查看是什么形状      	'''
    # np.save('./adj.txt',x) 
    adj = torch.tensor(adj)#.cuda()

    #adj = torch.FloatTensor(x).cuda()
    return adj


class GraphAttentionLayer(nn.Module):
	"""
	描述：
		再MPGA中，单层的GAT，输入输出的维度相同，attention的计算方式使用softmax
		in_features：输入的维度，
		down_ratio:降维的比例
		out_feature:在多头注意力之中需要用
	"""
	def __init__(self, in_features,down_ratio=8,sgat_on=True,cgat_on=True):
		super(GraphAttentionLayer, self).__init__()
		#self.dropout = dropout
		self.in_features = in_features
		self.hid_features = in_features//down_ratio #数据降维，使用//保证输出的结果为整数
		#alpha sigma两次降维后，做矩阵运算获得att，类似GAT中的先用w再用a获得注意力。
		self.use_sgat = sgat_on
		self.use_cgat = cgat_on
		if self.use_sgat:
			self.down_alpha = nn.Sequential( #无论是通道还是空域都需要先降维 图卷积的#默认使用hid_feature是
				nn.Conv2d(in_channels=self.in_features, out_channels=self.hid_features, 
						kernel_size=1, stride=1, padding=0, bias=False), #输入 in_features 2048 输出hid_features  in_features//down_ratio
				nn.BatchNorm2d(self.hid_features),
				nn.ReLU()
			)

			self.down_sigma = nn.Sequential( #无论是通道还是空域都需要先降维 图卷积的
				nn.Conv2d(in_channels=self.in_features, out_channels=self.hid_features, 
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.hid_features),
				nn.ReLU()
			)
	

	def forward(self, x):#v 是输入的各个节点， b c h w是输入feature map的shape
		#输入后都是先降维计算注意力，concat=false需要聚合特征前需要将输入的维度降低再聚合
		b,c,h,w = x.size()

		if self.use_sgat:
			adj = create_adj(h,w,self.in_features,8)
			# print('图片的维度：',x.size())                                   # torch.Size([2, 64, 224, 224])
			alpha = self.down_alpha(x)#concat的时候不太一样                 
			# print('alpha.shape:',alpha.shape)                               # alpha: torch.Size([2, 8, 224, 224])
			sigma = self.down_sigma(x)
			# print('sigma :',sigma.shape)                                    # sigma: torch.Size([2, 8, 224, 224])
			alpha = alpha.view(b, self.hid_features, -1).permute(0, 2, 1)	
			# print('转换后alpha :',alpha.shape)                               # alpha: torch.Size([2, 50176, 8])
			sigma = sigma.view(b, self.hid_features, -1)
			# print('转换后sigma :',sigma.shape)                               # sigma: torch.Size([2, 8, 50176])
			att = torch.matmul(alpha, sigma)                                   #这就是每个图的自注意力机制
			# print('alpha乘sigma得到大的att shape:',att.shape)                 # att shape: torch.Size([2, 50176, 50176])
			zero_vec = -9e15*torch.ones_like(att)
			attention = torch.where(adj.expand_as(att)> 0, att,zero_vec)
			# print('attention shape:',attention.shape)                        # attention shape: torch.Size([2, 50176, 50176])
			attention = F.softmax(attention, dim=2)  # 计算节点，临近节点的关系，因为attention为3维 b h w 按行求则为
			# print('softmax(attention) shape:',attention.shape)               # shape: torch.Size([2, 50176, 50176])
			h_s = torch.matmul(attention, x.view(b, c, -1).permute(0, 2, 1)).permute(0,2,1).view(b,c,h,w)  #聚合临近节点的信息表示该节点
			# print('图上传播后',h_s.shape)                                     # torch.Size([2, 64, 224, 224])
		if self.use_cgat:
			cadj = create_adj(h,w,c,2)#2表示通道的adj未进行节点维度的变化，直接点乘和sigmod计算的att
			theta_xc = x.view(b, c, -1)
			phi_xc = x.view(b, c, -1).permute(0, 2, 1) # 8 2048 256 1  batch_size 节点 channel
			Gc = torch.matmul(theta_xc, phi_xc) # bactchsiz n n   通道之间的关系 
			zero_vec = -9e15*torch.ones_like(Gc)
			catt = torch.where(cadj.expand_as(Gc)> 0, Gc, zero_vec)
			cattention = F.softmax(catt, dim=2)  # 计算节点，临近节点的关系，因为attention为3维 b h w 按行求则为
			h_prime = torch.matmul(cattention, x.view(b, c, -1)).view(b,c,h,w)   #聚合临近节点的信息表示该节点
		if self.use_cgat and self.use_sgat:
			return torch.add(h_s, h_prime)
		if self.use_cgat:
			return h_prime #残差

		return  h_s
      

import torch
import torch.nn as nn
import torch.nn.functional as F

# 创建示例输入张量
batch_size = 2
in_channels = 64
height = 224
width = 224
x = torch.randn(batch_size, in_channels, height, width)
print("x size:", x.shape)
# adj = create_adj(224, 224, 3, 8)
# print(adj)

# 实例化GraphAttentionLayer
layer = GraphAttentionLayer(in_features=in_channels, down_ratio=8, sgat_on=True, cgat_on=True)
out = layer(x)
print("ooutput size:", out.shape)

# # 只测试空间图注意力
# layer = GraphAttentionLayer(in_features=in_channels, down_ratio=8, sgat_on=True, cgat_on=False) 
# out = layer(x)

# # 只测试通道图注意力
# layer = GraphAttentionLayer(in_features=in_channels, down_ratio=8, sgat_on=False, cgat_on=True)
# out = layer(x)