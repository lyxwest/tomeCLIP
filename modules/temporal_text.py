import torch
from torch import nn
from collections import OrderedDict
import numpy as np

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class AfterReconstruction(nn.Identity):
    def __init__(self, inplanes):
        super().__init__()
        self.inplanes = inplanes

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, dropout = 0.):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)
        
        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.control_point = AfterReconstruction(d_model)

    def attention(self, x: torch.Tensor):
        # x: 50 bT c
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = self.control_point(x)
        h = x 
        x = h + self.drop_path(self.attention(self.ln_1(x)))

        x = self.control_point(x)
        h = x
        x = h + self.drop_path(self.mlp(self.ln_2(x)))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, dropout=None, ):
        super().__init__()
        if dropout is None:
            dropout = [0.0 for i in range(layers)] 
        print('dropout used:{}'.format(dropout))
        self.width = width
        self.layers = layers
        
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, dropout=dropout[i]) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    

class time_text(nn.Module):
    def __init__(self,
                frame_ids, 
                context_length,
                embed_dim,
                transformer_width,
                transformer_layers,
                transformer_heads,
                vision_layers,
                dropout, 
                emb_dropout,
                 ):
        super().__init__()

        if dropout > 0.:
            dpr = [x.item() for x in torch.linspace(0, dropout, vision_layers)]  # stochastic depth decay rule
        else:
            dpr = None

        self.context_length = context_length
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            dropout=dpr
        )

        self.frame_ids= frame_ids
        self.ln_final = LayerNorm(transformer_width)
        
        self.dropout = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        # self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

        self.initialize_parameters()

    def initialize_parameters(self):
                       
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # if self.text_projection is not None:
        #     nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        for block in self.transformer.resblocks:
            return block.attn.in_proj_weight.dtype
    
    def forward(self, text_emb:torch.tensor):
        #print(self.dtype)   torch.float32 [400,768]

        x = text_emb.unsqueeze(dim=1).repeat(1,16,1).type(self.dtype) + self.frame_ids.type(self.dtype)     #[400, 16, 768]


        if self.emb_dropout > 0:
            x = self.dropout(x)
        #x = x.permute(1, 0, 2)  # NLD -> LND
        # print(x.shape,'---------1')
        x = self.transformer(x)
        #x = x.permute(1, 0, 2)  # LND -> NLD
        # print(x.shape,'---------2')
        x = self.ln_final(x).type(self.dtype)
        # print(x.shape,'---------3')
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        #x = x[torch.arange(x.shape[0]), text_emb.argmax(dim=-1)] @ self.text_projection
        x = x.mean(dim=1, keepdim=False)
        # print(x.shape,'---------4')
        return x
    

#frame part
class LayerNormVisual(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNormVisual, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class ResidualAttentionBlockVisual(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.transformer_heads = n_head
        self.attn = nn.MultiheadAttention(d_model, n_head)    #d_model:total dimension of the model
        self.ln_1 = LayerNormVisual(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNormVisual(d_model)
        self.attn_mask = attn_mask


    def build_attention_mask(self,transformer_heads, batch, num_segs):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(num_segs, num_segs)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(dim=0).expand(batch*transformer_heads, -1, -1)
        return mask
    

    def attention(self, x: torch.Tensor):
        t, b, _ = x.size()
        #print(b, t, self.transformer_heads)
        # self.attn_mask = self.build_attention_mask(transformer_heads=self.transformer_heads, batch=b ,num_segs=t)
        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device)
            #print('--------Using masked attention-------------')  
        else:
            self.attn_mask = None
            #print('--------Not using masked attention-------------')
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]    # x x x对应q k v

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class TemporalTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlockVisual(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks((x))


class temporal_module(nn.Module):
    def __init__(self, clip_state_dict, frame_ids):
        super().__init__()

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        # visual_transformer_layers = len(
        #     set(k.split(".")[3] for k in clip_state_dict if k.startswith(f"visual.transformer.resblocks")))
        # print(visual_transformer_layers)
        #self.frame_position_embeddings = nn.Embedding(context_length, embed_dim)
        self.transformer = TemporalTransformer(width=embed_dim, layers=3, heads=transformer_heads)
        self.frame_ids = frame_ids
        self.time_text = time_text(
            frame_ids = frame_ids,
            context_length = context_length,
            embed_dim = embed_dim,
            transformer_width = transformer_width,
            transformer_heads = transformer_heads,
            transformer_layers = 3,
            vision_layers = 3,
            dropout = 0.1, 
            emb_dropout = 0.
        )
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNormVisual):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    #-------------mask attention-----------------
    # def mask(self, x):
    #     b, t, c = x.size()
    #     mask = self.build_attention_mask(transformer_heads=self.transformer_heads, batch=b, num_segs=t)
    #     self.mask_transformer = TemporalTransformer(width=self.embed_dim, layers=6, heads=self.transformer_heads, attn_mask=mask)

    def time_frame(self, x):
        b, t, c = x.size()
        x = x.contiguous()

        x_original = x


        #position_ids = torch.arange(frame_length, dtype=torch.long, device=x.device)
        #position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)
        #position_ids = self.frame_ids

        #frame_position_embeddings = self.frame_position_embeddings(position_ids)
        x = x + self.frame_ids

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x.type(x_original.dtype) + x_original

        return x.mean(dim=1, keepdim=False)

    
    def forward(self, frame, text):
        frame = self.time_frame(frame)
        text = self.time_text(text)

        frame = frame / frame.norm(dim=-1, keepdim=True)
        text = text / text.norm(dim=-1, keepdim=True)
        # logit_scale = self.logit_scale.exp()
        logits = frame @ text.T
        return logits