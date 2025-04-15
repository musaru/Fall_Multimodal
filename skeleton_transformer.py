import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision

class AbsolutePositionalEncoder(nn.Module):
    def __init__(self, emb_dim, max_position=512):
        super().__init__()
        self.position = torch.arange(max_position).unsqueeze(1)

        positional_encoding = torch.zeros(1, max_position, emb_dim)

        _2i = torch.arange(0, emb_dim, step=2).float()

        # PE(pos, 2i) = sin(pos/10000^(2i/d_model))
        positional_encoding[0, :, 0::2] = torch.sin(self.position / (10000 ** (_2i / emb_dim)))

        # PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        positional_encoding[0, :, 1::2] = torch.cos(self.position / (10000 ** (_2i / emb_dim)))
        
        self.register_buffer('pos_bias', positional_encoding)

    def forward(self, x):
        # batch_size, input_len, embedding_dim
        batch_size, seq_len, _ = x.size()

        return self.pos_bias[:batch_size, :seq_len, :]

class SqueezeAndExcitation3D(nn.Module):
    def __init__(self, in_channels, squeeze_rate=1/4, activation=nn.GELU):
        super().__init__()

        mid_channles = int(in_channels*squeeze_rate)

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.conv3d_1 = nn.Conv3d(in_channels,mid_channles,kernel_size=1)
        self.activation = activation()
        self.conv3d_2 = nn.Conv3d(mid_channles,in_channels,kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x:torch.Tensor):

        attn = self.global_pool(x)
        attn = self.conv3d_1(attn)
        attn = self.activation(attn)
        attn = self.conv3d_2(attn)
        attn = self.sigmoid(attn)

        return attn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self,input_dims:int,head_dim:int,n_heads:int, seq_len:int):
        super().__init__()
        
        
        self.input_dims = input_dims
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.embed_dims = head_dim*n_heads
        
        

        
        self.seq_len = seq_len
        
        self.w_qkv = nn.Linear(input_dims,embed_dims*3)
        self.scale_factor = embed_dims**-0.5

        self.merge = nn.Linear(embed_dims, input_dims)

        
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        N, C, T, V, M = x.size()
        H,HD = self.n_heads, self.head_dim
        
        x = x.permute(0,4,2,3,1) # N,M,T,V,C
        x = self.w_qkv(x)
        
        q,k,v = torch.chunk(x,3,dim=-1)
        
        q = q.reshape(N*M,T,V,H,HD).permute(0,1,3,2,4) # NM, T, H, V, HD
        k = k.reshape(N*M,T,V,H,HD).permute(0,1,3,2,4)
        v = v.reshape(N*M,T,V,H,HD).permute(0,1,3,2,4)
        
        k = k * self.scale_factor
        dot_prod = torch.einsum("B T H I D, B T H J D -> B T H I J",q,k)        
        dot_prod = f.softmax(dot_prod, dim=-1)
        
        out = torch.einsum("B T H I J, B T H J D -> B T H I D", dot_prod, v)
        out = out.permute(0,1,3,2,4).reshape(N,M,T,V,self.embed_dims)

        out = self.merge(out)
        out = out.permute(0,4,2,3,1)
        
        return out
    
    
class RelativePositionalMultiHeadSelfAttention(nn.Module):
    def __init__(self,input_dims:int,head_dim:int,n_heads:int, seq_len:int):
        super().__init__()
        
        
        self.input_dims = input_dims
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.embed_dims = head_dim*n_heads
        self.seq_len = seq_len
        
        self.w_qkv = nn.Linear(input_dims,self.embed_dims*3)
        self.scale_factor = self.embed_dims**-0.5

        self.merge = nn.Linear(self.embed_dims, input_dims)
        
        self.relative_position_bias_table = nn.parameter.Parameter(
            torch.empty(((2 * self.seq_len - 1), self.head_dim), dtype=torch.float32),
        )
        # initialize with truncated normal the bias
        torch.nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
    def compute_relative_positions(self, seq_len):
        # Compute a matrix with relative positions between each pair of tokens
        # This returns indices that are used to retrieve relative embeddings
        range_vec = torch.arange(seq_len)
        rel_pos_matrix = range_vec[:, None] - range_vec[None, :]
        rel_pos_matrix = rel_pos_matrix + seq_len - 1  # shift to get positive indices
        return rel_pos_matrix
        
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        N, C, V, T, M = x.size()
        H,HD = self.n_heads, self.head_dim
        
        x = x.permute(0,4,2,3,1) # N,M,V,T,C
        x = self.w_qkv(x)
        
        q,k,v = torch.chunk(x,3,dim=-1)
        
        q = q.reshape(N*M,V,T,H,HD).permute(0,1,3,2,4) # NM, V, H, T, HD
        k = k.reshape(N*M,V,T,H,HD).permute(0,1,3,2,4)
        v = v.reshape(N*M,V,T,H,HD).permute(0,1,3,2,4)
        
        dot_prod = torch.einsum("B V H I D, B V H J D -> B V H I J",q,k) # NM,V,H,T,T
        dot_prod = dot_prod * self.scale_factor
        
        pos_bias = self.relative_position_bias_table[self.compute_relative_positions(T)] # T,T,embedding_vector
        rel_attn_scores = torch.einsum('bvhld,lrd->bvhlr', q, pos_bias)  # (B, num_heads, L, L)
        
        dot_prod = f.softmax(dot_prod + rel_attn_scores, dim=-1)
        
        out = torch.einsum("B V H I J, B V H J D -> B V H I D", dot_prod, v)
        out = out.permute(0,1,3,2,4).reshape(N,M,V,T,self.embed_dims)

        out = self.merge(out)
        out = out.permute(0,4,2,3,1)
        
        return out
    
class PreNormTransformerBlock(nn.Module):
    def __init__(self,input_dims,head_dim,n_heads,n_joints,seq_len,ffn_expand_rate:int=4,ffn_dropout_rate:float=0.5,ffn_activation=nn.GELU):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(input_dims)
        self.multi_head_spatial_self_attention = RelativePositionalMultiHeadSelfAttention(input_dims,head_dim,n_heads,n_joints)
        
        self.norm2 = nn.LayerNorm(input_dims)
        self.multi_head_temporal_self_attention = RelativePositionalMultiHeadSelfAttention(input_dims,head_dim,n_heads,seq_len)
        
        self.norm3 = nn.LayerNorm(input_dims)
        self.feed_forward_network = nn.Sequential(
            nn.Linear(input_dims,int(input_dims*ffn_expand_rate)),
            ffn_activation(),
            nn.Linear(int(input_dims*ffn_expand_rate),input_dims),
            nn.Dropout(ffn_dropout_rate),
        )
        
        self.norm4 = nn.LayerNorm(input_dims)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        
        N, C, T, V, M = x.size()
        
        # spatial (vertex) self attention
        out = self.norm1(x.permute(0,4,2,3,1)).permute(0,4,2,3,1) # B,C,T,V,M --[permute]--> B,M,T,V,C --[LayerNorm]-> B,M,T,V,C --[permute]--> B,C,T,V,M
        out = self.multi_head_spatial_self_attention(out) # B,C,T,V,M --[self attention]-->  B,C,T,V,M
        x = x + out # residual connection
        
        # temporal (time) self attention
        out = self.norm2(x.permute(0,4,2,3,1)).permute(0,4,2,3,1) # B,C,T,V,M --[permute]--> B,M,T,V,C --[LayerNorm]-> B,M,T,V,C --[permute]--> B,C,T,V,M
        out = self.multi_head_temporal_self_attention(out.permute(0,1,3,2,4)).permute(0,1,3,2,4) # B,C,T,V,M --[permute]--> B,C,V,T,M --[self attention]-> B,C,V,T,M --[permute]--> B,C,T,V,M
        x = x + out # residual connection
        
        # fead forward network
        out = self.norm3(x.permute(0,4,2,3,1)) # B,C,T,V,M --[permute]--> B,M,T,V,C 
        out = self.feed_forward_network(out).permute(0,4,2,3,1) # B,M,T,V,C --[Feed Forward Network]--> B,M,T,V,C --[permute]--> B,C,T,V,M
        x = x + out # residual connection
        
        # last normalize
        out = self.norm4(out.permute(0,4,2,3,1)).permute(0,4,2,3,1) # B,C,T,V,M --[permute]--> B,M,T,V,C --[LayerNorm]-> B,M,T,V,C --[permute]--> B,C,T,V,M
        return out


############################################
# paper : https://arxiv.org/pdf/2206.00330v1
############################################
class B2TSpatialTenporalTransformerBlock(nn.Module):
    def __init__(self,input_dims,head_dim,n_heads,n_joints,seq_len,ffn_expand_rate:int=4,ffn_dropout_rate:float=0.5,ffn_activation=nn.GELU,normalization=nn.LayerNorm,stochastic_depth_rate=0):
        super().__init__()
        
        self.multi_head_spatial_self_attention = RelativePositionalMultiHeadSelfAttention(input_dims,head_dim,n_heads,n_joints)
        self.norm1 = normalization(input_dims)
        
        
        self.multi_head_temporal_self_attention = RelativePositionalMultiHeadSelfAttention(input_dims,head_dim,n_heads,seq_len)
        self.norm2 = normalization(input_dims)
        
        
        self.feed_forward_network = nn.Sequential(
            nn.Linear(input_dims,int(input_dims*ffn_expand_rate)),
            ffn_activation(),
            nn.Linear(int(input_dims*ffn_expand_rate),input_dims),
            nn.Dropout(ffn_dropout_rate),
        )
        self.norm3 = normalization(input_dims)

        self.stochastic_depth = torchvision.ops.StochasticDepth(stochastic_depth_rate,mode="batch") if stochastic_depth_rate > 0 else nn.Identity()

        
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        
        N, C, T, V, M = x.size()
        
        # spatial (vertex) self attention
        out = x + self.stochastic_depth(self.multi_head_spatial_self_attention(x)) # B,C,T,V,M --[self attention]-->  B,C,T,V,M
        out = self.norm1(out.permute(0,4,2,3,1)).permute(0,4,2,3,1) if isinstance(self.norm1,nn.LayerNorm) else self.norm1(out)# B,C,T,V,M --[permute]--> B,M,T,V,C --[LayerNorm]-> B,M,T,V,C --[permute]--> B,C,T,V,M
        
        # temporal (time) self attention
        out = out.permute(0,1,3,2,4) # B,C,T,V,M --[permute]--> B,C,V,T,M 
        out = out + self.stochastic_depth(self.multi_head_temporal_self_attention(out)) #B,C,V,T,M --[self attention]-> B,C,V,T,M  
        out = self.norm2(out.permute(0,4,2,3,1)).permute(0,4,2,3,1) if isinstance(self.norm2,nn.LayerNorm) else self.norm2(out) # B,C,V,T,M --[permute]--> B,M,V,T,C --[LayerNorm]-> B,M,V,T,C --[permute]--> B,C,V,T,M
        out = out.permute(0,1,3,2,4) # B,C,V,T,M --[permute]--> B,C,T,V,M 
        
        # fead forward network
        out = out + self.stochastic_depth(self.feed_forward_network(out.permute(0,4,2,3,1)).permute(0,4,2,3,1))  # B,C,T,V,M  --[permute]--> B,M,T,V,C --[Feed Forward Network]--> B,M,T,V,C --[permute]--> B,C,T,V,M
        out = x + out
        out = self.norm3(out.permute(0,4,2,3,1)).permute(0,4,2,3,1) if isinstance(self.norm3,nn.LayerNorm) else self.norm3(out) # B,C,T,V,M --[permute]--> B,M,T,V,C --[LayerNorm]-> B,M,T,V,C --[permute]--> B,C,T,V,M
        
        return out

############################################
# paper : https://arxiv.org/pdf/2206.00330v1
############################################
class B2TTransformerBlock_Parallel(nn.Module):
    def __init__(self,input_dims,head_dim,n_heads,n_joints,seq_len,ffn_expand_rate:int=4,ffn_dropout_rate:float=0.5,ffn_activation=nn.GELU,normalization=nn.LayerNorm):
        super().__init__()
        
        self.multi_head_spatial_self_attention = RelativePositionalMultiHeadSelfAttention(input_dims,head_dim,n_heads,n_joints)
        self.norm1 = normalization(input_dims)
        
        
        self.multi_head_temporal_self_attention = RelativePositionalMultiHeadSelfAttention(input_dims,head_dim,n_heads,seq_len)
        
        
        self.feed_forward_network = nn.Sequential(
            nn.Linear(input_dims,int(input_dims*ffn_expand_rate)),
            ffn_activation(),
            nn.Linear(int(input_dims*ffn_expand_rate),input_dims),
            nn.Dropout(ffn_dropout_rate),
        )
        self.norm3 = normalization(input_dims)
        
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        
        N, C, T, V, M = x.size()
        
        # spatial (vertex) self attention
        out = x + self.multi_head_spatial_self_attention(x) + self.multi_head_temporal_self_attention(x.permute(0,1,3,2,4)).permute(0,1,3,2,4) # B,C,T,V,M --[self attention]-->  B,C,T,V,M
        out = self.norm1(out.permute(0,4,2,3,1)).permute(0,4,2,3,1) # B,C,T,V,M --[permute]--> B,M,T,V,C --[LayerNorm]-> B,M,T,V,C --[permute]--> B,C,T,V,M
        
        
        # fead forward network
        out = out + self.feed_forward_network(out.permute(0,4,2,3,1)).permute(0,4,2,3,1)  # B,C,T,V,M  --[permute]--> B,M,T,V,C --[Feed Forward Network]--> B,M,T,V,C --[permute]--> B,C,T,V,M
        out = x + out
        out = self.norm3(out.permute(0,4,2,3,1)).permute(0,4,2,3,1) # B,C,T,V,M --[permute]--> B,M,T,V,C --[LayerNorm]-> B,M,T,V,C --[permute]--> B,C,T,V,M
        
        return out

############################################
# paper : https://arxiv.org/pdf/2206.00330v1
############################################
class B2TTransformerBlock(nn.Module):
    def __init__(self,input_dims,head_dim,n_heads,n_joints,seq_len,ffn_expand_rate:int=4,ffn_dropout_rate:float=0.5,ffn_activation=nn.GELU,normalization=nn.LayerNorm):
        super().__init__()
        
        self.multi_head_spatial_self_attention = RelativePositionalMultiHeadSelfAttention(input_dims,head_dim,n_heads,n_joints)
        self.norm1 = normalization(input_dims)        
        
        self.feed_forward_network = nn.Sequential(
            nn.Linear(input_dims,int(input_dims*ffn_expand_rate)),
            ffn_activation(),
            nn.Linear(int(input_dims*ffn_expand_rate),input_dims),
            nn.Dropout(ffn_dropout_rate),
        )
        self.norm3 = normalization(input_dims)
        
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        
        N, C, T, V, M = x.size()
        
        # spatial (vertex) self attention
        out = x + self.multi_head_spatial_self_attention(x) # B,C,T,V,M --[self attention]-->  B,C,T,V,M
        out = self.norm1(out.permute(0,4,2,3,1)).permute(0,4,2,3,1) # B,C,T,V,M --[permute]--> B,M,T,V,C --[LayerNorm]-> B,M,T,V,C --[permute]--> B,C,T,V,M
        
        
        # fead forward network
        out = out + self.feed_forward_network(out.permute(0,4,2,3,1)).permute(0,4,2,3,1)  # B,C,T,V,M  --[permute]--> B,M,T,V,C --[Feed Forward Network]--> B,M,T,V,C --[permute]--> B,C,T,V,M
        out = x + out
        out = self.norm3(out.permute(0,4,2,3,1)).permute(0,4,2,3,1) # B,C,T,V,M --[permute]--> B,M,T,V,C --[LayerNorm]-> B,M,T,V,C --[permute]--> B,C,T,V,M
        
        return out
    
class GrowthBlock(nn.Module):
    def __init__(self,input_dims,head_dim,n_heads,n_joints,seq_len,ffn_expand_rate:int=4,ffn_dropout_rate:float=0.5,ffn_activation=nn.GELU,normalization=nn.LayerNorm,growth=16):
        super().__init__()
        
        self.transformer_block = B2TTransformerBlock(
            input_dims,
            head_dim,
            n_heads,
            n_joints,
            seq_len,
            ffn_expand_rate,
            ffn_dropout_rate,
            ffn_activation,
            normalization,
        )
        
        self.squeeze = nn.Sequential(
            nn.Linear(input_dims,growth),
            ffn_activation(),
            normalization(growth),
        )
        
    def forward(self,x):
        _x = self.transformer_block(x)
        _x = self.squeeze(_x.permute(0,4,2,3,1)).permute(0,4,2,3,1)
        
        return torch.cat((x,_x),dim=1)


class TransposeAxis(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        if x.dim()!=5:
            raise RuntimeError("Input x does not have 5 dim")
        return x.permute(0,1,3,2,4)
    
class SkeletonTransformer(nn.Module):
    def __init__(self,in_channels:int,n_joints:int,seq_len:int,num_classes:int,embedding_dim:int=32,n_block:int=6,head_dim:int=16,n_heads:int=8,):
        super().__init__()
        
        self.in_channels = in_channels
        self.n_joints = n_joints
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        layers = []
        self.embedding = nn.Sequential(
            nn.Linear(in_channels,int(embedding_dim/2)),
            nn.GELU(),
            nn.Linear(int(embedding_dim/2),embedding_dim),
            nn.GELU(),
        )
        # self.vertex_pos_emb = nn.Parameter(
        #     torch.randn(1,1,self.n_joints,embedding_dim)
        # )
        stochastic_depth_rate = np.linspace(0,0.5,n_block)
        growth=int(embedding_dim/4)
        for n in range(n_block):
            layers += [
                B2TSpatialTenporalTransformerBlock(
                    embedding_dim,
                    head_dim,
                    n_heads,
                    n_joints,
                    seq_len,
                    normalization=nn.BatchNorm3d,
                    stochastic_depth_rate=stochastic_depth_rate[n],
                )
                # B2TTransformerBlock_Parallel(
                #     embedding_dim,
                #     head_dim,
                #     n_heads,
                #     n_joints,
                #     seq_len,
                # )
                # GrowthBlock(
                #     embedding_dim+int(n*growth),
                #     head_dim,
                #     n_heads,
                #     n_joints,
                #     seq_len,
                #     # normalization=nn.BatchNorm3d,
                #     growth=growth,
                # )
            ]
            
        self.extractor = nn.Sequential(*layers)
        
        self.fcn = nn.Sequential(
           # nn.Conv2d(embedding_dim + int(n_block*growth),num_classes,1),
            nn.Conv2d(embedding_dim,num_classes,1),
        )
        
    def forward(self,x:torch.Tensor) -> torch.Tensor:
       
        
        x = self.embedding(x.permute(0,4,2,3,1)).permute(0,4,2,3,1)
        
        x = self.extractor(x)
        
        B, C, T, V, M = x.size()
        x = x.permute(0,4,1,2,3).reshape(B*M,C,T,V)
        # global pooling
        x = f.avg_pool2d(x, x.size()[2:])
        x = x.view(B, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x

class SkeletonTransformer_Ablation1(nn.Module):
    def __init__(self,in_channels:int,n_joints:int,seq_len:int,num_classes:int,embedding_dim:int=32,n_block:int=6,head_dim:int=16,n_heads:int=8,):
        super().__init__()
        
        self.in_channels = in_channels
        self.n_joints = n_joints
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        layers = []
        self.embedding = nn.Sequential(
            nn.Linear(in_channels,int(embedding_dim/2)),
            nn.GELU(),
            nn.Linear(int(embedding_dim/2),embedding_dim),
            nn.GELU(),
        )
        stochastic_depth_rate = np.linspace(0,0.5,n_block)
        growth=int(embedding_dim/4)
        for n in range(int(n_block/2)):
            layers += [
                B2TTransformerBlock(
                    embedding_dim,
                    head_dim,
                    n_heads,
                    n_joints,
                    seq_len,
                    ffn_expand_rate=4,
                    ffn_dropout_rate=0.5,
                    ffn_activation=nn.GELU,
                    normalization=nn.LayerNorm
                )
            ]
        layers += [
            TransposeAxis()
        ]
        for n in range(int(n_block/2)):
            layers += [
                B2TTransformerBlock(
                    embedding_dim,
                    head_dim,
                    n_heads,
                    seq_len,
                    n_joints,
                    ffn_expand_rate=4,
                    ffn_dropout_rate=0.5,
                    ffn_activation=nn.GELU,
                    normalization=nn.LayerNorm
                )
            ]

        
            
        self.extractor = nn.Sequential(*layers)
        
        self.fcn = nn.Sequential(
           # nn.Conv2d(embedding_dim + int(n_block*growth),num_classes,1),
            nn.Conv2d(embedding_dim,num_classes,1),
        )
        
    def forward(self,x:torch.Tensor) -> torch.Tensor:
       
        
        x = self.embedding(x.permute(0,4,2,3,1)).permute(0,4,2,3,1)
        
        x = self.extractor(x)
        
        B, C, T, V, M = x.size()
        x = x.permute(0,4,1,2,3).reshape(B*M,C,T,V)
        # global pooling
        x = f.avg_pool2d(x, x.size()[2:])
        x = x.view(B, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x
if __name__=="__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = SkeletonTransformer(
        in_channels=3,
        n_joints=27,
        seq_len=32,
        num_classes=100,
    ).to(device).eval()
    
    print(model)
    inputs = torch.randn((1,3,32,27,1)).to(device)
    out = model(inputs)
    print(out.size())