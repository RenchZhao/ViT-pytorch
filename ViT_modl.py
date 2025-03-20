# 初步实现，可以训练
# Attention的核心代码可能多次维度变换运行效率不高
# 层的各种dropout没有考虑，很多是重复执行的代码


from torch_nn_train_toolkit import *
from einops import rearrange

class PatchEmbedding(nn.Module):
    """输入图像并将其变换到符合transformer编码器输入要求的层"""
    def __init__(self, img_size, in_chans, patch_size=(16,16), embed_dim=768, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        assert img_size[0] % patch_size[0]==0 and img_size[1] % patch_size[1]==0, "Can't split the image totally"
        self.num_patches = (img_size[0]//patch_size[0])*(img_size[1]//patch_size[1])
        patch_flat_l = img_size[0]*img_size[1]*in_chans//(self.num_patches)
        self.Linear_Proj = nn.Linear(patch_flat_l,embed_dim)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, X):
        batch,in_chans,img_h,img_w = X.shape
        assert img_h == self.img_size[0] and img_w == self.img_size[1], "Size of input images doesn't match the model"
        X = rearrange(X, 'b c h w -> b (h w c)')
        X = X.reshape(batch, self.num_patches, -1)
        return self.norm(self.Linear_Proj(X))
        
        

        
class MultiHeadAttention(nn.Module):
    """多头注意力层"""
    def __init__(self,
                 D,     # 输入的被投影到D维空间的序列，同论文附录A定义
                 k,     # 多头注意力head的数量，同论文附录A定义
                 Dh=None, # 每个注意力head的q,k,v行向量维度，同论文附录A定义
                 attn_dp=0.1,
                 proj_dp=0.1):
        super().__init__()
        self.k = k
        self.D = D
        assert D % k == 0, "Can't split the features totally"
        if type(Dh) == type(None):
            self.Dh = D//k
        else:
            self.Dh = Dh
        self.Uqkv = nn.Linear(D,3*D) # D = Dh * k
        self.Umsa = nn.Linear(k*self.Dh,D)
        self.attn_dropout = nn.Dropout(attn_dp)
        self.proj_dropout = nn.Dropout(proj_dp)
    
    def forward(self, X):
        B, N, D = X.shape
        assert D==self.D, "Shape of input feature doesn't match the model"
        qkv = self.Uqkv(X).reshape(B,N,self.Dh,3,self.k)
        qkv = qkv.permute(3,0,4,1,2)
        q, k, v = qkv[0], qkv[1], qkv[2] # shape (B,k,N,Dh)
        q = q.reshape(B*self.k,N,self.Dh)
        k = k.reshape(B*self.k,N,self.Dh)
        v = v.reshape(B*self.k,N,self.Dh)
        A = torch.bmm(q,k.transpose(1,2))/pow(self.Dh,0.5)
        A = torch.softmax(A,dim=-1) # shape(B*k,N,N)
        A = self.attn_dropout(A)
        A = torch.bmm(A,v) # shape(B*k,N,Dh)
        A = A.reshape(B,self.k,N,self.Dh)
        A = rearrange(A, 'B k N Dh -> B N (k Dh)')
        return self.proj_dropout(self.Umsa(A))

class MLP_Enc(nn.Module):
    """Transformer编码器里面的MLP"""
    def __init__(self, D, hidden_dim, dropout=0.1):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Linear(D, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, D),
            nn.Dropout(dropout)
        )
    def forward(self,X):
        return self.MLP(X)

class Encoder_Block(nn.Module):
    """Transformer编码器单个模块"""
    def __init__(self,
                 D,     # 输入的被投影到D维空间的序列，同论文附录A定义
                 k,     # 多头注意力head的数量，同论文附录A定义
                 mlp_hid_dim,   # 块内MLP的隐藏层神经元数
                 attn_dp=0.1,   # 多头注意力的attn_dp
                 proj_dp=0.1,   # 多头注意力的proj_dp
                 mlp_dp=0.1     # MLP的dropout rate
                 ):
        super().__init__()
        self.norm1 = nn.LayerNorm(D)
        self.Attention = MultiHeadAttention(D,k,attn_dp=attn_dp,proj_dp=proj_dp)
        self.norm2 = nn.LayerNorm(D)
        self.MLP = MLP_Enc(D,mlp_hid_dim,mlp_dp)

    def forward(self,X):
        X = X + self.Attention(self.norm1(X))
        X = X + self.MLP(self.norm2(X))
        return X

class TransformerEncoder(nn.Module):
    """ViT中的Transformer编码器"""
    def __init__(self,
                 num_blocks,   # 编码器含有的Encoder_Block数量
                 D,     # 输入的被投影到D维空间的序列，同论文附录A定义
                 k,     # 多头注意力head的数量，同论文附录A定义
                 mlp_hid_dim,   # 块内MLP的隐藏层神经元数
                 attn_dp=0.1,   # 多头注意力的attn_dp
                 proj_dp=0.1,   # 多头注意力的proj_dp
                 mlp_dp=0.1     # MLP的dropout rate
                 ):
        super().__init__()
        
        blk=[]
        for i in range(num_blocks):
            blk.append(Encoder_Block(D,k,mlp_hid_dim,attn_dp=attn_dp,proj_dp=proj_dp))
        self.net = nn.Sequential(*blk)

    def forward(self,X):
        return self.net(X)

class ViT(nn.Module):
    """ViT网络封装函数"""
    def __init__(self,
              Depth,    # ViT网络深度
              num_classes,# 分类数目
              img_size, # 输入图片高宽
              in_chans, # 输入图片通道数
              attn_head_num, # 多头注意力head的数量，同论文附录A定义
              mlp_hid_dim,   # 块内MLP的隐藏层神经元数
              patch_size=(16,16),   # 图片分片大小
              embed_dim=768,    # 输入的被投影到D维空间的序列，同论文附录A定义
              attn_dp=0.1,   # 多头注意力的attn_dp
              proj_dp=0.1,   # 多头注意力的proj_dp
              mlp_dp=0.1     # MLP的dropout rate
              ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = (img_size[0]//patch_size[0])*(img_size[1]//patch_size[1])
        self.PatchEmbedding = PatchEmbedding(img_size, in_chans, patch_size=patch_size, embed_dim=embed_dim)
        self.class_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_emb = nn.Parameter(torch.randn(1,self.num_patches+1,embed_dim))
        self.Enc_num_blocks = (Depth-2)//(2)
        self.TransformerEncoder = TransformerEncoder(self.Enc_num_blocks, embed_dim, attn_head_num, mlp_hid_dim=mlp_hid_dim, attn_dp=attn_dp, proj_dp=proj_dp,mlp_dp=mlp_dp)
        self.MLP_head = nn.Linear(embed_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, X):
        out = self.PatchEmbedding(X)
        B = X.shape[0]
        cls_tokens = self.class_token.repeat(B,1,1)
        out = torch.cat((cls_tokens,out),1) + self.pos_emb
        out = self.TransformerEncoder(out)
        out = self.MLP_head((out[:,0,:]))
        return self.softmax(out)

'''
test = torch.randn((3,3,32,32))
model=ViT(24, num_classes=10, img_size=(32,32),in_chans=3, attn_head_num=4, mlp_hid_dim=128)
Y=model(test)
print(Y.shape)

'''
# ViT-B/16 12层

model=ViT(12, num_classes=10, img_size=(32,32),in_chans=3, attn_head_num=12, mlp_hid_dim=3072, patch_size=(16,16), embed_dim=768, attn_dp=0.0, proj_dp=0.0, mlp_dp=0.0)
# 电脑跑不动，论文原设置是attn_head_num=16, mlp_hid_dim=4096, patch_size=(16,16), embed_dim=1024，只能适当调小参数

#print(model)

total_epoch=0

epoch=0

lr, num_epochs, batch_size = 4e-4, 100, 256
# 论文batchsize是512，这里跑不了所以只能256
# 不做预训练

lr_adj_step = 5
# 

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0)

'''
lr_lambda = lambda epoch: 1 - epoch/500
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
'''

train_iter, test_iter = load_data_CIFAR10(batch_size, data_agmt=False)

loss = nn.CrossEntropyLoss()

model,epoch,tr_hist_ls,success=load_model_params(model,path="ViT_CIFAR10.ckpt")

tr_hist_ls = train_modl(model, train_iter, test_iter, num_epochs, loss, lr, device=try_gpu(), init=not success, optimizer=optimizer, scheduler=None, tr_hist_ls=tr_hist_ls)

total_epoch=epoch+num_epochs

save_model(model,epoch=total_epoch,path="ViT_CIFAR10.ckpt", tr_hist_ls=tr_hist_ls)

