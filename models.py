from utlis import *
import math
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class MemoryUnit(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres):
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # Mem: MxC
        self.bias = None
        self.shrink_thres = shrink_thres
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        att_weight = F.linear(input, self.weight)  # Fea x Mem^T: (TxC) x (CxM) = TxM
        att_weight = F.softmax(att_weight, dim=1)  # TxM
        # ReLU based shrinkage, hard shrinkage for positive value
        if self.shrink_thres > 0:
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)
        mem_trans = self.weight.permute(1, 0)  # Mem^T: C x M
        output = F.linear(att_weight, mem_trans)  # AttWeight x (Mem^T)^T: (TxM) x (MxC) = TxC
        return {'output': output, 'att': att_weight}  # output, att_weight

    def extra_repr(self):
        return 'mem_dim={}, fea_dim={}'.format(
            self.mem_dim, self.fea_dim is not None
        )


# NxCxL -> (NxL)xC -> addressing Mem, (NxL)xC -> NxCxL
class MemModule(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres):
        super(MemModule, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = MemoryUnit(self.mem_dim, self.fea_dim, self.shrink_thres)

    def forward(self, input):
        s = input.data.shape
        l = len(s)

        if l == 3:
            x = input.permute(0, 2, 1)
        elif l == 4:
            x = input.permute(0, 2, 3, 1)
        elif l == 5:
            x = input.permute(0, 2, 3, 4, 1)
        else:
            x = []
            print('wrong feature map size')
        
        x = x.contiguous()
        x = x.view(-1, s[1])
        y_and = self.memory(x)
        y = y_and['output']
        att = y_and['att']

        if l == 3:
            y = y.view(s[0], s[2], s[1])
            y = y.permute(0, 2, 1)
            att = att.view(s[0], s[2], self.mem_dim)
            att = att.permute(0, 2, 1)
        elif l == 4:
            y = y.view(s[0], s[2], s[3], s[1])
            y = y.permute(0, 3, 1, 2)
            att = att.view(s[0], s[2], s[3], self.mem_dim)
            att = att.permute(0, 3, 1, 2)
        elif l == 5:
            y = y.view(s[0], s[2], s[3], s[4], s[1])
            y = y.permute(0, 4, 1, 2, 3)
            att = att.view(s[0], s[2], s[3], s[4], self.mem_dim)
            att = att.permute(0, 4, 1, 2, 3)
        else:
            y = x
            att = att
            print('wrong feature map size')

        return {'output': y, 'att': att}
    

# NxCxHxW -> (HxW)xCxN -> truncated SVD -> (HxW)xCxN -> NxCxHxW 
class TSVDModule(nn.Module):
    def __init__(self, low_rank):
        super(TSVDModule, self).__init__()
        self.low_rank = low_rank

    def forward(self, input):
        s = input.data.shape
        l = len(s)

        if l == 4:
            x = input.permute(2, 3, 1, 0)
        else:
            x = []
            print('wrong feature map size')

        x = x.contiguous()
        x = x.view(-1, s[1], s[0])
        U, S, Vh = torch.svd_lowrank(x, q=self.low_rank)
        y = U @ torch.diag_embed(S) @ torch.transpose(Vh, 1, 2)

        if l == 4:
            y = y.view(s[2], s[3], s[1], s[0])
            y = y.permute(3, 2, 0, 1)
        else:
            y = x
            att = att
            print('wrong feature map size')

        return y


class Autoencoder(nn.Module):
    def __init__(self, c_in, r):
        super().__init__()
        self.c_in = c_in
        self.r = r
        f1, f2, f3, f4 = 2, 4, 8, 16
        self.encoder = nn.Sequential(
            nn.Conv1d(c_in, f1, 3, stride=2, padding=1),
            nn.BatchNorm1d(f1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(f1, f2, 3, stride=2, padding=1),
            nn.BatchNorm1d(f2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(f2, f3, 3, stride=2, padding=1),
            nn.BatchNorm1d(f3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(f3, f4, 3, stride=2, padding=1),
            nn.BatchNorm1d(f4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(f4, f3, 3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm1d(f3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(f3, f2, 3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm1d(f2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(f2, f1, 3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm1d(f1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(f1, c_in, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        z = self.encoder(x)             # (B, 16, 18)
        out = self.decoder(z)           # (B, 1, 274)
        return out, z


class ProjectionHead(nn.Module):
    def __init__(self, c_in=16, h_dim=128, p_dim=128):
        super().__init__()
        c_mid1 = 32
        c_mid2 = 64
        self.feat = nn.Sequential(
            nn.Conv1d(c_in, c_mid1, 3, stride=2, padding=1),
            nn.BatchNorm1d(c_mid1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(c_mid1, c_mid2, 3, stride=2, padding=1),
            nn.BatchNorm1d(c_mid2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(c_mid2, h_dim, 3, stride=2, padding=1),
            nn.BatchNorm1d(h_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1)     # (B, h_dim, 1)
        )
        self.proj = nn.Sequential(
            nn.Linear(h_dim, p_dim, bias=False),
            nn.BatchNorm1d(p_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, z):
        h = self.feat(z).squeeze(-1)    # (B, h_dim)
        p = self.proj(h)                # (B, p_dim)
        return F.normalize(h, dim=-1), F.normalize(p, dim=-1)


class ContraAE(nn.Module):
    def __init__(self, c_in=1, r=4, h_dim=64, p_dim=128):
        super().__init__()
        self.ae   = Autoencoder(c_in=c_in, r=r)
        self.head = ProjectionHead(c_in=16, h_dim=h_dim, p_dim=p_dim)

    def reconstruct(self, x):
        """Decode full AE path (use for reconstruction loss or inference)."""
        out, _ = self.ae(x)
        return out

    def encode(self, x):
        """Encoder only: returns z (B,16,18)."""
        return self.ae.encoder(x)

    def encode_to_proj(self, x, return_fw: bool = False, decode: bool = False):
        """
        Returns z, h, p. Optionally also z_fw and/or reconstruction.
        - z:   (B, 16, 18)   encoder latent
        - h:   (B, h_dim)    normalized feature (downstream)
        - p:   (B, proj_dim) normalized projection (VICReg/contrastive)
        - z_fw (optional):  (B, 16*r)  first r columns flattened (consistency loss)
        - out  (optional):  (B, 1, 274) reconstruction if decode=True
        """
        z = self.ae.encoder(x)
        h, p = self.head(z)
        ret = (z, h, p)

        if return_fw:
            z_fw = z[:, :, :self.ae.r].contiguous()
            ret = ret + (z_fw,)

        if decode:
            out = self.ae.decoder(z)
            ret = ret + (out,)

        return ret


# ------------------------------
# Transformer encoder block (LayerNorm-only)
# ------------------------------
class TransformerBlock1D(nn.Module):
    def __init__(self, d_model=16, n_heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, T, d)
        # Self-attention with pre-norm
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        # MLP with pre-norm
        x = x + self.mlp(self.ln2(x))
        return x


# ------------------------------
# Transformer-style AE (LayerNorm, no BatchNorm)
#   - Keeps z as (B, 16, 18)
#   - Reconstructs to (B, 1, 274)
# ------------------------------
class TransformerAE(nn.Module):
    """
    Patchify (Conv1d) -> Positional Embedding -> Transformer Encoder (LayerNorm) -> 
    latent tokens -> (permute to B, 16, 18) as z
    Decode: ConvTranspose1d to 274
    """
    def __init__(
        self,
        c_in=1,
        r=4,
        d_model=16,      # feature (channel) dim == 16 to match your head
        n_tokens=18,     # number of patches/tokens
        n_layers=4,
        n_heads=4,
        mlp_ratio=4.0,
        dropout=0.0,
        patch_kernel=16,
        patch_stride=16,
        patch_pad=7,
        recon_kernel=16,
        recon_stride=16,
        recon_pad=7,
        recon_outpad=0,
    ):
        super().__init__()
        self.c_in = c_in
        self.r = r
        self.d_model = d_model
        self.n_tokens = n_tokens

        # --- Patchify: (B,1,274) -> (B, d_model, 18)
        self.patch_embed = nn.Conv1d(
            in_channels=c_in, out_channels=d_model,
            kernel_size=patch_kernel, stride=patch_stride, padding=patch_pad, bias=True
        )

        # --- Positional embeddings for 18 tokens (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, d_model))

        # --- Transformer encoder (LayerNorm only)
        self.blocks = nn.ModuleList([
            TransformerBlock1D(d_model=d_model, n_heads=n_heads,
                               mlp_ratio=mlp_ratio, dropout=dropout)
        for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(d_model)

        # --- Reconstruction: (B, d_model, 18) -> (B, 1, 274)
        self.decoder = nn.ConvTranspose1d(
            in_channels=d_model, out_channels=c_in,
            kernel_size=recon_kernel, stride=recon_stride,
            padding=recon_pad, output_padding=recon_outpad, bias=True
        )

        # init (Xavier is fine; pos_embed zeros is OK)
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def encoder_tokens(self, x):
        """
        x: (B,1,274) -> tokens: (B, 18, d_model)
        """
        # Patchify to (B, d_model, 18)
        x = self.patch_embed(x)
        # Permute to (B, 18, d_model)
        x = x.transpose(1, 2)
        # Add positional embeddings
        x = x + self.pos_embed
        # Transformer blocks (pre-norm)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_final(x)
        return x  # (B, 18, d_model)

    def encoder(self, x):
        """
        Return z shaped as (B, 16, 18) for compatibility with your head.
        """
        tok = self.encoder_tokens(x)          # (B, 18, 16)
        z = tok.transpose(1, 2).contiguous()  # (B, 16, 18)
        return z

    def forward(self, x):
        """
        Return: out (B,1,274), z (B,16,18)
        """
        z = self.encoder(x)                   # (B,16,18)
        out = self.decoder(z)                 # (B,1,274)
        return out, z


# ------------------------------
# Wrapper matching your previous ContraAE API
# ------------------------------
class TransAE(nn.Module):
    def __init__(self, c_in=1, r=4, h_dim=128, p_dim=128,
                 # transformer hyperparams
                 n_layers=4, n_heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        # Replace Autoencoder with TransformerAE
        self.ae = TransformerAE(
            c_in=c_in, r=r, d_model=16, n_tokens=18,
            n_layers=n_layers, n_heads=n_heads, mlp_ratio=mlp_ratio, dropout=dropout,
            # patchify & invert chosen so z is (B,16,18) and out is length 274
            patch_kernel=16, patch_stride=16, patch_pad=7,
            recon_kernel=16, recon_stride=16, recon_pad=7, recon_outpad=0,
        )
        self.head = ProjectionHead(c_in=16, h_dim=h_dim, p_dim=p_dim)

    def reconstruct(self, x):
        out, _ = self.ae(x)
        return out

    def encode(self, x):
        return self.ae.encoder(x)  # (B,16,18)

    def encode_to_proj(self, x, return_fw: bool = False, decode: bool = False):
        """
        Returns z, h, p. Optionally z_fw and/or reconstruction.
        - z:   (B, 16, 18)
        - h:   (B, h_dim)
        - p:   (B, p_dim)
        - z_fw (optional): first r columns of z, shape (B,16,r)
        - out  (optional): (B,1,274)
        """
        z = self.encode(x)              # (B,16,18)
        h, p = self.head(z)             # normalized
        ret = (z, h, p)

        if return_fw:
            z_fw = z[:, :, :self.ae.r].contiguous()
            ret = ret + (z_fw,)

        if decode:
            out = self.ae.decoder(z)
            ret = ret + (out,)

        return ret


