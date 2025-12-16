import torch
from torch import nn
from typing import List, Optional

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from timm.layers.patch_embed import PatchEmbed

try:
    from pytorch_wavelets import DWTForward, DWT1DForward
except ImportError:
    print("Attenzione: `pytorch_wavelets` non trovata. Uso un placeholder per DWTForward.")
    # Questo è un placeholder. Per favore, installa la libreria corretta.
    class DWTForward(nn.Module):
        def __init__(self, J, wave, mode):
            super().__init__()
            print(f"Placeholder DWTForward inizializzato. Installa pytorch_wavelets.")
        def forward(self, x):
            B, C, H, W = x.shape
            new_H, new_W = H // 2, W // 2
            return torch.randn(B, C, new_H, new_W), [torch.randn(B, C, 3, new_H, new_W)]


# ### Moduli e Classi di Supporto ###

def pair(t):
    """Converte un singolo valore in una tupla."""
    return t if isinstance(t, tuple) else (t, t)

# class WaveletTokenReducer(nn.Module):
#     """Applica la trasformata wavelet alla mappa di token e ne riduce la risoluzione."""
#     def __init__(self, wavelet='db4', mode='zero'):
#         super().__init__()
#         self.dwt = DWTForward(J=2, wave=wavelet, mode=mode)

#     def forward(self, x: torch.Tensor, H: int, W: int) -> (torch.Tensor, int, int):
#         B, N, C = x.shape
#         x_img = x.transpose(1, 2).reshape(B, C, H, W)
#         Yl, _ = self.dwt(x_img) # Mantiene solo i coefficienti di approssimazione (LL)
#         _, _, new_H, new_W = Yl.shape
#         x_reduced = Yl.reshape(B, C, -1).transpose(1, 2)
#         return x_reduced, new_H, new_W

class WaveletTokenReducer(nn.Module):
    """Applies a 1D Discrete Wavelet Transform (DWT) along the sequence dimension
    of a token tensor to reduce its length (number of tokens).
    """
    def __init__(self, wavelet='db4', mode='zero', num_levels=1):
        """
        Args:
            wavelet (str): Name of the wavelet to use (e.g., 'haar', 'db4').
            mode (str): Padding mode for the DWT (e.g., 'zero', 'periodization', 'reflect').
            num_levels (int): Number of decomposition levels for the 1D DWT.
                              Each level approximately halves the sequence length.
                              So, K = N / (2^num_levels).
        """
        super().__init__()
        # DWT1DForward operates on tensors of shape (B, C, L) where L is sequence length
        # In our case, B will be the original B, C will be D (features), and L will be N (tokens)
        self.dwt = DWT1DForward(J=num_levels, wave=wavelet, mode=mode)
        self.num_levels = num_levels

    def forward(self, x: torch.Tensor) -> (torch.Tensor, int):
        """
        Applies 1D DWT to reduce sequence length.

        Args:
            x (torch.Tensor): Input token tensor of shape (B, N, D),
                              where N is the sequence length.

        Returns:
            tuple:
                - x_reduced (torch.Tensor): Reduced token tensor of shape (B, K, D),
                                            where K is the new sequence length.
        """
        B, N, D = x.shape

        # 1. Reshape for 1D DWT: (B, N, D) -> (B, D, N)
        # pytorch_wavelets.DWT1DForward expects (N, C, L) where N is batch, C is channels, L is signal length.
        # Here, our 'channels' are the 'D' features, and 'signal length' is 'N' tokens.
        x_reshaped_for_dwt = x.transpose(1, 2) # Shape becomes (B, D, N)

        # 2. Apply 1D DWT.
        # Yl: Approximation coefficients (low-pass)
        # Yh: Detail coefficients (high-pass) - a list of tensors per level
        Yl, Yh = self.dwt(x_reshaped_for_dwt)

        # For resolution reduction, we typically only keep the approximation coefficients (Yl).
        # Yl will have a shape of (B, D, new_N)
        _, _, new_N_approx = Yl.shape

        # The new sequence length should theoretically be N / 2^J.
        # DWT1DForward handles padding, so the exact new_N might be slightly
        # different than N // (2**self.num_levels) if N is not a multiple of 2^J.

        # 3. Reshape the reduced sequence back into token format (B, K, D)
        # First transpose (B, D, K) -> (B, K, D)
        x_reduced = Yl.transpose(1, 2)

        return x_reduced

class FeedForward(nn.Module):
    """Rete Feed-Forward standard per un blocco Transformer."""
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    """Self-Attention Multi-Head standard."""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# ### Classi Transformer e DWTViT ###

class Transformer(nn.Module):
    """Transformer Encoder che include la logica di riduzione dei token."""
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.,
                 reduction_stages: Optional[List[int]] = None, wavelet: str = 'db4'):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        self.token_reducers = nn.ModuleList([])

        reduction_stages = reduction_stages if reduction_stages is not None else []

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))
            # Aggiunge un riduttore se lo stage corrente è nella lista
            if i in reduction_stages:
                self.token_reducers.append(WaveletTokenReducer(wavelet=wavelet))
            else:
                self.token_reducers.append(None)

    def forward(self, x):#, H, W):
        """
        Args:
            x (torch.Tensor): Tensor di input (cls_token + patch_tokens).
            H (int): Altezza iniziale della mappa di token.
            W (int): Larghezza iniziale della mappa di token.
        """
        for i, (attn, ff) in enumerate(self.layers):
            x = attn(x) + x          
            x = ff(x) + x

            reducer = self.token_reducers[i]
            if reducer is not None:
                cls_token = x[:, :1]
                patch_tokens = x[:, 1:]
                patch_tokens = reducer(patch_tokens)#, H, W)
                x = torch.cat((cls_token, patch_tokens), dim=1)

        return self.norm(x)

class DWTViT(nn.Module):
    """Vision Transformer con Riduzione dei Token basata su Wavelet."""
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.,
                 reduction_stages: Optional[List[int]] = None, wavelet: str = 'db4'):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Le dimensioni dell\'immagine devono essere divisibili per la dimensione della patch.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'Il tipo di pool deve essere \'cls\' o \'mean\'.'

        # Dimensioni iniziali della mappa di token
        self.H = image_height // patch_height
        self.W = image_width // patch_width

        self.patch_embed = self.patch_embed = PatchEmbed(img_size = image_size, patch_size = patch_size, in_chans = channels, embed_dim = dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Inizializzazione della classe Transformer con i parametri di riduzione
        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout,
            reduction_stages=reduction_stages,
            wavelet=wavelet
        )

        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.patch_embed(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # Passa i token e le dimensioni iniziali H, W al transformer
        x = self.transformer(x)#, self.H, self.W)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)