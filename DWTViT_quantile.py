import torch
from torch import nn
from typing import List, Optional, Tuple

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from timm.layers.patch_embed import PatchEmbed

try:
    from pytorch_wavelets import DWT1DForward
except ImportError:
    print("Attenzione: `pytorch_wavelets` non trovata. Uso un placeholder per DWT1DForward.")
    class DWT1DForward(nn.Module):
        def __init__(self, J, wave, mode):
            super().__init__()
            self.J = J
            print(f"Placeholder DWT1DForward inizializzato con J={J}. Per favore, installa `pytorch_wavelets` per la piena funzionalità.")
        def forward(self, x):
            B, C, L = x.shape
            new_L = L // (2**self.J)
            yl = torch.randn(B, C, new_L, device=x.device, dtype=x.dtype)
            yh = [torch.randn(B, C, new_L, device=x.device, dtype=x.dtype) for _ in range(self.J)]
            return yl, yh


def pair(t):
    """Converte un singolo valore in una tupla."""
    return t if isinstance(t, tuple) else (t, t)
  
class AdaptivePruner(nn.Module):
    """
    A parameter-free module that prunes tokens in a fully vectorized and optimized manner.
    The pruning decision is based on a robust, non-parametric quantile-based thresholding
    of attention entropy, stratifying the batch into three pruning levels.
    """
    def __init__(self, wavelet: str = 'db4', mode: str = 'zero', verbose: bool = False):
        """
        Initializes the pruner.

        Args:
            wavelet (str): The type of wavelet to use for the Discrete Wavelet Transform.
            mode (str): The padding mode for the DWT.
            verbose (bool): If True, prints detailed pruning information for each batch.
        """
        super().__init__()
        self.wavelet = wavelet
        self.mode = mode
        self.verbose = verbose
        
        # Define the number of pruning levels. Level 0 is no pruning.
        # Level 1 prunes once, Level 2 prunes twice.
        self.max_pruning_levels = 3
        
        # Create a list of DWT reducers, one for each actual pruning level (1 and 2).
        # The J parameter in DWT determines the number of decomposition levels,
        # which directly controls the reduction factor.
        self.reducers = nn.ModuleList([
            DWT1DForward(J=level + 1, wave=wavelet, mode=mode) for level in range(self.max_pruning_levels)
        ])
    def forward(self, x: torch.Tensor, cls_attention_map: torch.Tensor) -> Tuple:
        """
        Applies adaptive pruning to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (B, N, D), where B is batch size,
                              N is the number of tokens, and D is the feature dimension.
            cls_attention_map (torch.Tensor): The attention map from the token,
                                              shape (B, num_heads, num_patches).
                                              It is assumed to be averaged over heads.

        Returns:
            Tuple: A tuple containing:
                - final_x (torch.Tensor): The pruned tensor of shape (B, N_new, D).
                - attention_mask (torch.Tensor): A boolean mask for attention, shape (B, N_new).
        """
        B, N, D = x.shape
        cls_token, patch_tokens = x[:, :1], x[:, 1:]
        initial_patch_count = patch_tokens.shape[1] # -> Restituisce un intero

        # --- 1. Calculate Entropy and Quantile-Based Pruning Levels ---
        # Calculate Shannon entropy for each sample's attention map.
        # A small epsilon prevents log(0).

        entropies = -torch.sum(cls_attention_map * torch.log2(cls_attention_map + 1e-9), dim=-1)
        
        # Define quantile points for stratification (Q1 and Median).
        quantiles_to_compute = torch.tensor([0.25, 0.5], device=x.device)
        q_thresholds = torch.quantile(entropies, quantiles_to_compute)
        
        # Vectorized assignment of pruning levels based on quantiles.
        # This single line replaces all conditional logic and hyperparameter-based calculations.
        # (entropies > q_thresholds) creates a boolean tensor.
        #.sum(dim=1) counts how many thresholds are crossed (0, 1, or 2).
        # We subtract from max_pruning_levels to map (0, 1, 2) to levels (2, 1, 0).
        # Level 2: Most aggressive pruning (entropy <= Q1)
        # Level 1: Moderate pruning (Q1 < entropy <= Median)
        # Level 0: No pruning (entropy > Median)
        pruning_levels = self.max_pruning_levels - (entropies.unsqueeze(1) > q_thresholds.unsqueeze(0)).sum(dim=1)

        # --- 2. Fully Vectorized Pruning Execution ---
        # Initialize output tensors. final_patches will be filled based on pruning level.
        final_patches = patch_tokens.clone() # Start with original patches for level 0
        output_lengths = torch.full((B,), initial_patch_count, dtype=x.dtype, device=x.device)

        # This loop is highly efficient as it iterates over a small, fixed number of
        # pruning levels (e.g., 2), not the batch size B.
        for level in range(1, self.max_pruning_levels + 1):
            # Create a mask for all samples in the batch assigned to the current pruning level.
            level_mask = (pruning_levels == level)
            if not level_mask.any():
                continue

            # Select the tokens that need to be pruned at this level.
            tokens_to_prune = patch_tokens[level_mask]
            
            # Select the corresponding DWT reducer. level is 1-indexed.
            reducer = self.reducers[level - 1]
            
            # Apply the DWT. Input must be (B, D, N) for DWT1DForward.
            Yl, _ = reducer(tokens_to_prune.transpose(1, 2))
            pruned_tokens = Yl.transpose(1, 2) # Transpose back to (B, N_new, D)
            
            new_len = pruned_tokens.shape[1]
            
            # Place the pruned tokens into the final output tensor and pad with zeros.
            # This ensures all sequences in the temporary tensor have the same length.
            final_patches[level_mask, :new_len] = pruned_tokens.to(final_patches.dtype)
            final_patches[level_mask, new_len:] = 0 
            
            # Record the new, shorter length for these samples.
            output_lengths[level_mask] = new_len

        if self.verbose:
            print("--- Pruning Stage (V2 - Quantile) ---")
            print(f"  Quantile Thresholds (Q1, Median): {q_thresholds.cpu().numpy()}")
            for b in range(B):
                print(f"  Image {b}: Entropy={entropies[b]:.2f}, Pruning Level={pruning_levels[b].item()}, "
                      f"Tokens: {initial_patch_count} -> {output_lengths[b].item()}")

        # --- 3. Final Assembly and Mask Creation ---
        # Trim the final tensor to the maximum length of any sequence in the batch.
        max_len = int(output_lengths.max().item())
        final_patches = final_patches[:, :max_len, :]
        
        # Concatenate the token back with the (potentially pruned) patch tokens.
        final_x = torch.cat([cls_token, final_patches], dim=1)

        # Create the attention mask to ignore padded tokens in subsequent layers.
        # This is crucial for correct attention calculation.
        patch_attention_mask = torch.arange(max_len, device=x.device)[None, :] < output_lengths[:, None]
        cls_attention_mask = torch.ones(B, 1, device=x.device, dtype=torch.bool)
        attention_mask = torch.cat([cls_attention_mask, patch_attention_mask], dim=1)

        return final_x, attention_mask

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout))
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
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
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            mask_value = -torch.finfo(dots.dtype).max
            mask = mask[:, None, None, :]
            dots.masked_fill_(~mask.bool(), mask_value)

        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn

# ### Classi Transformer e DWTViT Aggiornate ###

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.,
                 pruning_locations: Optional[List[int]] = None, 
                 wavelet: str = 'db4',
                 verbose: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        
        self.pruning_locations = set(pruning_locations if pruning_locations is not None else [])
        
        self.pruners = nn.ModuleDict({
            str(loc): AdaptivePruner(wavelet=wavelet, verbose=verbose) 
            for loc in self.pruning_locations
        })

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x, attention_mask: Optional[torch.Tensor] = None, batch_pruning_stats: Optional[dict] = None):
        for i, (attn_layer, ff) in enumerate(self.layers):
            out, attn_weights = attn_layer(x, mask=attention_mask)
            x = out + x
            
            if i in self.pruning_locations:
                pruner = self.pruners[str(i)]
                cls_attn_map = attn_weights[:, :, 0, 1:].mean(dim=1)
                
                if attention_mask is not None:
                    patch_mask = attention_mask[:, 1:]
                    cls_attn_map = cls_attn_map.masked_fill(~patch_mask.bool(), 0)

                x, attention_mask = pruner(x, cls_attn_map)

            x = ff(x) + x

            # Dopo il layer, se era una postazione di pruning, registra le statistiche
            if batch_pruning_stats is not None and i in self.pruning_locations:
                if attention_mask is not None:
                    # Conta i token di patch attivi (escludendo il CLS token)
                    num_patch_tokens = attention_mask[:, 1:].sum(dim=1)
                else:
                    # Se non c'è maschera, non è avvenuto alcun pruning
                    num_patch_tokens = torch.full((x.shape[0],), x.shape[1] - 1, device=x.device, dtype=torch.long)
                batch_pruning_stats[i].append(num_patch_tokens)

        return self.norm(x), attention_mask

class DWTViT_quantile(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.,
                 pruning_locations: Optional[List[int]] = None, 
                 wavelet: str = 'db4',
                 verbose: bool = False):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        assert pool in {'cls', 'mean'}

        self.patch_embed = PatchEmbed(img_size=image_size, patch_size=patch_size, in_chans=channels, embed_dim=dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        self.pruning_locations = pruning_locations if pruning_locations is not None else []
        # Dizionario per memorizzare le statistiche del batch corrente
        self.batch_pruning_stats = {loc: [] for loc in self.pruning_locations}

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout,
            pruning_locations=self.pruning_locations,
            wavelet=wavelet,
            verbose=verbose
        )

        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)

    def report_batch_stats(self):
        """Calcola e stampa le medie delle patch usate per il batch più recente."""
        if not self.pruning_locations:
            print("No pruning locations configured.")
            return
            
        print("\n--- Utilizzo Medio Patch Per Layer (Ultimo Batch) ---")
        for loc in self.pruning_locations:
            counts_list = self.batch_pruning_stats.get(loc, [])
            if not counts_list:
                # Questo può accadere se il forward pass non ha registrato dati per questo layer
                print(f"  Layer {loc}: Nessuna statistica registrata.")
                continue
            
            # Concatena tutti i tensori registrati e calcola la media
            avg_tokens = torch.cat(counts_list).float().mean().item()
            print(f"  Layer {loc}: {avg_tokens:.2f} patch medie")
        print("---------------------------------------------------------")

    def forward(self, img):
        # Azzera le statistiche all'inizio di ogni forward pass
        for loc in self.pruning_locations:
            self.batch_pruning_stats[loc] = []

        x = self.patch_embed(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # Passa il dizionario delle statistiche al transformer
        x, _ = self.transformer(x, batch_pruning_stats=self.batch_pruning_stats)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)