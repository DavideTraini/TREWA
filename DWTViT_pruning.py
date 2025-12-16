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

# ### Moduli e Classi di Supporto ###

def pair(t):
    """Converte un singolo valore in una tupla."""
    return t if isinstance(t, tuple) else (t, t)

class AdaptivePruner(nn.Module):
    def __init__(self, wavelet='db4', mode='zero', pruning_aggressiveness: List[float] = [1.5, 0.5], strategy: str = "cA", verbose: bool = False):
        super().__init__()
        self.wavelet = wavelet
        self.mode = mode
        self.verbose = verbose
        self.strategy = strategy
        self.aggressiveness_factors = torch.tensor(sorted(pruning_aggressiveness, reverse=True))
        self.max_pruning_levels = len(self.aggressiveness_factors)

        self.reducers = nn.ModuleList([
            DWT1DForward(J=level + 1, wave=wavelet, mode=mode) for level in range(self.max_pruning_levels)
        ])

    def forward(self, x: torch.Tensor, cls_attention_map: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, D = x.shape
        cls_token, patch_tokens = x[:, :1], x[:, 1:]
        initial_patch_count = patch_tokens.shape[1]

        entropies = -torch.sum(cls_attention_map * torch.log2(cls_attention_map + 1e-9), dim=-1)
        mean_entropy = entropies.mean()
        std_entropy = entropies.std()
        
        if std_entropy < 1e-6:
            pruning_levels = torch.ones(B, dtype=torch.long, device=x.device)
            if self.verbose:
                print("--- Pruning Stage --- Deviazione standard dell'entropia troppo bassa. Forzo la riduzione (livello 1).")
        else:
            dynamic_thresholds = mean_entropy - self.aggressiveness_factors.to(x.device) * std_entropy
            pruning_levels = torch.sum(entropies.unsqueeze(1) < dynamic_thresholds, dim=1)
        
        pruning_levels.clamp_min_(1)

        final_patches = torch.zeros_like(patch_tokens)
        output_lengths = torch.full((B,), initial_patch_count, dtype=torch.long, device=x.device)

        for level in range(1, self.max_pruning_levels + 1):
            level_mask = pruning_levels == level
            if not level_mask.any():
                continue

            tokens_to_prune = patch_tokens[level_mask]
            reducer = self.reducers[level - 1]
            Yl, Yh = reducer(tokens_to_prune.transpose(1, 2))

            if self.strategy == "cA":
                pruned_tokens = Yl
            elif self.strategy == "cD":
                pruned_tokens = Yh[0]
            # elif self.strategy == "sum":
            #     pruned_tokens = Yl + Yh[0]
            else:
                raise ValueError(f"Strategia di pruning non riconosciuta: {self.strategy}")

            pruned_tokens = pruned_tokens.transpose(1, 2)
            new_len = pruned_tokens.shape[1]

            final_patches[level_mask, :new_len] = pruned_tokens.to(final_patches.dtype)
            final_patches[level_mask, new_len:] = 0
            output_lengths[level_mask] = new_len

        if self.verbose:
            print("--- Pruning Stage (Dettagli) ---")
            for b in range(B):
                print(f"  Immagine {b}: Entropia={entropies[b]:.2f}, Livello={pruning_levels[b].item()}, Token: {initial_patch_count} -> {output_lengths[b].item()}")

        max_len = output_lengths.max().item()
        final_patches = final_patches[:, :max_len, :]
        final_x = torch.cat([cls_token, final_patches], dim=1)

        attention_mask = torch.arange(max_len, device=x.device)[None, :] < output_lengths[:, None]
        attention_mask = torch.cat([torch.ones(B, 1, device=x.device, dtype=torch.bool), attention_mask], dim=1)

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
                 pruning_aggressiveness: List[float] = [1.5, 0.5],
                 strategy: str = "cA",
                 verbose: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])

        self.pruning_locations = set(pruning_locations if pruning_locations is not None else [])
        
        self.pruners = nn.ModuleDict({
            str(loc): AdaptivePruner(
                wavelet=wavelet, 
                pruning_aggressiveness=pruning_aggressiveness,
                strategy=strategy,
                verbose=verbose
            ) for loc in self.pruning_locations
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


class DWTViT_pruning(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.,
                 pruning_locations: Optional[List[int]] = None, 
                 wavelet: str = 'db4',
                 pruning_aggressiveness: List[float] = [1.5, 0.5],
                 strategy: str = "cA",
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
            pruning_aggressiveness=pruning_aggressiveness,
            strategy=strategy,
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