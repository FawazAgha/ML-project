import torch
import torch.nn as nn
import torch.nn.functional as F

def alibi_bias(n_heads, seq_len, device, dtype):
    # slopes from paper
    def get_slopes(n):
        def pow2_ceil(x): 
            p=1
            while p<x: p*=2
            return p
        m=pow2_ceil(n); slopes=[]
        for i in range(1, m+1):
            slopes.append(2**(-8*i/m))
        return torch.tensor(slopes[:n])
    slopes = get_slopes(n_heads).to(device=device, dtype=dtype)
    pos = torch.arange(seq_len, device=device)
    bias = pos[None, :] - pos[:, None]                  # [T,T]
    bias = bias.unsqueeze(0).unsqueeze(0).to(dtype)     # [1,1,T,T]
    return slopes.view(n_heads,1,1,1) * bias            # broadcast -> [H,1,T,T]

class MHA(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads; self.dk = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3*d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, alibi):
        B,T,C = x.size()
        qkv = self.qkv(x).view(B,T,3,self.h,self.dk).permute(2,0,3,1,4) # [3,B,H,T,dk]
        q,k,v = qkv[0], qkv[1], qkv[2]
        qf, kf, vf = q.float(), k.float(), v.float()

        causal_bias = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device, dtype=qf.dtype),
            diagonal=1,
        ).view(1, 1, T, T)
        attn_mask = causal_bias
        if alibi is not None:
            alibi_bias = alibi[:, 0, :T, :T].float().unsqueeze(0)       # [1,H,T,T]
            attn_mask = attn_mask + alibi_bias

        att = F.scaled_dot_product_attention(
            qf,
            kf,
            vf,
            attn_mask=attn_mask,
            dropout_p=self.drop.p if self.training else 0.0,
            is_causal=False,
        )
        y = att.transpose(1,2).contiguous().view(B,T,C).to(x.dtype)
        return self.o(y)

class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.att = MHA(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff  = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )
    def forward(self, x, alibi):
        x = x + self.att(self.ln1(x), alibi)
        x = x + self.ff(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab, n_layers=6, d_model=512, n_heads=8, d_ff=2048, max_seq=1024, dropout=0.1, use_alibi=True):
        super().__init__()
        self.max_seq = max_seq; self.n_heads = n_heads; self.use_alibi = use_alibi
        self.tok = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        # weight tying
        self.lm_head.weight = self.tok.weight
        if use_alibi:
            self.register_buffer("alibi_cache", torch.empty(0))
        else:
            self.alibi_cache = None

    def forward(self, idx):
        B,T = idx.shape
        if T > self.max_seq:
            raise ValueError(f"sequence length {T} exceeds max_seq {self.max_seq}")
        x = self.tok(idx)
        if self.use_alibi:
            if self.alibi_cache.numel() == 0 or self.alibi_cache.size(-1) < T or self.alibi_cache.dtype != x.dtype:
                self.alibi_cache = alibi_bias(self.n_heads, self.max_seq, x.device, x.dtype)
            alibi = self.alibi_cache[..., :T, :T]
        else:
            alibi = None
        for blk in self.blocks:
            x = blk(x, alibi)
        x = self.ln_f(x)
        return self.lm_head(x)                          # [B,T,V]
