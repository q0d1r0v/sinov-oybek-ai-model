"""Kichik GPT: o'zbek tili uchun noldan yozilgan model (~10 mln parametr)."""
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 8000   # tokenizer lug'ati
    block_size: int = 256    # model bir vaqtda ko'radigan token soni
    n_layer: int = 6         # bloklar soni
    n_head: int = 8          # diqqat boshlari
    n_embd: int = 320        # har bir token vektori o'lchami
    dropout: float = 0.1


class Attention(nn.Module):
    """Har bir token oldingi tokenlarga qaraydi (keyingilarini ko'rmaydi)."""

    def __init__(self, c: Config):
        super().__init__()
        assert c.n_embd % c.n_head == 0
        self.n_head, self.dropout = c.n_head, c.dropout
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd, bias=False)
        self.proj = nn.Linear(c.n_embd, c.n_embd, bias=False)
        self.proj_drop = nn.Dropout(c.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj_drop(self.proj(y.transpose(1, 2).reshape(B, T, C)))


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.n_embd)
        self.attn = Attention(c)
        self.ln2 = nn.LayerNorm(c.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(c.n_embd, 4 * c.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * c.n_embd, c.n_embd, bias=False),
            nn.Dropout(c.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.config = c
        self.tok = nn.Embedding(c.vocab_size, c.n_embd)
        self.pos = nn.Embedding(c.block_size, c.n_embd)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln = nn.LayerNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # kirish va chiqish vaznlari umumiy
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith(("proj.weight", "mlp.2.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        T = idx.size(1)
        assert T <= self.config.block_size, "matn block_size dan uzun"
        x = self.drop(self.tok(idx) + self.pos(torch.arange(T, device=idx.device)))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln(x))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, eos_id=None):
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.config.block_size:])
            logits = logits[:, -1] / max(temperature, 1e-5)
            if top_k:
                v, _ = logits.topk(min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat((idx, nxt), dim=1)
            if eos_id is not None and nxt.item() == eos_id:
                break
        return idx
