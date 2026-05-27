import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def forward(self, x):
        seq_len = x.shape[1]
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class SwiGLU(nn.Module):
    """SwiGLU activation function"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE"""
    def __init__(self, dim, n_heads, max_seq_len=512):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)

        # Causal mask
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer('mask', mask)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(x)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:T, :T], float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm"""
    def __init__(self, dim, n_heads, mlp_ratio=4, max_seq_len=512):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads, max_seq_len)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, dim * mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class TinyRecursiveNetwork(nn.Module):
    def __init__(self, dim, n_heads=8, n_layers=2, mlp_ratio=4, max_seq_len=512):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, mlp_ratio, max_seq_len)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class TinyRecursiveModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        dim=256,
        n_heads=8,
        n_layers=2,
        mlp_ratio=4,
        max_seq_len=256,
        n_latent_recursions=6,
        n_improvement_cycles=3,
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_latent_recursions = n_latent_recursions
        self.n_improvement_cycles = n_improvement_cycles

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.net = TinyRecursiveNetwork(dim, n_heads, n_layers, mlp_ratio, max_seq_len)

        self.combine_xyz = nn.Linear(dim * 3, dim, bias=False)
        self.combine_yz = nn.Linear(dim * 2, dim, bias=False)

        self.output_head = nn.Linear(dim, vocab_size, bias=False)
        self.halt_head = nn.Linear(dim, 1, bias=False)

        self.y_init = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.z_init = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_embeddings(self, input_ids):
        B, T = input_ids.shape
        input_ids = input_ids.clamp(0, self.vocab_size - 1)
        T = min(T, self.max_seq_len)
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        return self.token_emb(input_ids[:, :T]) + self.pos_emb(pos)

    def latent_recursion(self, x, y, z):
        for _ in range(self.n_latent_recursions):
            combined = self.combine_xyz(torch.cat([x, y, z], dim=-1))
            z = self.net(combined)

        combined_yz = self.combine_yz(torch.cat([y, z], dim=-1))
        y = self.net(combined_yz)
        return y, z

    def deep_recursion(self, x, y, z, use_grad=True):
        if not use_grad:
            with torch.no_grad():
                for _ in range(self.n_improvement_cycles):
                    y, z = checkpoint(self.latent_recursion, x, y, z, use_reentrant=False)
            return y.detach(), z.detach()

        with torch.no_grad():
            for _ in range(self.n_improvement_cycles - 1):
                y, z = self.latent_recursion(x, y, z)

        y, z = self.latent_recursion(x, y, z)
        return y.detach(), z.detach(), self.output_head(y), self.halt_head(y.mean(dim=1))

    def forward(self, input_ids, targets=None, n_supervision_steps=4):
        B, T = input_ids.shape
        T = min(T, self.max_seq_len)
        input_ids = input_ids[:, :T]

        x = self.get_embeddings(input_ids)

        y = self.y_init.expand(B, T, -1).clone()
        z = self.z_init.expand(B, T, -1).clone()

        if targets is None:
            y, z = self.deep_recursion(x, y, z, use_grad=False)
            return self.output_head(y)

        targets = targets[:, :T]
        total_loss = 0.0

        for step in range(n_supervision_steps):
            y, z, logits, halt_logit = self.deep_recursion(x, y, z, use_grad=True)
            ce_loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=-100
            )
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                mask = (targets != -100)
                correct = ((preds == targets) & mask).float().sum() / mask.float().sum().clamp(min=1)
            halt_loss = F.binary_cross_entropy_with_logits(
                halt_logit.squeeze(-1),
                correct.expand(B)
            )
            total_loss = total_loss + ce_loss + 0.1 * halt_loss

        return total_loss / n_supervision_steps

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=0.8, top_k=40):
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -(self.max_seq_len - 1):]
            idx_cond = idx_cond.clamp(0, self.vocab_size - 1)
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

class CausalLMOutput:
    def __init__(self, logits):
        self.logits = logits

class FinanceTRMWrapper(nn.Module):
    """Wrapper bọc TRM model đóng giả làm Hugging Face model để chạy thẳng vào pipeline.
    """
    def __init__(self, model, max_new_tokens=20):
        super().__init__()
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.config = type('Config', (), {'is_encoder_decoder': False})()
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Inferencer choice_losses expects output.logits
        logits = self.model(input_ids=input_ids)
        return CausalLMOutput(logits)
    
    def generate(self, input_ids, attention_mask=None, max_new_tokens=None, **kwargs):
        max_tokens = max_new_tokens or self.max_new_tokens
        
        # Lấy model gốc nếu đang bị bọc trong DataParallel
        actual_model = self.model.module if hasattr(self.model, "module") else self.model
        
        with torch.no_grad():
            output_ids = actual_model.generate(
                input_ids=input_ids, 
                max_new_tokens=max_tokens,
                temperature=kwargs.get("temperature", 0.1),
                top_k=kwargs.get("top_k", 10)
            )
        
        return output_ids
