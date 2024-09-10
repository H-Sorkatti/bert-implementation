from math import sqrt
import torch
from torch import nn
import torch.nn.functional as F


# input_ids: Optional[torch.Tensor] = None,
#     attention_mask: Optional[torch.Tensor] = None,
#     head_mask: Optional[torch.Tensor] = None,
#     inputs_embeds: Optional[torch.Tensor] = None,
#     output_attentions: Optional[bool] = None,
#     output_hidden_states: Optional[bool] = None,
#     return_dict: Optional[bool] = None,


class Embeddings(nn.Module):
    """
    An Embedding layer that combines token embeddings with positional embeddings
    """

    def __init__(self, config):
        super().__init__()
        self.token_embeds = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embeds = nn.Embedding(config.max_seq_length, config.embed_dim)
        self.layernorm = nn.LayerNorm(config.embed_dim, 1e-12)
        self.dropout = nn.Dropout()

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        # TODO: add a torch device to position_ids
        position_ids = torch.arange(seq_len, dtype=torch.long)
        token_embeddings = self.token_embeds(input_ids)
        position_embeddings = self.position_embeds(position_ids)
        combined_embeds = token_embeddings + position_embeddings
        norm_combined_embeds = self.layernorm(combined_embeds)
        embeddings = self.dropout(norm_combined_embeds)
        return embeddings


class AttentionHead(nn.Module):
    """
    Impelements Dot-Product Self-Attention
    """

    def __init__(self, config):
        super().__init__()
        self.q_net = nn.Linear(config.embed_dim, config.head_dim, bias=False)
        self.k_net = nn.Linear(config.embed_dim, config.head_dim, bias=False)
        self.v_net = nn.Linear(config.embed_dim, config.head_dim, bias=False)

    def forward(self, input_ids):
        q = self.q_net(input_ids)
        k = self.k_net(input_ids)
        v = self.q_net(input_ids)

        scores = torch.bmm(q, k.transpose(1, 2)) / sqrt(q.shape[1])
        attn_weights = F.softmax(scores, dim=1)
        return torch.bmm(attn_weights, v)


class MultiHeadAttention(nn.Module):
    """Multi-Headed Attention Layer"""

    def __init__(self, config):
        super().__init__()
        self.head_dim = config.embed_dim // config.num_heads
        self.heads = nn.Sequential(
            *[
                AttentionHead(config.embed_dim, self.head_dim)
                for _ in range(config.num_heads)
            ]
        )

    def forward(self, input_ids):
        outs = [head(input_ids) for head in self.heads]
        final_attn = torch.cat(outs, dim=-1)
        return final_attn


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(
            in_features=config.input_dim, out_features=config.intermediate_dim
        )
        self.linear_2 = nn.Linear(
            in_features=config.intermediate_dim, out_features=config.input_dim
        )
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids):
        x = self.linear_1(input_ids)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(config.input_dim)
        self.layernorm_2 = nn.LayerNorm(config.input_dim)
        self.multihead_attention = MultiHeadAttention(
            config.input_dim, config.num_heads
        )
        self.feed_forward = FeedForward(config.input_dim, config.intermediate_dim)

    def forward(self, input_ids):
        x = self.layernorm_1(input_ids)
        attn = self.multihead_attention(x)
        x = x + attn
        x = self.layernorm_2(x)
        ff = self.feed_forward(x)
        x = x + ff
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(
            config.embed_dim, config.vocab_size, config.max_seq_length
        )
        self.encoder_layers = nn.Sequential(
            *[
                TransformerEncoderLayer(
                    config.embed_dim, config.num_heads, config.intermediate_dim
                )
                for _ in range(config.num_encoder_blocks)
            ]
        )

    def forward(self, input_ids):
        embeddings = self.embeddings(input_ids)
        x = self.encoder_layers(embeddings)
        return x
