import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt


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

    def __init__(self, config, head_dim):
        super().__init__()
        self.q_net = nn.Linear(config.embed_dim, head_dim, bias=True)
        self.k_net = nn.Linear(config.embed_dim, head_dim, bias=True)
        self.v_net = nn.Linear(config.embed_dim, head_dim, bias=True)

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
            *[AttentionHead(config, self.head_dim) for _ in range(config.num_heads)]
        )
        self.attn_output = nn.Linear(config.embed_dim, config.embed_dim, bias=True)

    def forward(self, input_ids):
        outs = [head(input_ids) for head in self.heads]
        final_attn = torch.cat(outs, dim=-1)
        final_attn = self.attn_output(final_attn)
        return final_attn


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(
            in_features=config.embed_dim, out_features=config.intermediate_dim
        )
        self.linear_2 = nn.Linear(
            in_features=config.intermediate_dim, out_features=config.embed_dim
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
        self.layernorm_1 = nn.LayerNorm(config.embed_dim)
        self.layernorm_2 = nn.LayerNorm(config.embed_dim)
        self.multihead_attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

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
        self.embeddings = Embeddings(config)
        self.encoder_layers = nn.Sequential(
            *[TransformerEncoderLayer(config) for _ in range(config.num_encoder_blocks)]
        )

    def forward(self, input_ids):
        embeddings = self.embeddings(input_ids)
        x = self.encoder_layers(embeddings)
        return x


class BertConfig:
    def __init__(
        self,
        embed_dim=768,
        vocab_size=30_000,
        max_seq_length=50,
        num_encoder_blocks=12,
        num_heads=8,
        intermediate_dim=3072,
        classification_head=False,
        num_labels=None
    ):
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.num_encoder_blocks = num_encoder_blocks
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.classification_head = classification_head
        self.num_labels = num_labels


class BERT(nn.Module):
    """
    This is an implementation of BERT (Bidirectional Encoder Representation) architecture.

    parameters:
        config:  The model configuration. Must be a BertConfig object that contains:
                    embed_dim: Inner-model hidden-state dimension.
                    vocab_size: Vocabulary size of the used dataset.
                    max_seq_length: Maximum sequence length.
                    num_encoder_blocks: Number of encoder blocks.
                    num_heads: Number of heads in the Self-Attention layer.
                    intermediate_dim: Dimension of the inner feed-forward neural network.
                    classification_head: bool. Default 'False'.
                    num_labels: Number of labels for the classification head.
                    
    methods:
        forward: Takes a torch tensor of int token IDs of shape (N, max_seq_length), where N is the batch size.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = TransformerEncoder(config)
        if self.config.classification_head:
            self.dropout = nn.Dropout()
            self.classifier = nn.Linear(config.embed_dim, config.num_labels)

    def forward(self, input_ids):
        x = self.encoder(input_ids)
        if self.config.classification_head:
            x = self.dropout(x[:, 0, :])
            x = self.classifier(x)
        return x
