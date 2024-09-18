My implementation of the BERT transformer. As per "[Attention Is All You Need](https://arxiv.org/pdf/1706.03762)" and "[BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding](https://arxiv.org/pdf/1810.04805)".

# Usage

```python
from bert import BERT, BertConfig

config = BertConfig(
        embed_dim=768,
        vocab_size=30_000,
        max_seq_length=64,
        num_encoder_blocks=12,
        num_heads=8,
        intermediate_dim=3072,
        classification_head=True,
        num_labels=2
    )

model = BERT(config)
print(model)
```

## Output
<pre>
BERT(
  (encoder): TransformerEncoder(
    (embeddings): Embeddings(
      (token_embeds): Embedding(30000, 768)
      (position_embeds): Embedding(64, 768)
      (layernorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.5, inplace=False)
    )
    (encoder_layers): ModuleList(
      (0-11): 12 x TransformerEncoderLayer(
        (layernorm_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (layernorm_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (multihead_attention): MultiHeadAttention(
          (heads): ModuleList(
            (0-7): 8 x AttentionHead(
              (q_net): Linear(in_features=768, out_features=96, bias=True)
              (k_net): Linear(in_features=768, out_features=96, bias=True)
              (v_net): Linear(in_features=768, out_features=96, bias=True)
            )
          )
          (attn_output): Linear(in_features=768, out_features=768, bias=True)
        )
        (feed_forward): FeedForward(
          (linear_1): Linear(in_features=768, out_features=3072, bias=True)
          (linear_2): Linear(in_features=3072, out_features=768, bias=True)
          (gelu): GELU(approximate='none')
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
  )
  (dropout): Dropout(p=0.5, inplace=False)
  (classifier): Linear(in_features=768, out_features=2, bias=True)
)
</pre>