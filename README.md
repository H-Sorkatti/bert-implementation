My implementation of the BERT transformer. As per "[Attention Is All You Need](https://arxiv.org/pdf/1706.03762)" and "[BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding](https://arxiv.org/pdf/1810.04805)".

# Usage
```python
from bert import BERT, BertConfig

config = BertConfig(
        embed_dim=768,
        vocab_size=30_000,
        max_seq_length=50,
        num_encoder_blocks=12,
        num_heads=8,
        intermediate_dim=3072,
        classification_head=True,
        num_labels=2
    )

model = BERT(config)
print(model)
```