"""R2: corrected visual masking and question-conditioned multi-query pooling.

The original ViewSelector stays unchanged for the published-checkpoint baseline.
Only this lightweight selector is trained; inputs are frozen LLaVA features.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .ViewSelector import MCAN_ED, AttFlat


class QueryViewSelector(nn.Module):
    def __init__(self, input_channel=3584, embedding_dim=256, flat_out_size=512,
                 num_layers=2, num_heads=8, num_queries=4):
        super().__init__()
        self.text_feat_linear = nn.Sequential(nn.Linear(input_channel, embedding_dim, bias=False), nn.GELU())
        self.image_feat_linear = nn.Sequential(nn.Linear(input_channel, embedding_dim, bias=False), nn.GELU())
        self.fusion_backbone = MCAN_ED(embedding_dim, num_heads, num_layers, pdrop=0.1)
        self.attflat_lang = AttFlat(embedding_dim, embedding_dim, 1, flat_out_size, 0.1)
        self.question_query = nn.Linear(flat_out_size, embedding_dim)
        self.queries = nn.Parameter(torch.empty(num_queries, embedding_dim))
        self.visual_pool = nn.MultiheadAttention(embedding_dim, num_heads, dropout=0.1, batch_first=True)
        self.pool_norm = nn.LayerNorm(embedding_dim)
        self.visual_output = nn.Linear(num_queries * embedding_dim, flat_out_size)
        nn.init.normal_(self.queries, std=0.02)

    def forward(self, images, text, image_padding, text_padding):
        # [views, patches, input_dim], [views, text_tokens, input_dim].
        # True means padding only; real image tokens are never all masked.
        text_mask = text_padding[:, None, None, :]
        image_mask = image_padding[:, None, None, :]
        language, visual = self.fusion_backbone(
            self.text_feat_linear(text), self.image_feat_linear(images), text_mask, image_mask)
        question = self.attflat_lang(language, text_mask)
        queries = self.queries[None] + self.question_query(question)[:, None]
        summaries, _ = self.visual_pool(queries, visual, visual,
                                        key_padding_mask=image_padding, need_weights=False)
        view = self.visual_output(self.pool_norm(summaries).flatten(1))
        return F.cosine_similarity(question, view, dim=-1)


def binary_losses(scores, labels):
    # Preserve the upstream cosine-to-probability BCE; clamp numerical endpoints.
    probabilities = ((scores.float() + 1) / 2).clamp(1e-6, 1 - 1e-6)
    return F.binary_cross_entropy(probabilities, labels.float(), reduction="none")
