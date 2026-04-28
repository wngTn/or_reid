import torch
from einops import rearrange

from ..base_model import BaseModel
from ..modules import HorizontalPoolingPyramid, PackSequenceWrapper, SeparateBNNecks, SeparateFCs, SetBlockWrapper


class Baseline(BaseModel):
    def build_network(self, model_cfg):
        self.Backbone = self.get_backbone(model_cfg["backbone_cfg"])
        self.Backbone = SetBlockWrapper(self.Backbone)
        self.FCs = SeparateFCs(**model_cfg["SeparateFCs"])
        if self._training_mode:
            self.BNNecks = SeparateBNNecks(**model_cfg["SeparateBNNecks"])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg["bin_num"])

    def forward(self, inputs):
        if not isinstance(inputs, tuple):
            return self.forward_single(inputs)
        ipts, labs, _, _, seqL = inputs

        sils = ipts[0]
        # Single view 2D
        sils = rearrange(sils, "n s c h w -> n c s h w")

        del ipts
        outs = self.Backbone(sils)  # [n, c, s, h, w]

        # Temporal Pooling, TP
        outs = self.TP(outs, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        embed_1 = self.FCs(feat)  # [n, c, p]

        retval = {
            "inference_feat": {"embeddings": embed_1},
        }
        if self.training:
            embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]
            retval["training_feat"] = {
                "triplet": {"embeddings": embed_1, "labels": labs},
                "softmax": {"logits": logits, "labels": labs},
            }
        return retval

    def forward_single(self, inputs):
        sils = inputs
        # Single view 2D
        sils = rearrange(sils, "n s c h w -> n c s h w")

        outs = self.Backbone(sils)  # [n, c, s, h, w]

        # Temporal Pooling, TP
        outs = self.TP(outs, None, options={"dim": 2})[0]  # [n, c, h, w]
        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        embed_1 = self.FCs(feat)  # [n, c, p]
        return embed_1
