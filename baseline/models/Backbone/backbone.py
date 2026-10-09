import torch.nn as nn
import timm

# name -> (timm model, output channels); only the backbones used in the experiments
BACKBONES = {
    "ViTS-224": ("vit_small_patch16_224", 384),
    "DeitS-224": ("deit_small_distilled_patch16_224", 384),
    "Swinv2T-256": ("swinv2_tiny_window16_256", 768),
}


def make_backbone(opt):
    return Backbone(opt)


class Backbone(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.img_size = (opt.h, opt.w)
        if opt.backbone not in BACKBONES:
            raise NameError("{} not in the backbone list {}".format(opt.backbone, list(BACKBONES)))
        timm_name, self.output_channel = BACKBONES[opt.backbone]
        kwargs = {"img_size": self.img_size} if opt.backbone == "ViTS-224" else {}
        self.backbone = timm.create_model(timm_name, pretrained=True, **kwargs)

    def forward(self, image):
        return self.backbone.forward_features(image)
