import torch.nn as nn
from .SingleBranch import SingleBranch, SingleBranchSwin
from .FSRA import FSRA
from .LPN import LPN

HEADS = {"SingleBranch": SingleBranch, "SingleBranchSwin": SingleBranchSwin, "FSRA": FSRA, "LPN": LPN}


def make_head(opt):
    return Head(opt)


class Head(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        if opt.head not in HEADS:
            raise NameError("{} not in the head list {}".format(opt.head, list(HEADS)))
        self.head = HEADS[opt.head](opt)
        self.opt = opt

    def forward(self, features):
        return self.head(features)
