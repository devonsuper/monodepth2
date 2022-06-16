import numpy as np
import torch
import torch.nn as nn

class MergedModel(nn.Module):
    def __init__(self, modelA, modelB):
        super(MergedModel, self).__init__()
        self.modelA = modelA
        self.modelB = modelB

    def forward(self, x):
        x = self.modelA(x)
        x = self.modelB(x)

        return x