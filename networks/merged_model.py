import numpy as np
import torch
import torch.nn as nn

class MergedModel(nn.Module):
    def __init__(self, modelA, modelB, output_index=0):
        super(MergedModel, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.output_index = output_index

    def forward(self, x):
        x = self.modelA(x)
        x = self.modelB(x)

        x = x[self.output_index]

        return x