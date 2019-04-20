import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class FirstModel(BaseModel):
    def __init__(self, num_classes=10):
        super(FirstModel, self).__init__()
        

    def forward(self, x):
        return