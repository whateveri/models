
import torch
from torch import nn

class TinyVGG(nn.Module):
    def __init__(self,input_features:int,
                 hidden_features:int,
                 output_features:int
                ) -> None:
        super().__init__()

        self.block1=nn.Sequential(
            nn.Conv2d(in_channels=input_features,
                      out_channels=hidden_features,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_features,
                      out_channels=hidden_features,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.block2=nn.Sequential(
            nn.Conv2d(in_channels=hidden_features,
                      out_channels=hidden_features,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_features,
                      out_channels=hidden_features,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
            
        )

        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=13*13*hidden_features,
                      out_features=output_features)
        )

    def forward(self,x:torch.Tensor):
        x=self.block1(x)
        x=self.block2(x)
        x=self.classifier(x)

        return x
