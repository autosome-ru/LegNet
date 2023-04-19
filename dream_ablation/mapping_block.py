from torch import nn

class MappingBlock(nn.Module):
    def __init__(self, in_ch, out_ch, activation):
        super().__init__()
        self.block =  nn.Sequential(
                        nn.Conv1d(
                            in_channels=in_ch,
                            out_channels=out_ch,
                            kernel_size=1,
                            padding='same',
                       ),
                       activation()
        )
        
    def forward(self, x):
        return self.block(x)