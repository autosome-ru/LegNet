from torch import nn 
from se_layer import SELayer
from se_complex import SELayerComplex

class EffBlock(nn.Module):
    def __init__(self, 
                 in_ch, 
                 ks, 
                 resize_factor,
                 filter_per_group,
                 activation, 
                 out_ch=None,
                 se_reduction=None,
                 se_type="complex",
                 inner_dim_calculation="out"
                 ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = self.in_ch if out_ch is None else out_ch
        self.resize_factor = resize_factor
        self.se_reduction = resize_factor if se_reduction is None else se_reduction
        self.ks = ks
        self.inner_dim_calculation = inner_dim_calculation
        if inner_dim_calculation == "out":
            self.inner_dim = self.out_ch * self.resize_factor
        elif inner_dim_calculation == "in":
            self.inner_dim = self.in_ch * self.resize_factor
        else:
            raise Exception(f"Wrong inner_dim_calculation: {inner_dim_calculation}")
            
        self.filter_per_group = filter_per_group
        self.se_type = se_type
        
        if se_type == "simple":
            se_constructor = SELayer
        elif se_type == "complex":
            se_constructor = SELayerComplex
        elif se_type == "none":
            se_constructor = lambda *args, **kwargs: nn.Identity()
        else:
            raise Exception(f"Wrong se_type: {se_type}")
            
        
        
        block = nn.Sequential(
                        nn.Conv1d(
                            in_channels=self.in_ch,
                            out_channels=self.inner_dim,
                            kernel_size=1,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(self.inner_dim),
                       activation(),
                       
                       nn.Conv1d(
                            in_channels=self.inner_dim,
                            out_channels=self.inner_dim,
                            kernel_size=ks,
                            groups=self.inner_dim // self.filter_per_group,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(self.inner_dim),
                       activation(),
                       se_constructor(self.in_ch, 
                                      self.inner_dim,
                                      reduction=self.se_reduction), # self.in_ch is not good
                       nn.Conv1d(
                            in_channels=self.inner_dim,
                            out_channels=self.in_ch,
                            kernel_size=1,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(self.in_ch),
                       activation(),
        )
        
      
        self.block = block
    
    def forward(self, x):
        return self.block(x)
    