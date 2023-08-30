import torch.nn as nn
from ..builder import ATTR_HEAD
from mmcv.cnn import Linear, bias_init_with_prob
@ATTR_HEAD.register_module()
class Attribute_Classifier(nn.Module):
    def __init__(self,
                 attr_classes=[6, 8, 5, 4, 7],
                 mlp_layers=3,
                 in_channels=256, 
                 feedforward_channels=256):
        super(Attribute_Classifier, self).__init__()
        self.mlp_layers = mlp_layers
        assert self.mlp_layers > 2
        self.attr_classes=attr_classes
        self.in_channels=in_channels
        self.feedforward_channels=feedforward_channels
        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        for i, out_class in enumerate(self.attr_classes):
            layer_name = 'attr_{}_out_{}_class_head'.format(i, out_class)
            self.add_module(layer_name, self.base_mlp(out_class))
    
    def base_mlp(self, out_value_num):
        attr_branches = []
        attr_branches.append(Linear(self.in_channels, self.feedforward_channels))
        attr_branches.append(nn.LayerNorm(self.feedforward_channels))
        attr_branches.append(nn.ReLU())
        for _ in range(self.mlp_layers - 2):
            attr_branches.append(Linear(self.feedforward_channels, self.feedforward_channels))
            attr_branches.append(nn.LayerNorm(self.feedforward_channels))
            attr_branches.append(nn.ReLU())
        attr_branches.append(Linear(self.feedforward_channels, out_value_num))
        mlp = nn.Sequential(*attr_branches)
        return mlp
    
    def init_weights(self):
        bias_init = bias_init_with_prob(0.01)
        for m in self.children():
            nn.init.constant_(m[-1].bias, bias_init)

    def forward(self, pts_embed):
        attr_score=[]
        for i, out_class in enumerate(self.attr_classes):
            layer_name = 'attr_{}_out_{}_class_head'.format(i, out_class)
            attr_layer = getattr(self, layer_name)
            attr_score.append(attr_layer(pts_embed))
        return attr_score