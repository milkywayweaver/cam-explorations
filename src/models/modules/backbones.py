import torch
from torch import nn
from torchvision import models

class BackboneMeta(type):
    def __str__(cls):
        return f'{cls.__name__}'

class BaseBackbone(nn.Module,metaclass=BackboneMeta):
    '''
    Base class for backbone class.
    Child backbone class needs to implement self.backbone with nn.Module object
    '''
    def __init__(self):
        super().__init__()

    @property
    def output_shape(self):
        dummy_input = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            dummy_output = self.backbone(dummy_input) # type: ignore
        output_shape = dummy_output.shape
        return output_shape

class VGGBackbone(BaseBackbone):
    def __init__(self,):
        super().__init__()
        self.pretrain = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(self.pretrain.features.children())[:-13])
        del self.pretrain

    def forward(self,x:torch.Tensor):
        return self.backbone(x)

    def __str__(self,):
        return 'VGG'

class ResNetBackbone(BaseBackbone):
    def __init__(self,):
        super().__init__()
        self.pretrain = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(self.pretrain.children())[:-3])
        del self.pretrain

    def forward(self,x:torch.Tensor):
        return self.backbone(x)

    def __str__(self,):
        return 'ResNet'

class EffNetBackbone(BaseBackbone):
    def __init__(self,):
        super().__init__()
        self.pretrain = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(list(self.pretrain.children())[0].children())[:6])
        del self.pretrain

    def forward(self,x:torch.Tensor):
        return self.backbone(x)

    def __str__(self,):
        return 'EfficientNet'

if __name__ == '__main__':
    model = VGGBackbone()
    print(model.backbone)