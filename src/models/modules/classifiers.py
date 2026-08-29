import torch
from torch import nn

class ClassifierMeta(type):
    def __str__(cls):
        return f'{cls.__name__}'

class BaseClassifier(nn.Module,metaclass=ClassifierMeta):
    '''
    Base class for classifier class.
    Child classifier class needs to implement self.classifier with nn.Module object
    '''
    def __init__(self,num_classes,fmaps_shape):
        super().__init__()
        self.num_classes = num_classes
        self.fmaps_shape = fmaps_shape

    def forward(self,x:torch.Tensor):
        return self.classifier(x)
    
class MLPClassifier(BaseClassifier):
    def __init__(self,num_classes,fmaps_shape):
        super().__init__(num_classes=num_classes,fmaps_shape=fmaps_shape)
        flattened_shape = self.fmaps_shape[1]*self.fmaps_shape[2]*self.fmaps_shape[3]
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1,end_dim=-1),
            nn.Linear(in_features=flattened_shape,out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128,out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128,out_features=self.num_classes)
        )

    def __str__(self,):
        return 'MLP Classifier'

class GAPClassifier(BaseClassifier):
    def __init__(self,num_classes,fmaps_shape):
        super().__init__(num_classes=num_classes,fmaps_shape=fmaps_shape)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1,end_dim=-1),
            nn.Linear(in_features=self.fmaps_shape[1],out_features=self.num_classes)
        )

    def __str__(self,):
        return 'GAP Classifier'

class ConvClassifier(BaseClassifier):
    def __init__(self,num_classes,fmaps_shape):
        super().__init__(num_classes=num_classes,fmaps_shape=fmaps_shape)
        self.classifier  = nn.Sequential(
            nn.Conv2d(self.fmaps_shape[1],128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(),
            nn.Conv2d(128,self.num_classes,kernel_size=1,stride=1,padding=0)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self,x:torch.Tensor):
        fmaps_cls = self.classifier(x)
        logits = self.gap(fmaps_cls).squeeze(-1).squeeze(-1)

        return logits

    def __str__(self,):
        return 'Convolutional Classifier'