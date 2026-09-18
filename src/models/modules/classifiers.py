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
    '''
    Multi Layer Perceptron classifier class.
    Args:
        num_classes (int): Number of classes
        fmaps_shape (torch.Size): [Batch, Channel, Height, Width] shape of the last feature maps
        layers (list[int], default=[128,128]): MLP layer configurations.
            Each element represents one layer of N neurons represented by the integer value.
            The total amount of layers are always length of layers + 1 with the last additional layer having output features of num_classes.
    Returns:
        None
    '''
    def __init__(self,num_classes,fmaps_shape,layers=[128,128]):
        super().__init__(num_classes=num_classes,fmaps_shape=fmaps_shape)
        flattened_shape = self.fmaps_shape[1]*self.fmaps_shape[2]*self.fmaps_shape[3]
        layers.insert(0,flattened_shape)

        sequence_list:list[nn.Module] = [nn.Flatten(start_dim=1,end_dim=-1)]
        for i in range(len(layers)-1):
            sequence_list.append(nn.Linear(in_features=layers[i],out_features=layers[i+1]))
            sequence_list.append(nn.LeakyReLU())
        sequence_list.append(nn.Linear(in_features=layers[-1],out_features=self.num_classes))


        self.classifier = nn.Sequential(*sequence_list)

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