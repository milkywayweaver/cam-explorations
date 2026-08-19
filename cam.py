import torch
from torch import nn
from torchvision import models

class CAM(nn.Module):
    def __init__(self,num_class:int,backbone:str='vgg',ch_project:str='mapper'):
        '''
        Class Activation Mapping (CAM) model using method proposed in ACoL paper (Zhang et al, 2018)
        Args:
            num_class (int): The number of output classes
            backbone (str, default="vgg"): Backbone model to use
                Supported values are:
                    - "vgg" : VGG 19 BN
                    - "resnet" : ResNet50
                    - "effnet" : EfficientNet B0
            ch_project (str, default="mapper"): Grayscale to RGB converter to fit the model into backbone channel requirement
                Supported values are:
                    - "mapper" : Uses a Conv2d layer that projects the singular channel 1C --> 3C
                    = "dupliacte" : Duplicates the singular channel 1C --> 3C
        Returns:
            None
        '''
        super().__init__()
        if backbone == 'vgg':
            self.pretrain = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(self.pretrain.features.children())[:-13])
            del self.pretrain.avgpool
            del self.pretrain.classifier
        elif backbone == 'resnet':
            self.pretrain = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).children())[:-3])
            del self.pretrain.avgpool
            del self.pretrain.fc
        elif backbone == 'effnet':
            self.pretrain = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(list(models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1).children())[0].children())[:6])
            del self.pretrain.avgpool
            del self.pretrain.classifier
        else:
            raise ValueError('Unknown backbone! Use "vgg", "resnet", or "effnet".')
        if ch_project not in ['mapper','duplicate']:
            raise ValueError('Unknown style! Use "mapper",or "duplicate".')
    
        self.ch_project = ch_project
        
        dummy_input = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            dummy_output = self.backbone(dummy_input)
        fmaps_count = dummy_output.shape[1]
        
        self.mapper = nn.Conv2d(1,3,kernel_size=3,stride=1,padding=1)
        self.classifier  = nn.Sequential(
            nn.Conv2d(fmaps_count,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(128,num_class,kernel_size=1,stride=1,padding=0)
        )
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1,1))
        
    def forward(self,x):
        # 1C --> 3C Projection
        if self.ch_project == 'mapper':
            x = self.mapper(x)
        else:
            x = x.expand(-1,3,-1,-1)

        fmaps = self.backbone(x)
        self.__fmaps_cls = self.classifier(fmaps)
        logits = self.gap(self.__fmaps_cls).squeeze()
        self.__preds = torch.argmax(logits,dim=1)

        return logits
    
    def get_cam(self,relu=True,threshold=0.7):
        '''
        Gets processed CAMs
        Args:
            relu (bool, default=True): Whether to apply relu to the raw cam before processing or not
            threshold (float, default=0.7): Threshold value to convert raw cam into segmentation mask
        '''
        cams = self.__cam_to_mask(self.__fmaps_cls,self.__preds,relu=relu,threshold=threshold)
        return cams
    
    def __cam_to_mask(self,cams,preds,relu,threshold):
        '''
        Helper function to convert raw CAM to prediction mask
        '''
        if relu:
            cams = torch.relu(cams)
        cams = cams[torch.arange(cams.shape[0]),preds,:,:]
        cams_min = cams.min(dim=-1)[0].min(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
        cams_max = cams.max(dim=-1)[0].max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)

        norm_cam = ((cams-cams_min)/(cams_max-cams_min+1e-8)).unsqueeze(1)
        norm_cam = torch.nn.functional.interpolate(norm_cam,(224,224),mode='bilinear')
        
        preds_masks = norm_cam >= threshold
        return preds_masks