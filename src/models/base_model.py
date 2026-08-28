import torch
from torch import nn
from torchvision import models

class BaseModel(nn.Module):
    def __init__(self,num_class:int,backbone_model_type:str='vgg',ch_project:str='mapper'):
        super().__init__()
        self.num_class = num_class

        self.backbone_model_type = backbone_model_type
        if self.backbone_model_type == 'vgg':
            self.pretrain = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(self.pretrain.features.children())[:-13])
            del self.pretrain.avgpool
            del self.pretrain.classifier
        elif self.backbone_model_type == 'resnet':
            self.pretrain = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).children())[:-3])
            del self.pretrain.avgpool
            del self.pretrain.fc
        elif self.backbone_model_type == 'effnet':
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
            # nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
            # nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(),
            nn.Conv2d(128,num_class,kernel_size=1,stride=1,padding=0)
        )
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1,1))

    def forward(self,x:torch.Tensor):
        if self.ch_project == 'mapper':
            x = self.mapper(x)
        else:
            x = x.expand(-1,3,-1,-1)

        self._fmaps = self.backbone(x)
        self._fmaps_cls = self.classifier(self._fmaps)
        logits = self.gap(self._fmaps_cls).squeeze()
        self._preds = torch.argmax(logits,dim=1)

        return logits

    def get_cam(self,threshold:float=0.7) -> torch.Tensor:
        '''
        Helper function to create CAMs based on instance variable.
        Args:
            threshold (float || str, default=0.7): Threshold to use for CAM to mask conversion. To be passed to _cam_to_mask method to convert CAMs into masks. Pass "raw" to get raw cam instead of mask.
        Returns:
            cams (torch.Tensor): CAMs with a shape of Bx1xHxW.
        '''
        raise NotImplementedError('get_cam method has not been implemented.')

    def _process_cam(self,cams:torch.Tensor,threshold:float|str=0.7,img_size:tuple[int,int]=(224,224)) -> torch.Tensor:
        '''
        Helper function to convert raw CAM to prediction mask
        Args:
            cams (torch.Tensor): CAMs with Bx1xHxW.
            threshold (float | str, default=0.7): Threshold to use for CAM to mask conversion. Use "raw" to keep raw CAM.
            img_size (tuple[int,int], default=(224,224)): The size of the original input image, used to scale the obtained masks back to original size.
        Returns:
            masks (torch.Tensor): Masks obtained from thresholding CAMs.
        '''
        cams_min = cams.min(dim=-1)[0].min(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
        cams_max = cams.max(dim=-1)[0].max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)

        norm_cam = ((cams-cams_min)/(cams_max-cams_min+1e-8))
        norm_cam = torch.nn.functional.interpolate(norm_cam,img_size,mode='bilinear')

        if threshold != 'raw':
            norm_cam = norm_cam >= threshold
        return norm_cam