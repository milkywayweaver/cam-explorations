import torch
from torch import nn
from torchvision import models

class CAM(nn.Module):
    def __init__(self,num_class,backbone='vgg'):
        super().__init__()
        if backbone == 'vgg':
            self.pretrain = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1)
            self.backbone = self.pretrain.features
            del self.pretrain.avgpool
            del self.pretrain.classifier
        elif backbone == 'resnet':
            self.pretrain = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).children())[:-2])
            del self.pretrain.avgpool
            del self.pretrain.fc
        elif backbone == 'effnet':
            self.pretrain = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1).children())[:-2])
            del self.pretrain.avgpool
            del self.pretrain.classifier
        else:
            raise ValueError('Unknown backbone! Use "vgg", "resnet", or "effnet".')
        
        dummy_input = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            dummy_output = self.backbone(dummy_input)
        fmaps_count = dummy_output.shape[1]
        
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
        x = x.expand(-1,3,-1,-1)
        fmaps = self.backbone(x)
        fmaps_cls = self.classifier(fmaps)
        logits = self.gap(fmaps_cls).squeeze()
        preds = torch.argmax(logits,dim=1)

        self.cams = self.cam_to_mask(fmaps_cls,preds)
        return logits
    
    def cam_to_mask(self,cams,preds,threshold=0.7):
        cams = torch.relu(cams)
        cams = cams[np.arange(cams.shape[0]),preds,:,:]
        cams_min = cams.min(dim=-1)[0].min(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
        cams_max = cams.max(dim=-1)[0].max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)

        norm_cam = ((cams-cams_min)/(cams_max-cams_min+1e-8)).unsqueeze(1)
        norm_cam = torch.nn.functional.interpolate(norm_cam,(224,224))
        
        preds_masks = norm_cam >= threshold
        return preds_masks