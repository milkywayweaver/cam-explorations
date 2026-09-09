import torch
from torch import nn
from torchvision import models

class ModelMeta(type):
    def __str__(cls):
        return f'{cls.__name__}'

class BaseModel(nn.Module,metaclass=ModelMeta):
    def __init__(self):
        super().__init__()

    def forward(self,x:torch.Tensor):
        pass

    def set_backbone(self,backbone):
        '''
        Sets the backbone of the selected CAM model.
        Args:
            backbone (BaseBackbone): Backbone class of choice.
        Returns:
            None
        '''
        self.backbone = backbone

    def set_classifier(self,classifier):
        '''
        Sets the classifier of the selected CAM model.
        Args:
            classifier (BaseClassifier): Classifier class of choice.
        Returns:
            None
        '''
        self.classifier = classifier

    def set_projection(self,projection):
        '''
        Sets the projection method that converts grayscale images to 3 channels for the selected CAM model.
        Args:
            projection (str): Projection method.
                Supported values are:
                    - "mapper": Uses convolutional layer that outputs 3 channels.
                    - "duplicate": Duplicates the grayscale image to form 3 channels.
        Returns:
            None
        '''
        if projection not in ['mapper','duplicate']:
            raise ValueError('Unknown style! Use "mapper",or "duplicate".')
        self.projection = projection
        if self.projection == 'mapper':
            self.mapper = nn.Conv2d(1,3,kernel_size=3,stride=1,padding=1)

    def _attr_check(self):
        '''
        Helper function to make sure that all three required attributes (backbone, classifier, and projection) are set before forward() method is called.
        '''
        if (not hasattr(self,'backbone')) | (not hasattr(self,'classifier')) | (not hasattr(self,'projection')):
            raise AttributeError('backbone, classifier, and projection must be set! Use set_backbone(), set_classifier(), and set_projection() to set the required attributes resepectively.')

    def get_cam(self,threshold:float=0.7) -> torch.Tensor:
        '''
        Helper function to create CAMs based on instance variable.
        Args:
            threshold (float || str, default=0.7): Threshold to use for CAM to mask conversion. To be passed to _cam_to_mask method to convert CAMs into masks. Pass "raw" to get raw cam instead of mask.
        Returns:
            cams (torch.Tensor): CAMs with a shape of Bx1xHxW.
        '''
        raise NotImplementedError('get_cam method has not been implemented.')

    def _process_cam(self,cams:torch.Tensor,threshold:float|str=0.7,img_size:tuple[int,int]|None=(224,224)) -> torch.Tensor:
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
        if img_size:
            norm_cam = torch.nn.functional.interpolate(norm_cam,img_size,mode='bilinear')

        if threshold != 'raw':
            norm_cam = norm_cam >= threshold # type: ignore
        return norm_cam