import torch
from .base_model import BaseModel

class CAM(BaseModel):
    def __init__(self,num_class:int,backbone_model_type:str='vgg',ch_project:str='mapper'):
        super().__init__(num_class=num_class,backbone_model_type=backbone_model_type,ch_project=ch_project)

    def get_cam(self,threshold:float=0.7):
        
        cams = self._fmaps_cls[torch.arange(self._fmaps_cls.shape[0]),self._preds,:,:].unsqueeze(1)
        cams = self._process_cam(cams,threshold=threshold)
        return cams