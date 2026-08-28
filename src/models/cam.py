import torch
from src.models.modules.base_model import BaseModel
from src.models.modules.classifiers import GAPClassifier,ConvClassifier

class CAM(BaseModel):
    '''
    CAM generation method based on Zhou et al. (2016).
    Uses Global Average Pooling (GAP) layer that connects directly to output layer.
    CAM overrides any classifiers into GAPClassifier.
    '''
    def __init__(self,num_classes:int,backbone:torch.nn.Module,classifier:torch.nn.Module,ch_project:str='mapper'):
        super().__init__(num_classes=num_classes,backbone=backbone,classifier=classifier,ch_project=ch_project)

        self.classifier = GAPClassifier(self.num_classes,fmaps_shape=self.backbone.output_shape)

    def forward(self,x:torch.Tensor):
        if self.ch_project == 'mapper':
            x = self.mapper(x)
        else:
            x = x.expand(-1,3,-1,-1)

        self._fmaps = self.backbone(x)
        logits = self.classifier(self._fmaps)
        self._preds = torch.argmax(logits,dim=1)

        return logits

    def get_cam(self,threshold:float=0.7):
        weights = self.classifier.classifier[-1].weight.data
        preds_weights = weights[self._preds,:].unsqueeze(-1).unsqueeze(-1)
        cams = torch.sum(self._fmaps*preds_weights,dim=1,keepdim=True)
        cams = self._process_cam(cams,threshold=threshold)
        return cams

class ConvCAM(BaseModel):
    '''
    CAM generation method based on Zhang et al. (2018).
    Uses convolutional layers that squeeze the channels to the number of classes before taking the channel-wise average as the output logit.
    ConvCAM overrides any classifiers into ConvClassifier.
    '''
    def __init__(self,num_classes:int,backbone:torch.nn.Module,classifier:torch.nn.Module,ch_project:str='mapper'):
        super().__init__(num_classes=num_classes,backbone=backbone,classifier=classifier,ch_project=ch_project)

        self.classifier = ConvClassifier(self.num_classes,fmaps_shape=self.backbone.output_shape)

    def forward(self,x:torch.Tensor):
        if self.ch_project == 'mapper':
            x = self.mapper(x)
        else:
            x = x.expand(-1,3,-1,-1)

        self._fmaps = self.backbone(x)
        self._fmaps_cls = self.classifier.classifier(self._fmaps)
        logits = self.classifier.gap(self._fmaps_cls).squeeze(-1).squeeze(-1)
        self._preds = torch.argmax(logits,dim=1)

        return logits

    def get_cam(self,threshold:float=0.7):
        
        cams = self._fmaps_cls[torch.arange(self._fmaps_cls.shape[0]),self._preds,:,:].unsqueeze(1)
        cams = self._process_cam(cams,threshold=threshold)
        return cams