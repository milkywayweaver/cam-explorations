import torch
from .base_model import BaseModel

class GradCAM(BaseModel):
    def __init__(self,num_class:int,backbone_model_type:str='vgg',ch_project:str='mapper'):
        super().__init__(num_class=num_class,backbone_model_type=backbone_model_type,ch_project=ch_project)

    def forward(self,x:torch.Tensor):
        self.x = x.clone()
        # Overrides the default forward by applying gradient hook
        if self.ch_project == 'mapper':
            x = self.mapper(x)
        else:
            x = self.x.expand(-1,3,-1,-1)

        self._fmaps = self.backbone(x)
        # Register hook 
        if self._fmaps.requires_grad:
            self._fmaps.register_hook(self._get_grads)
        self._fmaps_cls = self.classifier(self._fmaps)
        logits = self.gap(self._fmaps_cls).squeeze()
        self._preds = torch.argmax(logits,dim=1)

        return logits

    def _get_grads(self,grads):
         self.grads = grads

    def get_cam(self,threshold:float=0.7,x=None):
        self.zero_grad()
        with torch.enable_grad():
            if x is None:
                x = self.x
            logits = self.forward(x)
            y_preds = logits.argmax(1)
            scores = logits[torch.arange(logits.shape[0]),y_preds]
            scores.backward(gradient=torch.ones_like(scores))

        grad_weights = self.grads.mean(dim=(2,3)).unsqueeze(-1).unsqueeze(-1)
        cams = self._fmaps*grad_weights
        cams = torch.relu(torch.sum(cams,dim=1)).unsqueeze(1)
        cams = self._cam_to_mask(cams,threshold=threshold)
        return cams