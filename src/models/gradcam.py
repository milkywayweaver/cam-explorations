import torch
from src.models.modules.base_model import BaseModel

class GradCAM(BaseModel):
    def __init__(self,num_classes:int,backbone:torch.nn.Module,classifier:torch.nn.Module,ch_project:str='mapper'):
        super().__init__(num_classes=num_classes,backbone=backbone,classifier=classifier,ch_project=ch_project)

    def forward(self,x:torch.Tensor):
        self.x = x.clone()
        if self.ch_project == 'mapper':
            x = self.mapper(x)
        else:
            x = self.x.expand(-1,3,-1,-1)

        self._fmaps = self.backbone(x)
        # Register hook 
        if self._fmaps.requires_grad:
            self._fmaps.register_hook(self._get_grads)

        logits = self.classifier(self._fmaps)
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
        cams = self._process_cam(cams,threshold=threshold)
        return cams

