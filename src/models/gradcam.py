import torch
from src.models.modules.base_model import BaseModel
from src.models.modules.classifiers import MLPClassifier

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
        logits = self.classifier(self._fmaps)
        self._preds = torch.argmax(logits,dim=1)

        return logits

    def get_cam(self,threshold:float=0.7,x=None):
        self.zero_grad()
        with torch.enable_grad():
            if x is None:
                x = self.x
            logits = self.forward(x)
            y_preds = logits.argmax(1)
            scores = logits[torch.arange(logits.shape[0]),y_preds]

            self.grads = torch.autograd.grad(scores,self._fmaps,grad_outputs=torch.ones_like(scores),create_graph=False)[0]

        grad_weights = self.grads.mean(dim=(2,3)).unsqueeze(-1).unsqueeze(-1)
        cams = self._fmaps*grad_weights
        cams = torch.relu(torch.sum(cams,dim=1,keepdim=True))
        cams = self._process_cam(cams,threshold=threshold)
        return cams

    def __str__(self,):
        return 'GradCAM (Selvaraju et al, 2017)'

class GradCAMPlusPlus(GradCAM):
    def __init__(self,num_classes:int,backbone:torch.nn.Module,classifier:torch.nn.Module,ch_project:str='mapper'):
        super().__init__(num_classes=num_classes,backbone=backbone,classifier=classifier,ch_project=ch_project)
        self.classifier = MLPClassifier(self.num_classes,fmaps_shape=self.backbone.output_shape)

    def get_cam(self,threshold:float=0.7,x=None):
        self.zero_grad()
        with torch.enable_grad():
            if x is None:
                x = self.x
            logits = self.forward(x)
            y_preds = logits.argmax(1)
            scores = logits[torch.arange(logits.shape[0]),y_preds]

            self.grads = torch.autograd.grad(scores,self._fmaps,grad_outputs=torch.ones_like(scores),create_graph=True)[0]

        alpha = self.grads**2/(2*self.grads**2+torch.sum(self._fmaps*self.grads**3,dim=(2,3),keepdim=True))
        grad_weights = torch.sum(alpha*torch.relu(self.grads),dim=(2,3),keepdim=True)
        cams = self._fmaps*grad_weights
        cams = torch.sum(cams,dim=1,keepdim=True)
        cams = self._process_cam(cams,threshold=threshold)
        return cams

    def __str__(self,): # type: ignore
        return 'GradCAM++ (Chattopadhyay et al, 2017)'