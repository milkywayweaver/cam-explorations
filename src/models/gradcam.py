import torch
from src.models.modules.base_model import BaseModel
from src.models.modules.classifiers import MLPClassifier

class GradCAM(BaseModel):
    '''
    CAM generation method based on Selvaraju et al (2017)'.
    Uses gradients on the feature maps wrt selected class to weight feature maps.
    '''
    def __init__(self):
        super().__init__()

    def forward(self,x:torch.Tensor):
        self._attr_check()
        self.x = x.clone()
        if self.projection == 'mapper':
            x = self.mapper(x)
        else:
            x = x.expand(-1,3,-1,-1)

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
    '''
    CAM generation method based on Chattopadhyay et al (2017)'.
    Improves GradCAM by using higher order derivative to weight each pixel in each feature maps.
    '''
    def __init__(self):
        super().__init__()

    def set_classifier(self, classifier):
        '''
        Overrides the provided classifier with MLP classifier as the model assumption require.
        '''
        self.num_classes = classifier.num_classes
        self.fmaps_shape = classifier.fmaps_shape
        self.classifier = MLPClassifier(self.num_classes,fmaps_shape=self.fmaps_shape)
        

    def get_cam(self,threshold:float=0.7,x=None):
        self._attr_check()
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

class LayerCAM(GradCAM):
    '''
    CAM generation method based on Jiang et al (2021)'.
    Improves GradCAM by weighing each pixel individually and compute CAMs on different layers.
    '''
    def __init__(self):
        super().__init__()
        self.activations = []
        self.grads = []

    def set_backbone(self, backbone):
        self.backbone = backbone
        self.layers = self.backbone.layer_cam()
        
        def activation_hook(module,input,output):
            self.activations.append(output.detach())
            return None
        def grads_hook(module,grads_in,grads_out):
            self.grads.append(grads_out[0].detach())
            return None

        for layer in self.layers:
            layer.register_forward_hook(activation_hook)
            layer.register_full_backward_hook(grads_hook)

    def forward(self,x:torch.Tensor):
        self._attr_check()
        self.activations = []
        self.grads = []

        self.x = x.clone()
        if self.projection == 'mapper':
            x = self.mapper(x)
        else:
            x = x.expand(-1,3,-1,-1)

        self._fmaps = self.backbone(x)
        logits = self.classifier(self._fmaps)
        self._preds = torch.argmax(logits,dim=1)

        return logits

    def scale_function(self,cams,gamma):
        return torch.tanh(gamma*cams/torch.amax(cams,dim=(2,3),keepdim=True))

    def get_cam(self,threshold:float=0.7,x=None):
        self.zero_grad()
        with torch.enable_grad():
            if x is None:
                x = self.x
            logits = self.forward(x)
            y_preds = logits.argmax(1)
            scores = logits[torch.arange(logits.shape[0]),y_preds]
            scores.backward(torch.ones_like(scores))
        self.grads = self.grads[::-1]
        cams = []
        for i in range(len(self.activations)):
            weight = torch.relu(self.grads[i])
            cam = torch.relu(torch.sum(weight*self.activations[i],dim=1,keepdim=True))
            cam = self._process_cam(cam,threshold='raw')
            cams.append(cam)
        cams = torch.cat(cams,dim=1)
        cams = self.scale_function(cams,2)
        cams = torch.amax(cams,dim=1,keepdim=True)
        cams = self._process_cam(cams,threshold=threshold)
        return cams

    def __str__(self,): # type: ignore
        return 'LayerCAM (Jiang et al, 2021)'