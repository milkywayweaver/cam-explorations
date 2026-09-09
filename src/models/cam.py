import torch
from src.models.modules.base_model import BaseModel
from src.models.modules.classifiers import GAPClassifier,ConvClassifier

class CAM(BaseModel):
    '''
    CAM generation method based on Zhou et al. (2016).
    Uses Global Average Pooling (GAP) layer that connects directly to output layer.
    CAM overrides any classifiers into GAPClassifier.
    '''
    def __init__(self):
        super().__init__()

    def set_classifier(self, classifier):
        '''
        Overrides the provided classifier with GAP classifier as the model require.
        '''
        self.num_classes = classifier.num_classes
        self.fmaps_shape = classifier.fmaps_shape
        self.classifier = GAPClassifier(self.num_classes,fmaps_shape=self.fmaps_shape)

    def forward(self,x:torch.Tensor):
        self._attr_check()
        if self.projection == 'mapper':
            x = self.mapper(x)
        else:
            x = x.expand(-1,3,-1,-1)

        self._fmaps = self.backbone(x)
        logits = self.classifier(self._fmaps)
        self._preds = torch.argmax(logits,dim=1)

        return logits

    def get_cam(self,threshold:float=0.7):
        weights = self.classifier.classifier[-1].weight.data
        preds_weights = weights[self._preds,:].unsqueeze(-1).unsqueeze(-1) # type: ignore
        cams = torch.sum(self._fmaps*preds_weights,dim=1,keepdim=True)
        cams = self._process_cam(cams,threshold=threshold)
        return cams

    def __str__(self,):
        return 'CAM (Zhou et al, 2016)'

class ConvCAM(BaseModel):
    '''
    CAM generation method based on Zhang et al. (2018).
    Uses convolutional layers that squeeze the channels to the number of classes before taking the channel-wise average as the output logit.
    ConvCAM overrides any classifiers into ConvClassifier.
    '''
    def __init__(self):
        super().__init__()

    def set_classifier(self, classifier):
        '''
        Overrides the provided classifier with convolutional classifier as the model require.
        '''
        self.num_classes = classifier.num_classes
        self.fmaps_shape = classifier.fmaps_shape
        self.classifier = ConvClassifier(num_classes=self.num_classes,fmaps_shape=self.fmaps_shape)

    def forward(self,x:torch.Tensor):
        self._attr_check()
        if self.projection == 'mapper':
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

    def __str__(self,):
        return 'CAM (Zhang et al, 2018)'

class ScoreCAM(BaseModel):
    '''
    CAM generation method based on Wang et al. (2020).
    Uses CIC to weight feature maps.
    ScoreCAM compresses final feature maps into smaller dimension to save computational resources.
    Args:
        compressed_size (int, default=64): The amount of channels to compress the final feature maps into.
    '''
    def __init__(self,compressed_size=64):
        super().__init__()
        self.compressed_size = compressed_size
        
    def set_classifier(self, classifier):
        self.num_classes = classifier.num_classes
        self.compressor = torch.nn.Conv2d(
            self.backbone.output_shape[1],self.compressed_size,kernel_size=(1,1),stride=1,padding=0
            )
        self.fmaps_shape = list(self.backbone.output_shape)
        self.fmaps_shape[1] = self.compressed_size 
        self.classifier = type(classifier)(num_classes=self.num_classes,fmaps_shape=self.fmaps_shape)
        
    def forward(self,x:torch.Tensor):
        self._attr_check()
        self.x = x.clone()
        if self.projection == 'mapper':
            x = self.mapper(x)
        else:
            x = x.expand(-1,3,-1,-1)

        original_fmaps = self.backbone(x)
        self._fmaps = self.compressor(original_fmaps)
        logits = self.classifier(self._fmaps)
        self._preds = torch.argmax(logits,dim=1)

        return logits

    def get_cam(self,threshold:float=0.7):
        Ss = [] # BxC 
        with torch.inference_mode():
            for i in range(self._fmaps.shape[1]):
                mask = self._fmaps[:,i].unsqueeze(1)
                mask = self._process_cam(mask,threshold='raw')
                masked_x = self.x*mask 

                masked_logits = self.forward(masked_x)
                masked_scores = masked_logits[torch.arange(masked_logits.shape[0]),self._preds]

                S = masked_scores # Bx1
                Ss.append(S.unsqueeze(1))
        Ss = torch.cat(Ss,dim=1)
        alpha = torch.softmax(Ss,dim=1).unsqueeze(-1).unsqueeze(-1)
        
        cams = torch.relu(torch.sum(torch.relu(alpha*self._fmaps),dim=1,keepdim=True))
        cams = self._process_cam(cams,threshold=threshold)
        return cams

    def __str__(self,):
        return 'ScoreCAM (Wang et al, 2020)'

class FIMFScoreCAM(BaseModel):
    '''
    CAM generation method based on Li et al. (2022).
    Based on ScoreCAM (Wang et al, 2020) with improvements on the performance side.
    '''
    def __init__(self):
        super().__init__()

    def forward(self,x:torch.Tensor):
        self._attr_check()
        if self.projection == 'mapper':
            x = self.mapper(x)
        else:
            x = x.expand(-1,3,-1,-1)

        self._fmaps = self.backbone(x)
        logits = self.classifier(self._fmaps)
        self._preds = torch.argmax(logits,dim=1)

        return logits

    def get_cam(self,threshold:float=0.7):
        Cs = [] # BxC 
        with torch.inference_mode():
            for i in range(self._fmaps.shape[1]):
                mask = self._fmaps[:,i].unsqueeze(1)
                mask = torch.relu(mask)
                mask = self._process_cam(mask,threshold='raw',img_size=None)
                masked_fmaps = self._fmaps*mask 

                masked_logits = self.classifier(masked_fmaps)
                masked_scores = masked_logits[torch.arange(masked_logits.shape[0]),self._preds]

                C = masked_scores
                Cs.append(C.unsqueeze(1))
        Cs = torch.cat(Cs,dim=1)
        alpha = torch.softmax(Cs,dim=1).unsqueeze(-1).unsqueeze(-1)
    
        cams = torch.relu(torch.sum(torch.relu(alpha*self._fmaps),dim=1,keepdim=True))
        cams = self._process_cam(cams,threshold=threshold)
        return cams

    def __str__(self,):
        return 'FIMF ScoreCAM (Li et al, 2022)'