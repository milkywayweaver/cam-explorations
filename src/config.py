from src.models.modules.backbones import VGGBackbone,ResNetBackbone,EffNetBackbone
from src.models.modules.classifiers import ConvClassifier,GAPClassifier,MLPClassifier
from src.models.gradcam import GradCAM,GradCAMPlusPlus,LayerCAM
from src.models.cam import ConvCAM,CAM,ScoreCAM,FIMFScoreCAM

from datetime import datetime

NUM_CLASSES = 3

MODEL = GradCAM()
MODEL.set_backbone(VGGBackbone())
MODEL.set_classifier(ConvClassifier(NUM_CLASSES,MODEL.backbone.output_shape))
MODEL.set_projection('duplicate')

CONFIG = {
    'experiment':str(MODEL),
    'run_name':datetime.now(), # String
    'model':MODEL,
    'backbone':MODEL.backbone, # Used in script.py 
    'classifier':MODEL.classifier,
    'seed':42, # Used in script.py
    'augment':['geometric','color',], # Used in augment.py -- List of strings: "geometric", "color", "blur", "erasing"
    'batch_size':32, # Used in script.py
    'ch_project':MODEL.projection, # Used in script.py # String: "mapper", "duplicate"
    'threshold':0.7, # Used in trainloop.py
    'epochs':50, # Used in script.py
}