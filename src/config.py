from src.models.modules.backbones import VGGBackbone,ResNetBackbone,EffNetBackbone
from src.models.modules.classifiers import ConvClassifier,GAPClassifier,MLPClassifier
from src.models.gradcam import GradCAM,GradCAMPlusPlus,LayerCAM
from src.models.cam import ConvCAM,CAM,ScoreCAM,FIMFScoreCAM

CONFIG = {
    'run_name':'VGG - Batch Size 32', # String
    'cam_method':LayerCAM,
    'backbone': ResNetBackbone, # Used in script.py 
    'classifier': MLPClassifier,
    'seed':42, # Used in script.py
    'augment':['geometric','color',], # Used in augment.py -- List of strings: "geometric", "color", "blur", "erasing"
    'batch_size':16, # Used in script.py
    'ch_project':'mapper', # Used in script.py # String: "mapper", "duplicate"
    'threshold':0.7, # Used in trainloop.py
    'epochs':20, # Used in script.py
}