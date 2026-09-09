from src.models.modules.backbones import VGGBackbone,ResNetBackbone,EffNetBackbone
from src.models.modules.classifiers import ConvClassifier,GAPClassifier,MLPClassifier
from src.models.gradcam import GradCAM,GradCAMPlusPlus,LayerCAM
from src.models.cam import ConvCAM,CAM,ScoreCAM,FIMFScoreCAM

from datetime import datetime
import torch

### SEED #############################################################
SEED = 42
NUM_CLASSES = 3
######################################################################

torch.random.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
generator = torch.Generator().manual_seed(SEED)

### MODEL ############################################################
MODEL = GradCAM()
MODEL.set_backbone(EffNetBackbone())
MODEL.set_classifier(MLPClassifier(NUM_CLASSES,MODEL.backbone.output_shape))
MODEL.set_projection('duplicate')
######################################################################

### OTHERS ###########################################################
CONFIG = {
    'experiment':str(MODEL),
    'run_name':datetime.now(), # String
    'model':MODEL,
    'backbone':MODEL.backbone, # Used in script.py 
    'classifier':MODEL.classifier,
    'seed':SEED, # Used in script.py
    'augment':['geometric','color',], # Used in augment.py -- List of strings: "geometric", "color", "blur", "erasing"
    'batch_size':32, # Used in script.py
    'ch_project':MODEL.projection, # Used in script.py # String: "mapper", "duplicate"
    'threshold':0.7, # Used in trainloop.py
    'epochs':50, # Used in script.py
}
######################################################################