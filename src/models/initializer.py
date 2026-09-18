from src.models.modules.backbones import VGGBackbone,ResNetBackbone,EffNetBackbone
from src.models.modules.classifiers import ConvClassifier,GAPClassifier,MLPClassifier
from src.models.gradcam import GradCAM,GradCAMPlusPlus,LayerCAM
from src.models.cam import ConvCAM,CAM,ScoreCAM,FIMFScoreCAM

def get_model_class(model,backbone,classifier,projection):
    if model not in ['CAM','ConvCAM','GradCAM','GradCAM++','ScoreCAM','FIMF ScoreCAM','LayerCAM']:
        raise ValueError('Unknown model! Please use "CAM", "ConvCAM", "GradCAM", "GradCAM++", "ScoreCAM", "FIMF ScoreCAM", or "LayerCAM".')
    if backbone not in ['VGG','ResNet','EffNet']:
        raise ValueError('Unknown backbone! Please use "VGG", "ResNet", or "EffNet".')
    if classifier not in ['MLP','GAP','Conv']:
        raise ValueError('Unknown classifier! Please use "MLP", "GAP", or "Conv".')
    if projection not in ['duplicate','mapper']:
        raise ValueError('Unknown projection! Please use "duplicate" or "mapper".')

    match model:
        case 'CAM':
            MODEL = CAM
        case 'ConvCAM':
            MODEL = ConvCAM
        case 'GradCAM':
            MODEL = GradCAM
        case 'GradCAM++':
            MODEL = GradCAMPlusPlus
        case 'ScoreCAM':
            MODEL = ScoreCAM
        case 'FIMF ScoreCAM':
            MODEL = FIMFScoreCAM
        case 'LayerCAM':
            MODEL = LayerCAM

    match backbone:
        case 'VGG':
            BACKBONE = VGGBackbone
        case 'ResNet':
            BACKBONE = ResNetBackbone
        case 'EffNet':
            BACKBONE = EffNetBackbone

    match classifier:
        case 'MLP':
            CLASSIFIER = MLPClassifier
        case 'GAP':
            CLASSIFIER = GAPClassifier
        case 'Conv':
            CLASSIFIER = ConvClassifier

    return [MODEL,BACKBONE,CLASSIFIER] # type: ignore