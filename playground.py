from src.models.modules.backbones import VGGBackbone,ResNetBackbone,EffNetBackbone
from src.models.modules.classifiers import ConvClassifier,GAPClassifier,MLPClassifier
from src.models.gradcam import GradCAM,GradCAMPlusPlus
from src.models.cam import ConvCAM,CAM,ScoreCAM,FIMFScoreCAM
import torch


if __name__ == '__main__':
    cam = FIMFScoreCAM(4,backbone=VGGBackbone,classifier=MLPClassifier,ch_project='mapper')
    # print(cam.classifier)

    X = torch.rand(
        (8,1,224,224))
    logits = cam(X)
    # print(logits.shape)

    mask = cam.get_cam()
    # print(mask.shape)
