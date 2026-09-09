from src.models.modules.backbones import VGGBackbone,ResNetBackbone,EffNetBackbone
from src.models.modules.classifiers import ConvClassifier,GAPClassifier,MLPClassifier
from src.models.gradcam import GradCAM,GradCAMPlusPlus,LayerCAM
from src.models.cam import ConvCAM,CAM,ScoreCAM,FIMFScoreCAM
import torch


if __name__ == '__main__':
    cam = GradCAM()
    cam.set_backbone(VGGBackbone())
    cam.set_classifier(MLPClassifier(4,cam.backbone.output_shape))
    cam.set_projection('duplicate')

    print(cam)
    print(cam.backbone)
    print(cam.classifier)
    print(cam.projection)

    X = torch.rand((8,1,224,224))
    logits = cam(X)
    masks = cam.get_cam()

    print(f'Logits: {logits.shape}')
    print(f'Masks: {masks.shape}')
