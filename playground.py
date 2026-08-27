from src.models.gradcam import GradCAM
from src.models.cam import CAM
import torch

if __name__ == '__main__':
    gradcam = GradCAM(4,backbone_model_type='effnet',ch_project='mapper')
    cam = CAM(4,backbone_model_type='effnet',ch_project='mapper')

    X = torch.rand((8,1,224,224))

    with torch.enable_grad():
        logits = gradcam(X)
        prob = torch.softmax(logits,1)
        y_pred = prob.argmax(1)
        score = logits[torch.arange(8),y_pred]

        grad_tensor = torch.ones_like(score)
        score.backward(gradient=grad_tensor)

        gradcam_mask = gradcam.get_cam()
        print(gradcam_mask.shape)
