import numpy as np
import torch
from torch import nn
from torchmetrics.functional.segmentation import dice_score
from torchmetrics.detection.iou import IntersectionOverUnion
from torchvision.ops import masks_to_boxes
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.trainloop import val_step

def evaluate(model,test_dataloader,criterion=nn.CrossEntropyLoss()):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _, preds = val_step(model,criterion,test_dataloader,device=device,return_data=True)
    Xs,ys,y_preds,Ms,cams = preds
    # Accuracy
    acc = accuracy_score(ys,y_preds)
    # Dice Similarity Coef.
    dsc = dice_score(cams,Ms,num_classes=2).median()
    # IoU (n = number of target PER IMAGE)
    cam_bbox = [{
        'boxes': masks_to_boxes(cams[i].unsqueeze(0)), # nhw --> xyxy
        'labels': y_preds[i].unsqueeze(0) # n
    } for i in range(Xs.shape[0])]
    M_bbox = [{
        'boxes': masks_to_boxes(Ms[i].unsqueeze(0)), # nhw --> xyxy
        'labels': ys[i].unsqueeze(0) # n 
    } for i in range(Xs.shape[0])]
    iou = IntersectionOverUnion()(cam_bbox,M_bbox)['iou']

    results = {
        'metrics':(acc,dsc,iou),
        'data':(Xs,ys,y_preds,Ms,cams),
        'bbox':(M_bbox,cam_bbox)
    }
    return results

def plot_confmat(ys,y_preds,classes=None):
    confmat = confusion_matrix(ys,y_preds)
    classes = classes if classes else [i for i in range(len(np.unique(ys)))]
    sns.heatmap(confmat,annot=True,fmt='d',cmap='viridis',xticklabels=classes,yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

def plot_sample(Xs,ys,y_preds,Ms,cams,index=None):
    if index is None:
        index = np.random.randint(Xs.shape[0])
    
    plt.figure(figsize=(10,4))
    plt.subplot(1,3,1)
    plt.imshow(Xs[index].squeeze().cpu())
    plt.axis(False)
    plt.title(f'True: {ys[index]} | Pred: {y_preds[index]}')

    M = (Ms[index].squeeze().cpu() >= 0.5).long()
    plt.subplot(1,3,2)
    plt.imshow(M)
    plt.axis(False)
    plt.title('GT Mask')

    cam = (cams[index].squeeze().cpu() >= 0.5).long()
    plt.subplot(1,3,3)
    plt.imshow(cam)
    plt.axis(False)
    plt.title('CAM')

    plt.suptitle(f'DSC = {dice_score(cam.unsqueeze(0),M.unsqueeze(0),num_classes=2,include_background=False,average='macro',input_format='index').item():.2f}',y=0.9)

def plot_history(epochs,history):
    plt.subplot(1,3,1)
    sns.lineplot(x=np.arange(epochs),y=history['train_loss'],marker='o')
    sns.lineplot(x=np.arange(epochs),y=history['val_loss'],marker='o')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(1,3,2)
    sns.lineplot(x=np.arange(epochs),y=history['train_acc'],marker='o')
    sns.lineplot(x=np.arange(epochs),y=history['val_acc'],marker='o')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.subplot(1,3,3)
    sns.lineplot(x=np.arange(epochs),y=history['train_dsc'],marker='o',label='Train')
    sns.lineplot(x=np.arange(epochs),y=history['val_dsc'],marker='o',label='Val')
    plt.legend(loc='center right',bbox_to_anchor=(1.26,0.5))
    plt.title('DSC')
    plt.xlabel('Epochs')
    plt.ylabel('DSC')
    plt.suptitle('Training History Curve')
    plt.tight_layout()