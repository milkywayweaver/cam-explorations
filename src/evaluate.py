import numpy as np
import torch
from torch import nn
from torchmetrics.functional.segmentation import dice_score
from torchmetrics.functional.detection.iou import intersection_over_union
from torchvision.ops import masks_to_boxes
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import patches
import seaborn as sns

from src.config import CONFIG
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_preds(model,dataloader) -> dict:
    model.to(device)
    model.eval()
    data = {
        'X':[],
        'M':[],
        'y':[],
        'y_pred':[],
        'cam':[],
        'mask':[]
    }
    with torch.no_grad():
        for batch,(X,M,y) in enumerate(dataloader):
            X,M,y = X.to(device),M.to(device),y.argmax(1).to(device).to(torch.float32)
            y_logits = model(X)
            y_preds = torch.softmax(y_logits,1).argmax(1)
            y_cams = model.get_cam(threshold='raw')
            y_masks = model.get_cam(threshold=CONFIG['threshold'])

            data['X'].extend(X)
            data['M'].extend(M)
            data['y'].extend(y)
            data['y_pred'].extend(y_preds)
            data['cam'].extend(y_cams)
            data['mask'].extend(y_masks)
        data['X'] = torch.cat(data['X'],dim=0).cpu() # type: ignore
        data['M'] = torch.cat(data['M'],dim=0).cpu().to(torch.long)  # type: ignore
        data['y'] = torch.stack(data['y']).cpu()  # type: ignore
        data['y_pred'] = torch.stack(data['y_pred']).cpu()  # type: ignore
        data['cam'] = torch.cat(data['cam'],dim=0).cpu()  # type: ignore
        data['mask'] = torch.cat(data['mask'],dim=0).cpu().to(torch.long)  # type: ignore
    return data 

def evaluate(model,dataloader):
    data = make_preds(model,dataloader)
    # Accuracy
    acc = accuracy_score(data['y'],data['y_pred'])
    # Dice Similarity Coef.
    dsc = dice_score(data['mask'],data['M'],num_classes=2,include_background=False,average='macro',input_format='index').median()
    # IoU
    M_bbox = masks_to_boxes(data['M'])
    mask_bbox = masks_to_boxes(data['mask'])
    iou = intersection_over_union(mask_bbox,M_bbox,aggregate=True)

    data['M_bbox'] = M_bbox
    data['mask_bbox'] = mask_bbox

    results = {
        'metrics':(acc,dsc,iou),
        'data':data
    }
    return results

def plot_confmat(ys,y_preds,classes=None,title=None):
    confmat = confusion_matrix(ys,y_preds)
    classes = classes if classes else [i for i in range(len(np.unique(ys)))]
    sns.heatmap(confmat,annot=True,fmt='d',cmap='viridis',xticklabels=classes,yticklabels=classes)  # type: ignore
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    if title:
        plt.title(title,weight='bold')

def plot_sample(data,index=None,classes=None):
    '''
    Plots sample image, its ground truth (GT) mask, predicted mask, and raw CAM.
    Args:
        data (dict): Data dictionary obtained from evaluate() function.
        index (int, default=None): Index of the selected sample. If none, randomly pick a sample form the provided data.
        classes (list, default=None): List of class labels. If none, numeric-encoded class labels will be used.
    Returns:
        None
    '''
    if index is None:
        index = np.random.randint(data['X'].shape[0])

    X = data['X'][index]
    M = data['M'][index]
    y = data['y'][index].to(int)
    y_pred = data['y_pred'][index].to(int)
    cam = data['cam'][index]
    mask = data['mask'][index]
    M_bbox = data['M_bbox'][index]
    mask_bbox = data['mask_bbox'][index]

    M_patch = patches.Rectangle((M_bbox[0],M_bbox[1]),M_bbox[2]-M_bbox[0],M_bbox[3]-M_bbox[1],fill=False,ec='lime',fc=None)
    mask_patch = patches.Rectangle((mask_bbox[0],mask_bbox[1]),mask_bbox[2]-mask_bbox[0],mask_bbox[3]-mask_bbox[1],fill=False,ec='magenta',fc=None)

    plt.subplot(1,4,1)
    ax = plt.gca()
    ax.imshow(X)
    ax.add_patch(M_patch)
    ax.add_patch(mask_patch)
    plt.text(M_bbox[0],M_bbox[1],s='GT',color='lime')
    plt.text(mask_bbox[0],mask_bbox[1],s='Pred',color='magenta')
    plt.axis(False)
    plt.title('Image')

    plt.subplot(1,4,2)
    plt.imshow(M)
    plt.axis(False)
    plt.title('GT Mask')

    plt.subplot(1,4,3)
    plt.imshow(mask)
    plt.axis(False)
    plt.title('Predicted Mask')

    plt.subplot(1,4,4)
    plt.imshow(cam)
    plt.axis(False)
    plt.title('Upscaled CAM')

    dsc = dice_score(mask.unsqueeze(0),M.unsqueeze(0),num_classes=2,include_background=False,average='macro',input_format='index')[0]
    iou = intersection_over_union(mask_bbox.unsqueeze(0),M_bbox.unsqueeze(0),aggregate=True)
    true_class = classes[y] if classes else y
    pred_class = classes[y_pred] if classes else y_pred
    plt.suptitle(f'True: {true_class} | Pred: {pred_class}\nIoU: {iou:.4f} | DSC = {dsc:.2f}',y=0.98,weight='bold',fontsize=14)
    plt.tight_layout()

def plot_history(epochs,history,title=None):
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
    if title:
        plt.suptitle(title,weight='bold')
    else:
        plt.suptitle('Training History Curve',weight='bold')

def plot_distribution(data:dict,metric:str='dsc'):
    '''
    Plots the distribution of selected metric.
    Args:
        data (dict): Data dictionary obtained from evaluate() function.
        metric (str, default="dsc"): Selected metric.
            Availabel metrics are: "dsc" and "iou".
    Returns:
        None
    '''
    if metric.lower() == 'dsc':
        items = dice_score(data['mask'],data['M'],num_classes=2,include_background=False,average='macro',input_format='index')
    elif metric.lower() == 'iou':
        items = intersection_over_union(data['mask_bbox'],data['M_bbox'],aggregate=False).diagonal()
    else:
        raise NotImplementedError(f'Metric {metric} not yet implemented!')

    sns.histplot(items)
    plt.xlabel(metric)
    plt.title(f'Median {metric.upper()}: {items.median():.4f}')