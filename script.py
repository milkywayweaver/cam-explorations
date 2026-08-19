import numpy as np
import matplotlib.pyplot as plt

import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from augment import train_transform,test_transform
from briscloader import LoadBRISC
from cam import CAM
from trainloop import train_loop
from evaluate import evaluate,plot_confmat,plot_history,plot_sample

import mlflow
from config import CONFIG

# SETUP ENVIRONMENT =========================================================================================================
os.makedirs('saves',exist_ok=True)

np.random.seed(CONFIG['seed'])
torch.random.manual_seed(CONFIG['seed'])
torch.cuda.manual_seed(CONFIG['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
generator = torch.Generator().manual_seed(CONFIG['seed'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

mlflow.set_experiment('CAM (Zhang et al, 2018)')

# READ DATA ================================================================================================================
loader = LoadBRISC()
train_ds,val_ds,test_ds = loader.load(classes='tumor',
                                      planes='all',
                                      encoding_type='onehot',
                                      split_val=True,
                                      train_transform=train_transform,
                                      test_transform=test_transform,
                                      generator=generator)
train_dl = DataLoader(train_ds,
                      batch_size=CONFIG['batch_size'],
                      shuffle=True,
                      drop_last=True)
val_dl = DataLoader(val_ds,
                    batch_size=CONFIG['batch_size'],
                    shuffle=False,
                    drop_last=True)
test_dl = DataLoader(test_ds,
                     batch_size=CONFIG['batch_size'],
                     shuffle=False,
                     drop_last=False)

# MODEL TRAINING
MODEL= CAM(len(loader.classes),backbone=CONFIG['backbone'],ch_project=CONFIG['ch_project'])
EPOCHS = 20
LR = 1e-3
WD = 1e-4
CRITERION = nn.CrossEntropyLoss()
OPTIMIZER = torch.optim.AdamW(MODEL.parameters(),lr=LR,weight_decay=WD)
SCHEDULER = torch.optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER, mode='min', factor=0.1, patience=10)
t0 = time.time()
history = train_loop(MODEL,EPOCHS,CRITERION,OPTIMIZER,device,train_dl,val_dl,SCHEDULER)
t1 = time.time()
torch.save(MODEL.state_dict(),f'saves/CAM_{CONFIG['run_name']}.pth')

# MODEL EVALUATION
results = evaluate(MODEL,test_dl)
(acc,dsc,iou) = results['metrics']
(Xs,ys,y_preds,Ms,cams) = results['data']
M_bbox,cam_bbox = results['bbox']
fit_time = t1-t0
print(f'----- MODEL EVALUATION -----')
print(f'Accuracy: {acc:.4f}')
print(f'IoU     : {iou:.4f}')
print(f'DSC     : {dsc:.4f}')
print(f'Time    : {fit_time:.4f} s')

hist_fig = plt.figure(figsize=(15,4))
plot_history(EPOCHS,history)

confmat_fig = plt.figure(figsize=(7,6))
plot_confmat(ys,y_preds,classes=loader.classes)

# LOGGING
with mlflow.start_run(run_name=CONFIG['run_name']):
    mlflow.log_params({
        'seed':CONFIG['seed'],
        'backbone':CONFIG['backbone'],
        'augment':CONFIG['augment'],
        'batchsize':CONFIG['batch_size'],
        'relu':CONFIG['relu'],
        'ch_project':CONFIG['ch_project'],
        'epochs':EPOCHS,
        'lr':LR,
        'wd':WD
    })
    mlflow.log_metrics({
        'accuracy':acc,
        'detection_iou':iou,
        'segmentation_dsc':dsc,
        'fit_time':fit_time
    })
    mlflow.log_figure(hist_fig,'training_history.png')
    mlflow.log_figure(confmat_fig,'confmat.png')

plt.close(hist_fig)
plt.close(confmat_fig)


