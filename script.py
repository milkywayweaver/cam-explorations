import numpy as np
import matplotlib.pyplot as plt

import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.augment import train_transform,test_transform
from src.briscloader import LoadBRISC
from src.trainloop.forward_trainer import ForwardTrainer
from src.evaluate import evaluate,plot_confmat,plot_history,plot_sample,plot_distribution

import mlflow
from src.config import CONFIG

# SETUP ENVIRONMENT =========================================================================================================
os.makedirs('saves',exist_ok=True)

np.random.seed(CONFIG['seed'])
torch.random.manual_seed(CONFIG['seed'])
torch.cuda.manual_seed(CONFIG['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
generator = torch.Generator().manual_seed(CONFIG['seed'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'Starting run with CONFIG:')
for key,value in CONFIG.items():
    print(f'{key}:', value)
print(f'device: {device}')


mlflow.set_experiment('CAM (Zhang et al, 2018)')

# READ DATA ================================================================================================================
loader = LoadBRISC()
### !!! IMPORTANT !!!
### Tumor only selection has not been implemented in the evaluation module
train_ds,val_ds,test_ds = loader.load(classes='tumor', # type: ignore
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
classes = loader.classes

# MODEL TRAINING
MODEL = CONFIG['cam_method'](len(classes),backbone=CONFIG['backbone'],classifier=CONFIG['classifier'],ch_project=CONFIG['ch_project'])
LR = 1e-3
WD = 1e-4
CRITERION = nn.CrossEntropyLoss()
OPTIMIZER = torch.optim.AdamW(MODEL.parameters(),lr=LR,weight_decay=WD)
SCHEDULER = torch.optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER, mode='min', factor=0.1, patience=10)

loop = ForwardTrainer(MODEL,CRITERION,OPTIMIZER,device,SCHEDULER)
history = loop.fit(train_dl,val_dl,CONFIG['epochs'])
torch.save(MODEL.state_dict(),f'saves/CAM_{CONFIG['run_name']}.pth')
print(f'Training completed in {loop.fit_time:.4f} seconds.')

# MODEL EVALUATION
results = evaluate(MODEL,test_dl)
(acc,dsc,iou) = results['metrics']
data = results['data']
fit_time = loop.fit_time
print(f'----- TEST SET MODEL EVALUATION -----')
print(f'Accuracy: {acc:.4f}')
print(f'IoU     : {iou:.4f}')
print(f'DSC     : {dsc:.4f}')
print(f'Time    : {fit_time:.4f} s')

os.makedirs('figs',exist_ok=True)

hist_fig = plt.figure(figsize=(15,4))
plot_history(CONFIG['epochs'],history)
plt.savefig('figs/training_curve.png')

confmat_fig = plt.figure(figsize=(7,6))
plot_confmat(data['y'],data['y_pred'],classes=loader.classes)
plt.savefig('figs/confmat.png')

samples = []
for i in range(9):
    sample = plt.figure(figsize=(12,4))
    plot_sample(data,classes=classes)
    plt.savefig(f'figs/sample_{i+1}.png')
    samples.append(sample)

plt.figure(figsize=(10,3))
plt.subplot(1,2,1)
plot_distribution(data,'iou')
plt.subplot(1,2,2)
plot_distribution(data,'dsc')

# # LOGGING
# with mlflow.start_run(run_name=CONFIG['run_name']):
#     mlflow.log_params({
#         'seed':CONFIG['seed'],
#         'backbone_model_type':CONFIG['backbone_model_type'],
#         'augment':CONFIG['augment'],
#         'batchsize':CONFIG['batch_size'],
#         'ch_project':CONFIG['ch_project'],
#         'threshold':CONFIG['threshold'],
#         'epochs':CONFIG['epochs'],
#         'lr':LR,
#         'wd':WD
#     })
#     mlflow.log_metrics({
#         'accuracy':acc,
#         'detection_iou':iou,
#         'segmentation_dsc':dsc,
#         'fit_time':fit_time
#     })
#     mlflow.log_figure(hist_fig,'training_history.png')
#     mlflow.log_figure(confmat_fig,'confmat.png')

# plt.close(hist_fig)
# plt.close(confmat_fig)


