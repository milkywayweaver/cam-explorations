import numpy as np
import matplotlib.pyplot as plt

import os
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.augment import train_transform,test_transform
from src.briscloader import LoadBRISC
from src.trainloop.forward_trainer import ForwardTrainer
from src.evaluate import evaluate,plot_confmat,plot_history,plot_sample,plot_distribution
from src.models.initializer import get_model_class

import mlflow
import yaml

# IMPORT CONFIG =============================================================================================================
with open('./config.yaml') as f:
    CONFIG = yaml.safe_load(f)

# SETUP ENVIRONMENT =========================================================================================================
os.makedirs('saves',exist_ok=True)

np.random.seed(CONFIG['seed'])
torch.random.manual_seed(CONFIG['seed'])
torch.cuda.manual_seed(CONFIG['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
generator = torch.Generator().manual_seed(CONFIG['seed'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'Starting run with CONFIG:')
for key,value in CONFIG.items():
    print(f'{key}: {str(value)}')
print(f'device: {device}')

# READ DATA ================================================================================================================
loader = LoadBRISC()
### !!! IMPORTANT !!!
### Tumor only selection has not been implemented in the evaluation module
train_ds,val_ds,test_ds = loader.load(classes=CONFIG['data_contents'],
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

# INITIALIZE MODEL
model_cls, backbone_cls, classifier_cls = get_model_class(
    CONFIG['model']['class'],
    CONFIG['backbone']['class'],
    CONFIG['classifier']['class'],
    CONFIG['projection']
)

MODEL = model_cls(**CONFIG['model']['kwargs'])
MODEL.set_backbone(backbone_cls(**CONFIG['backbone']['kwargs']))
MODEL.set_classifier(classifier_cls(len(classes),MODEL.backbone.output_shape,**CONFIG['classifier']['kwargs']))
MODEL.set_projection('duplicate')

CRITERION = nn.CrossEntropyLoss()
OPTIMIZER = torch.optim.AdamW(MODEL.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['wd'])
SCHEDULER = torch.optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER, mode='min', factor=0.1, patience=10)

# MODEL TRAINING
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

dist_fig = plt.figure(figsize=(10,3))
plt.subplot(1,2,1)
plot_distribution(data,'iou')
plt.subplot(1,2,2)
plot_distribution(data,'dsc')
plt.tight_layout()
plt.savefig('figs/distribution.png')

# LOGGING
CONFIG['experiment_name'] = str(MODEL) if CONFIG['experiment_name'] == 'default' else CONFIG['experiment_name']
CONFIG['run_name'] = datetime.now() if CONFIG['run_name'] == 'now' else CONFIG['run_name']
mlflow.set_experiment(CONFIG['experiment_name'])
with mlflow.start_run(run_name=CONFIG['run_name']):
    mlflow.log_params({
        'seed':CONFIG['seed'],
        'model':CONFIG['model'],
        'backbone':CONFIG['backbone'],
        'classifier':CONFIG['classifier'],
        'projection':CONFIG['projection'],
        'augment':CONFIG['augment'],
        'batchsize':CONFIG['batch_size'],
        'epochs':CONFIG['epochs'],
        'lr':CONFIG['lr'],
        'wd':CONFIG['wd'],
        'threshold':CONFIG['threshold'],
    })
    mlflow.log_metrics({
        'accuracy':acc,
        'detection_iou':iou,
        'segmentation_dsc':dsc,
        'fit_time':fit_time
    })
    mlflow.log_figure(hist_fig,'training_history.png')
    mlflow.log_figure(confmat_fig,'confmat.png')
    mlflow.log_figure(dist_fig,'distribution.png')
    for i,sample in enumerate(samples):
        mlflow.log_figure(sample,f'sample_{i}.png')
        plt.close(sample)

plt.close(hist_fig)
plt.close(confmat_fig)
plt.close(dist_fig)


