import torch
from torch import nn
from tqdm.auto import tqdm
from torchmetrics.functional.segmentation import dice_score
from sklearn.metrics import accuracy_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_step(model,criterion,optimizer,dataloader,device=device):
    model.to(device)
    model.train()
    losses, accs, dscs = 0,0,0
    for batch,(X,M,y) in enumerate(dataloader):
        X,M,y = X.to(device),M.to(device),y.to(device)
        y_logit = model(X)
        y_mask = model.cams
        y = y.to(torch.float32)
        y_pred = torch.softmax(y_logit,1)
        loss = criterion(y_logit,y)
        losses += loss
        accs += accuracy_score(y.argmax(1).cpu(),y_pred.argmax(1).cpu().detach().numpy())
        dscs += dice_score(y_mask,M,num_classes=2).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    losses /= len(dataloader)
    losses = losses.cpu().detach().item()
    accs /= len(dataloader)
    dscs /= len(dataloader)
    return losses,accs,dscs

def val_step(model,criterion,dataloader,device=device):
    model.to(device)
    model.eval()
    losses, accs, dscs = 0,0,0
    with torch.inference_mode():
        for batch,(X,M,y) in enumerate(dataloader):
            X,M,y = X.to(device),M.to(device),y.to(device)
            y_logit = model(X)
            y_mask = model.cams
            y = y.to(torch.float32)
            y_pred = torch.softmax(y_logit,1)

            losses += criterion(y_logit,y).cpu().detach().item()
            accs += accuracy_score(y.argmax(1).cpu(),y_pred.argmax(1).cpu().detach().numpy())
            dscs += dice_score(y_mask,M,num_classes=2).sum().item()
        losses /= len(dataloader)
        accs /= len(dataloader)
        dscs /= len(dataloader)
    return losses,accs,dscs

def train_loop(model,epochs,criterion,optimizer,device,train_dataloader,val_dataloader,scheduler=None):
    metrics = {'train_loss':[],
               'train_acc':[],
               'train_dsc':[],
               'val_loss':[],
               'val_acc':[],
               'val_dsc':[]}

    for epoch in tqdm(range(epochs)):
        print(f'Epoch {epoch}:')
        train_loss,train_acc,train_dsc = train_step(model,criterion,optimizer,train_dataloader,device)
        val_loss,val_acc,val_dsc = val_step(model,criterion,val_dataloader,device)

        if scheduler:
            scheduler.step(val_loss)
            
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['train_dsc'].append(train_dsc)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        metrics['val_dsc'].append(val_dsc)
        print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f} | Train DSC: {train_dsc:.2f}| Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f} | Val DSC: {val_dsc:.2f}')
    return metrics