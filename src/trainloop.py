import torch
from tqdm.auto import tqdm
from torchmetrics.functional.segmentation import dice_score
from sklearn.metrics import accuracy_score

from src.config import CONFIG

def train_step(model,criterion,optimizer,dataloader,device):
    model.to(device)
    model.train()
    losses, accs, dscs = 0,0,0
    for batch,(X,M,y) in enumerate(dataloader):
        X,M,y = X.to(device),M.to(device),y.to(device)
        y_logit = model(X)
        y_mask = model.get_cam(threshold=CONFIG['threshold'])
        y = y.to(torch.float32)
        y_pred = torch.softmax(y_logit,1)
        
        loss = criterion(y_logit,y)
        losses += loss
        accs += accuracy_score(y.argmax(1).cpu(),y_pred.argmax(1).cpu().detach())
        
        dscs += dice_score(y_mask.long().cpu(),M.long().cpu(),num_classes=2,input_format='index',average='macro',include_background=False).mean().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    losses /= len(dataloader)
    losses = losses.cpu().detach().item()
    accs /= len(dataloader)
    dscs /= len(dataloader)
    return losses,accs,dscs

def val_step(model,criterion,dataloader,device,return_data=False):
    model.to(device)
    model.eval()
    losses, accs, dscs = 0,0,0
    if return_data:
        Xs = []
        ys = []
        y_preds = []
        Ms = []
        cams = []
    with torch.inference_mode():
        for batch,(X,M,y) in enumerate(dataloader):
            X,M,y = X.to(device),M.to(device),y.to(device)
            y_logit = model(X)
            y_mask = model.get_cam(threshold=CONFIG['threshold'])
            y = y.to(torch.float32)
            y_pred = torch.softmax(y_logit,1)

            losses += criterion(y_logit,y).cpu().detach().item()
            accs += accuracy_score(y.argmax(1).cpu(),y_pred.argmax(1).cpu().detach())
            dscs += dice_score(y_mask.long().cpu(),M.long().cpu(),num_classes=2,input_format='index',average='macro',include_background=False).mean().item()

            if return_data:
                Xs.extend(X.cpu())
                ys.extend(y.argmax(1).cpu())
                y_preds.extend(y_pred.argmax(1).cpu())
                Ms.extend(M.cpu())
                cams.extend(y_mask.cpu())
        losses /= len(dataloader)
        accs /= len(dataloader)
        dscs /= len(dataloader)
    if return_data:
        Xs = torch.cat(Xs,dim=0)
        ys = torch.stack(ys)
        y_preds = torch.stack(y_preds)
        Ms = torch.cat(Ms,dim=0).to(torch.long)
        cams = torch.cat(cams,dim=0)
        return (losses,accs,dscs),(Xs,ys,y_preds,Ms,cams)
    else:
        return losses,accs,dscs

def train_loop(model,epochs,criterion,optimizer,device,train_dataloader,val_dataloader,scheduler=None):
    '''
    Function to train CAM model
    Args:
        model (torch.nn.Module): The CAM model
            Requirement of the model:
                - Must split into images, masks, and labels
                - Must have .cams attribute that returns the cam of each item in the batch
                - Must be onehot encoded multiclass problem
        epochs (int): The number of epochs to train the model for
        criterion (torch.nn.Module): Loss function for the classification task
        optimizer (torch.optim): Optimizer function
        device (str): Device to compute calculations on
        train_dataloader (DataLoader): PyTorch DataLoader object containing the train data
        test_dataloader (DataLoader): PyTorch DataLoader object containing the test data
        scheduler (torch.optim.lr_scheduler): Scheduler object to alter learning rate mid-training
    Returns:
        A dictionary containing the loss, accuracy, and dice similarity coefficients at each epoch for both train and validation
    '''
    metrics = {'train_loss':[],
               'train_acc':[],
               'train_dsc':[],
               'val_loss':[],
               'val_acc':[],
               'val_dsc':[]}

    for epoch in tqdm(range(epochs)):
        print(f'Epoch {epoch}:')
        train_loss,train_acc,train_dsc = train_step(model,criterion,optimizer,train_dataloader,device)
        val_loss,val_acc,val_dsc = val_step(model,criterion,val_dataloader,device,return_data=False)

        if scheduler:
            scheduler.step(val_loss)
            
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['train_dsc'].append(train_dsc)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        metrics['val_dsc'].append(val_dsc)
        print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f} | Train DSC: {train_dsc:.2f} | Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f} | Val DSC: {val_dsc:.2f}')
    return metrics