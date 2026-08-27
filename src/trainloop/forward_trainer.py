import torch
from tqdm.auto import tqdm
from torchmetrics.functional.segmentation import dice_score
from sklearn.metrics import accuracy_score
import time

from src.config import CONFIG
from src.trainloop.base_trainer import BaseTrainer

class ForwardTrainer(BaseTrainer):
    '''
    Trainer class to fit models.
    Inherits from BaseTrainer.
    '''
    def __init__(self,model,criterion,optimizer,device,scheduler=None):
        super().__init__(model=model,criterion=criterion,optimizer=optimizer,device=device,scheduler=scheduler)

    def _train_step(self,dataloader) -> tuple[float,float,float]:
        self.model.to(self.device)
        self.model.train()
        losses, accs, dscs = 0,0,0
        for batch,(X,M,y) in enumerate(dataloader):
            X,M,y = X.to(self.device),M.to(self.device),y.to(self.device).to(torch.float32)
            y_logits = self.model(X)
            y_masks = self.model.get_cam(threshold=CONFIG['threshold'])
            y_probs = torch.softmax(y_logits,1)
            
            loss = self.criterion(y_logits,y)
            losses += loss.item()
            accs += accuracy_score(y.argmax(1).cpu(),y_probs.argmax(1).cpu().detach())
            
            dscs += dice_score(y_masks.long().cpu(),M.long().cpu(),num_classes=2,input_format='index',average='macro',include_background=False).mean().item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        losses /= len(dataloader)
        losses = losses
        accs /= len(dataloader)
        dscs /= len(dataloader)
        return (losses,accs,dscs) # type: ignore

    def _val_step(self,dataloader) -> tuple[float,float,float]:
        self.model.to(self.device)
        self.model.eval()
        losses, accs, dscs = 0,0,0
        with torch.no_grad():
            for batch,(X,M,y) in enumerate(dataloader):
                X,M,y = X.to(self.device),M.to(self.device),y.to(self.device).to(torch.float32)
                y_logits = self.model(X)
                y_masks = self.model.get_cam(threshold=CONFIG['threshold'])
                y_probs = torch.softmax(y_logits,1)

                losses += self.criterion(y_logits,y).item()
                accs += accuracy_score(y.argmax(1).cpu(),y_probs.argmax(1).cpu().detach())
                dscs += dice_score(y_masks.long().cpu(),M.long().cpu(),num_classes=2,input_format='index',average='macro',include_background=False).mean().item()

            losses /= len(dataloader)
            accs /= len(dataloader)
            dscs /= len(dataloader)
        return (losses,accs,dscs) # type: ignore

    def fit(self,train_loader:torch.utils.data.DataLoader,val_loader:torch.utils.data.DataLoader,epochs:int=20) -> dict:
        '''
        Fits the model to train data.
        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader containing the training data.
            val_loader (torch.utils.data.DataLoader): DataLoader containing the validation data.
            epochs (int, default=20): Number of epochs the model will be trained for.
        Returns:
            metrics (dict[list,list,list,list,list,list]): Training history consisting of the loss, accuracy, and DSC, for both training and validation steps.
        '''
        metrics = {'train_loss':[],
                   'train_acc':[],
                   'train_dsc':[],
                   'val_loss':[],
                   'val_acc':[],
                   'val_dsc':[]}
        
        t0 = time.time()
        for epoch in tqdm(range(epochs)):
            print(f'Epoch {epoch}:')
            train_loss,train_acc,train_dsc = self._train_step(train_loader)
            val_loss,val_acc,val_dsc = self._val_step(val_loader)
    
            if self.scheduler:
                self.scheduler.step(val_loss)
                
            metrics['train_loss'].append(train_loss)
            metrics['train_acc'].append(train_acc)
            metrics['train_dsc'].append(train_dsc)
            metrics['val_loss'].append(val_loss)
            metrics['val_acc'].append(val_acc)
            metrics['val_dsc'].append(val_dsc)
            print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f} | Train DSC: {train_dsc:.2f} | Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f} | Val DSC: {val_dsc:.2f}')
        t1 = time.time()
        self.fit_time = t1-t0
        return metrics