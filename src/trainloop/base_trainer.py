class BaseTrainer():
    def __init__(self,model,criterion,optimizer,device,scheduler=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

    def _train_step(self,dataloader)  -> tuple[float,float,float]:
        raise NotImplementedError('Method not yet implemented!')

    def _val_step(self,dataloader) -> tuple[float,float,float]:
        raise NotImplementedError('Method not yet implemented!')

    def get_fit_time(self):
        '''
        Gets the latest fit time. Requires to run .fit() method first!
        '''
        raise NotImplementedError('Method not yet implemented!')