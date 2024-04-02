import sys
sys.path.append('../../')

import os
import time
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import hydra
from collections import deque
import matplotlib.pyplot as plt
from src.losses.loss import Loss

class Trainer(object):
    '''
    Trainer base class: object used to run training and eval routines
    '''
    def __init__(self, model, logger, config):
        '''
        '''
        self.logger = logger
        self.curEpoch = 0
        self.print_freq = config['print_freq']

        # choosing the device
        if(torch.cuda.is_available()):
            self.logger.info('CUDA is available, using GPU for computations.')
        else:
            self.logger.info('CUDA is unavailable, using CPU for computations.')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # metrics and loss???

        # model
        self.model = model.to(self.device)

        # init training
        self.trainableParams = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = hydra.utils.instantiate(config['optimizer'], params=self.trainableParams)
        self.lrScheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config['lrScheduler']['factor'],
            patience=config['lrScheduler']['patience'],
            verbose=True,
            min_lr=config['lrScheduler']['min_lr']
        )
        
        # clip norm
        if config['clip_norm']:
            self.clip_norm = config['clip_norm']
            self.logger.info('Gradient clipping by {}.'.format(self.clip_norm))
        else:
            self.clip_norm = 0
        
        #checkpoint handling
        checkpointPath = config['checkpointPath']
        if checkpointPath is not None:
            self.logger.info(f'Continue training from checkpoint: {checkpointPath}.')
            checkpoint = torch.load(checkpointPath, map_location='cpu')
            try:
                self.model.load_state_dict(checkpoint['model'])
            except Exception as e:
                self.logger.info(e)
                self.logger.info('WARNING! load_state_dict_failed, expected load_state_dict in the checkpoint init.')
            self.model = self.model.to(self.device)
        else:
            self.logger.info('Starting new training run.') 
        
        #checkpoint configuration
        self.checkpointQueue = deque(maxlen=config.nCheckpoints)
        self.newCheckpointsPath = config['newCheckpointsPath']
        self.logger.info(self.newCheckpointsPath)
    
    def train(self, dataLoader):
        self.logger.info('Set train mode...')
        self.model.train()
        numSteps = len(dataLoader)
        total_loss = 0.0

        start_time = time.time()
        for step, (mix, ref) in enumerate(dataLoader):
            ref = ref.permute(1, 0, 2)
            mix = mix.to(self.device)
            ref = [ref[i].to(self.device) for i in range(len(ref))]
            self.optimizer.zero_grad()
            out = self.model(mix)

            l = Loss(out, ref)
            epoch_loss = l
            total_loss += epoch_loss.item()
            epoch_loss.backward()

            if self.clip_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

            self.optimizer.step()
            
            if step % self.print_freq == 0:
                message = '<epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}>.'.format(
                    self.curEpoch,
                    step, 
                    self.optimizer.param_groups[0]['lr'],
                    total_loss / (step + 1)
                )
                self.logger.info(message)
        end_time = time.time()
        total_loss = total_loss / numSteps
        message = 'Finished *** <epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, Total time:{:.3f} min>.'.format(
            self.curEpoch,
            numSteps,
            self.optimizer.param_groups[0]['lr'],
            total_loss,
            (end_time - start_time) / 60
        )
        self.logger.info(message)
        return total_loss
    
    def eval(self, dataLoader):
        self.logger.info('Set eval mode...')
        self.model.eval()
        numSteps = len(dataLoader)
        total_loss = 0.0
        start_time = time.time()
        with torch.no_grad():
            for step, (mix, ref) in enumerate(dataLoader):
                ref = ref.permute(1, 0, 2)

                mix = mix.to(self.device)
                ref = [ref[i].to(self.device) for i in range(len(ref))]
                self.optimizer.zero_grad()

                out = self.model(mix) 
                
                l = Loss(out, ref)
                epoch_loss = l
                total_loss += epoch_loss.item()

                if step % self.print_freq == 0:
                    message = '<epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}>.'.format(
                        self.curEpoch,
                        step,
                        self.optimizer.param_groups[0]['lr'],
                        total_loss / (step + 1)
                    )
                    self.logger.info(message)
        end_time = time.time()
        total_loss = total_loss / numSteps
        message = 'Finished *** <epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, Total time:{:.3f} min>.'.format(
            self.curEpoch,
            step,
            self.optimizer.param_groups[0]['lr'],
            total_loss,
            (end_time - start_time) / 60
        )
        self.logger.info(message)
        return total_loss

    def run(self, trainLoader, testLoader, nEpochs=50, early_stop=10):
        self.save_checkpoint(best=False)
        best_loss = self.eval(testLoader)
        
        train_losses = []
        eval_losses = []
        no_improve_cnt = 0

        while self.curEpoch < nEpochs:
            self.logger.info('Initiating epoch '+ str(self.curEpoch) + '.')
            self.curEpoch += 1

            train_loss = self.train(trainLoader)
            eval_loss = self.eval(testLoader)

            train_losses.append(train_loss)
            eval_losses.append(eval_loss)

            self.lrScheduler.step(eval_loss)

            if eval_loss >= best_loss:
                no_improve_cnt += 1
                self.logger.info('No improvement, Best Loss: {:.4f}.'.format(best_loss))
            else:
                best_loss = eval_loss
                no_improve_cnt = 0
                self.save_checkpoint(best=True)
                self.logger.info('Epoch: {:d}, Now Best Loss Change: {:.4f}.'.format(self.curEpoch, best_loss))
            
            if no_improve_cnt == early_stop:
                self.logger.info('Stop training cause no impr for {:d} epochs'.format(no_improve_cnt))
                break
        
        self.save_checkpoint(best=False)
        self.logger.info('Training for {:d}/{:d} epoches done!'.format(self.curEpoch, nEpochs))

        plt.title('Loss of train and test')
        x = [i for i in range(self.curEpoch)]
        plt.plot(x, train_losses, 'b-', label=u'train losses', linewidth=0.8)
        plt.plot(x, eval_losses, 'c-', label=u'eval losses', linewidth=0.8)
        plt.legend()
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig('loss.png')
    
    def processCheckpoint(self, path):
        if len(self.checkpointQueue) == self.checkpointQueue.maxlen:
            removedCheckpoint = self.checkpointQueue[0]
            os.remove(removedCheckpoint)
        self.checkpointQueue.append(path)
        
    def save_checkpoint(self, best=False):
        '''
        Saves the checkpoint.
        '''
        cpt = {
            'epoch': self.curEpoch,
            'optimizer': self.optimizer.state_dict(),
            'model': self.model.state_dict()
        }
        pathToSave = os.path.join(self.newCheckpointsPath, '{0}_{1}.pt'.format(str(self.curEpoch), 'best' if best else 'last'))
        torch.save(cpt, pathToSave)
        self.processCheckpoint(pathToSave)