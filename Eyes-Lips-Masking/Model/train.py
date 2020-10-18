import torch
from torch.optim import  Adam
import torch.nn as nn
from tqdm import tqdm 

class Trainer():

    def __init__(self, epochs, data, network, device = 'cuda'):

        if device == 'cuda':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print('Device Name:', torch.cuda.get_device_name(0))
            else:
                print('CUDA Device not available...')
                self.device = torch.device('cpu')
        
        self.model = network
        self.optimizer =  Adam(self.model.parameters(), lr = 0.0002, betas = (0.5, 0.99))
        self.epochs = epochs
        self.dataloader = data 
        self.criterion = nn.CrossEntropyLoss()
        
    def train(self):
        
        self.model = self.model.to(self.device)
        
        for epoch in range(self.epochs):

            epoch_loss = 0
            
            for batch in tqdm(self.dataloader):
                
                img = (batch['img'] / 255).to(self.device)
                masks = batch['masks'].to(self.device)

                preds = self.model(img)
                loss = self.criterion(preds, masks)
                epoch_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print('Epoch: {} Loss Value: {}'.format(epoch + 1, epoch_loss / self.epochs))











        
