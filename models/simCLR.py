import math
from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from losses import InfoNCE

class SimCLR(pl.LightningModule):
    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs):
        super(SimCLR, self).__init__()
        self.save_hyperparameters()
        self.convnet = torchvision.models.resnet18(num_classes=4*hidden_dim)
        self.convnet.fc = nn.Sequential(
            self.convnet.fc,
            nn.ReLU(inplace=True),
            nn.Linear(4*hidden_dim, hidden_dim)
        )
        self.loss = InfoNCE(self.convnet, temperature)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs,
                                                            eta_min=self.hparams.lr / 50)
        return [optimizer], [lr_scheduler]
    
    def training_step(self, batch, batch_idx):
        return self.loss(batch)
    
    def validation_step(self, batch, batch_idx):
        self.loss(batch)