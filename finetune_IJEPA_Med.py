import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    ModelSummary,
)
from model import IJEPA_base
from pretrain_IJEPA import IJEPA
import os
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter




class IJEPADataset(Dataset):
    def __init__(self,
                 dataset_path,
                 stage='train',
                 transform=None,
                 ):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.225, 0.225, 0.225])
        ])
        self.data = datasets.ImageFolder(os.path.join(dataset_path, stage), transform=self.transform)
        self.num_classes = len(self.data.classes)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img, label = self.data[index]
        
        one_hot_label = np.zeros(self.num_classes)
        one_hot_label[label] = 1
        
        return img, one_hot_label
    
'''
Placeholder for datamodule in pytorch lightning
'''
class D2VDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_path,
                 batch_size=16,
                 num_workers=4,
                 pin_memory=True,
                 shuffle=True
                 ):
        super().__init__()
        
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle

        
    def setup(self, stage=None):
        self.train_dataset = IJEPADataset(dataset_path=self.dataset_path, stage='train')
        self.val_dataset = IJEPADataset(dataset_path=self.dataset_path, stage='val')
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )


'''
Finetune IJEPA
'''
class IJEPA_FT(pl.LightningModule):
    #take pretrained model path, number of classes, learning rate, weight decay, and drop path as input
    def __init__(self, pretrained_model_path, num_classes, lr=1e-3, weight_decay=0, drop_path=0.1):

        super().__init__()
        self.save_hyperparameters()

        #set parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.drop_path = drop_path

        #define model layers
        self.pretrained_model = IJEPA.load_from_checkpoint(pretrained_model_path)
        self.pretrained_model.model.mode = "test"
        self.pretrained_model.model.layer_dropout = self.drop_path
        self.average_pool = nn.AvgPool1d((self.pretrained_model.embed_dim), stride=1)
        #mlp head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.pretrained_model.num_tokens),
            nn.Linear(self.pretrained_model.num_tokens, num_classes),
        )

        #define loss
        self.criterion = nn.CrossEntropyLoss()

        self.writer = SummaryWriter()


    def forward(self, x):
        x = self.pretrained_model.model(x)
        x = self.average_pool(x) #conduct average pool like in paper
        x = x.squeeze(-1)
        x = self.mlp_head(x) #pass through mlp head
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch   
        y_hat = self(x)
        loss = self.criterion(y_hat, y) #calculate loss
        accuracy = (y_hat.argmax(dim=1) == y.argmax(dim=1)).float().mean() #calculate accuracy
        self.log('train_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.writer.add_scalar("Train/accuracy", accuracy, self.global_step)
        self.writer.add_scalar("Train/loss", loss, self.global_step)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch  
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        accuracy = (y_hat.argmax(dim=1) == y.argmax(dim=1)).float().mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.writer.add_scalar("Val/accuracy", accuracy, self.global_step)
        self.writer.add_scalar("Val/loss", loss, self.global_step)
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx):
        return self(batch[1])
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer


if __name__ == '__main__':

    dataset = D2VDataModule(dataset_path='/projects/earlydetection/Pneumonia')

    model = IJEPA_FT(pretrained_model_path='pretrain-checkpoints-Med/last.ckpt', num_classes=2)

    logger = TensorBoardLogger("lightning_logs", name="IJEPA_finetune_logs_Med")

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)

    checkpoint_callback = ModelCheckpoint(
    monitor='val_accuracy',  
    dirpath='finetune-checkpoints-Med/',  
    filename='{epoch:02d}-{val_accuracy:.2f}',  
    save_top_k=1,  
    mode='max',
    save_last=True  
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        precision="bf16-mixed",
        max_epochs=100,
        callbacks=[lr_monitor, model_summary, checkpoint_callback],
        gradient_clip_val=.1,
        logger=logger, 
    )

    trainer.fit(model, dataset)

