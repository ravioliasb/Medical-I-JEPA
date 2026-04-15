import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    ModelSummary,
)
from pytorch_lightning.loggers import WandbLogger
from model import IJEPA_base
import os
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter



class IJEPADataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

class D2VDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path, batch_size=16, num_workers=4, pin_memory=True, shuffle=True):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.225, 0.225, 0.225])
        ])

    def setup(self, stage=None):
        self.train_dataset = datasets.ImageFolder(os.path.join(self.dataset_path, 'train'), transform=self.transform)

        self.val_dataset = datasets.ImageFolder(os.path.join(self.dataset_path, 'val'), transform=self.transform)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class IJEPA(pl.LightningModule):
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3, 
            embed_dim=64,
            enc_heads=8,
            enc_depth=8,
            decoder_depth=6,
            lr=1e-6,
            weight_decay=0.05,
            target_aspect_ratio = (0.75,1.5),
            target_scale = (0.15, .2),
            context_aspect_ratio = 1,
            context_scale = (0.85,1.0),
            M = 4, #number of different target blocks
            m=0.996, #momentum
            m_start_end = (.996, 1.)

    ):
        super().__init__()
        self.save_hyperparameters()
        
        #define models
        self.model = IJEPA_base(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, 
                                enc_depth = enc_depth, num_heads=enc_heads, pred_depth=decoder_depth, M=M)

        #define hyperparameters
        self.M = M
        self.lr = lr
        self.weight_decay = weight_decay
        self.m = m
        self.target_aspect_ratio = target_aspect_ratio
        self.target_scale = target_scale
        self.context_aspect_ratio = context_aspect_ratio
        self.context_scale = context_scale
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_tokens = (img_size // patch_size) ** 2
        self.m_start_end = m_start_end

        #define loss
        self.criterion = nn.MSELoss()

        self.writer = SummaryWriter()

    
    def forward(self, x, target_aspect_ratio, target_scale, context_aspect_ratio, context_scale):
        return self.model(x, target_aspect_ratio, target_scale, context_aspect_ratio, context_scale)
    
    '''Update momentum for teacher encoder'''
    def update_momentum(self, m):
        student_model = self.model.student_encoder.eval()
        teacher_model = self.model.teacher_encoder.eval()
        with torch.no_grad():
            for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
                teacher_param.data.mul_(other=m).add_(other=student_param.data, alpha=1 - m)


    def training_step(self, batch, batch_idx):
        x = batch[0]
        #generate random target and context aspect ratio and scale
        target_aspect_ratio = np.random.uniform(self.target_aspect_ratio[0], self.target_aspect_ratio[1])
        target_scale = np.random.uniform(self.target_scale[0], self.target_scale[1])
        context_aspect_ratio = self.context_aspect_ratio
        context_scale = np.random.uniform(self.context_scale[0], self.context_scale[1])

        y_student, y_teacher = self(x, target_aspect_ratio, target_scale, context_aspect_ratio, context_scale)
        loss = self.criterion(y_student, y_teacher)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.writer.add_scalar("Train/loss", loss, self.global_step)
                    
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[0]
        target_aspect_ratio = np.random.uniform(self.target_aspect_ratio[0], self.target_aspect_ratio[1])
        target_scale = np.random.uniform(self.target_scale[0], self.target_scale[1])
        context_aspect_ratio = self.context_aspect_ratio
        context_scale = np.random.uniform(self.context_scale[0], self.context_scale[1])

        y_student, y_teacher = self(x, target_aspect_ratio, target_scale, context_aspect_ratio, context_scale)
        loss = self.criterion(y_student, y_teacher)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.writer.add_scalar("Val/loss", loss, self.global_step)

        
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx):
        target_aspect_ratio = np.random.uniform(self.target_aspect_ratio[0], self.target_aspect_ratio[1])
        target_scale = np.random.uniform(self.target_scale[0], self.target_scale[1])
        context_aspect_ratio = self.context_aspect_ratio
        context_scale = 1
        self.model.mode = "test"

        return self(batch, target_aspect_ratio, target_scale, context_aspect_ratio, context_scale) #just get teacher embedding

    def on_after_backward(self):
        self.update_momentum(self.m)
        self.m += (self.m_start_end[1] - self.m_start_end[0]) / self.trainer.estimated_stepping_batches


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


if __name__ == '__main__':
    dataset = D2VDataModule(dataset_path='/projects/earlydetection/CTData')


    model = IJEPA(img_size=224, patch_size=16, in_chans=3, embed_dim=64, enc_heads=8, enc_depth=8, decoder_depth=6, lr=1e-3)

    logger = TensorBoardLogger("lightning_logs", name="IJEPA_pretrain_logs_Med")


    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)

        
    checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',  
    dirpath='pretrain-checkpoints-Med/',  
    filename='{epoch:02d}-{val_loss:.5f}',  
    save_top_k=3,  
    mode='min',
    save_last=True  
    )

    #accumulate_steps = 16

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        precision="bf16-mixed",
        max_epochs=40,
        callbacks=[lr_monitor, model_summary, checkpoint_callback],
        gradient_clip_val=.1,
        #gradient_clip_algorithm="value",
        logger=logger,
        #accumulate_grad_batches=accumulate_steps,
    )

    trainer.fit(model, dataset)