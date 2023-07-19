import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from tqdm import tqdm

from models.simCLR import SimCLR

def train(root_dir, args, train_loader, val_loader, **kwargs) -> SimCLR:
    trainer = pl.Trainer(default_root_dir=os.path.join(root_dir, 'SimCLR'),
                         accelerator='gpu' if str(args.device).startswith('cuda') else 'cpu',
                         devices=1, max_epochs=args.max_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode='max', monitor='val_acc_top5'),
                                    LearningRateMonitor('epoch')])
    pretrained_filename = os.path.join(root_dir, 'SimCLR.ckpt')
    if args.skip_train and os.path.isfile(pretrained_filename):
        print(f'Found pretrained model at {pretrained_filename}, loading...')
        model = SimCLR.load_from_checkpoint(pretrained_filename)
    else:
        model = SimCLR(**kwargs)
        trainer.fit(model, train_loader, val_loader)
        model = SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    return model