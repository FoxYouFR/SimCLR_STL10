import os
import argparse
import torch

from torchvision import datasets

from data_handler.transformation import ContrastiveTransformation, contrast_transforms
from utils.imviz import plot_images_from_dataset
from trainer import train

CHECKPOINT_PATH = './saved/checkpoints'
DATASET_PATH = './data'
SAVE_PATH = './saved'

def main(args):
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = os.cpu_count()
    print(f'Device: {args.device}')
    print(f'Number of workers: {NUM_WORKERS}')

    unlabeled_data = datasets.STL10(root=DATASET_PATH, split='unlabeled', download=True,
                                    transform=ContrastiveTransformation(contrast_transforms, n_views=2))
    train_data_contrast = datasets.STL10(root=DATASET_PATH, split='train', download=True, 
                                         transform=ContrastiveTransformation(contrast_transforms, n_views=2))
    train_loader = torch.utils.data.DataLoader(unlabeled_data, batch_size=args.batch_size, shuffle=True,
                                               drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)
    val_loader = torch.utils.data.DataLoader(train_data_contrast, batch_size=args.batch_size, shuffle=False,
                                             drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)

    plot_images_from_dataset(unlabeled_data, num=6, save=True,
                             title='Augmented image examples of the STL10 dataset')

    simclr_model = train(
        CHECKPOINT_PATH,
        args,
        train_loader,
        val_loader,
        hidden_dim=128,
        lr=args.lr,
        temperature=args.temperature,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Contrastive Auto-Encoder')
    parser.add_argument('-b', '--batch-size', type=int, default=128, help='Input batch size for training (default: 128)')
    parser.add_argument('-e', '--max-epochs', type=int, default=500, help='Maximum number of epochs to train (default: 500)')
    parser.add_argument('-w', '--weight-decay', type=float, default=1e-4, help='Regularization term for weight decay (default: 1e-4)')
    parser.add_argument('-l', '--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('-t', '--temperature', type=float, default=0.07, help='Temperature for contrastive loss (default: 0.07)')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Activate verbosity')
    parser.add_argument('--skip-train', action='store_true', default=False, help='Skips the training procedure, requires to have a model saved')
    args = parser.parse_args()

    main(args)