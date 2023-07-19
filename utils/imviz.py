import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

VIZ_PATH = './visualization'

def plot_image(image, label=None):
    image = image.detach().cpu().numpy().transpose(1, 2, 0)
    plt.figure(figsize=(20, 20))
    plt.imshow(image, cmap='gray')
    if label: plt.title(label)
    plt.axis('off')
    plt.show()

def plot_images_from_dataset(dataset, num, title='', figsize=(10, 5), save=False):
    images = torch.stack([img for i in range(num) for img in dataset[i][0]], dim=0)
    grid = make_grid(images, nrow=num, normalize=True, pad_value=0.9)
    
    if save:
        save_image(grid, f'{VIZ_PATH}/image_{title.replace(" ", "_")}.png')
    else:
        grid = grid.permute(1, 2, 0)
        plt.figure(figsize=figsize)
        plt.title(title)
        plt.imshow(grid)
        plt.axis('off')
        plt.show()