import torch
from sklearn.metrics import confusion_matrix
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os
'''
    Calculate accuracy
'''
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))
    return acc, preds
'''
    Calculate F1_score
'''
def F1_score(outputs, labels):
    _, preds = torch.max(outputs, dim = 1)
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*((precision*recall)/(precision+recall))

    return precision, recall, f1, preds

'''
    get learning rate from optimizer
'''
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

'''
    Pick GPU if available, else CPU
'''
def get_default_device():
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
'''
    Move tensor(s) to chosen device
'''
def to_device(data, device):
    
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking = True)

def denormalize(images, means, stds):
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means

def show_batch(dl, mean, stds):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]); ax.set_yticks([])
        denorm_images = denormalize(images, mean, stds)
        ax.imshow(make_grid(denorm_images[:64], nrow=8).permute(1, 2, 0).clamp(0,1))
        break