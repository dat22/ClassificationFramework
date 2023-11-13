from torchvision import transforms as T

class Transform:
    def __init__(self, resize, mean,  std):
        self.tranformers = {
            'train': T.Compose([
                # T.RandomResizedCrop(resize),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean, std, inplace = True)
            ]),
            'val': T.Compose([
                T.ToTensor(),
                T.Normalize(mean, std, inplace = True)
            ])
        }
        
    def __call__(self, img, phase = 'train'):
        return self.tranformers[phase](img)
 