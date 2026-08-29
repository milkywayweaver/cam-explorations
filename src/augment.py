from torchvision.transforms import v2
from src.config import CONFIG
    
train_list = [
    v2.ToPILImage(),
    v2.Resize((224,224))
    ]

if 'geometric' in CONFIG['augment']:
    geometric_transform = [
        v2.RandomRotation(180),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5)
        ]
    train_list.extend(geometric_transform)
if 'color' in CONFIG['augment']:
    train_list.append(v2.ColorJitter(hue=0,saturation=0,brightness=0.2,contrast=0.2))
if 'blur' in CONFIG['augment']:
    train_list.append(v2.GaussianBlur(kernel_size=(5,5)))
train_list.append(v2.ToTensor())
if 'erasing' in CONFIG['augment']:
    train_list.append(v2.RandomErasing(p=0.25,scale=(0.05,0.15)))

train_transform = v2.Compose(train_list)
test_transform = v2.Compose([
                v2.ToPILImage(),
                v2.Resize((224,224)),
                v2.ToTensor()
                ])