from torchvision import transforms
from src.config import CONFIG
    
train_list = [
    transforms.ToPILImage(),
    transforms.Resize((224,224))
    ]

if 'geometric' in CONFIG['augment']:
    geometric_transform = [
        transforms.RandomRotation(180),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5)
        ]
    train_list.extend(geometric_transform)
if 'color' in CONFIG['augment']:
    train_list.append(transforms.ColorJitter(hue=0,saturation=0,brightness=0.2,contrast=0.2))
if 'blur' in CONFIG['augment']:
    train_list.append(transforms.GaussianBlur(kernel_size=(5,5)))
train_list.append(transforms.ToTensor())
if 'erasing' in CONFIG['augment']:
    train_list.append(transforms.RandomErasing(p=0.25,scale=(0.05,0.15)))


train_transform = transforms.Compose(train_list)
test_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224,224)),
                transforms.ToTensor()
                ])