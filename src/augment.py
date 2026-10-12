from torchvision.transforms import v2

def get_transforms(augment_params):
    if augment_params is None:
        augment_params = []
    train_list = [
        v2.ToPILImage(),
        v2.Resize((224,224))
        ]

    if 'geometric' in augment_params:
        geometric_transform = [
            v2.RandomRotation(180),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5)
            ]
        train_list.extend(geometric_transform)
    if 'zoom' in augment_params:
        color_transform = [
            v2.RandomResizedCrop(size=(224,224),scale=(0.8**2,1.0),ratio=(1.0,1.0)),
            v2.RandomInvert(p=0.5)
        ]
        train_list.extend(color_transform)
    if 'color' in augment_params:
        train_list.append(v2.ColorJitter(hue=0,saturation=0,brightness=0.2,contrast=0.2))
    if 'blur' in augment_params:
        train_list.append(v2.GaussianBlur(kernel_size=(5,5)))
    train_list.append(v2.ToTensor())
    if 'erasing' in augment_params:
        train_list.append(v2.RandomErasing(p=0.25,scale=(0.05,0.15)))

    train_transform = v2.Compose(train_list)
    test_transform = v2.Compose([
                    v2.ToPILImage(),
                    v2.Resize((224,224)),
                    v2.ToTensor()
                    ])
    return train_transform,test_transform