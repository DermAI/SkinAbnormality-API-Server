
import torch
import torchvision

def defineTransform(inference=False):
    layout = (256, 256)
    size = (224, 224)
    if(inference):
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(layout),
                torchvision.transforms.CenterCrop(size),
                torchvision.transforms.ToTensor()
            ]
        )
        pass
    else:
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(layout),
                torchvision.transforms.RandomRotation(degrees=(0, 360)),
                torchvision.transforms.RandomVerticalFlip(p=0.5),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.ColorJitter(brightness=1, contrast=1, saturation=1, hue=0.5),
                torchvision.transforms.RandomCrop(size),
                torchvision.transforms.ToTensor()
            ]
        )
        pass
    return(transform)

class SeniorModel(torch.nn.Module):
    
    def __init__(self, weight='ResNet152_Weights.IMAGENET1K_V1'):
        super().__init__()
        if(weight=='ResNet152_Weights.IMAGENET1K_V1'):
            component = torchvision.models.resnet152(weights=weight)
            backbone = torch.nn.Sequential(*list(component.children())[:-1])
            ##  Output of shape is `[-1, 2048, 1, 1]` array.
            layer = torch.nn.Sequential(
                backbone,
                torch.nn.AvgPool2d(kernel_size=1),
                torch.nn.Flatten(1, -1),
                torch.nn.Linear(2048, 23)
            )
            pass
        if(weight=='DenseNet121_Weights.IMAGENET1K_V1'):
            component = torchvision.models.densenet121(weights=weight)
            backbone = torch.nn.Sequential(*list(component.children())[:-1])
            ##  Output of shape is `[-1, 1024, 7, 7]` array.
            layer = torch.nn.Sequential(
                backbone,
                torch.nn.AvgPool2d(kernel_size=7),
                torch.nn.Flatten(1, -1),
                torch.nn.Linear(1024, 23)
            )
            pass
        if(weight=='EfficientNet_B0_Weights.IMAGENET1K_V1'):
            component = torchvision.models.efficientnet_b0(weights=weight)
            backbone = torch.nn.Sequential(*list(component.children())[:-1])
            ##  Output of shape is `[-1, 1280, 1, 1]` array.
            layer = torch.nn.Sequential(
                backbone,
                torch.nn.AvgPool2d(kernel_size=1),
                torch.nn.Flatten(1, -1),
                torch.nn.Linear(1280, 23)
            )
            pass
        if(weight=='MobileNet_V2_Weights.IMAGENET1K_V1'):
            component = torchvision.models.mobilenet_v2(weights=weight)
            backbone = torch.nn.Sequential(*list(component.children())[:-1])
            ##  Output of shape is `[-1, 1280, 7, 7]` array.
            layer = torch.nn.Sequential(
                backbone,
                torch.nn.AvgPool2d(kernel_size=7),
                torch.nn.Flatten(1, -1),
                torch.nn.Linear(1280, 23)
            )
            pass        
        self.layer = layer
        return

    def forward(self, image):
        score = self.layer(image)        
        return(score)

    pass

class downstreamModel(torch.nn.Module):
    
    def __init__(self, weight={'backbone':'', 'senior':''}):
        super().__init__()
        senior = SeniorModel(weight=weight['backbone'])
        if(weight['senior']): senior.load_state_dict(torch.load(weight['senior']))
        if(weight['backbone']=='EfficientNet_B0_Weights.IMAGENET1K_V1'):
            senior.layer[3] = torch.nn.Linear(1280, 2)
        if(weight['backbone']=='ResNet152_Weights.IMAGENET1K_V1'):
            senior.layer[3] = torch.nn.Linear(2048, 2)    
        if(weight['backbone']=='DenseNet121_Weights.IMAGENET1K_V1'):
            senior.layer[3] = torch.nn.Linear(1024, 2)    
        if(weight['backbone']=='MobileNet_V2_Weights.IMAGENET1K_V1'):
            senior.layer[3] = torch.nn.Linear(1280, 2)    
        self.layer = senior.layer
        return

    def forward(self, image):
        score = self.layer(image)        
        return(score)

    def loadWeight(self, path):
        self.load_state_dict(torch.load(path, 'cpu'))
        self.eval()
        return

    pass
