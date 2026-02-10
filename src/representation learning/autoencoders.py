# Source of Resnet Autoencoder:
# https://github.com/Horizon2333/imagenet-autoencoder/blob/main/models/resnet.py

#for typing
from typing import List, Callable

# Ml modules
import torch
import torch.nn as nn

# training visualization
from tqdm import tqdm
import matplotlib.pyplot as plt


class ResNet(nn.Module):

    def __init__(self, configs: List, bottleneck: bool= False, num_classes: int= 1000):
        super(ResNet, self).__init__()

        self.encoder = ResNetEncoder(configs, bottleneck)

        self.avpool = nn.AdaptiveAvgPool2d((1,1))

        if bottleneck:
            self.fc = nn.Linear(in_features=2048, out_features=num_classes)
        else:
            self.fc = nn.Linear(in_features=512, out_features=num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.encoder(x)

        x = self.avpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


class ResNetEncoder(nn.Module):

    def __init__(self, configs: List, bottleneck: bool= False):
        super(ResNetEncoder, self).__init__()

        if len(configs) != 4:
            raise ValueError("Only 4 layers can be configued")

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
        )

        if bottleneck:

            self.conv2 = EncoderBottleneckBlock(in_channels=64,   hidden_channels=64,  up_channels=256,
                                                layers=configs[0], downsample_method="pool")
            self.conv3 = EncoderBottleneckBlock(in_channels=256,  hidden_channels=128, up_channels=512,
                                                layers=configs[1], downsample_method="conv")
            self.conv4 = EncoderBottleneckBlock(in_channels=512,  hidden_channels=256, up_channels=1024,
                                                layers=configs[2], downsample_method="conv")
            self.conv5 = EncoderBottleneckBlock(in_channels=1024, hidden_channels=512, up_channels=2048,
                                                layers=configs[3], downsample_method="conv")
            # self.linear = nn.Linear(in_features=2*1024*3*3, out_features=1024)

        else:

            self.conv2 = EncoderResidualBlock(in_channels=64,  hidden_channels=64,  layers=configs[0],
                                              downsample_method="pool")
            self.conv3 = EncoderResidualBlock(in_channels=64,  hidden_channels=128, layers=configs[1],
                                              downsample_method="conv")
            self.conv4 = EncoderResidualBlock(in_channels=128, hidden_channels=256, layers=configs[2],
                                              downsample_method="conv")
            self.conv5 = EncoderResidualBlock(in_channels=256, hidden_channels=512, layers=configs[3],
                                              downsample_method="conv")
            # self.linear = nn.Linear(in_features=512*3*3, out_features=512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # x = torch.flatten(x, start_dim=1)
        # x = self.linear(x)

        return x

class ResNetDecoder(nn.Module):

    def __init__(self, configs: List, bottleneck: bool= False):
        super(ResNetDecoder, self).__init__()

        if len(configs) != 4:
            raise ValueError("Only 4 layers can be configued")

        if bottleneck:

            # self.linear = nn.Linear(in_features=1024, out_features=2*1024*3*3)

            self.conv1 = DecoderBottleneckBlock(in_channels=2048, hidden_channels=512, down_channels=1024,
                                                layers=configs[0])
            self.conv2 = DecoderBottleneckBlock(in_channels=1024, hidden_channels=256, down_channels=512,
                                                layers=configs[1])
            self.conv3 = DecoderBottleneckBlock(in_channels=512,  hidden_channels=128, down_channels=256,
                                                layers=configs[2])
            self.conv4 = DecoderBottleneckBlock(in_channels=256,  hidden_channels=64,  down_channels=64,
                                                layers=configs[3])


        else:

            # self.linear = nn.Linear(in_features=512, out_features=512*3*3)
            self.conv1 = DecoderResidualBlock(hidden_channels=512, output_channels=256, layers=configs[0])
            self.conv2 = DecoderResidualBlock(hidden_channels=256, output_channels=128, layers=configs[1])
            self.conv3 = DecoderResidualBlock(hidden_channels=128, output_channels=64,  layers=configs[2])
            self.conv4 = DecoderResidualBlock(hidden_channels=64,  output_channels=64,  layers=configs[3])

        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=7, stride=2, padding=3,
                               output_padding=1, bias=False),
        )

        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.linear(x)
        # x = x.reshape((x.shape[0],-1,3,3))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gate(x)

        return x

class EncoderResidualBlock(nn.Module):

    def __init__(self, in_channels: int, hidden_channels: int, layers: int, downsample_method : str= "conv"):
        super(EncoderResidualBlock, self).__init__()

        if downsample_method == "conv":

            for i in range(layers):

                if i == 0:
                    layer = EncoderResidualLayer(in_channels=in_channels, hidden_channels=hidden_channels,
                                                 downsample=True)
                else:
                    layer = EncoderResidualLayer(in_channels=hidden_channels, hidden_channels=hidden_channels,
                                                 downsample=False)
                
                self.add_module('%02d EncoderLayer' % i, layer)
        
        elif downsample_method == "pool":

            maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.add_module('00 MaxPooling', maxpool)

            for i in range(layers):

                if i == 0:
                    layer = EncoderResidualLayer(in_channels=in_channels, hidden_channels=hidden_channels,
                                                 downsample=False)
                else:
                    layer = EncoderResidualLayer(in_channels=hidden_channels, hidden_channels=hidden_channels,
                                                 downsample=False)
                
                self.add_module('%02d EncoderLayer' % (i+1), layer)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for name, layer in self.named_children():

            x = layer(x)

        return x

class EncoderBottleneckBlock(nn.Module):

    def __init__(self, in_channels:int, hidden_channels:int, up_channels:int, layers:int,
                 downsample_method: str= "conv"):
        super(EncoderBottleneckBlock, self).__init__()

        if downsample_method == "conv":

            for i in range(layers):

                if i == 0:
                    layer = EncoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels,
                                                   up_channels=up_channels, downsample=True)
                else:
                    layer = EncoderBottleneckLayer(in_channels=up_channels, hidden_channels=hidden_channels,
                                                   up_channels=up_channels, downsample=False)
                
                self.add_module('%02d EncoderLayer' % i, layer)
        
        elif downsample_method == "pool":

            maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.add_module('00 MaxPooling', maxpool)

            for i in range(layers):

                if i == 0:
                    layer = EncoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels,
                                                   up_channels=up_channels, downsample=False)
                else:
                    layer = EncoderBottleneckLayer(in_channels=up_channels, hidden_channels=hidden_channels,
                                                   up_channels=up_channels, downsample=False)
                
                self.add_module('%02d EncoderLayer' % (i+1), layer)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for name, layer in self.named_children():

            x = layer(x)

        return x


class DecoderResidualBlock(nn.Module):

    def __init__(self, hidden_channels: int, output_channels: int, layers: int):
        super(DecoderResidualBlock, self).__init__()

        for i in range(layers):

            if i == layers - 1:
                layer = DecoderResidualLayer(hidden_channels=hidden_channels, output_channels=output_channels,
                                             upsample=True)
            else:
                layer = DecoderResidualLayer(hidden_channels=hidden_channels, output_channels=hidden_channels,
                                             upsample=False)
            
            self.add_module('%02d EncoderLayer' % i, layer)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for name, layer in self.named_children():

            x = layer(x)

        return x

class DecoderBottleneckBlock(nn.Module):

    def __init__(self, in_channels: int, hidden_channels:int, down_channels:int, layers:int):
        super(DecoderBottleneckBlock, self).__init__()

        for i in range(layers):

            if i == layers - 1:
                layer = DecoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels,
                                               down_channels=down_channels, upsample=True)
            else:
                layer = DecoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels,
                                               down_channels=in_channels, upsample=False)
            
            self.add_module('%02d EncoderLayer' % i, layer)
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for name, layer in self.named_children():

            x = layer(x)

        return x


class EncoderResidualLayer(nn.Module):

    def __init__(self, in_channels: int, hidden_channels: int, downsample: bool):
        super(EncoderResidualLayer, self).__init__()

        if downsample:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3,
                          stride=2, padding=1, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3,
                          stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )

        self.weight_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=hidden_channels),
        )

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1,
                          stride=2, padding=0, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
            )
        else:
            self.downsample = None

        self.relu = nn.Sequential(
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x = x + identity

        x = self.relu(x)

        return x

class EncoderBottleneckLayer(nn.Module):

    def __init__(self, in_channels: int, hidden_channels: int, up_channels: int, downsample: bool):
        super(EncoderBottleneckLayer, self).__init__()

        if downsample:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=2,
                          padding=0, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )

        self.weight_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.weight_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=up_channels, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=up_channels),
        )

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=up_channels, kernel_size=1,
                          stride=2, padding=0, bias=False),
                nn.BatchNorm2d(num_features=up_channels),
            )
        elif (in_channels != up_channels):
            self.downsample = None
            self.up_scale = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=up_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=up_channels),
            )
        else:
            self.downsample = None
            self.up_scale = None

        self.relu = nn.Sequential(
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)
        x = self.weight_layer3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)
        elif self.up_scale is not None:
            identity = self.up_scale(identity)

        x = x + identity

        x = self.relu(x)

        return x

class DecoderResidualLayer(nn.Module):

    def __init__(self, hidden_channels: int, output_channels: int, upsample: bool):
        super(DecoderResidualLayer, self).__init__()

        self.weight_layer1 = nn.Sequential(
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
        )

        if upsample:
            self.weight_layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=3,
                                   stride=2, padding=1, output_padding=1, bias=False)                
            )
        else:
            self.weight_layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=3,
                          stride=1, padding=1, bias=False),
            )

        if upsample:
            self.upsample = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=1,
                                   stride=2, output_padding=1, bias=False)   
            )
        else:
            self.upsample = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)

        if self.upsample is not None:
            identity = self.upsample(identity)

        x = x + identity

        return x

class DecoderBottleneckLayer(nn.Module):

    def __init__(self, in_channels: int, hidden_channels: int, down_channels: int, upsample: bool):
        super(DecoderBottleneckLayer, self).__init__()

        self.weight_layer1 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1,
                      stride=1, padding=0, bias=False),
        )

        self.weight_layer2 = nn.Sequential(
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
        )

        if upsample:
            self.weight_layer3 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=down_channels, kernel_size=1,
                                   stride=2, output_padding=1, bias=False)
            )
        else:
            self.weight_layer3 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=hidden_channels, out_channels=down_channels, kernel_size=1,
                          stride=1, padding=0, bias=False)
            )

        if upsample:
            self.upsample = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=down_channels, kernel_size=1,
                                   stride=2, output_padding=1, bias=False)
            )
        elif (in_channels != down_channels):
            self.upsample = None
            self.down_scale = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=in_channels, out_channels=down_channels, kernel_size=1,
                          stride=1, padding=0, bias=False)
            )
        else:
            self.upsample = None
            self.down_scale = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)
        x = self.weight_layer3(x)

        if self.upsample is not None:
            identity = self.upsample(identity)
        elif self.down_scale is not None:
            identity = self.down_scale(identity)

        x = x + identity

        return x
    
class ResNetAutoEncoder(nn.Module):
    
    @staticmethod
    def get_configs(arch: str='resnet50'):
        # True or False means wether to use BottleNeck

        if arch == 'test':
            return [1, 1, 1, 1], False
        elif arch == 'resnet18':
            return [2, 2, 2, 2], False
        elif arch == 'resnet34':
            return [3, 4, 6, 3], False
        elif arch == 'resnet50':
            return [3, 4, 6, 3], True
        elif arch == 'resnet101':
            return [3, 4, 23, 3], True
        elif arch == 'resnet152':
            return [3, 8, 36, 3], True
        else:
            raise ValueError("Undefined model")

    def __init__(self, arch:str, num_workers: int, batch_size: int, eval_step: int):

        super(ResNetAutoEncoder, self).__init__()
        
        configs, bottleneck = self.get_configs(arch)
        
        self.encoder = ResNetEncoder(configs=configs, bottleneck=bottleneck)
        self.decoder = ResNetDecoder(configs=configs, bottleneck=bottleneck)
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.eval_step = eval_step
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.encoder(x)
        x = self.decoder(x)

        return x
    
    def data_loader(self, dataset: torch.utils.data.Dataset, mode: str,
                   ) -> torch.utils.data.DataLoader:
        
        mode_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=self.batch_size,
                                              shuffle=True if mode=="train" else False,
                                              num_workers=self.num_workers, pin_memory=True,)

        return mode_loader
    
    def eval_model(self, epoch: int, data_loader: torch.utils.data.DataLoader, optimizer: torch.optim,
                   criterion: nn.modules.loss, device: torch.device, mode: str):
        
        with tqdm(data_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            n_loss = 0
            loss_ctr = 0
            for img, _ in tepoch:
                img = img.to(device)

                recon = self.forward(img)
                loss = criterion(recon, img)
                
                if mode == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                n_loss += loss.item()
                loss_ctr += 1

                tepoch.set_postfix(mode=mode, loss=n_loss/loss_ctr)

        if epoch % self.eval_step == 0 or mode == "test":
            self.hist[mode].append((n_loss/loss_ctr, img[:9], recon[:9]))
    
    @staticmethod
    def plot_encoder_decoder(outputs: List, title: str):
        fig= plt.figure(figsize=(9, 2))
        fig.suptitle(title)
        imgs = outputs[1].detach().cpu().numpy()
        recon = outputs[2].detach().cpu().numpy()
        for i, item in enumerate(imgs):
                plt.subplot(2, 9, i+1)
                plt.axis("off")
                plt.imshow(item[0], cmap="gray")

        for i, item in enumerate(recon):
            plt.subplot(2, 9, 9+i+1)
            plt.axis("off")
            plt.imshow(item[0], cmap="gray")
            
    def plot_autoencoder_results(self, num_epochs):
        
        for k in range(0, num_epochs//self.eval_step):
            ResNetAutoEncoder.plot_encoder_decoder(self.hist["train"][k], title=f"train_{k}")
            ResNetAutoEncoder.plot_encoder_decoder(self.hist["val"][k], title=f"val_{k}")
            
        ResNetAutoEncoder.plot_encoder_decoder(self.hist["test"][0], title="test")
            
            
        
    
    def fit(self, dataset: Callable, optimizer: torch.optim, loss: nn.modules.loss, num_epochs: int):
        # check to run training on cpu or gpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        # Point to training loop video
        self.hist = {"train":[], "test":[], "val":[]}
        
        # load data
        train_loader = self.data_loader(dataset(mode="train"), mode="train")
        val_loader = self.data_loader(dataset(mode="val"), mode="val")
        test_loader = self.data_loader(dataset(mode="test"), mode="test")
        
        for epoch in range(1, num_epochs+1):
            self.train()
            self.eval_model(epoch = epoch, data_loader = train_loader, optimizer = optimizer,
                            criterion = loss, device = device, mode = "train")
            
            if epoch % self.eval_step == 0:
                self.eval()
                self.eval_model(epoch = epoch, data_loader = val_loader, optimizer = optimizer,
                                criterion = loss, device = device, mode = "val")
        
        
        self.eval()
        self.eval_model(epoch = epoch, data_loader = test_loader, optimizer = optimizer,
                        criterion = loss, device = device, mode = "test")
        
        self.plot_autoencoder_results(num_epochs)