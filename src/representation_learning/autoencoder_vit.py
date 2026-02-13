import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore


class ResNetDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.initial_linear = nn.Sequential(
            nn.Linear(latent_dim, 512 * 6 * 6),
            nn.BatchNorm1d(512 * 6 * 6),
            nn.ReLU(inplace=True)
        )
        
        # ResNet-style decoder blocks
        self.decoder = nn.Sequential(
            # 6x6 -> 12x12
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 12x12 -> 24x24
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 24x24 -> 48x48
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 48x48 -> 96x96
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Final convolution to get 3 channels
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.initial_linear(x)
        x = x.view(-1, 512, 6, 6)
        x = self.decoder(x)
        return x
    
class ShallowResNetDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        
        # Reduce initial dimension to save parameters
        self.initial_linear = nn.Sequential(
            nn.Linear(latent_dim, 256 * 3 * 3),
            nn.BatchNorm1d(256 * 3 * 3),
            nn.ReLU(inplace=True)
        )
        
        # More efficient decoder with fewer channels
        self.decoder = nn.Sequential(
            # 3x3 -> 6x6
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 6x6 -> 12x12
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 12x12 -> 24x24
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 24x24 -> 48x48
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # 48x48 -> 96x96
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            
            # Final convolution
            nn.Conv2d(8, 3, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.initial_linear(x)
        x = x.view(-1, 256, 3, 3)
        x = self.decoder(x)
        return x

class LightViTDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.patch_size = 8
        self.hidden_dim = 128

        # Project to patches with a more direct approach
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 4 * 4 * self.hidden_dim),  # Directly project to 4x4 spatial dimension
            nn.LayerNorm(4 * 4 * self.hidden_dim),
            nn.GELU(),
        )

        # Simpler progressive upsampling
        self.decoder = nn.Sequential(
            # 4x4 -> 12x12
            nn.ConvTranspose2d(self.hidden_dim, 64, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(64),
            nn.GELU(),
            
            # 12x12 -> 24x24 
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.GELU(),
            
            # 24x24 -> 48x48
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(16),
            nn.GELU(),
            
            # 48x48 -> 96x96
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(8),
            nn.GELU(),
            
            # Final conv
            nn.Conv2d(8, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Project features
        x = self.projector(x)
        
        # Reshape to spatial representation
        B = x.shape[0]
        x = x.reshape(B, self.hidden_dim, 4, 4)
        
        # Decode to image
        x = self.decoder(x)
        return x

class DinoV2Autoencoder(nn.Module):
    def __init__(self, batch_size=32, num_workers=4, eval_step=5, freeze_encoder=True):
        super().__init__()
        self.encoder = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
        
        # Freeze encoder weights if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        self.latent_dim = 768  # ViT-B/8 output dimension
        self.decoder = LightViTDecoder(self.latent_dim)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.eval_step = eval_step
        self.hist = {"train": [], "val": [], "test": []}
        
    def encode(self, x):
        with torch.set_grad_enabled(self.training):
            features = self.encoder(x)
            if isinstance(features, tuple):
                features = features[0]  # Take CLS token for ViT
        return features

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, **kwargs):
        z = self.encode(x)
        recon = self.decode(z)
        return recon

    def data_loader(self, dataset, mode):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(mode == "train"),
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def eval_model(self, epoch, data_loader, optimizer, criterion, device, mode, plotting=True) -> None:
        with torch.set_grad_enabled(mode == "train"):
            with tqdm(data_loader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                running_loss = 0
                batch_count = 0

                tepoch = tqdm(data_loader, unit="batch", desc=f"Epoch {epoch+1} [{mode}]")
                for i, (img, _) in enumerate(tepoch):
                    img = img.to(device)
                    recon = self(img)
                    loss = criterion(recon, img)

                    if mode == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item()
                    batch_count += 1
                    tepoch.set_postfix(mode=mode, loss=running_loss/batch_count)

            if epoch % self.eval_step == 0 or mode == "test":
                # Detach tensors before converting to numpy
                self.hist[mode].append((
                    running_loss/batch_count,
                    img[:9].detach().cpu().numpy(),
                    recon[:9].detach().cpu().numpy()
                ))
                if plotting:
                    # Plot current results with detached tensors
                    self.plot_current_samples(
                        img[:9].detach().cpu().numpy(),
                        recon[:9].detach().cpu().numpy(),
                        epoch,
                        mode
                    )

    def plot_current_samples(self, imgs, recons, epoch, mode):
        """Plot current batch of original and reconstructed images"""
        plt.figure(figsize=(15, 3))
        plt.suptitle(f"{mode.capitalize()} Results - Epoch {epoch}")
        
        # Plot original images
        for i in range(min(9, imgs.shape[0])):
            plt.subplot(2, 9, i + 1)
            plt.axis("off")
            plt.imshow(np.transpose(imgs[i], (1, 2, 0)))  # Changed to use np.transpose directly
        
        # Plot reconstructions
        for i in range(min(9, recons.shape[0])):
            plt.subplot(2, 9, i + 10)
            plt.axis("off")
            plt.imshow(np.transpose(recons[i], (1, 2, 0)))  # Changed to use np.transpose directly
        
        plt.tight_layout()
        plt.show()
        plt.close()

    def fit(self, dataset, num_epochs=100, learning_rate=1e-4, plotting=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # Setup data loaders
        train_loader = self.data_loader(dataset(mode="train"), "train")
        val_loader = self.data_loader(dataset(mode="val"), "val")
        test_loader = self.data_loader(dataset(mode="test"), "test")

        # Initialize optimizer and loss
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        for epoch in range(1, num_epochs + 1):
            # Training phase
            self.train()
            self.eval_model(epoch, train_loader, optimizer, criterion, device, "train", plotting=False)

            # Validation phase
            if epoch % self.eval_step == 0:
                self.eval()
                with torch.no_grad():
                    self.eval_model(epoch, val_loader, optimizer, criterion, device, "val", plotting=plotting)
    
        # Test phase
        self.eval()
        with torch.no_grad():
            self.eval_model(epoch, test_loader, optimizer, criterion, device, "test", plotting=plotting)

    def plot_results(self, num_epochs):
        for k in range(0, num_epochs // self.eval_step):
            plt.figure(figsize=(15, 6))
            
            # Plot training results
            plt.subplot(2, 1, 1)
            plt.title(f"Training - Epoch {k * self.eval_step}")
            self._plot_samples(self.hist["train"][k])
            
            # Plot validation results
            plt.subplot(2, 1, 2)
            plt.title(f"Validation - Epoch {k * self.eval_step}")
            self._plot_samples(self.hist["val"][k])
            
            plt.tight_layout()
            plt.show()

        # Plot test results
        plt.figure(figsize=(15, 3))
        plt.title("Test Results")
        self._plot_samples(self.hist["test"][0])
        plt.show()

    def _plot_samples(self, outputs):
        imgs = outputs[1].cpu().numpy()
        recon = outputs[2].cpu().numpy()
        
        for i in range(9):
            plt.subplot(2, 9, i + 1)
            plt.axis("off")
            plt.imshow(np.transpose(imgs[i], (1, 2, 0)))
            
            plt.subplot(2, 9, i + 10)
            plt.axis("off")
            plt.imshow(np.transpose(recon[i], (1, 2, 0)))

