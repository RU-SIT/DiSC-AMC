# %%
import os
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import random_split, Subset, DataLoader



from .autoencoder_vit import DinoV2Autoencoder
from .autoencoders import ResNetAutoEncoder
from .data_loader import SpectogramDataset, ConstilationDataset

# %%
class ImageClassifier(nn.Module):
    def __init__(self, backbone, num_classes, pretrained_path=None, freeze_encoder=False):
        super().__init__()
        self.backbone_type = backbone

        if backbone == 'dino':
            autoencoder = DinoV2Autoencoder(freeze_encoder=freeze_encoder)
            latent_dim = 768
            if pretrained_path:
                print(f"Loading DINO pretrained weights from {pretrained_path}")
                checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=True)
                autoencoder.load_state_dict(checkpoint, strict=False)
            self.encoder: nn.Module = autoencoder.encoder # type: ignore
            self.classifier_head = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, num_classes)
            )
        elif backbone == 'resnet':
            autoencoder = ResNetAutoEncoder(
                arch='resnet34',
                num_workers=1,
                batch_size=32,
                eval_step=5
            )
            
            latent_dim = 512
            if pretrained_path:
                print(f"Loading ResNet pretrained weights from {pretrained_path}")
                checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=True)
                autoencoder.load_state_dict(checkpoint, strict=False)
            self.encoder: nn.Module = autoencoder.encoder
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier_head = nn.Linear(latent_dim, num_classes)
        elif backbone == 'denomae':
            from functools import partial as _partial
            from src.denoMAE2.main import DenoMAE2
            from src.denoMAE2.finetune import DownstreamClassifier

            base_encoder = DenoMAE2(
                img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12,
                decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8,
                norm_layer=_partial(nn.LayerNorm, eps=1e-6),
            )
            if pretrained_path:
                print(f"Loading DenoMAE2 pretrained weights from {pretrained_path}")
                checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
                checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
                base_encoder.load_state_dict(checkpoint, strict=False)

            self._denomae_classifier = DownstreamClassifier(
                base_encoder, num_classes=num_classes, freeze_encoder=freeze_encoder,
            )
            # Expose encoder/classifier_head so the rest of the codebase
            # can access them uniformly (e.g. for embedding extraction).
            self.encoder = self._denomae_classifier.encoder
            self.classifier_head = self._denomae_classifier.classification_head
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if freeze_encoder and backbone != 'denomae':
            # denomae handles freezing inside DownstreamClassifier
            print("Freezing encoder weights.")
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        if self.backbone_type == 'denomae':
            return self._denomae_classifier(x)
        features = self.encoder(x)
        if self.backbone_type == 'dino':
            # DINO encoder returns CLS token
            output = self.classifier_head(features)
        elif self.backbone_type == 'resnet':
            # ResNet encoder returns a feature map
            pooled_features = self.pool(features)
            flattened_features = torch.flatten(pooled_features, 1)
            output = self.classifier_head(flattened_features)
        else:
            raise ValueError(f"Unsupported backbone type: {self.backbone_type}")
        return output

def topk_accuracy(output, target, k=3):
    """Computes the top-k accuracy."""
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        return correct_k.item()

def topk_centroid_accuracy(features, centroid_tensor, target, k=3):
    """Computes the top-k accuracy based on centroid distances.
    
    For each sample, computes the Euclidean distance to every class centroid
    and checks whether the correct class is among the k closest centroids.
    
    Args:
        features: Encoder output features, shape ``(batch_size, latent_dim)``.
        centroid_tensor: Class centroids, shape ``(num_classes, latent_dim)``.
            Row order must match the integer label indices.
        target: Ground-truth integer labels, shape ``(batch_size,)``.
        k: Number of top predictions to consider.
    
    Returns:
        Number of correct top-k predictions in the batch (float).
    """
    with torch.no_grad():
        # Compute distances to centroids: (batch_size, num_classes)
        distances = torch.cdist(features, centroid_tensor)

        # Get top-k closest centroids (smallest distances)
        _, pred = distances.topk(k, 1, largest=False, sorted=True)
        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        return correct_k.item()

class TopKLoss(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, output, target):
        # Get the top k predictions and their indices
        _, topk_indices = torch.topk(output, self.k, dim=1)
        
        # Create a mask for samples where the target is in the top k
        # target.unsqueeze(1) expands target to be [batch_size, 1]
        # topk_indices is [batch_size, k]
        # The comparison creates a boolean tensor [batch_size, k]
        mask = torch.any(topk_indices == target.unsqueeze(1), dim=1)
        
        # We only want to calculate loss for the samples that were misclassified (target not in top k)
        # Invert the mask to get the misclassified samples
        misclassified_mask = ~mask
        
        if not torch.any(misclassified_mask):
            # If all samples are correctly classified in top-k, loss is 0
            return torch.tensor(0.0, device=output.device, requires_grad=True)

        # Filter the output and target for misclassified samples
        misclassified_output = output[misclassified_mask]
        misclassified_target = target[misclassified_mask]

        # For the misclassified samples, apply standard cross-entropy loss
        # This encourages the model to push the correct class into the top-k
        return nn.CrossEntropyLoss()(misclassified_output, misclassified_target)

# %%
def main(args):
    torch.cuda.empty_cache()
    # --- Configuration from args ---
    BASE_DATA_PATH = args.base_data_path
    CLASSES = args.classes.split(',')
    NUM_EPOCHS = args.num_epochs
    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    EVAL_STEP = args.eval_step
    IMAGE_SIZE = args.image_size

    # --- Data Preparation ---
    TRANSFORM = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
    full_dataset = ConstilationDataset(
        dataset_path=BASE_DATA_PATH,
        classes=CLASSES,
        transform=TRANSFORM
    )
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # --- Model Initialization ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageClassifier(
        backbone=args.model,
        num_classes=len(CLASSES),
        pretrained_path=args.pretrained_path,
        freeze_encoder=args.freeze_encoder
    )

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    # criterion = TopKLoss(k=5)  # Use TopKLoss for top-3 accuracy
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    print(f"--- Starting Training for {args.model.upper()} Classifier ---")

    # --- Training and Evaluation Loop ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=False)
        for images, labels in train_loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())
        
        train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}")

        if (epoch + 1) % EVAL_STEP == 0:
            model.eval()
            val_loss, correct, topk_correct_val, total = 0.0, 0, 0, 0
            with torch.no_grad():
                val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]", leave=False)
                for images, labels in val_loop:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    topk_correct_val += topk_accuracy(outputs, labels, k=5)
            
            val_accuracy = 100 * correct / total
            val_topk_accuracy = 100 * topk_correct_val / total
            print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%, Top-5 Accuracy: {val_topk_accuracy:.2f}%")

    print("\n--- Training Finished. Evaluating on Test Set ---")
    # --- Final Evaluation ---
    model.eval()
    test_correct, topk_correct_test, test_total = 0, 0, 0
    with torch.no_grad():
        test_loop = tqdm(test_loader, desc="Testing", leave=True)
        for images, labels in test_loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            topk_correct_test += topk_accuracy(outputs, labels, k=5)

    test_accuracy = 100 * test_correct / test_total
    test_topk_accuracy = 100 * topk_correct_test / test_total
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")
    print(f"Final Test Top-5 Accuracy: {test_topk_accuracy:.2f}%")

    # --- Save Model ---
    if args.save_path:
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), args.save_path)
        else:
            torch.save(model.state_dict(), args.save_path)
        print(f"Model saved to {args.save_path}")

# %%
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a classifier model.")
    parser.add_argument('--model', type=str, required=True, choices=['dino', 'resnet'], help="Backbone model type.")
    parser.add_argument('--base_data_path', type=str, default='../../data/own/unlabeled_10k/train', help='Path to the training data.')
    parser.add_argument('--classes', type=str, default='OOK,4ASK,8ASK,OQPSK,CPFSK,GFSK,4PAM,DQPSK,16PAM,GMSK', help='Comma-separated class names.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers.')
    parser.add_argument('--eval_step', type=int, default=5, help='Validation frequency (in epochs).')
    parser.add_argument('--image_size', type=int, default=96, help='Image size for transformations.')
    parser.add_argument('--pretrained_path', type=str, default='../../exp/dino_autoencoder.pth', help='Path to pretrained autoencoder weights.')
    parser.add_argument('--save_path', type=str, default='../../exp/dino_classifier.pth', help='Path to save the trained classifier.')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze the encoder weights during training.')

    args = parser.parse_args()
    main(args)




