# %%
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import random_split, Subset, DataLoader

sys.path.append('../src/spectrogram')

from autoencoder_vit import DinoV2Autoencoder
from autoencoders import ResNetAutoEncoder
from data_loader import SpectogramDataset, ConstilationDataset

# %%
class DinoClassifier(nn.Module):
    def __init__(self, num_classes, freeze_encoder=False):
        super().__init__()
        # Instantiate the autoencoder to access its encoder part
        autoencoder = DinoV2Autoencoder(freeze_encoder=freeze_encoder)
        # autoencoder = ResNetAutoEncoder(arch='resnet34',
        #         batch_size=32,
        #         num_workers=1,
        #         eval_step=5)
        # Load pretrained weights if a path is provided
        # self.pretrained_path = '../exp/dino_autoencoder.pth'
        self.pretrained_path = '../exp/resnet_autoencoder.pth'
        if hasattr(self, 'pretrained_path') and self.pretrained_path is not None:
            checkpoint = torch.load(self.pretrained_path, map_location='cpu')
            autoencoder.load_state_dict(checkpoint, strict=False)
        self.encoder = autoencoder.encoder
        
        # DINO ViT-B/8 has a latent dimension of 768
        latent_dim = 768
        # latent_dim = 512

        # Add a pooling layer to convert the feature map to a vector
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
       
        
        # Define a classification head
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        )

    def forward(self, x):
        # The encoder returns the CLS token features
        features = self.encoder(x)
        # Pass features to the classification head
        # features = self.pool(features)
        # features = torch.flatten(features, 1)
    
        output = self.classifier_head(features)
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
    
def topk_centriod_accuracy(features, centroid_tensor, target, k=3):
    """Computes the top-k accuracy based on centroid distances."""
    with torch.no_grad():
        batch_size = target.size(0)

        # Compute distances to centroids
        distances = torch.cdist(features, centroid_tensor)  # Shape: (batch_size, num_classes)
        
        # Get top-k closest centroids
        _, pred = distances.topk(k, 1, largest=False, sorted=True)  # Get indices of closest centroids
        pred = pred.t()
        
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        return correct_k.item()

# %%
torch.cuda.empty_cache()  # Clear CUDA cache to avoid memory issues
# --- Configuration ---
# Adjust these paths to point to your dataset
BASE_DATA_PATH = '../data/own/unlabeled_10k/train' # Example path
CLASSES = ['OOK', '4ASK', '8ASK', 'OQPSK', 'CPFSK', 'GFSK', '4PAM', 'DQPSK', '16PAM', 'GMSK']

# Hyperparameters
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_WORKERS = 4 # Adjust based on your system
EVAL_STEP = 5   # Evaluate on validation set every 5 epochs

# Define the transformation to apply to the spectrogram images
# The models expect a certain input size, e.g., 224x224 for ViT.
TRANSFORM = transforms.Compose([
    transforms.Resize((96, 96)), # Resizing to 96x96 as seen in your decoder architectures
    transforms.ToTensor()
])

# 2. Prepare Dataset and DataLoaders
# Instantiate the full dataset
full_dataset = ConstilationDataset(
    dataset_path=BASE_DATA_PATH,
    classes=CLASSES,
    transform=TRANSFORM
)

# Split dataset into training, validation, and test sets
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# 3. Initialize Model, Loss, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DinoClassifier(num_classes=len(CLASSES), freeze_encoder=False).to(device)

# Loss function for multi-class classification
criterion = nn.CrossEntropyLoss()

# Optimizer (only training the classifier head)
optimizer = optim.AdamW(model.classifier_head.parameters(), lr=LEARNING_RATE)

print("Setup complete. Starting training...")


# %%
# 4. Training and Evaluation Loop
for epoch in range(NUM_EPOCHS):
    # --- Training Phase ---
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}")

    # --- Validation Phase ---
    if (epoch + 1) % EVAL_STEP == 0:
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                topk_correct_val += topk_accuracy(outputs, labels, k=3)
        
        val_accuracy = 100 * correct / total
        val_topk_accuracy = 100 * topk_correct_val / total
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%, Top-3 Accuracy: {val_topk_accuracy:.2f}%")


print("\nTraining finished. Evaluating on the test set...")

# 5. Final Evaluation on Test Set
model.eval()
test_correct = 0
topk_correct_test = 0
test_total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        topk_correct_test += topk_accuracy(outputs, labels, k=3)

test_accuracy = 100 * test_correct / test_total
test_topk_accuracy = 100 * topk_correct_test / test_total
print(f"Final Test Accuracy: {test_accuracy:.2f}%")
print(f"Final Test Top-3 Accuracy: {test_topk_accuracy:.2f}%")



# %%
import json
import numpy as np

model = DinoClassifier(num_classes=len(CLASSES), freeze_encoder=False)
model.load_state_dict(torch.load('../exp/dino_classifier.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
test_correct = 0
topk_correct_test = 0
centroid_topk_correct = 0
test_total = 0
topk=5

full_dataset = ConstilationDataset(
    dataset_path='../data/own/-30dB/test',
    classes=CLASSES,
    transform=TRANSFORM
)

with open('../data/own/unlabeled_10k/train/class_centers.json', 'r') as f:
        data = json.load(f)
        # Ensure correct order using CLASSES list
        c_list = [data[c_name] for c_name in CLASSES]
        centroid_tensor = torch.tensor(np.array(c_list), dtype=torch.float32).to(device)


test_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        features = model.encoder(images) # Shape: (Batch, Feature_Dim)
        outputs = model.classifier_head(features)
        
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        topk_correct_test += topk_accuracy(outputs, labels, k=topk)
        centroid_topk_correct += topk_centriod_accuracy(features, centroid_tensor, labels, k=topk)

test_accuracy = 100 * test_correct / test_total
test_topk_accuracy = 100 * topk_correct_test / test_total
print(f"Final Test Accuracy: {test_accuracy:.2f}%")
print(f"Final Test Top-{topk} Accuracy: {test_topk_accuracy:.2f}%")
centroid_topk_accuracy = 100 * centroid_topk_correct / test_total
print(f"Final Centroid Top-{topk} Accuracy: {centroid_topk_accuracy:.2f}%")


# %%
from glob import glob
from PIL import Image
import json

topk = 5
dataset_folder = '-30dB'
model = DinoClassifier(num_classes=len(CLASSES), freeze_encoder=False)
model.load_state_dict(torch.load('../exp/dino_classifier.pth', map_location='cpu', weights_only=True))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

dataset_path=f'../data/own/{dataset_folder}/test'
noiseless_image_path = os.path.join(dataset_path, 'noiseLessImg')
noisy_image_path = os.path.join(dataset_path, 'noisyImg')
signal_images = glob(os.path.join(noiseless_image_path, '*.png'))
signal_images.extend(glob(os.path.join(noisy_image_path, '*.png')))


predictions = {}
with torch.no_grad():
    for image_path in signal_images:
        image = Image.open(image_path).convert("RGB")
        image = TRANSFORM(image).unsqueeze(0).to(device)
        outputs = model(image)
        # Get top 4 predictions
        probs, indices = torch.topk(outputs, topk, dim=1)

        # Convert indices to class names
        pred_classes = [CLASSES[i] for i in indices[0]]
        # Store predictions
        type_img = os.path.basename(os.path.dirname(image_path))
        file_name = os.path.basename(image_path)
        predictions[file_name] = predictions.get(file_name, {})
        predictions[file_name][type_img] = pred_classes


# Save predictions to a JSON file
with open(f'../data/own/{dataset_folder}/top{topk}_predictions.json', 'w') as f:
    json.dump(predictions, f, indent=4)

print(f"Top-{topk} predictions saved to top{topk}_predictions.json")

# %%
from scipy.spatial.distance import cdist
import numpy as np
import json

# --- 1. Setup Model and Data ---
print("--- Setting up model and data for feature extraction ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained classifier model
model = DinoClassifier(num_classes=len(CLASSES), freeze_encoder=False) # Freeze encoder as we only need features
model.load_state_dict(torch.load('../exp/dino_classifier.pth', map_location=device, weights_only=True))
model.to(device)
model.eval()

# Load the full training dataset
# We need to modify the dataset to return file paths along with images and labels
class DatasetWithPath(ConstilationDataset):
    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        path = self.signal_images[idx]
        return image, label, path

train_dataset = DatasetWithPath(
    dataset_path='../data/own/unlabeled_10k/train', # Using the training data path
    classes=CLASSES,
    transform=TRANSFORM
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# --- 2. Extract Features for All Training Samples ---
print("--- Extracting features from all training samples ---")
features_by_class = {i: [] for i in range(len(CLASSES))}
paths_by_class = {i: [] for i in range(len(CLASSES))}

with torch.no_grad():
    for images, labels, paths in train_loader:
        images = images.to(device)
        # Use the model's encoder to get feature vectors
        # The forward pass of the encoder is what we need
        features = model.encoder(images)
        
        for i in range(len(labels)):
            label_idx = labels[i].item()
            features_by_class[label_idx].append(features[i].cpu().numpy())
            paths_by_class[label_idx].append(paths[i])

# --- 3. Calculate Class Centers (Centroids) ---
print("--- Calculating class centers ---")
class_centers = {}
for i in range(len(CLASSES)):
    if features_by_class[i]:
        class_features = np.array(features_by_class[i])
        class_centers[i] = np.mean(class_features, axis=0)

centriods = {CLASSES[i]: class_centers[i] for i in class_centers}
with open('../data/own/unlabeled_10k/train/class_centers.json', 'w') as f:
    json.dump({CLASSES[i]: class_centers[i].tolist() for i in class_centers}, f, indent=4)

# --- 4. Find the Closest Sample to Each Center ---
print("--- Finding the closest sample to each class center ---")
closest_samples = {}
for i in range(len(CLASSES)):
    if i in class_centers:
        center = class_centers[i].reshape(1, -1)
        features = np.array(features_by_class[i])
        
        # Calculate Euclidean distance from each sample to the center
        distances = cdist(features, center, 'euclidean')
        
        # Find the index of the sample with the minimum distance
        closest_sample_index = np.argmin(distances)
        
        # Get the path of the closest sample
        closest_path = paths_by_class[i][closest_sample_index]
        closest_samples[CLASSES[i]] = closest_path

# --- 5. Print the Results ---
print("\n--- Closest Sample to the Center of Each Class ---")
for class_name, path in closest_samples.items():
    print(f"{class_name}: {path}")


# %%
from glob import glob
from PIL import Image
import json
import torch
import numpy as np

topk = 5
dataset_folder = 'unlabeled_10k'
dataset_path = f'../data/own/{dataset_folder}/test'
noiseless_image_path = os.path.join(dataset_path, 'noiseLessImg')
noisy_image_path = os.path.join(dataset_path, 'noisyImg')

# Load Model
model = DinoClassifier(num_classes=len(CLASSES), freeze_encoder=False)
model.load_state_dict(torch.load('../exp/dino_classifier.pth', map_location='cpu', weights_only=True))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Prepare Centroids Tensor
# We convert the dictionary of numpy centroids to a single PyTorch tensor for fast batch calculation
# We must keep track of which class name corresponds to which row in the tensor
centroid_list = []
centroid_class_names = []

for i in sorted(class_centers.keys()):
    centroid_list.append(class_centers[i])
    centroid_class_names.append(CLASSES[i])

# Shape: (Num_Classes, Feature_Dim)
centroid_tensor = torch.tensor(np.array(centroid_list), dtype=torch.float32).to(device)

# Load Images
signal_images = glob(os.path.join(noiseless_image_path, '*.png'))
signal_images.extend(glob(os.path.join(noisy_image_path, '*.png')))

predictions = {}
print(f"Processing {len(signal_images)} images using Centroids from {dataset_folder}...")

with torch.no_grad():
    for image_path in signal_images:
        image = Image.open(image_path).convert("RGB")
        image = TRANSFORM(image).unsqueeze(0).to(device)
        
        # 1. Extract Features from the Encoder (instead of classifier head)
        features = model.encoder(image) # Shape: (1, Feature_Dim)
        
        # 2. Calculate Euclidean Distance to all Centroids
        # torch.cdist computes distance between each pair of rows
        dists = torch.cdist(features, centroid_tensor, p=2) # Shape: (1, Num_Classes)
        
        # 3. Find Top-K Closest Centroids (Smallest Distances)
        # largest=False because we want the mimimum distance
        topk_vals, topk_indices = torch.topk(dists, k=min(topk, len(centroid_class_names)), dim=1, largest=False)

        # 4. Convert indices to class names
        # topk_indices is a tensor of indices into our centroid_tensor/centroid_class_names list
        pred_classes = [centroid_class_names[i] for i in topk_indices[0].cpu().numpy()]
        
        # Store predictions
        type_img = os.path.basename(os.path.dirname(image_path))
        file_name = os.path.basename(image_path)
        predictions[file_name] = predictions.get(file_name, {})
        predictions[file_name][type_img] = pred_classes


# Save predictions to a JSON file
output_path = f'../data/own/{dataset_folder}/top{topk}_centroid_predictions.json'
with open(output_path, 'w') as f:
    json.dump(predictions, f, indent=4)

print(f"Top-{topk} centroid-based predictions saved to {output_path}")

# %%
import pickle

# Assuming the pickle file is named 'train_data.pkl' and located in a 'data' directory
# Please adjust the path to your pickle file accordingly.
with open('../data/own/unlabeled_10k/test_noisySignal_5_4_data.pkl', 'rb') as f:
    train_data = pickle.load(f)

# Now, train_data contains the data from the pickle file.
# You can print it to verify
# print(train_data)
train_data.keys()

# %%
i = 1
print(train_data['discret_prompts'][i], train_data['labels'][i])
print(len(train_data['old_prompts'][0]), len(train_data['labels']))

# %%
"""
### Instructions
    You are an expert quantitative analyst in wireless communication modulation.
    Based on your knowledge in wireless communication modulation and the detailed signal statistics provided below, determine the modulation type.
"""

# %%
import os
import pickle

# with open('../data/own/unlabeled_10k/test_noiselessSignal_data.pkl', 'rb') as f:
#     noiseless_data = pickle.load(f)

# print(noiseless_data.keys())
N_BINS = 20
TOP_K = 5
NOISE_MODE = 'noisySignal'
file_name = '../data/own/unlabeled_10k/test/noisySignal/4ASK_-5.57dB__076_20250127_145624.npy'
with open(f'../data/own/unlabeled_10k/test_{NOISE_MODE}_{N_BINS}_{TOP_K}_data.pkl', 'rb') as f:
    noisy_data = pickle.load(f)

print(noisy_data.keys())

# Compare the prompts of two data (noiseless and noisy) for the same index

idx = noisy_data['signal_paths'].index(os.path.abspath(file_name))  # You can change this index to compare other samples

# idx = 0

# print("Noiseless prompt:")
# print(noiseless_data['prompts'][idx])
# print("\nNoisy prompt:")
# print(noisy_data['prompts'][idx])

# print("\nNoiseless discrete prompt:")
# print(noiseless_data['discret_prompts'][idx])
print("\nNoisy discrete prompt:")
print(noisy_data['discret_prompts'][idx])

# print("\nLabel (should be the same):")
# print(noiseless_data['labels'][idx], noisy_data['labels'][idx])


# Compare the number of tokens for discret_prompts with old_prompts

# import re

# from transformers import GPT2TokenizerFast

# # Initialize GPT-2 tokenizer (compatible with OpenAI GPT tokenization)
gpt_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# def count_tokens(text):
#     # Use GPT tokenizer to count tokens
#     return len(gpt_tokenizer.encode(text))

# idx = 0  # You can change this index to compare other samples

# print(len(noiseless_data['prompts'][idx]))
# print(len(noiseless_data['old_prompts'][idx]))
# print(len(noiseless_data['discret_prompts'][idx]))


# discret_prompt = noiseless_data['discret_prompts'][idx]
# old_prompt = noiseless_data['old_prompts'][idx]

# discret_tokens = count_tokens(discret_prompt)
# old_tokens = count_tokens(old_prompt)

# print(f"Number of tokens in discret_prompt: {discret_tokens}")
# print(f"Number of tokens in old_prompt: {old_tokens}")





# %%
noisy_data['signal_paths']

# %%
with open('../data/own/unlabeled_10k/test_noisySignal_data.pkl', 'rb') as f:
    new_old_data = pickle.load(f)

print(len(new_old_data['discret_prompts'][idx]))

# %%
print(len(noisy_data['discret_prompts'][idx]))


