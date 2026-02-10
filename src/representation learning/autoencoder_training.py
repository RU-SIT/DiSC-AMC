# %%
import os
import sys
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torchvision import transforms  # type: ignore
from torch.utils.data import random_split, Subset # type: ignore

sys.path.append('../src/spectrogram')

from autoencoder_vit import DinoV2Autoencoder
from autoencoders import ResNetAutoEncoder
from data_loader import SpectogramDataset, ConstilationDataset

# %%
class DatasetFactory:
    """A callable class that creates SpectogramDataset instances for different modes."""
    def __init__(self, dataset_path, classes, transform):
        self.dataset_path = dataset_path
        self.classes = classes
        self.transform = transform

        # Create the full dataset
        full_dataset = ConstilationDataset(
            dataset_path=self.dataset_path,
            classes=self.classes,
            transform=self.transform
        )

        # Split the dataset into train, validation, and test sets
        dataset_size = len(full_dataset)
        val_size = int(0.1 * dataset_size)
        test_size = int(0.1 * dataset_size)
        train_size = dataset_size - val_size - test_size

        # Ensure reproducibility with a fixed generator
        generator = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size], generator=generator
        )

    def __call__(self, mode='train'):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
        Returns:
            torch.utils.data.Subset: A subset of the dataset for the specified mode.
        """
        if mode == 'train':
            print(f"Using {len(self.train_dataset)} samples for training.")
            return self.train_dataset
        elif mode == 'val':
            print(f"Using {len(self.val_dataset)} samples for validation.")
            return self.val_dataset
        elif mode == 'test':
            print(f"Using {len(self.test_dataset)} samples for testing.")
            return self.test_dataset
        else:
            raise ValueError(f"Mode '{mode}' not recognized. Use 'train', 'val', or 'test'.")


def main(args):
    """Main function to train the specified autoencoder."""
    # --- Configuration from args ---
    BASE_DATA_PATH: str = args.base_data_path
    CLASSES: list = args.classes.split(',')
    NUM_EPOCHS: int = args.num_epochs
    LEARNING_RATE: float = args.learning_rate
    BATCH_SIZE: int = args.batch_size
    NUM_WORKERS: int = args.num_workers
    EVAL_STEP: int = args.eval_step
    IMAGE_SIZE: int = args.image_size

    # Define the transformation to apply to the spectrogram images
    TRANSFORM = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])


    # Create the dataset factory instance
    dataset_factory = DatasetFactory(BASE_DATA_PATH, CLASSES, TRANSFORM)
    save_path = args.save_path

    if args.model == 'dino':
        print("--- Training DinoV2Autoencoder ---")
        # 1. Initialize the model
        model = DinoV2Autoencoder(
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            eval_step=EVAL_STEP,
            freeze_encoder=False # Set to False to fine-tune the DINOv2 encoder
        )
        # 2. Start the training process
        model.fit(
            dataset=dataset_factory,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            plotting=False,
        )
        torch.save(model.state_dict(), save_path)
        print(f"--- DinoV2Autoencoder Training Complete ---")
        print(f"Model saved to {save_path}")

    elif args.model == 'resnet':
        print("\n--- Training ResNetAutoEncoder ---")
        # 1. Initialize the model
        model = ResNetAutoEncoder(
            arch='resnet34',
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            eval_step=EVAL_STEP
        )
        # 2. Define optimizer and loss
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()
        # 3. Start the training process
        model.fit(
            dataset=dataset_factory,
            optimizer=optimizer,
            loss=criterion,
            num_epochs=NUM_EPOCHS,
        )
        torch.save(model.state_dict(), save_path)
        print(f"--- ResNetAutoEncoder Training Complete ---")
        print(f"Model saved to {save_path}")


# %%
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train an autoencoder model.")
    parser.add_argument(
        '--model', 
        type=str, 
        required=True, 
        choices=['dino', 'resnet'],
        help="The type of model to train ('dino' or 'resnet')."
    )
    parser.add_argument(
        '--save_path', 
        type=str, 
        required=True, 
        help="Path to save the trained model."
    )
    parser.add_argument('--base_data_path', type=str, default='../../data/own/unlabeled_10k/train/', help='Path to the training data.')
    parser.add_argument('--classes', type=str, default='OOK,4ASK,8ASK,OQPSK,CPFSK,GFSK,4PAM,DQPSK,16PAM,GMSK', help='Comma-separated list of class names.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for data loading.')
    parser.add_argument('--eval_step', type=int, default=5, help='Evaluate on validation set every N epochs.')
    parser.add_argument('--image_size', type=int, default=96, help='The size to resize images to (e.g., 96 for 96x96).')
    
    args = parser.parse_args()
    main(args)

