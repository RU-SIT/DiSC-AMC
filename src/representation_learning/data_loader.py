# %%
# Python library imports
import os
from glob import glob

# Standard library imports
import numpy as np
from PIL import Image
from scipy.signal import stft

# Torch imports
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms # Import transforms

# Custom imports
from .constants import FFT_SIZE
from .processing import get_power_spectrogram_db, get_color_img
import torchvision


class SpectogramDataset(Dataset):
    def __init__(self,
                 dataset_path: str = None,
                 classes: list = None,
                 fft_size: int = FFT_SIZE,
                 stft_overlap: int = 32,
                 colormap: str='viridis',
                 transform: callable = transforms.ToTensor()):
        
        self.signal_files = glob(os.path.join(dataset_path, '*.npy'))
        self.classes = classes
        self.label_to_int = {label: i for i, label in enumerate(self.classes)}
        self.int_to_label = {i: label for label, i in self.label_to_int.items()}
        self.num_classes = len(self.classes)

        self.colormap = colormap # for multi-modal we might want to use different colormaps
        self.transform = transform
        self.fft_size = fft_size
        self.stft_overlap = stft_overlap
        
    def __len__(self):
        return len(self.signal_files)

    def _convert_to_pil(self, spectrum_db: np.ndarray, colormap='viridis') -> Image.Image:
        """Converts a spectrogram (numpy array) to a PIL Image."""
        rgb_image_np = get_color_img(spectrum_db=spectrum_db, colormap=colormap)
        rgb_image_uint8 = (rgb_image_np * 255).astype(np.uint8)
        pil_image = Image.fromarray(rgb_image_uint8, 'RGB')
        return pil_image

    def _create_spectrogram_image(self,
                                  signal_data: np.ndarray,
                                  colormap: str = None) -> Image.Image:
        """
        Converts signal data into an RGB PIL Image of its spectrogram, optionally merging background.
        """

        colormap = self.colormap if colormap is None else colormap
        
        _,_, stft_matrix = stft(signal_data, nperseg=self.fft_size, noverlap=self.stft_overlap, window= 'hann', return_onesided=False)
        stft_matrix = np.fft.fftshift(stft_matrix, axes=0) 
        
        spectrum_db = get_power_spectrogram_db(stft_matrix=stft_matrix,
                                               window_size=self.fft_size,
                                               overlap=self.stft_overlap)
        
        return self._convert_to_pil(spectrum_db=spectrum_db, colormap=colormap)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        # 1. Load signal
        signal_file_path = self.signal_files[idx]
        
        # Extract label from filename
        filename = os.path.basename(signal_file_path)
        target_label_str = filename.split('_')[0] # Basic parsing, adjust if needed
        if target_label_str not in self.label_to_int:
            # Handle cases where the label might not be in the predefined classes,
            # for example, by assigning a default or raising an error.
            # Here, we'll skip the sample by raising an index error,
            # which is a common pattern for DataLoaders to handle problematic data.
            raise IndexError(f"Label '{target_label_str}' from file {filename} not in known classes.")
            
        target_int = self.label_to_int[target_label_str]

        # Load the entire signal
        signal_data = np.load(signal_file_path, mmap_mode='r')

        targets = target_int

        # 4. Create the main spectrogram image (merged with background)
        main_spectrogram_image = self._create_spectrogram_image(
            signal_data=signal_data,
            colormap=None,
        )

        # 6. Apply transformations to the main spectrogram image
        final_main_tensor = self.transform(main_spectrogram_image)
            
        return final_main_tensor, target_int
    
class DatasetWithPath(Dataset):
    """Wraps a ConstilationDataset to also return the file path of each sample.
    
    Useful for centroid analysis and inference pipelines where the
    source file path is needed alongside the image and label.
    """
    def __init__(self,
                 dataset_path: str = None,
                 classes: list = None,
                 transform: callable = transforms.ToTensor()):
        self._inner = ConstilationDataset(dataset_path=dataset_path,
                                          classes=classes,
                                          transform=transform)

    def __len__(self):
        return len(self._inner)

    def __getitem__(self, idx):
        image, label = self._inner[idx]
        path = self._inner.signal_images[idx]
        return image, label, path

    # Expose inner dataset attributes for convenience
    @property
    def classes(self):
        return self._inner.classes

    @property
    def label_to_int(self):
        return self._inner.label_to_int

    @property
    def int_to_label(self):
        return self._inner.int_to_label

    @property
    def signal_images(self):
        return self._inner.signal_images


class ConstilationDataset(Dataset):
    def __init__(self,
                 dataset_path: str = None,
                 classes: list = None,
                 transform: callable = transforms.ToTensor()):
        
        noiseless_image_path = os.path.join(dataset_path, 'noiseLessImg')
        noisy_image_path = os.path.join(dataset_path, 'noisyImg')
        self.signal_images = glob(os.path.join(noiseless_image_path, '*.png'))
        self.signal_images.extend(glob(os.path.join(noisy_image_path, '*.png')))
        self.classes = classes
        self.label_to_int = {label: i for i, label in enumerate(self.classes)}
        self.int_to_label = {i: label for label, i in self.label_to_int.items()}
        self.num_classes = len(self.classes)

        self.transform = transform

    def __len__(self):
        return len(self.signal_images)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        # 1. Load signal image
        signal_image_path = self.signal_images[idx]
        
        # Extract label from filename
        filename = os.path.basename(signal_image_path)
        target_label_str = filename.split('_')[0]
        # print(f"Processing file: {filename}, target label: {target_label_str}")
        if target_label_str not in self.label_to_int:
            # Handle cases where the label might not be in the predefined classes,
            # for example, by assigning a default or raising an error.
            # Here, we'll skip the sample by raising an index error,
            # which is a common pattern for DataLoaders to handle problematic data.
            raise IndexError(f"Label '{target_label_str}' from file {filename} not in known classes.")
            
        target_int = self.label_to_int[target_label_str]

        # Load the image
        signal_image = Image.open(signal_image_path).convert('RGB')
        
        # 6. Apply transformations to the image
        final_image_tensor = self.transform(signal_image)
        
        return final_image_tensor, target_int
# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create a dummy dataset for demonstration
    dummy_data_path = '../../data/own/unlabeled_10k/train/'
    classes = ['OOK', '4ASK', '8ASK', 'OQPSK', 'CPFSK', 'GFSK', '4PAM', 'DQPSK', '16PAM', 'GMSK']

    # 1. Define transformations
    # The output size of the spectrogram depends on the input signal length.
    # For batching, all tensors must have the same size. So we resize them.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 2. Instantiate the Dataset
    # dataset = SpectogramDataset(
    #     dataset_path=dummy_data_path,
    #     classes=classes,
    #     transform=transform
    # )
    dataset = ConstilationDataset(
        dataset_path=dummy_data_path,
        classes=classes,
        transform=transform
    )

    # 3. Create a DataLoader
    batch_size = 4
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # 4. Get a batch of data
    try:
        images, labels = next(iter(data_loader))

        # 5. Create a grid of images for plotting
        img_grid = torchvision.utils.make_grid(images)

        # 6. Plot the grid
        plt.figure(figsize=(10, 10))
        plt.imshow(img_grid.permute(1, 2, 0))
        label_names = [dataset.int_to_label[label.item()] for label in labels]
        plt.title(f"Batch of Spectrograms\nLabels: {label_names}")
        plt.axis('off')
        plt.show()

    except StopIteration:
        print("DataLoader is empty. This might happen if the dataset path is incorrect or no files were found.")
    except IndexError as e:
        print(f"Caught an IndexError: {e}. This can happen if a file's label is not in the provided classes list.")
    
# %%
