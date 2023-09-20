import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt

def shuffle_patches(image, label, patch_size):
    # Get the number of patches per row/column
    num_patches = image.size[0] // patch_size
    
    # Extract patches from image and label
    image_patches = [np.array(image.crop((j*patch_size, i*patch_size, (j+1)*patch_size, (i+1)*patch_size))) for i in range(num_patches) for j in range(num_patches)]
    
    label_patches = [np.array(label.crop((j*patch_size, i*patch_size, (j+1)*patch_size, (i+1)*patch_size))) for i in range(num_patches) for j in range(num_patches)]
    
    # Shuffle the patches
    combined_patches = list(zip(image_patches, label_patches))
    np.random.shuffle(combined_patches)
    image_patches, label_patches = zip(*combined_patches)
    
    # Create a new image by concatenating the shuffled patches
    new_image = Image.fromarray(np.concatenate([np.concatenate(image_patches[j*num_patches:(j+1)*num_patches], axis=1) for j in range(num_patches)], axis=0))
    new_label = Image.fromarray(np.concatenate([np.concatenate(label_patches[j*num_patches:(j+1)*num_patches], axis=1) for j in range(num_patches)], axis=0))
    
    return new_image, new_label