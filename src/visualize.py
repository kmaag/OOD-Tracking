import cv2
from skimage import measure as ms
import numpy as np
from matplotlib import cm
from PIL import Image
from .datasets import cityscapes_color_palette_train_ids, cityscapes_class_names_to_train_id


def plot_heatmap(numpy_array):
    color_map = cm.get_cmap('viridis', 8)
    img_array = color_map(numpy_array)
    return Image.fromarray((img_array[..., :3] * 255).astype(np.uint8))
    
def cityscapes_colorize(arr):
    imc = Image.fromarray(arr.astype(np.uint8)).convert('P')
    imc.putpalette(cityscapes_color_palette_train_ids)
    return imc

def road_as_roi(arr):
    mask_roi = np.array(arr==cityscapes_class_names_to_train_id["road"], dtype="uint8") * 255
    return Image.fromarray(mask_roi)
