import os
import random
import concurrent.futures
import matplotlib.colors as colors
from pathlib import Path

from src.tracking_utils import segments_tracking_i, visualize_tracking_i, eval_tracking_seq, merge_results



# -----------------------------------------------------------
# tracking segments
# -----------------------------------------------------------

def tracking_segments(dataset, io_root, cross_val, num_cpus=1):

    if num_cpus == 1:
        for seq in dataset.sequences:
            segments_tracking_i(io_root, dataset.all_basenames, seq, cross_val)
    else:
        p_args = [ (io_root, dataset.all_basenames, seq, cross_val) for seq in dataset.sequences ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus) as executer:
            executer.map(segments_tracking_i, *zip(*p_args))


# -----------------------------------------------------------
# visualize tracking 
# -----------------------------------------------------------

def visualize_tracking_per_image(dataset, io_root, cross_val, num_cpus=1):

    colors_list_tmp = list(colors._colors_full_map.values()) 
    colors_list = []
    for color in colors_list_tmp:
        if len(color) == 7:
            colors_list.append(color)
    random.shuffle(colors_list)

    for seq in dataset.sequences:
        if not os.path.exists( Path(io_root) / f"tracking_img_{cross_val}" / seq ):
            os.makedirs( Path(io_root) / f"tracking_img_{cross_val}" / seq )

    if num_cpus == 1:
        for idx in range(len(dataset.all_basenames)):
            visualize_tracking_i(io_root, cross_val, dataset.all_basenames[idx], dataset.all_images[idx], colors_list)
    else:
        p_args = [ (io_root, cross_val, dataset.all_basenames[idx], dataset.all_images[idx], colors_list) for idx in range(len(dataset.all_basenames)) ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
            executor.map(visualize_tracking_i, *zip(*p_args))

# -----------------------------------------------------------
# evaluate tracking 
# -----------------------------------------------------------

def evaluate_tracking(dataset, dataset_name, io_root, cross_val, num_cpus=1):

    if not os.path.exists(Path(io_root) / f"tracking_eval_{cross_val}"): 
        os.makedirs(Path(io_root) / f"tracking_eval_{cross_val}")

    if num_cpus == 1:
        for seq in dataset.sequences:
            eval_tracking_seq(seq, io_root, cross_val, dataset.all_basenames, dataset.all_images, dataset_name)
    else:
        p_args = [ (seq, io_root, cross_val, dataset.all_basenames, dataset.all_images, dataset_name) for seq in dataset.sequences ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus) as executer:
            executer.map(eval_tracking_seq, *zip(*p_args))
    
    merge_results(io_root, cross_val, dataset.sequences)