import pickle
import numpy as np
from pathlib import Path
from p_tqdm import p_map
from functools import partial


# -----------------------------------------------------------
# some simple helper function to process meta data
# -----------------------------------------------------------

def load_meta_data(basename, load_dir):
    load_path = Path(load_dir) / "meta_data" / f"{basename}.p"
    return pickle.load(open(load_path, "rb"))

def get_meta_data_per_image(dataset, load_dir=None, num_cpus=1):
    print("Load MetaSeg files from here: {}".format(Path(load_dir) / "meta_data"))
    meta_data = p_map(partial(load_meta_data, load_dir=load_dir), dataset.basenames, num_cpus=num_cpus)
    return meta_data

def concatenate_metrics(metrics):
    num_segments = sum(m.num_segments for m in metrics)
    all_metrics = {k: np.empty(num_segments, dtype=v.dtype) for k, v in metrics[0].metrics.items()}
    curr_idx = 0
    for m in metrics:
        next_idx = curr_idx + m.num_segments
        for k, v in m.metrics.items():
            all_metrics[k][curr_idx:next_idx] = v
        curr_idx = next_idx
    return all_metrics

def metrics_dict_as_predictors(metrics, exclude=None):
    if exclude is None:
        exclude = ["iou", "iou0", "class"]
    else:
        print("Exclude the metrics:", exclude)
    return np.array([v for k, v in metrics.items() if k not in exclude], dtype=np.float32).T.copy()



