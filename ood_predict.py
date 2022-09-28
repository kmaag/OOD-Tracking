import torch
import numpy as np

import torch.nn.functional as F

from PIL import Image
from pathlib import Path
from tqdm import tqdm
from easydict import EasyDict as edict
from yaml import load
from src.semseg_models.deepv3 import DeepWV3Plus
from torchvision.transforms import Compose, ToTensor, Normalize
from src.visualize import plot_heatmap, cityscapes_colorize, road_as_roi
from functools import partial
from p_tqdm import p_map
from multiprocessing import Pool
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, auc
from src.utils import counts_array_to_data_list



# -----------------------------------------------------------
# initialize model
# -----------------------------------------------------------

def DeepLabV3Plus_WideResNet38(num_classes=19):
    model = DeepWV3Plus(num_classes, trunk='WideResnet38')
    return torch.nn.DataParallel(model)

def init_DeepLabV3Plus_WideResNet38(checkpoint_path, num_classes=19):
    torch.cuda.empty_cache()
    print("Load PyTorch model", end="", flush=True)
    network = DeepLabV3Plus_WideResNet38(num_classes)
    print("... ok")
    if checkpoint_path is not None:
        print("Checkpoint: %s" % checkpoint_path, end="", flush=True)
        network.load_state_dict(torch.load(checkpoint_path)['state_dict'], strict=False)
        print(" ... ok\n")
    return network.cuda().eval()


# -----------------------------------------------------------
# compute functions
# -----------------------------------------------------------

def compute_softmax(model, image):
    x = image.cuda()
    with torch.no_grad():
        y = model(x)
    softmax = F.softmax(y, 1).data.cpu()
    del x, y
    return torch.squeeze(softmax)

def compute_softmax_entropy(model, image, normalize=True):
    softmax = compute_softmax(model, image)
    entropy = torch.sum(-softmax * torch.log(softmax), dim=0)
    if normalize:
        entropy = torch.div(entropy, torch.log(torch.tensor(softmax.shape[0])))
    return entropy


# -----------------------------------------------------------
# compute and save data
# -----------------------------------------------------------

def compute_and_save_softmax_probs(dataset, checkpoint_path, save_root):    
    """create folders to save files"""
    sub_folder = "probs"
    for sequence in dataset.sequences:
        (Path(save_root) / sub_folder / sequence).mkdir(parents=True, exist_ok=True)
    
    net = init_DeepLabV3Plus_WideResNet38(checkpoint_path)
    transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    print("compute and save softmax probabilities")
    for image, basename in tqdm(zip(dataset, dataset.all_basenames), total=len(dataset.all_basenames)):
        input_image = transform(image[0]).unsqueeze_(0)
        softmax = compute_softmax(net, input_image).numpy()
        softmax_save_path = Path(save_root) / sub_folder / f"{basename}"
        np.save(softmax_save_path, softmax.astype("float16"))
    del net
    
def compute_and_save_softmax_entropy(dataset, checkpoint_path, save_root):
    """create folders to save files"""
    for sequence in dataset.sequences:
        (Path(save_root) / "entropy" / sequence).mkdir(parents=True, exist_ok=True)

    net = init_DeepLabV3Plus_WideResNet38(checkpoint_path)
    transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    print("compute and save softmax entropy")
    for image, basename in tqdm(zip(dataset, dataset.all_basenames), total=len(dataset.all_basenames)):
        input_image = transform(image[0]).unsqueeze_(0)
        entropy = compute_softmax_entropy(net, input_image)
        entropy_save_path = Path(save_root) / "entropy" / f"{basename}"
        np.save(entropy_save_path, entropy.numpy().astype("float16"))
    del net


# -----------------------------------------------------------
# load and visualize data
# -----------------------------------------------------------
    
def visualize_segmentation(dataset, load_dir, semantic=False, cityscapes_color=False, road_roi=False, num_cpus=5):
    print("visualize semantic segmentation")
    basenames = dataset.all_basenames
    load_paths = [Path(load_dir) / "probs" / f"{basename}.npy" for basename in basenames]
    p_map(partial(visualize_segmentation_worker, semantic=semantic, cityscapes_color=cityscapes_color, road_roi=road_roi), load_paths, num_cpus=num_cpus)
    
def visualize_segmentation_worker(load_path, semantic=False, cityscapes_color=False, road_roi=False):
    probs = np.load(load_path)
    semantic_mask = np.argmax(probs, axis=0)
    seg_parent = Path(str(load_path.parent).replace("probs", "segmentation"))
    roi_parent = Path(str(load_path.parent).replace("probs", "road_roi"))
    if semantic:
        semantic_save_path = seg_parent / f"{load_path.stem}.png"
        semantic_save_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(semantic_mask.astype("uint8")).save(semantic_save_path)
    if cityscapes_color:
        semantic_color_save_path = seg_parent / f"{load_path.stem}_color.png"
        semantic_color_save_path.parent.mkdir(parents=True, exist_ok=True)
        cityscapes_colorize(semantic_mask).save(semantic_color_save_path)
    if road_roi:
        roi_save_path = roi_parent / f"{load_path.stem}.png"
        roi_save_path.parent.mkdir(parents=True, exist_ok=True)
        road_as_roi(semantic_mask).save(roi_save_path)
        
def visualize_ood_score_prediction(dataset, load_dir, ood_heatmap=False, num_cpus=5):
    print("visualize OOD prediction")
    basenames = dataset.basenames
    load_paths = [Path(load_dir) / "entropy" / f"{basename}.npy" for basename in basenames]
    p_map(partial(visualize_ood_score_prediction_worker, ood_heatmap=ood_heatmap), load_paths, num_cpus=num_cpus)
    
def visualize_ood_score_prediction_worker(load_path, ood_heatmap=False):
    if ood_heatmap:
        entropy = np.load(load_path)
        entropy_heatmap_save_path = str(load_path).replace(".npy", "_heatmap.png")
        plot_heatmap(entropy).save(entropy_heatmap_save_path)
        
        
# -----------------------------------------------------------
# evaluate ood scores
# -----------------------------------------------------------

def pixel_wise_evaluation(dataset, load_dir, num_cpus=5):
    print("evaluate OOD scores")
    basenames = dataset.basenames
    load_paths = [Path(load_dir) / "entropy" / f"{basename}.npy" for basename in basenames]
    target_paths = dataset.targets_semantic_ood
    target_paths = list(filter(lambda ele:ele is not None, target_paths))
    pool_args = [(load_paths[i], target_paths[i]) for i in range(len(load_paths))]
        
    with Pool(num_cpus) as pool:
        results = pool.starmap(pixel_wise_metrics_i, tqdm(pool_args, total=pool_args.__len__()), chunksize = 4)
    aggregate_pixel_metrics(results, load_dir)
    
    
    
def get_counts(load_paths, target_paths, num_bins=100):
    bins = np.linspace(start=0, stop=1, num=num_bins + 1)
    counts = {"in": np.zeros(num_bins, dtype="int64"), "out": np.zeros(num_bins, dtype="int64")}
    for load_path, target_path in tqdm(zip(load_paths, target_paths)):
        entropy = np.load(load_path)
        gt = np.array(Image.open(target_path))
        counts["in"] += np.histogram(entropy[gt == 0], bins=bins, density=False)[0]
        counts["out"] += np.histogram(entropy[gt == 254], bins=bins, density=False)[0]
    return counts

def pixel_wise_metrics_i(ood_heat_load_path, gt_load_path):
    ood_heat = np.load(ood_heat_load_path)
    gt = np.array(Image.open(gt_load_path))

    return pixel_metrics(ood_heat, gt)


def pixel_metrics(ood_heat, gt, num_bins=100, in_label=0, out_label=254):
    bins = np.linspace(start=0, stop=1, num=num_bins + 1)
    counts = {"in": np.zeros(num_bins, dtype="int32"), "out": np.zeros(num_bins, dtype="int32")}
    counts["in"] += np.histogram(ood_heat[gt == in_label], bins=bins, density=False)[0]
    counts["out"] += np.histogram(ood_heat[gt == out_label], bins=bins, density=False)[0]
    return counts


def aggregate_pixel_metrics(frame_results: list, tmp_path):
    print("Could take a moment...")
    num_bins = len(frame_results[0]["out"])
    counts = {"in": np.zeros(num_bins, dtype="int64"), "out": np.zeros(num_bins, dtype="int64")}
    for r in frame_results:
        counts["in"] += r["in"]
        counts["out"] += r["out"]
    fpr, tpr, _, auroc = calc_sensitivity_specificity(counts, balance=True)
    fpr95 = fpr[(np.abs(tpr - 0.95)).argmin()]
    _, _, _, auprc = calc_precision_recall(counts)
    # print("AUROC : {:6.2f} %".format(auroc*100))
    print("FPR95 : {:6.2f} %".format(fpr95*100))
    print("AUPRC : {:6.2f} %".format(auprc*100))
        


def calc_precision_recall(data, balance=False):
    if balance:
        x1 = counts_array_to_data_list(np.array(data["in"]), 1e+5)
        x2 = counts_array_to_data_list(np.array(data["out"]), 1e+5)
    else:
        ratio_in = np.sum(data["in"]) / (np.sum(data["in"]) + np.sum(data["out"]))
        ratio_out = 1 - ratio_in
        x1 = counts_array_to_data_list(np.array(data["in"]), 1e+7 * ratio_in)
        x2 = counts_array_to_data_list(np.array(data["out"]), 1e+7 * ratio_out)
    probas_pred1 = np.array(x1) / 100
    probas_pred2 = np.array(x2) / 100
    y_true = np.concatenate((np.zeros(len(probas_pred1)), np.ones(len(probas_pred2))))
    y_scores = np.concatenate((probas_pred1, probas_pred2))
    return precision_recall_curve(y_true, y_scores) + (average_precision_score(y_true, y_scores), )


def calc_sensitivity_specificity(data, balance=False):
    if balance:
        x1 = counts_array_to_data_list(np.array(data["in"]), max_size=1e+5)
        x2 = counts_array_to_data_list(np.array(data["out"]), max_size=1e+5)
    else:
        x1 = counts_array_to_data_list(np.array(data["in"]))
        x2 = counts_array_to_data_list(np.array(data["out"]))
    probas_pred1 = np.array(x1) / 100
    probas_pred2 = np.array(x2) / 100
    y_true = np.concatenate((np.zeros(len(probas_pred1)), np.ones(len(probas_pred2)))).astype("uint8")
    y_scores = np.concatenate((probas_pred1, probas_pred2))
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    return fpr, tpr, thresholds, auc(fpr, tpr)



# -----------------------------------------------------------
# entire routines
# -----------------------------------------------------------

def routine_segmentation(dataset, checkpoint_path, save_root, cityscapes_color=False, save_probs=False):    
    """create folders to save files"""
    sub_folders = edict({"seg": "segmentation", "road_roi": "road_roi", "probs": "probs"})
    for _, sub_folder in sub_folders.items():
        for sequence in dataset.sequences:
            (Path(save_root) / sub_folder / sequence).mkdir(parents=True, exist_ok=True)
    
    net = init_DeepLabV3Plus_WideResNet38(checkpoint_path)
    transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    print("Start semantic segmentation")
    for image, basename in tqdm(zip(dataset, dataset.all_basenames), total=len(dataset)):
        input_image = transform(image).unsqueeze_(0)
        
        """compute and save softmax probabilities"""
        softmax = compute_softmax(net, input_image)
        if save_probs:
            softmax_save_path = Path(save_root) / sub_folders.probs / f"{basename}"
            np.save(softmax_save_path, softmax.numpy().astype("float16"))
            
        """compute and save semantic segmentation mask"""
        semantic_mask = torch.argmax(softmax, dim=0)
        semantic_save_path = Path(save_root) / sub_folders.seg / f"{basename}.png"
        Image.fromarray(semantic_mask.numpy().astype("uint8")).save(semantic_save_path)
        if cityscapes_color:
            semantic_color_save_path = Path(save_root) / sub_folders.seg / f"{basename}_color.png"
            cityscapes_colorize(semantic_mask.numpy()).save(semantic_color_save_path)
            
        """compute and save region-of-interest mask"""
        road_roi_save_path = Path(save_root) / sub_folders.road_roi / f"{basename}.png"
        road_as_roi(semantic_mask).save(road_roi_save_path)
    del net
        
def routine_entropy(dataset, checkpoint_path, save_root, save_entropy=False):
    """create folders to save files"""
    sub_folder = "entropy"
    for sequence in dataset.sequences:
        (Path(save_root) / sub_folder / sequence).mkdir(parents=True, exist_ok=True)

    net = init_DeepLabV3Plus_WideResNet38(checkpoint_path)
    transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    print("Start entropy")
    for image, basename in tqdm(zip(dataset, dataset.all_basenames), total=len(dataset)):
        input_image = transform(image).unsqueeze_(0)

        """compute and save softmax entropy"""
        entropy = compute_softmax_entropy(net, input_image).numpy()
        if save_entropy:
            entropy_save_path = Path(save_root) / sub_folder / f"{basename}"
            np.save(entropy_save_path, entropy.astype("float16"))
        
        """compute and save softmax entropy heatmap"""
        entropy_heatmap_save_path = Path(save_root) / "entropy" / f"{basename}_heatmap.png"
        plot_heatmap(entropy).save(entropy_heatmap_save_path)
    del net
