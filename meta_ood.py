import cv2
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
from easydict import EasyDict as edict
from functools import partial
from p_tqdm import p_map, p_umap
from sklearn.metrics import average_precision_score, roc_auc_score
from src.metaseg.metrics import Metrics
from src.metaseg.utils import get_meta_data_per_image
from src.metaseg.compute import prepare_meta_training_data, meta_prediction, segment_wise_metrics_worker, aggregate_segment_metrics


# -----------------------------------------------------------
# some simple helper functions
# -----------------------------------------------------------

def thresholding_prediction(score_map, threshold):
    """thresholding on score map"""
    prediction = np.zeros(score_map.shape, dtype="uint8")
    prediction[score_map > threshold] = 1
    return prediction
    
def smooth_roi(roi_array, size=70):
    """morphological image operations to region of interest"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    smoothed_roi = cv2.morphologyEx(roi_array, cv2.MORPH_OPEN, kernel)
    smoothed_roi = cv2.morphologyEx(roi_array, cv2.MORPH_CLOSE, kernel)
    return smoothed_roi

def binarize_array(array, id):
    """standardize target format by binarizing"""
    binary = np.zeros(array.shape, dtype="uint8")
    binary[array == id] = 1
    return binary


# -----------------------------------------------------------
# compute and save OOD prediction masks
# -----------------------------------------------------------

def compute_and_save_ood_prediction(dataset, io_root, ood_score_threshold=0.8, smoother=None, num_cpus=5):
    
    basenames = dataset.all_basenames 
    score_load_paths = [Path(io_root) / "entropy" / f"{basename}.npy" for basename in basenames]
    roi_load_paths = [Path(io_root) / "road_roi" / f"{basename}.png" for basename in basenames]
    ood_pred_save_paths = [Path(io_root) / "ood_prediction" / f"{basename}.png" for basename in basenames]
    roi_save_paths = [Path(io_root) / "roi" / f"{basename}.png" for basename in basenames]
    
    """let the workers work"""
    print("predict and save ood prediction masks")
    p_map(partial(compute_ood_prediction_worker, ood_score_threshold=ood_score_threshold, smoother=smoother),
          score_load_paths, ood_pred_save_paths, roi_load_paths, roi_save_paths, num_cpus=num_cpus)
    

def compute_ood_prediction_worker(score_load_path, ood_pred_save_path, roi_load_path=None,
                                  roi_smoothed_save_path=None, ood_score_threshold=0.8, smoother=None):
    score = np.load(score_load_path)
    pred = thresholding_prediction(score, ood_score_threshold)
    if roi_load_path is not None:
        roi = np.array(Image.open(roi_load_path))
        if smoother is not None:
            roi = smoother.smooth(roi)
        if roi_smoothed_save_path is not None:
            Path(roi_smoothed_save_path).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(roi).save(roi_smoothed_save_path)
        pred[roi==0] = 0

    Path(ood_pred_save_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pred*254).save(ood_pred_save_path)


# -----------------------------------------------------------
# compute and save meta classification input data
# -----------------------------------------------------------
    
def compute_and_save_meta_data(dataset, io_root, num_cpus=5):
    
    basenames = dataset.all_basenames 
    targets_load_paths = dataset.targets_semantic_ood 
    
    probs_load_paths = [Path(io_root) / "probs" / f"{basename}.npy" for basename in basenames]
    ood_pred_load_paths = [Path(io_root) / "ood_prediction" / f"{basename}.png" for basename in basenames]
    meta_data_save_paths = [Path(io_root) / "meta_data" / f"{basename}.p" for basename in basenames]
    
    """let the workers work"""
    print("compute and save meta data")
    p_map(partial(compute_and_save_meta_data_worker, ood_target_id=dataset.ood_id),
          probs_load_paths, ood_pred_load_paths, meta_data_save_paths, targets_load_paths, num_cpus=num_cpus)
    

def compute_and_save_meta_data_worker(probs_load_path, ood_pred_load_path, meta_data_save_path,
                                      targets_load_path=None, ood_target_id=None):
    """prepare input data"""
    probs = np.load(probs_load_path).astype("float32")
    pred = binarize_array(np.array(Image.open(ood_pred_load_path)), id=254)
    if targets_load_path is not None and ood_target_id is not None:
        target = np.array(Image.open(targets_load_path))
        gt = binarize_array(target, ood_target_id)
    else:
        gt = None

    """compute meta data and save"""
    meta_data_full = Metrics(probs=probs, labels=pred, gt_labels=gt, num_classes=2)
    meta_data_slim = edict({"metrics": {k: v[1:] for k, v in meta_data_full.metrics.items()},
                            "segments": meta_data_full.segments,
                            "num_segments": meta_data_full.num_segments,
                            "boundary_mask": meta_data_full.boundary_mask
                            })
    meta_data_save_path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(meta_data_slim, open(meta_data_save_path, "wb"))


# -----------------------------------------------------------
# training meta classification
# -----------------------------------------------------------

def train_meta_classifier(training_dataset, meta_model, io_root, num_cpus=1):
    
    """load meta data"""
    meta_data = get_meta_data_per_image(training_dataset, io_root, num_cpus)
    
    """prepare meta data and fit meta classification model"""
    x, y, mu, sigma = prepare_meta_training_data(meta_data, meta_target_key="iou0")
    x =np.nan_to_num(x, nan=0.0) # change nan to zero
    classifier = meta_model.fit(x, y)
    print("training auroc : {:.2f} %".format(roc_auc_score(y, classifier.predict_proba(x)[:, 1]) * 100))
    print("training auprc : {:.2f} %".format(average_precision_score(~y, classifier.predict_proba(x)[:, 0]) * 100))

    """save the trained meta classification model"""
    meta_model_dict = {"classifier": classifier, "mu": mu, "sigma": sigma}
    meta_model_save_path = Path(io_root) / "meta_classifier" / f"{type(meta_model).__name__}.p"
    meta_model_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_model_save_path, 'wb') as file:
        pickle.dump(meta_model_dict, file)
    print("Saved meta classification model:", meta_model_save_path)


# -----------------------------------------------------------
# kick false positive OOD predictions with meta classifier
# -----------------------------------------------------------

def meta_kick(dataset, meta_model_name, io_root, io_root_train, cross_val, num_cpus=5):
    
    """load meta model"""
    meta_model_load_path = Path(io_root_train) / "meta_classifier" / f"{meta_model_name}.p"

    with open(meta_model_load_path, 'rb') as file:
        model_dict = pickle.load(file)
        meta_classifier = model_dict["classifier"]
        mu = model_dict["mu"]
        sigma = model_dict["sigma"]
    
    """load and save paths"""
    basenames = dataset.all_basenames
    
    ood_pred_load_paths = [Path(io_root) / "ood_prediction" / f"{basename}.png" for basename in basenames]
    meta_data_load_paths = [Path(io_root) / "meta_data" / f"{basename}.p" for basename in basenames]
    meta_classified_save_paths = [Path(io_root) / f"ood_prediction_meta_classified_{cross_val}" / f"{basename}.png" for basename
                                  in basenames]

    """let the workers work"""    
    pool_args = [(
        meta_classifier, meta_data_load_paths[i], mu, sigma, ood_pred_load_paths[i], meta_classified_save_paths[i]
    ) for i in range(len(basenames))]
    print("kick false positive OOD prediction")
    p_umap(lambda arg: meta_kick_worker(*arg), pool_args, num_cpus=num_cpus)


def meta_kick_worker(meta_classifier, meta_data_load_path, mu, sigma, ood_pred_load_path=None,
                     meta_classified_save_path=None):
    
    """load meta inference data"""
    with open(meta_data_load_path, 'rb') as file:
        meta_data = pickle.load(file)
    metrics = meta_data.metrics
    segments = meta_data.segments
    num_segments = meta_data.num_segments
    # boundary_mask = meta_data.boundary_mask
    """meta classification"""
    if num_segments > 0:
        y_hat = meta_prediction(meta_classifier, metrics, mu, sigma)

    """save meta classified ood prediction"""
    if ood_pred_load_path is not None and meta_classified_save_path is not None:
        meta_classified_save_path.parent.mkdir(parents=True, exist_ok=True)
        ood_pred = np.array(Image.open(ood_pred_load_path))
        for seg_id in range(1, num_segments + 1):
            if y_hat[seg_id - 1]:
                ood_pred[segments == seg_id] = 0
        Image.fromarray(ood_pred).save(meta_classified_save_path)

    
# -----------------------------------------------------------
# OOD meta classification evaluation
# -----------------------------------------------------------

def segment_wise_evaluation(dataset, io_root, cross_val, tfs1=None, tfs2=None, ent=None, mfs=None, num_cpus=5):
    basenames = dataset.basenames
    ood_pred_load_paths = [Path(io_root) / "ood_prediction" / f"{basename}.png" for basename in basenames]
    
    meta_classified_load_paths = [Path(io_root) / f"ood_prediction_meta_classified_{cross_val}" /f"{basename}.png" for basename
                                  in basenames]
    targets_load_paths = dataset.targets_semantic_ood
    targets_load_paths = list(filter(lambda ele:ele is not None, targets_load_paths))

    print("segment-wise evaluation")
    print("before meta classification")
    results = p_umap(segment_wise_metrics_worker, ood_pred_load_paths, targets_load_paths, num_cpus=num_cpus)
    aggregate_segment_metrics(results, tmp_path=Path(io_root), tfs=tfs1, entr_thresh = ent)
    print("\nafter meta classification")
    results = p_umap(segment_wise_metrics_worker, meta_classified_load_paths, targets_load_paths, num_cpus=num_cpus)
    aggregate_segment_metrics(results, tmp_path=Path(io_root), tfs=tfs2, entr_thresh = ent, mfs=mfs)
