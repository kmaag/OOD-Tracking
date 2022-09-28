import numpy as np
from .utils import concatenate_metrics, metrics_dict_as_predictors
from scipy.ndimage.measurements import label
from PIL import Image
from easydict import EasyDict
from pathlib import Path

def prepare_meta_training_data(meta_data, meta_target_key="iou"):
    metrics = concatenate_metrics(meta_data)
    x = metrics_dict_as_predictors(metrics)
    y = metrics[meta_target_key]
    x = x[~np.isnan(y)]
    mu = x.mean(0)
    sigma = x.std(0) + 1e-10
    x = (x - mu) / sigma
    y = y[~np.isnan(y)]
    return x, y, mu, sigma

def meta_prediction(meta_model, metrics, mu=None, sigma=None, standardize_predictors=True):
    x = metrics_dict_as_predictors(metrics)
    if standardize_predictors:
        try:
            x = (x - mu) / sigma
        except TypeError:
            print("Oops! No valid mu and sigma for standardization were provided. Please, try again...")
    x = np.nan_to_num(x, nan=0.0)
    return meta_model.predict(x)


def anomaly_instances_from_mask(segmentation: np.ndarray, label_pixel_gt: np.ndarray):
    """connected components"""
    structure = np.ones((3, 3), dtype=np.int)
    anomaly_instances, n_anomaly = label(label_pixel_gt, structure)
    anomaly_seg_pred, n_seg_pred = label(segmentation, structure)
    return anomaly_instances, anomaly_seg_pred


def segment_wise_metrics_worker(ood_pred_load_path, gt_load_path, anomaly_id=254):
    ood_pred = np.array(Image.open(ood_pred_load_path))
    gt = np.array(Image.open(gt_load_path))
    anomaly_instances, anomaly_seg_pred = anomaly_instances_from_mask((ood_pred==anomaly_id), (gt==anomaly_id))
    return segment_metrics(anomaly_instances, anomaly_seg_pred)


def segment_metrics(anomaly_instances, anomaly_seg_pred, iou_thresholds=np.linspace(0.25, 0.75, 11, endpoint=True)):
    """
    function that computes the segments metrics based on the adjusted IoU
    anomaly_instances: (numpy array) anomaly instance annoation
    anomaly_seg_pred: (numpy array) anomaly instance prediction
    iou_threshold: (float) threshold for true positive
    """

    """Loop over ground truth instances"""
    sIoU_gt = []
    size_gt = []

    for i in np.unique(anomaly_instances[anomaly_instances>0]):
        tp_loc = anomaly_seg_pred[anomaly_instances == i]
        seg_ind = np.unique(tp_loc[tp_loc != 0])

        """calc area of intersection"""
        intersection = len(tp_loc[np.isin(tp_loc, seg_ind)])
        adjustment = len(
            anomaly_seg_pred[np.logical_and(~np.isin(anomaly_instances, [0, i]), np.isin(anomaly_seg_pred, seg_ind))])

        adjusted_union = np.sum(np.isin(anomaly_seg_pred, seg_ind)) + np.sum(
            anomaly_instances == i) - intersection - adjustment
        sIoU_gt.append(intersection / adjusted_union)
        size_gt.append(np.sum(anomaly_instances == i))

    """Loop over prediction instances"""
    sIoU_pred = []
    size_pred = []
    prec_pred = []
    for i in np.unique(anomaly_seg_pred[anomaly_seg_pred>0]):
        tp_loc = anomaly_instances[anomaly_seg_pred == i]
        seg_ind = np.unique(tp_loc[tp_loc != 0])
        intersection = len(tp_loc[np.isin(tp_loc, seg_ind)])
        adjustment = len(
            anomaly_instances[np.logical_and(~np.isin(anomaly_seg_pred, [0, i]), np.isin(anomaly_instances, seg_ind))])
        adjusted_union = np.sum(np.isin(anomaly_instances, seg_ind)) + np.sum(
            anomaly_seg_pred == i) - intersection - adjustment
        sIoU_pred.append(intersection / adjusted_union)
        size_pred.append(np.sum(anomaly_seg_pred == i))
        prec_pred.append(intersection / np.sum(anomaly_seg_pred == i))

    sIoU_gt = np.array(sIoU_gt)
    sIoU_pred = np.array(sIoU_pred)
    size_gt = np.array((size_gt))
    size_pred = np.array(size_pred)
    prec_pred = np.array(prec_pred)

    """create results dictionary"""
    results = EasyDict(sIoU_gt=sIoU_gt, sIoU_pred=sIoU_pred, size_gt=size_gt, size_pred=size_pred, prec_pred=prec_pred)
    for t in iou_thresholds:
        results["tp_" + str(int(t*100))] = np.count_nonzero(sIoU_gt >= t)
        results["fn_" + str(int(t*100))] = np.count_nonzero(sIoU_gt < t)
        results["fp_" + str(int(t*100))] = np.count_nonzero(prec_pred < t)

    return results


def aggregate_segment_metrics(frame_results: list, iou_thresholds=np.linspace(0.25, 0.75, 11, endpoint=True), tmp_path = None, tfs= None, entr_thresh=None, mfs=None):

    sIoU_gt_mean = sum(np.sum(r.sIoU_gt) for r in frame_results) / sum(len(r.sIoU_gt) for r in frame_results)
    sIoU_pred_mean = sum(np.sum(r.sIoU_pred) for r in frame_results) / sum(len(r.sIoU_pred) for r in frame_results)
    prec_pred_mean = sum(np.sum(r.prec_pred) for r in frame_results) / sum(len(r.prec_pred) for r in frame_results)
    ag_results = {"tp_mean" : 0., "fn_mean" : 0., "fp_mean" : 0., "f1_mean" : 0.,
                  "sIoU_gt" : sIoU_gt_mean, "sIoU_pred" : sIoU_pred_mean, "prec_pred": prec_pred_mean}
    for t in iou_thresholds:
        tp = sum(r["tp_" + str(int(t*100))] for r in frame_results)
        fn = sum(r["fn_" + str(int(t*100))] for r in frame_results)
        fp = sum(r["fp_" + str(int(t*100))] for r in frame_results)
        f1 = (2 * tp) / (2 * tp + fn + fp)
        if t in [0.25, 0.50, 0.75]:
            ag_results["tp_" + str(int(t * 100))] = tp
            ag_results["fn_" + str(int(t * 100))] = fn
            ag_results["fp_" + str(int(t * 100))] = fp
            ag_results["f1_" + str(int(t * 100))] = f1
        ag_results["tp_mean"] += tp
        ag_results["fn_mean"] += fn
        ag_results["fp_mean"] += fp
        ag_results["f1_mean"] += f1

    ag_results["tp_mean"] /= len(iou_thresholds)
    ag_results["fn_mean"] /= len(iou_thresholds)
    ag_results["fp_mean"] /= len(iou_thresholds)
    ag_results["f1_mean"] /= len(iou_thresholds)
    print("--- averaged over sIoU thresholds", iou_thresholds)
    print("Number of TPs  : {:8.2f}".format(ag_results["tp_mean"]))
    print("Number of FNs  : {:8.2f}".format(ag_results["fn_mean"]))
    print("Number of FPs  : {:8.2f}".format(ag_results["fp_mean"]))
    print("Mean F1 score  : {:8.2f} %".format(ag_results["f1_mean"]*100))
    
        
    # if mfs is not None:
    #     mfs.add_f1_score(ag_results["f1_mean"]*100)

    # if tfs is not None:
    #     tfs.add_f1_score(np.round(entr_thresh, decimals=2), ag_results["f1_mean"]*100)
