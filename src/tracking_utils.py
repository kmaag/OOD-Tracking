import os
import cv2
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from scipy.stats import linregress
from skimage import measure as ms

from src.datasets import Cityscapes


def nearest_neighbour(comp_n, eps_near):

    for j1 in np.unique(comp_n)[1:]:
        tmp_j1 = np.zeros(comp_n.shape)
        tmp_j1[comp_n == j1] = 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (int(eps_near*2)+1, int(eps_near*2)+1))
        dilation = cv2.dilate(tmp_j1, kernel, iterations=1)

        for j2 in np.unique(comp_n)[1:]:
            if j1 != j2 and np.sum(comp_n[dilation == 1] == j2) > 0:
                comp_n[comp_n == j2] = j1

    return comp_n


def shift(i, imx, imy, time_series_components, comp_n, percentage, epsilon):

    comp_n_1_tmp = time_series_components[1].copy()
    x_y_indices = np.sum(np.asarray(np.where(comp_n_1_tmp == i)), axis=1)
    mean_x_n_1 = x_y_indices[1] / np.sum(comp_n_1_tmp == i)
    mean_y_n_1 = x_y_indices[0] / np.sum(comp_n_1_tmp == i)

    comp_n_2_tmp = time_series_components[2].copy()
    x_y_indices = np.sum(np.asarray(np.where(comp_n_2_tmp == i)), axis=1)
    mean_x_n_2 = x_y_indices[1] / np.sum(comp_n_2_tmp == i)
    mean_y_n_2 = x_y_indices[0] / np.sum(comp_n_2_tmp == i)

    # print("shift overlap")

    x_shift = int(mean_x_n_1 - mean_x_n_2)
    y_shift = int(mean_y_n_1 - mean_y_n_2)

    shifted_comp = np.zeros((imx, imy))
    shifted_comp[comp_n_1_tmp == i] = 1
    if x_shift < 0:
        shifted_comp = np.concatenate(
            (np.zeros((-x_shift, imy)), shifted_comp), axis=0)
        shifted_comp = shifted_comp[:imx, :]
    elif x_shift > 0:
        shifted_comp = np.concatenate(
            (shifted_comp, np.zeros((x_shift, imy))), axis=0)
        shifted_comp = shifted_comp[x_shift:, :]
    if y_shift < 0:
        shifted_comp = np.concatenate(
            (np.zeros((imx, -y_shift)), shifted_comp), axis=1)
        shifted_comp = shifted_comp[:, :imy]
    elif y_shift > 0:
        shifted_comp = np.concatenate(
            (shifted_comp, np.zeros((imx, y_shift))), axis=1)
        shifted_comp = shifted_comp[:, y_shift:]

    max_iou = 0
    max_index = 0

    for j in np.unique(comp_n)[1:]:

        intersection = np.sum(np.logical_and(shifted_comp == 1, comp_n == j))
        union = np.sum(np.logical_or(shifted_comp == 1, comp_n == j))

        if union > 0:
            if (intersection / union) >= percentage:
                time_series_components[0, comp_n == j] = i
                comp_n[comp_n == j] = 0
                #print("geometric center and overlapping match with percentage: IoU, index i and j",intersection / union, i, j)

            if (intersection / union) > max_iou:
                max_iou = (intersection / union)
                max_index = j

    if max_iou > 0:
        time_series_components[0, comp_n == max_index] = i
        comp_n[comp_n == max_index] = 0
        #print("geometric center and overlapping match: IoU, index i and j", max_iou, i, max_index)

    else:
        # print("shift distance")

        min_distance = 2000
        min_index = 0

        dir_n_2_n_1_x = mean_x_n_1 - mean_x_n_2
        dir_n_2_n_1_y = mean_y_n_1 - mean_y_n_2

        for j in np.unique(comp_n)[1:]:

            comp_n_tmp = comp_n.copy()
            x_y_indices = np.sum(np.asarray(np.where(comp_n_tmp == j)), axis=1)
            mean_x_n = x_y_indices[1] / np.sum(comp_n_tmp == j)
            mean_y_n = x_y_indices[0] / np.sum(comp_n_tmp == j)

            dir_n_1_n_x = mean_x_n - mean_x_n_1
            dir_n_1_n_y = mean_y_n - mean_y_n_1

            dist = (dir_n_1_n_x**2 + dir_n_1_n_y**2) ** 0.5 + ((dir_n_2_n_1_x -
                                                                dir_n_1_n_x)**2 + (dir_n_2_n_1_y - dir_n_1_n_y)**2)**0.5
            if dist < min_distance:
                # print("i,j, distance with direction", i,j,dist)
                min_distance = dist
                min_index = j

        if min_index > 0 and min_distance <= epsilon:
            time_series_components[0, comp_n == min_index] = i
            comp_n[comp_n == min_index] = 0
            #print("geometric center match: distance with direction, index i and j", min_distance, i, min_index)

    return time_series_components, comp_n


def shift_simplified(i, time_series_components, comp_n, epsilon):
    # print("shift simplified")

    min_distance = 2000
    min_index = 0

    comp_n_1_tmp = time_series_components[1].copy()
    x_y_indices = np.sum(np.asarray(np.where(comp_n_1_tmp == i)), axis=1)
    mean_x_n_1 = x_y_indices[1] / np.sum(comp_n_1_tmp == i)
    mean_y_n_1 = x_y_indices[0] / np.sum(comp_n_1_tmp == i)

    for j in np.unique(comp_n)[1:]:

        comp_n_tmp = comp_n.copy()
        x_y_indices = np.sum(np.asarray(np.where(comp_n_tmp == j)), axis=1)
        mean_x_n = x_y_indices[1] / np.sum(comp_n_tmp == j)
        mean_y_n = x_y_indices[0] / np.sum(comp_n_tmp == j)

        dist = ((mean_x_n - mean_x_n_1)**2 + (mean_y_n - mean_y_n_1)**2)**0.5
        if dist < min_distance:
            #print("i,j, distance with direction", i, j, dist)
            min_distance = dist
            min_index = j

    if min_index > 0 and min_distance <= epsilon:
        time_series_components[0, comp_n == min_index] = i
        comp_n[comp_n == min_index] = 0
        #print("geometric center match: distance with direction, index i and j", min_distance, i, min_index)

    return time_series_components, comp_n


def overlap(i, time_series_components, comp_n, percentage):
    # print("overlap")

    max_iou = 0
    max_index = 0

    for j in np.unique(comp_n)[1:]:

        intersection = np.sum(np.logical_and(time_series_components[1] == i, comp_n == j))
        union = np.sum(np.logical_or(time_series_components[1] == i, comp_n == j))

        if union > 0:
            if (intersection / union) >= percentage:
                time_series_components[0, comp_n == j] = i
                comp_n[comp_n == j] = 0
                #print("overlapping match with percentage: IoU, index i and j", intersection / union, i, max_index)

            if (intersection / union) > max_iou:
                max_iou = (intersection / union)
                max_index = j

    if max_iou > 0:
        time_series_components[0, comp_n == max_index] = i
        comp_n[comp_n == max_index] = 0
        #print("overlapping match: IoU, index i and j", max_iou, i, max_index)

    return time_series_components, comp_n


def regression(i, imx, imy, time_series_components, comp_n, eps_time, percentage, reg_steps):

    comp_n_all_tmp = time_series_components[1:].copy()

    mean_x_list = []
    mean_y_list = []
    mean_t_list = []
    max_counter_mean_idx = np.zeros((4))

    for c in range(reg_steps):
        idx = reg_steps-1-c
        if np.sum(comp_n_all_tmp[idx] == i) > 0:
            x_y_indices = np.sum(np.asarray(
                np.where(comp_n_all_tmp[idx] == i)), axis=1)
            mean_x_list.append(
                x_y_indices[1] / np.sum(comp_n_all_tmp[idx] == i))
            mean_y_list.append(
                x_y_indices[0] / np.sum(comp_n_all_tmp[idx] == i))
            mean_t_list.append(c)
            if np.sum(comp_n_all_tmp[idx] == i) > max_counter_mean_idx[0]:
                max_counter_mean_idx[0] = np.sum(comp_n_all_tmp[idx] == i)
                max_counter_mean_idx[1] = mean_x_list[-1]
                max_counter_mean_idx[2] = mean_y_list[-1]
                max_counter_mean_idx[3] = idx

    if len(mean_t_list) >= 2:
        # linear regression of geometric centers
        b_x, a_x, _, _, _ = linregress(mean_t_list, mean_x_list)
        b_y, a_y, _, _, _ = linregress(mean_t_list, mean_y_list)
        pred_x = a_x + b_x * reg_steps
        pred_y = a_y + b_y * reg_steps

        # print("regression distance")

        min_distance = 2000
        min_index = 0

        for j in np.unique(comp_n)[1:]:

            comp_n_tmp = comp_n.copy()
            x_y_indices = np.sum(np.asarray(np.where(comp_n_tmp == j)), axis=1)
            mean_x_n = x_y_indices[1] / np.sum(comp_n_tmp == j)
            mean_y_n = x_y_indices[0] / np.sum(comp_n_tmp == j)

            dist = ((mean_x_n - pred_x)**2 + (mean_y_n - pred_y)**2)**0.5
            if dist < min_distance:
                print("i,j, distance with direction", i, j, dist)
                min_distance = dist
                min_index = j

        if min_index > 0 and min_distance <= eps_time:
            time_series_components[0, comp_n == min_index] = i
            comp_n[comp_n == min_index] = 0
            print("time series match: distance, index i and j",
                  min_distance, i, min_index)

        else:
            # print("regression overlap")

            x_shift = int(pred_x - max_counter_mean_idx[1])
            y_shift = int(pred_y - max_counter_mean_idx[2])

            shifted_comp = np.zeros((imx, imy))
            shifted_comp[comp_n_all_tmp[int(max_counter_mean_idx[3])] == i] = 1
            if x_shift < 0:
                shifted_comp = np.concatenate(
                    (np.zeros((-x_shift, imy)), shifted_comp), axis=0)
                shifted_comp = shifted_comp[:imx, :]
            elif x_shift > 0:
                shifted_comp = np.concatenate(
                    (shifted_comp, np.zeros((x_shift, imy))), axis=0)
                shifted_comp = shifted_comp[x_shift:, :]
            if y_shift < 0:
                shifted_comp = np.concatenate(
                    (np.zeros((imx, -y_shift)), shifted_comp), axis=1)
                shifted_comp = shifted_comp[:, :imy]
            elif y_shift > 0:
                shifted_comp = np.concatenate(
                    (shifted_comp, np.zeros((imx, y_shift))), axis=1)
                shifted_comp = shifted_comp[:, y_shift:]

            max_iou = 0
            max_index = 0

            for j in np.unique(comp_n)[1:]:

                intersection = np.sum(np.logical_and(
                    shifted_comp == 1, comp_n == j))
                union = np.sum(np.logical_or(shifted_comp == 1, comp_n == j))

                if union > 0:
                    if (intersection / union) >= percentage:
                        time_series_components[0, comp_n == j] = i
                        comp_n[comp_n == j] = 0
                        #print("geometric center and overlapping match with percentage: IoU, index i and j", intersection / union, i, max_index)

                    if (intersection / union) > max_iou:
                        max_iou = (intersection / union)
                        max_index = j

            if max_iou > 0:
                time_series_components[0, comp_n == max_index] = i
                comp_n[comp_n == max_index] = 0
                #print("geometric center and overlapping match: IoU, index i and j", max_iou, i, max_index)

    return time_series_components, comp_n


def segments_tracking_i(io_root, basenames, seq, cross_val):

    print('start', seq)

    if not os.path.exists(Path(io_root) / f"ood_prediction_tracked_{cross_val}" / seq):
        os.makedirs(Path(io_root) / f"ood_prediction_tracked_{cross_val}" / seq)

    epsilon = 50  
    num_reg = 5
    eps_near = epsilon/5  
    eps_time = epsilon 
    percentage = 0.35
    
    tmp_img = np.array(Image.open(Path(io_root) / f"ood_prediction_meta_classified_{cross_val}" / f"{basenames[0]}.png"))
    imx = tmp_img.shape[0]
    imy = tmp_img.shape[1]

    # print(np.unique(tmp_img))

    for index in range(len(basenames)):
        if seq + '/' in basenames[index]:
            break

    num_imgs = 0
    for basename in basenames:
        if seq + '/' in basename:
            num_imgs += 1
    
    max_seg_num = 0

    for n in range(num_imgs):
        start = time.time()

        l_n = n + index

        if os.path.isfile(Path(io_root) / f"ood_prediction_tracked_{cross_val}" / f"{basenames[l_n]}.npy"):
            print("skip image", n, seq)
            comp = np.load(Path(io_root) / f"ood_prediction_tracked_{cross_val}" / f"{basenames[l_n]}.npy")
            max_seg_num = max(max_seg_num, comp.max())
            print('max_seg_num', max_seg_num, l_n, np.unique(comp))

        else:

            time_series_components = np.zeros((num_reg+1, imx, imy))
            for m in range(1, num_reg+1):
                if n >= m:
                    time_series_components[m] = np.load(Path(io_root) / f"ood_prediction_tracked_{cross_val}" / f"{basenames[l_n-m]}.npy")

            comp_n = np.array(Image.open(Path(io_root) / f"ood_prediction_meta_classified_{cross_val}" / f"{basenames[l_n]}.png"))
            comp_n = ms.label(comp_n, background=0)
            # print(np.unique(comp_n))

            print("start nearest neighbour matching")
            comp_n = nearest_neighbour(comp_n, eps_near)

            if n == 0:
                time_series_components[0] = comp_n.copy()

            else:
                comp_pixel_counter = np.zeros((int(max_seg_num)+1))
                comp_n_1_tmp = time_series_components[1].copy()
                val, counts = np.unique(np.asarray(comp_n_1_tmp, dtype='int64'), return_counts=True)
                comp_pixel_counter[val] = counts
                idx_field_n_1 = np.argsort(-1*comp_pixel_counter)

                print("start geometric center matching")
                for i in idx_field_n_1:
                    if i != 0 and np.sum(time_series_components[1] == i) > 0:
                        if np.sum(time_series_components[2] == i) > 0:
                            time_series_components, comp_n = shift(i, imx, imy, time_series_components, comp_n, percentage, epsilon)
                        else:
                            time_series_components, comp_n = shift_simplified(
                                i, time_series_components, comp_n, epsilon)

                print("start overlapping matching")
                for i in idx_field_n_1:
                    if i != 0 and np.sum(time_series_components[1] == i) > 0:
                        time_series_components, comp_n = overlap(
                            i, time_series_components, comp_n, percentage)

                print("start time_series matching")
                if n >= 3:
                    reg_steps = min(n, num_reg)
                    for i in range(1, int(max_seg_num)+1):
                        if np.sum(time_series_components[0] == i) == 0 and np.sum(time_series_components[1:] == i) > 0:
                            time_series_components, comp_n = regression(
                                i, imx, imy, time_series_components, comp_n, eps_time, percentage, reg_steps)

                print("start remaining numbers")
                for j in np.unique(comp_n)[1:]:
                    max_seg_num += 1
                    # print("new number: number, index j", max_seg_num, j)
                    time_series_components[0, comp_n == j] = max_seg_num

            max_seg_num = max(max_seg_num, time_series_components[0].max())
            np.save(Path(io_root) / f"ood_prediction_tracked_{cross_val}" / f"{basenames[l_n]}.npy", time_series_components[0])
            print("image", basenames[l_n], "processed in {}s\r".format(round(time.time()-start, 4)))


def visualize_tracking_i(io_root, cross_val, basename, image_path, colors_list):

    def hex_to_rgb(input1):
        value1 = input1.lstrip('#')
        return tuple(int(value1[i:i+2], 16) for i in (0, 2, 4))
    
    target_path = image_path.replace('raw_data','semantic_ood').replace('jpg','png')
    if not os.path.isfile(target_path):
        target_path = None
    
    inp_image = np.array(Image.open(image_path))
    if target_path != None:
        target = np.array(Image.open(target_path))
    entropy = np.load(Path(io_root) / "entropy" / f"{basename}.npy")
    roi = np.array(Image.open(Path(io_root) / "roi" / f"{basename}.png"))
    comp_all = np.array(Image.open(Path(io_root) / "ood_prediction" / f"{basename}.png"))
    comp_track =np.load(Path(io_root) / f"ood_prediction_tracked_{cross_val}" / f"{basename}.npy")

    I1 = inp_image.copy()
    I3 = inp_image.copy()
    I4 = inp_image.copy()

    if target_path != None:
        I1[target==0,:] = np.asarray(Cityscapes.train_id2label[0].color)
        I1[target==254,:] = np.asarray(Cityscapes.train_id2label[254].color)
        I1 = I1 * 0.6 + inp_image * 0.4
    
    plt.imsave(Path(io_root) / f"tracking_img_{cross_val}" / f"{basename}_tmp.png", entropy, cmap='inferno')
    I2 = np.asarray(Image.open(Path(io_root) / f"tracking_img_{cross_val}" / f"{basename}_tmp.png").convert('RGB'))
    os.remove(Path(io_root) / f"tracking_img_{cross_val}" / f"{basename}_tmp.png")

    I3[roi==255,:] = np.asarray(Cityscapes.train_id2label[0].color)
    I3[comp_all==254,:] = np.asarray(Cityscapes.train_id2label[254].color)
    I3 = I3 * 0.6 + inp_image * 0.4

    I4[roi==255,:] = np.asarray(Cityscapes.train_id2label[0].color)
    for c in np.unique(comp_track)[1:]:
        I4[comp_track==c,:] = np.asarray(hex_to_rgb(colors_list[(int(c)-1) % len(colors_list)]))
    I4 = I4 * 0.6 + inp_image * 0.4

    img = np.concatenate((I1, I2), axis=1)
    img2 = np.concatenate((I3, I4), axis=1)
    img = np.concatenate((img, img2), axis=0)
    image = Image.fromarray(img.astype('uint8'), 'RGB')
    image = image.resize((int(inp_image.shape[1]),int(inp_image.shape[0])))
    image.save(Path(io_root) / f"tracking_img_{cross_val}" / f"{basename}.png") 
    plt.close()
    print("stored:", basename+".png")


def eval_tracking_seq(seq, io_root, cross_val, basenames, img_paths, dataset_name):

    print('start', seq)
    start = time.time()

    gt_max = 5
    results = { "num_frames": np.zeros((1)), \
            "gt_objects": np.zeros((gt_max,2)), \
            "tp": np.zeros((1)), \
            "fp": np.zeros((1)), \
            "fn": np.zeros((1)), \
            'precision': np.zeros((1)), \
            'recall': np.zeros((1)), \
            "matching": list([]), \
            'tracking_length': np.zeros((2)), \
            'mot_a': np.zeros((1)), \
            'switch_id': np.zeros((1)), \
            'mismatches': np.zeros((1)), \
            'mot_p': np.zeros((1)), \
            'center_dist': np.zeros((1)), \
            "GT": np.zeros((1)), \
            "mostly_tracked": np.zeros((1)), \
            "partially_tracked": np.zeros((1)), \
            "mostly_lost": np.zeros((1)) }

    for idx in range(len(basenames)):
        if seq in basenames[idx]:
            break
    
    target_paths = []
    num_imgs = 0
    for i in range(len(basenames)):
        target_paths.append(None)
        if seq in basenames[i]:
            num_imgs += 1
            target_path = img_paths[i].replace('raw_data','instance_ood').replace('jpg','png')
            if os.path.isfile(target_path):
                target_paths[-1] = target_path
    print('seq/index/num img/num gt: ', seq, idx, num_imgs, results["GT"])

    tp_tmp = np.zeros((gt_max))
    matching = {}
    for j in range(gt_max):
        matching['lengths_'+str(j)] = list([])
        matching['ids_'+str(j)] = list([])
    
    for n in range(num_imgs):

        l_n = n + idx
        print(seq, n, basenames[l_n])
        
        results["num_frames"] += 1

        components =np.load(Path(io_root) / f"ood_prediction_tracked_{cross_val}" / f"{basenames[l_n]}.npy")
        if target_paths[l_n] != None:
            gt = np.array(Image.open(target_paths[l_n]))

            gt_unique = np.unique(gt)
            results["GT"] = max(results["GT"],len(gt_unique)-1)
            for j in range(gt_max):
                if j+1 in gt_unique:
                    results["gt_objects"][j,0] += 1
                    results["gt_objects"][j,1] += 1

            comp_unique = np.unique(components)
            results["fp"] += len(comp_unique) - 1
            
            # gt ids - max iou, corresponding seg id
            iou_id = np.ones((gt_max,2))*-1

            for i in comp_unique[1:]:
                for j in range(gt_max):
                    intersection = np.sum(np.logical_and(gt==(j+1), components==i))
                    union = np.sum(np.logical_or(gt==(j+1), components==i))
                    if union > 0 and intersection > 0:
                        if intersection / union > iou_id[j,0]:
                            iou_id[j,0] = intersection / union
                            iou_id[j,1] = i
                            
            match_ids = np.unique(iou_id[:,1])
            if match_ids[0] == -1:
                match_ids = match_ids[1:]
            results["fp"] -= len(match_ids)

            for j in range(gt_max):
                if iou_id[j,0] > 0: 
                    tp_tmp[j] += 1
                    x_y_component = np.sum(np.asarray(np.where(components == iou_id[j,1])), axis=1)
                    x_y_gt = np.sum(np.asarray(np.where(gt==(j+1))), axis=1)
                    results["center_dist"] += ( (x_y_component[1]/np.sum(components==iou_id[j,1]) - x_y_gt[1]/np.sum(gt==(j+1)))**2 + (x_y_component[0]/np.sum(components==iou_id[j,1]) - x_y_gt[0]/np.sum(gt==(j+1)))**2 )**0.5 
                elif j+1 in gt_unique: 
                    results["fn"] += 1
                
                # new match with pred segment for a gt segment
                if (len(matching['ids_'+str(j)]) == 0 and iou_id[j,0] > 0) or (len(matching['ids_'+str(j)]) != 0 and iou_id[j,0] > 0 and iou_id[j,1] != matching['ids_'+str(j)][-1]): 
                    matching['ids_'+str(j)].append(iou_id[j,1])
                    matching['lengths_'+str(j)].append(0)
                # increase tracking lengths
                if len(matching['ids_'+str(j)]) != 0 and iou_id[j,1] == matching['ids_'+str(j)][-1]:
                    matching['lengths_'+str(j)][-1] += 1

        else:
            for j in range(gt_max):
                if results["gt_objects"][j,0] > 0:
                    results["gt_objects"][j,0] += 1
                
                if len(matching['ids_'+str(j)]) != 0 and np.sum(components == matching['ids_'+str(j)][-1]) > 0:
                    matching['lengths_'+str(j)][-1] += 1
    
    for j in range(gt_max):
        if len(matching['ids_'+str(j)]) > 0:
            results['switch_id'] += len(matching['ids_'+str(j)]) - 1

    for j in range(gt_max):
        results["tracking_length"][0] += np.sum(matching['lengths_'+str(j)])
    results["tracking_length"][1] = np.sum(results["gt_objects"][:,0])
    
    results["tp"] += np.sum(tp_tmp)
    for j in range(gt_max):
        if results["gt_objects"][j,1] > 0:
            quotient = tp_tmp[j] / results["gt_objects"][j,1]
            if quotient >= 0.8:
                results['mostly_tracked'] += 1
            elif quotient >= 0.2:
                results['partially_tracked'] += 1
            else:
                results['mostly_lost'] += 1
    
    results['matching'] = matching
    
    print('results', results)

    name = 'results_'+seq+'.p'
    pickle.dump(results, open(Path(io_root) / f"tracking_eval_{cross_val}" / name, "wb"))
    print("sequence", seq, "processed in {}s\r".format(round(time.time()-start, 4)))


def comp_remaining_metrics(tracking_metrics):
    """
    helper function for metrics calculation
    """ 
     
    tracking_metrics['mot_a'] = 1 - ( (tracking_metrics['fn'] + tracking_metrics['fp'] + tracking_metrics['switch_id']) / (tracking_metrics['tp'] + tracking_metrics['fn']) )
    
    tracking_metrics['mismatches'] = tracking_metrics['switch_id'] / (tracking_metrics['tp'] + tracking_metrics['fn'])
    
    tracking_metrics['precision'] = tracking_metrics['tp'] / ( tracking_metrics['tp'] + tracking_metrics['fp'])
    
    tracking_metrics['recall'] = tracking_metrics['tp'] / ( tracking_metrics['tp'] + tracking_metrics['fn'])
    
    tracking_metrics['mot_p'] = tracking_metrics['center_dist'] / tracking_metrics['tp']
    
    return tracking_metrics


def merge_results(io_root, cross_val, sequences):
    
    print('merge results over sequences')
    
    tracking_metrics = { "num_frames": np.zeros((1)), \
        "gt_objects": np.zeros((5,2)), \
        "tp": np.zeros((1)), \
        "fp": np.zeros((1)), \
        "fn": np.zeros((1)), \
        'precision': np.zeros((1)), \
        'recall': np.zeros((1)), \
        "matching": list([]), \
        'tracking_length': np.zeros((2)), \
        'mot_a': np.zeros((1)), \
        'switch_id': np.zeros((1)), \
        'mismatches': np.zeros((1)), \
        'mot_p': np.zeros((1)), \
        'center_dist': np.zeros((1)), \
        "GT": np.zeros((1)), \
        "mostly_tracked": np.zeros((1)), \
        "partially_tracked": np.zeros((1)), \
        "mostly_lost": np.zeros((1)) }

    with open(Path(io_root) / f"tracking_eval_{cross_val}" / "results_table.txt" , 'wt') as fi:
    
        for seq in sequences:
            print(seq)

            name = 'results_'+seq+'.p'
            results = pickle.load(open(Path(io_root) / f"tracking_eval_{cross_val}" / name, "rb"))
            
            tracking_results = comp_remaining_metrics(results)
            print(seq, tracking_results, file=fi)
            print(' ', file=fi)
            
            for tm in tracking_metrics:
                if tm in ['num_frames', 'gt_objects', 'tp', 'fp', 'fn', 'tracking_length', 'switch_id', 'center_dist', 'GT', 'mostly_tracked', 'partially_tracked', 'mostly_lost']:
                    tracking_metrics[tm] += results[tm]
        
        tracking_results = comp_remaining_metrics(tracking_metrics)
        print(tracking_results, file=fi)
