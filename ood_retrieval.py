import numpy as np
import os
import pickle
import time
import tqdm
import random
import torch
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.stats import trim_mean

from torchvision.transforms import Compose, Normalize, ToTensor
from pathlib import Path


# -----------------------------------------------------------
# model functions
# -----------------------------------------------------------

def init_feature_extractor(model):
    return model.cuda().eval()


def get_embedding(transform, image, model):
    with torch.no_grad():
        inp = transform(image).unsqueeze_(0).cuda()
        out = model(inp)
    return out.data.cpu().squeeze().numpy()

# -----------------------------------------------------------
# cutout functions
# -----------------------------------------------------------


def wrapper_cutout_components(args):
    return cutout_components(*args)


def cutout_components(load_path,
                      im_path,
                      min_height,
                      min_width,
                      min_crop_height,
                      min_crop_width,
                      max_id
                      ):
  
    components = np.load(load_path)

    
    crops = {'embeddings': [],
             'image_path': im_path,
             'boxes': [],
             'segment_indices': [],
             'tracking_id': [],
             }

    for cindex in np.unique(components)[1:]:
        segment_indices = np.argwhere(components == cindex)

        if segment_indices.shape[0] > 0:
            upper, left = segment_indices.min(0)
            lower, right = segment_indices.max(0)
            if (lower - upper) < min_height or (right - left) < min_width:
                continue

            if (right - left) < min_crop_width:
                margin = min_crop_width - (right - left)
                if left - (margin // 2) < 0:
                    left = 0
                    right = left + min_crop_width
                elif right + (margin // 2) > components.shape[1]:
                    right = components.shape[1]
                    left = right - min_crop_width

                if right > components.shape[1] or left < 0:
                    raise IndexError('Image with shape {} is too small for a {} x {} crop'.format(
                        components.shape, min_crop_height, min_crop_width))
            if (lower - upper) < min_crop_height:
                margin = min_crop_height - (lower - upper)
                if upper - (margin // 2) < 0:
                    upper = 0
                    lower = upper + min_crop_height
                elif lower + (margin // 2) > components.shape[0]:
                    lower = components.shape[0]
                    upper = lower - min_crop_height

                if lower > components.shape[0] or upper < 0:
                    raise IndexError('Image with shape {} is too small for a {} x {} crop'.format(
                        components.shape, min_crop_height, min_crop_width))

            crops['boxes'].append((left, upper, right, lower))
            crops['segment_indices'].append(segment_indices)
            crops['tracking_id'].append(int(cindex + max_id))

    return crops, len(np.unique(components)[1:])

# -----------------------------------------------------------
# dimensionality reduction functions
# -----------------------------------------------------------


def pca_emb(data):
    print('Computing PCA...')
    n_comp = 50 if 50 < min(len(data['embeddings']),
                            data['embeddings'][0].shape[0]) else min(len(data['embeddings']),
                                                                     data['embeddings'][0].shape[0])
    embeddings = PCA(
        n_components=n_comp
    ).fit_transform(np.stack(data['embeddings']).reshape((-1, data['embeddings'][0].shape[0])))

    return embeddings


def tsne_emb(data):
    print('Computing t-SNE for plotting')
    tsne_embedding = TSNE(n_components=2,
                          perplexity=30,
                          learning_rate=200.0,
                          early_exaggeration=12.0,
                          n_iter=1000,
                          verbose=3,
                          ).fit_transform(data)
    return tsne_embedding

# -----------------------------------------------------------
# clustering
# -----------------------------------------------------------


def dbscan_detect(data, eps, min_samples, metric, min_cluster_size):
    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(data)
    data = np.asarray(data)
    label = model.labels_ + 1
    cluster = []
    n_clusters = len(np.unique(label))
    v = 0
    for i in range(1, n_clusters):
        cluster.append([data[label == i]])
        v += 1
    tic = 0
    nmb = []
    for i in range(1, n_clusters):
        if len(cluster[tic][0]) > min_cluster_size:
            nmb.append(i)
            tic += 1
        else:
            del cluster[tic]

    return cluster, label, nmb


# -----------------------------------------------------------
# perform image retrieval
# -----------------------------------------------------------

def compute_embeddings(dataset, dataset_name, load_dir, crop_size, model, mode):
  
    start = time.time()

    load_name = "embedding_" + dataset_name + "_" + str(mode) + ".p"

    if not os.path.exists(Path(load_dir) / load_name):
        net = init_feature_extractor(model)
        print('embedding network initialized...')
        transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
        basenames = dataset.basenames
        tracking_paths = [os.path.join(load_dir, "ood_prediction_tracked_{}".format(str(mode)), f"{basename}.npy") for basename in basenames]

        print("cut out OOD components")
        r = []
        max_id = 0

        for seq in tqdm.tqdm(dataset.sequences):
            used_id = 0

            for index in range(len(basenames)):
                if seq + '/' in basenames[index]:
                    break

            num_imgs = 0
            for basename in basenames:
                if seq + '/' in basename:
                    num_imgs += 1

            for n in range(num_imgs):
                l_n = n + index
                r_i, id_n = cutout_components(tracking_paths[l_n],
                   dataset.images[l_n],
                   crop_size,
                   crop_size,
                   crop_size,
                   crop_size,
                   max_id)
                r.append(r_i)
                used_id = max(used_id, id_n)
            max_id += used_id


        crops = {
            'embeddings': [],
            'image_path': [],
            'box': [],
            'tracking_id': [],
            'gt': [],
            'seq': [],
            'inst': []
        }
        
        print("compute feature embeddings")
        for c in tqdm.tqdm(r):
            image = Image.open(c['image_path'])
            if os.path.exists(c['image_path'].replace('raw_data', 'semantic').replace('jpg', 'png')):
                gt = np.array(Image.open(c['image_path'].replace('raw_data', 'semantic').replace('jpg', 'png')))
            else:
                gt = 255 * np.ones((np.array(image).shape[0], np.array(image).shape[1]))
            if os.path.exists(c['image_path'].replace('raw_data', 'instance_ood').replace('jpg', 'png')):
                inst = np.array(Image.open(c['image_path'].replace('raw_data', 'instance_ood').replace('jpg', 'png')))
            else:
                inst = np.zeros((np.array(image).shape[0], np.array(image).shape[1]))


            for i, b in enumerate(c['boxes']):

                x0, y0 = np.hsplit(c['segment_indices'][i], 2)
                values0, counts0 = np.unique(gt[x0, y0], return_counts=True)
                ind0 = np.argmax(counts0)
                values1, counts1 = np.unique(inst[x0, y0], return_counts=True)
                ind1 = np.argmax(counts1)


                crops['image_path'].append(c['image_path'])
                crops['embeddings'].append(get_embedding(transform, image.crop(b), net))
                crops['box'].append(b)
                crops['tracking_id'].append(c['tracking_id'][i])
                crops['gt'].append(values0[ind0])
                crops['seq'].append(c['image_path'].split('/')[-2])
                crops['inst'].append(values1[ind1])

        print('features: ', len(crops['embeddings']))

        print('saving data...')
        dump_path = Path(load_dir) / load_name
        dump_dir = os.path.dirname(dump_path)
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
        pickle.dump(crops, open(dump_path, 'wb'))

    else:
        print("load embeddings")
        crops = pickle.load(open(Path(load_dir) / load_name, 'rb'))

    random.seed(123)

    data_pca = pca_emb(crops)
    data_tsne = tsne_emb(data_pca)

    save_name = "tsne_" + dataset_name + "_" + str(mode) + ".p"
    dump_path = Path(load_dir) / save_name
    dump_dir = os.path.dirname(dump_path)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    pickle.dump(data_tsne, open(dump_path, 'wb'))

    print("tsne embeddings saved: ", dump_path)
    print("embeddings computed in {}s\r".format(round(time.time()-start, 4)))



def cluster_data(load_dir, dataset_name, mode):

    crops_load_name = "embedding_" + dataset_name + "_" + str(mode) + ".p"
    tsne_load_name = "tsne_" + dataset_name + "_" + str(mode) + ".p"

    crops = pickle.load(open(Path(load_dir) / crops_load_name, 'rb'))
    data = pickle.load(open(Path(load_dir) / tsne_load_name, 'rb'))

    t_noise = 25
    filter = tidy_cluster(crops, data, t_noise, load_dir)

    tidy_data = data[filter > t_noise]
    print("DBSCAN")
    cluster, lbl, nmb = dbscan_detect(tidy_data, 4.0, 15, "euclidean", 1)
    
    mapping_dict = dict()
    all_boxes = np.array(crops['box'])[filter > t_noise]
    all_paths = np.array(crops['image_path'])[filter > t_noise]
    all_gt = np.array(crops['gt'])[filter > t_noise]
    all_seq = np.array(crops['seq'])[filter > t_noise]
    all_inst = np.array(crops['inst'])[filter > t_noise]

    for l, n in enumerate(nmb):
        if not os.path.exists(os.path.join(load_dir, str(mode), 'cluster_{}'.format(n))):
            os.makedirs(os.path.join(load_dir, str(mode), 'cluster_{}'.format(n)))
        ind = np.flatnonzero(lbl == n)
        boxes = all_boxes[ind]
        paths = all_paths[ind]
        gt = all_gt[ind]
        seq = all_seq[ind]
        inst = all_inst[ind]

        c_l = find_center_i(cluster[l])
        if not os.path.exists(Path(load_dir) / Path(str(mode)) / "center_of_cluster"):
            os.makedirs(Path(load_dir) / Path(str(mode))  / "center_of_cluster")
        save_name_l = 'center_of_cluster/center_of_cluster_{}.png'.format(n)
        Image.open(paths[c_l]).crop(boxes[c_l]).save(Path(load_dir) / Path(str(mode))  / save_name_l)


        if not os.path.exists(os.path.join(load_dir, str(mode), 'cluster_{}/images/'.format(n))):
            os.makedirs(os.path.join(load_dir, str(mode), 'cluster_{}/images/'.format(n)))
        if not os.path.exists(os.path.join(load_dir, str(mode), 'cluster_{}/ood-objects/'.format(n))):
            os.makedirs(os.path.join(load_dir, str(mode), 'cluster_{}/ood-objects/'.format(n)))

        for k, path in enumerate(paths):
            img = Image.open(path)
            img.save(os.path.join(load_dir, str(mode), 'cluster_{}/images/'.format(n), os.path.splitext(os.path.basename(path))[0] +'.png') )
            
            x, y = np.array_split(cluster[l][0][k], 2, axis=0)
            img.crop(boxes[k]).save(os.path.join(load_dir, str(mode), 'cluster_{}/ood-objects/'.format(n),  os.path.splitext(os.path.basename(path))[0] + '_{}_{}.png'.format(x, y)))

        
        cluster_dict = {
            'embedding': cluster[l][0],
            'image_path': paths,
            'boxes': boxes,
            'gt': gt,
            'seq': seq,
            'inst': inst
        }

        mapping_dict.update({'cluster_{}'.format(n): cluster_dict})

    load_name = 'cluster_dict_' + dataset_name + '_' + str(mode) + '.p'
    dump_path = Path(load_dir) / load_name
    dump_dir = os.path.dirname(dump_path)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    pickle.dump(mapping_dict, open(dump_path, 'wb'))

  

# -----------------------------------------------------------
# combine clustering and tracking
# -----------------------------------------------------------

def tidy_cluster(crops, data, t_noise, load_dir, visualize = False):
    tracking_ids = np.asarray(crops['tracking_id']).flatten()
    ids, len_ids = np.unique(tracking_ids, return_counts=True)
    
    if visualize:
        plt.bar(ids, len_ids, color='deeppink')
        plt.plot(ids, t_noise*np.ones(len(ids)), c='r')
        plt.title("number of frequent ID's: {}".format(np.sum(np.array(len_ids) > t_noise)))
        save_name = 'histogram_ids.png'
        plt.savefig(Path(load_dir) / save_name)
        plt.clf()

    filter = np.zeros(data.shape[0])
    for i, k in enumerate(ids):
        filter[tracking_ids == k] = len_ids[i]

    tidy_data = data[filter > t_noise]
    tidy_id = tracking_ids[filter > t_noise]
    all_paths = np.array(crops['image_path'])[filter > t_noise]
    all_boxes = np.array(crops['box'])[filter > t_noise]
    
    color_code = np.copy(tidy_id)
    for i, k in enumerate(np.unique(tidy_id)):
        color_code[tidy_id == k] = i + 1
    
    if not os.path.exists(Path(load_dir) / "center_of_id"):
            os.makedirs(Path(load_dir) / "center_of_id")

    for i, id in enumerate(np.unique(tidy_id)):
        data_i = tidy_data[tidy_id == id]
        paths = all_paths[tidy_id == id]
        boxes = all_boxes[tidy_id == id]
        c_i = find_center_i(data_i)
        save_name_i = 'center_of_id/center_of_id_{}.png'.format(i+1)
        Image.open(paths[c_i]).crop(boxes[c_i]).save(Path(load_dir) / save_name_i)
    
    return filter

def find_center_i(data):
    center = trim_mean(data, proportiontocut=0.2, axis=0)
    min_dist = np.inf
    for k, p in enumerate(data):
        dist = np.linalg.norm(center - p)
        if dist < min_dist:
            min_dist = dist
            cp = k
    return cp


# -----------------------------------------------------------
# evaluate clustering
# -----------------------------------------------------------

def eval_clustering(dataset, dataset_name, load_dir, mode):

    id_dict = dataset.id_dict
    classes = dataset.ood_classes
    load_name = "cluster_dict_" + dataset_name + '_' + str(mode) + ".p"

    cluster_dir = Path(load_dir) / load_name

    cl_inst_score = mean_cl_inst(id_dict, cluster_dir)
    cl_imp_score = mean_cl_imp(cluster_dir)
    cl_frag_score = mean_cl_frag(cluster_dir, classes)

    print("Clustering results for", dataset_name, ": $CL_{inst} = $", cl_inst_score, ", $CL_{imp} = $",  cl_imp_score, ", $CL_{frag} = $", cl_frag_score)



def cl_inst(load_dir, i, seq):
    cluster = pickle.load(open(load_dir, 'rb'))
    counter = np.zeros(len(cluster.keys()))
    for k, key in enumerate(cluster.keys()):
        counter[k] = np.sum(np.logical_and(cluster[key]['seq'] == seq, cluster[key]['inst'] == i))
    if np.sum(counter) > 0:
        return np.max(counter) / np.sum(counter)
    else:
        return np.nan

def mean_cl_inst(id_dict, load_dir):
    
    cl_inst_scores = []
    for key in id_dict.keys():
        for id in id_dict[key]:
            cl_inst_scores.append(cl_inst(load_dir, id, key))
    cl_inst_scores = np.array(cl_inst_scores)
    return np.nanmean(cl_inst_scores)


def mean_cl_imp(load_dir):
    cluster = pickle.load(open(load_dir, 'rb'))
    cl_imp_scores = []
    for key in cluster.keys():
        cl_imp_scores.append(len(np.unique(cluster[key]['gt'])))
    cl_imp_scores = np.array(cl_imp_scores)
    return np.nanmean(cl_imp_scores)



def cl_frag(load_dir, i):
    cluster = pickle.load(open(load_dir, 'rb'))
    cnt = 0
    for key in cluster.keys():
        if i in cluster[key]['gt']:
            cnt += 1
    return cnt

def mean_cl_frag(load_dir, classes):
    cl_frag_score = []
    for c in classes:
        cl_frag_score.append(cl_frag(load_dir, c))
    cl_frag_score = np.array(cl_frag_score)
    return np.nanmean(cl_frag_score)

