import os
import hydra
from omegaconf import DictConfig
from ood_predict import compute_and_save_softmax_probs, compute_and_save_softmax_entropy, visualize_segmentation, visualize_ood_score_prediction
from meta_ood import train_meta_classifier, compute_and_save_meta_data, compute_and_save_ood_prediction, meta_kick, segment_wise_evaluation
from tracking import tracking_segments, visualize_tracking_per_image, evaluate_tracking
from ood_retrieval import compute_embeddings, cluster_data
from src.utils import MeanF1Score
from ood_retrieval import compute_embeddings, cluster_data, eval_clustering
import numpy as np



@hydra.main(config_path=".", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    dataset_name = cfg.dataset
    dataset = hydra.utils.instantiate(cfg["dataloader_" + dataset_name])
    print(dataset)
    total_num_sequences = len(dataset.sequences)
    io_root = os.path.join(cfg.outputs_root, cfg.names[dataset_name])

    if cfg.tasks.segmentation:
        compute_and_save_softmax_probs(dataset, cfg.checkpoints.segmentation, io_root)
        visualize_segmentation(dataset, io_root, cfg.save.semantic, cfg.save.color, cfg.save.road_roi, cfg.num_cpus)
    if cfg.tasks.ood_prediction:
        compute_and_save_softmax_entropy(dataset, cfg.checkpoints.ood_prediction, io_root)
        visualize_ood_score_prediction(dataset, io_root, cfg.save.ood_heat, cfg.num_cpus)
    if cfg.tasks.meta_classification:
        roi_smoother = hydra.utils.instantiate(cfg["roi_" + dataset_name])
        entropy_thresh = cfg.params[dataset_name].entropy_thresh
        compute_and_save_ood_prediction(dataset, io_root, entropy_thresh, roi_smoother, cfg.num_cpus)
        print('compute_and_save_ood_prediction done')
        compute_and_save_meta_data(dataset, io_root, cfg.num_cpus)
        print('compute_and_save_meta_data done')
        mfs = MeanF1Score(io_root)
        if cfg.cross_val: 
            for step in range(total_num_sequences):
                list_total_sequences = np.arange(1, total_num_sequences+1)
                """initialize data splits for cross validation"""
                dataloader = "dataloader_" + cfg.dataset
                train_sequences = np.delete(list_total_sequences, step)
                val_sequences = [step+1]
                training_dataset = hydra.utils.instantiate(cfg[dataloader], sequences=train_sequences.tolist())
                validation_dataset = hydra.utils.instantiate(cfg[dataloader], sequences=val_sequences)
                
                """meta classification training"""
                meta_model = hydra.utils.instantiate(cfg[cfg.meta_model])
                training_meta_dataset = training_dataset
                train_meta_classifier(training_meta_dataset, meta_model, io_root, cfg.num_cpus)
                print('meta classification training done')

                """evaluate meta classification"""
                validation_meta_dataset = validation_dataset
                meta_kick(validation_meta_dataset, cfg.meta_model, io_root, io_root, cfg.cross_val, cfg.num_cpus)
                print('meta kick done')
                
        else:
            dataloader = "dataloader_" + cfg.dataset
            if dataloader == 'dataloader_carla':
                training_dataset = hydra.utils.instantiate(cfg["dataloader_sos"])
                io_root_train = os.path.join(cfg.outputs_root, cfg.names['sos'])
            if dataloader == 'dataloader_sos':
                training_dataset = hydra.utils.instantiate(cfg["dataloader_carla"])
                io_root_train = os.path.join(cfg.outputs_root, cfg.names['carla'])
            validation_dataset = dataset

            """meta classification training"""
            meta_model = hydra.utils.instantiate(cfg[cfg.meta_model])
            training_meta_dataset = training_dataset
            train_meta_classifier(training_meta_dataset, meta_model, io_root_train, cfg.num_cpus)
            print('meta classification training done')

            """evaluate meta classification"""
            validation_meta_dataset = validation_dataset
            meta_kick(validation_meta_dataset, cfg.meta_model, io_root, io_root_train, cfg.cross_val, cfg.num_cpus)
            print('meta kick done')
        
        segment_wise_evaluation(dataset, io_root, cfg.cross_val)
        print('segment wise eval done')
        print('meta_classification finished')

    if cfg.tasks.tracking:
        tracking_segments(dataset, io_root, cfg.cross_val, cfg.num_cpus)
    
    if cfg.tasks.plot_tracking:
        visualize_tracking_per_image(dataset, io_root, cfg.cross_val, cfg.num_cpus)
    
    if cfg.tasks.eval_tracking:
        evaluate_tracking(dataset, dataset_name, io_root, cfg.cross_val, cfg.num_cpus)

    if cfg.tasks.compute_embeddings:
        embed_model = hydra.utils.instantiate(cfg[cfg.embedding_model])
        compute_embeddings(dataset, dataset_name, io_root, cfg.crop_size, embed_model, cfg.cross_val)

    if cfg.tasks.clustering:
        cluster_data(io_root, dataset_name, cfg.cross_val)

    if cfg.tasks.eval_clustering:
        eval_clustering(dataset, dataset_name, io_root, cfg.cross_val)


if __name__ == "__main__":
    main()
