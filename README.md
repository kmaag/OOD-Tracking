## OOD Segmentation, Tracking and Retrieval:

### Packages and their versions:
Code tested with ```Python 3.6.10``` and ```pip 21.3.1```.
Install Python packages via
```sh
pip install -r requirements.txt
```

### Download weights:
```sh
mkdir checkpoints
wget -O ./checkpoints/DeepLabV3+_WideResNet38_cityscapes.pth https://uni-wuppertal.sciebo.de/s/WVFTc4ka37xASZV/download
wget -O ./checkpoints/DeepLabV3+_WideResNet38_entropy_maximized.pth https://uni-wuppertal.sciebo.de/s/kCgnr0LQuTbrArA/download
```

### Preparation:
Edit all necessary paths stored in "config.yaml". By default the outputs will be saved in "./outputs".
Also, in the same file, select the tasks to be executed by setting the corresponding boolean variable (True/False). These functions are CPU based and parts are parallized over the number of input images, adjust "num_cpus" in "config.yaml" to make use of this. 

### Run the code:
```python
python main.py
```

Code adapted from:
- https://github.com/SegmentMeIfYouCan/road-anomaly-benchmark
- https://github.com/mrottmann/MetaSeg
- https://github.com/robin-chan/meta-ood
- https://github.com/kmaag/Time-Dynamic-Prediction-Reliability 
- https://github.com/RonMcKay/OODRetrieval


## Authors:
Kira Maag, Robin Chan, Svenja Uhlemeyer and Kamil Kowol
