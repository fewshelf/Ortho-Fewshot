# Ortho-shot

### CAM PLOTS

<p align="center">
  <img width="1000" height="230" src="imgs/OrthoShot-CAM.png">
</p>

## Architecture

<p align="center">
  <img width="900" height="400" src="imgs/OrthoShot3.png">
</p>


### Installation

This repo was tested with Ubuntu 16.04.5 LTS, Python 3.6, PyTorch 0.4.0, and CUDA release 7.5.
| PyTorch versions >=0.4.0 | we used PyTorch versions 1.1.0

### Datasets
- [StanfordDog](http://vision.stanford.edu/aditya86/ImageNetDogs/).
- [StanfordCar](https://ai.stanford.edu/~jkrause/cars/car_dataset.html).
- [CUB-200](http://www.vision.caltech.edu/visipedia/CUB-200.html). <br>
- [miniimagenet](https://www.dropbox.com/sh/6yd1ygtyc3yd981/AADaG1GvNdXkHnjynhZY6TBia/miniImageNet.tar.gz?dl=0)
- [tieredImageNet](https://www.dropbox.com/sh/6yd1ygtyc3yd981/AABg-ODoQp1JEzhIt7q5GofVa/tieredImageNet.tar.gz?dl=0)

# Pre-trained Models

# Running

Exemplar commands for running the code can be found in `scripts/run.sh`.

# supervised pre-training
python train_orth_classifier.py --trial pretrain --model_path /path/to/save --tb_path /path/to/tensorboard --data_root /path/to/data_root

# distillation
# setting '-a 1.0' should give simimlar performance
python train_orth_distillation.py -r 0.5 -a 0.5 --path_t /path/to/teacher.pth --trial born1 --model_path /path/to/save --tb_path /path/to/tensorboard --data_root /path/to/data_root

# evaluation
python eval_fewshot.py --model_path /path/to/student.pth --data_root /path/to/data_root



