# Crop Field Health

This project aims to assist large-scale agriculture using neural networks to detect anomalies in crop fields,
such as weed clusters, skipped planting, and water destruction. Considering the expansiveness of global crop fields,
it is near-impossible to patrol crop fields on foot and resource-consuming and often largely expensive to try and have 
humans analyze aerial images using existing technologies. This project simplifies existing solutions and provides an 
speed and cost-efficient solution for analyzing images.

## Structure 

The data pipelines and system is developed in multiple stages, owing to the complexity of the Agriculture-Vision dataset (see below).
First, the dataset is inflated from its compressed files using the `scripts/expand.sh` script. Then, the `scripts/generate.sh` calls the
`preprocessing/generate.py` script, which creates JSON files containing all image paths for a unique image ID. Finally, the `preprocessing/dataset.py`
file contains the `AgricultureVisionDataset` object which is called from implementation scripts as training data.

For data inspection, including viewing all images associated with an ID or viewing all images belonging to a category, the `preprocessing/inspect.py`
contains functionality for viewing and saving these images.

## Usage

You can install the repository from the command line:

```shell script
git clone https://github.com/amogh7joshi/crop-field-health.git
```

A Makefile is included for installation. To use it, run the following.

```shell script
make install
```

Otherwise, in the proper directory, execute the following to install system requirements.

```shell script
python3 -m pip install -r requirements.txt
```

Once you have acquired the compressed dataset, place it in the `data/` directory and execute the 
`expand.sh` script. To process the dataset, then run the `preprocess.sh` script. After that, the dataset can be 
imported by calling:

```python
from preprocessing.dataset import AgricultureVisionDataset
```

You can work with it as follows:

```python
dataset = AgricultureVisionDataset()
dataset.construct()

# Access dataset attributes.
dataset.train_data # Training Data.
dataset.val_data # Val Data.
dataset.test_data # Test Data.

# Convert dataset from tensors to numpy arrays.
dataset.as_numpy()
```

## Agriculture-Vision Dataset

This project makes use of the Agriculture-Vision dataset, containing aerial crop field images from multiple classes.
The dataset can be requested from the challenge [website](https://www.agriculture-vision.com/contact-us), and for compatibility
the compressed file should be placed in the `data` directory.

```bibtex
@article{chiu2020agriculture,
         title={Agriculture-Vision: A Large Aerial Image Database for Agricultural Pattern Analysis},
         author={Mang Tik Chiu and Xingqian Xu and Yunchao Wei and Zilong Huang and Alexander Schwing 
                 and Robert Brunner and Hrant Khachatrian and Hovnatan Karapetyan and Ivan Dozier and Greg Rose 
                 and David Wilson and Adrian Tudor and Naira Hovakimyan and Thomas S. Huang and Honghui Shi},
         journal={arXiv preprint arXiv:2001.01306},
         year={2020}
}
```


