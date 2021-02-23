# Crop Field Health

![example](examples/evaluated-o1-21.png)

## Background

There are over 400 million acres of farmland in the United States, which traditionally must be monitored on foot to detect anomalies, an extensively time-consuming task. 
A farmland anomaly is any object, region, or event that disrupts the normal growth stages of crops, which if left unchecked, can drastically decrease the yield of a farm.

Some of the most harmful farmland anomalies to potential crop yield are:
- **clusters of weeds**, which inhibit crop growth and nutrient gains,
- **stagnant water**, which can serve as a breeding ground for harmful bacteria and pests,
- **unintended waterways**, which can destroy plants in their paths, and 
- **missed** or **double planting**, which prevents maximum planting efficiency.

I have used deep neural networks to conduct semantic image segmentation on aerial images of farmland, to classify and determine the 
locations of these anomalies
such as weed clusters, skipped planting, and water destruction. Considering the expansiveness of global crop fields,
it is near-impossible to patrol crop fields on foot and resource-consuming and often largely expensive to try and have 
humans analyze aerial images using existing technologies. 

This project simplifies existing solutions and provides an 
accurate and efficient solution for analyzing agricultural images, a relatively untouched field.

## Usage

You can install the repository from the command line:

```shell script
git clone https://github.com/amogh7joshi/crop-field-health.git
```

### Python Setup

A Makefile is included for Python installation. To use it, run the following.

```shell script
make install
```

Otherwise, in the proper directory, execute the following to install system requirements.

```shell script
python3 -m pip install -r requirements.txt
```

From here, the `scripts/expand.sh` script inflates the dataset into its permanent file structure, and 
the `scripts/preprocess.sh` processes the dataset into JSON files containing image paths for each image ID.

### C++ Setup

If you want to work with the C++ extensions of the project, which are located in the `cc` directory, then 
follow the above steps for repository installation and Python setup. 

From there, you need to build the C++ project. You will need CMake installed, as well as 
[OpenCV](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html) (for working with images) 
and [nlohmann-json](https://github.com/nlohmann/json#package-managers) (for working with JSON files). Once you have those
installed, execute:

```shell script
cmake -DCMAKE_BUILD_TYPE=Debug -G "CodeBlocks - Unix Makefiles" path/to/farmland-anomalies/cc
```

You can run the compiled C++ executables from there, by running `make` in the `cc` directory to build the files and then:

```shell script
./cc
```

## Data Gathering and Preprocessing

The data pipelines and system is developed in multiple stages, owing to the complexity of the Agriculture-Vision dataset (see below).
First, the dataset is inflated from its compressed files using the `scripts/expand.sh` script. Then, the `scripts/generate.sh` calls the
`preprocessing/generate.py` script, which creates JSON files containing all image paths for a unique image ID. (This can also be done using 
the optimized C++ extension in the `cc` directory, see the below C++ installation section for more information) Finally, the `preprocessing/dataset.py`
file contains the `AgricultureVisionDataset` object which is called from implementation scripts as training data.

For data inspection, including viewing all images associated with an ID, viewing all images belonging to a category, or viewing all images related to a 
certain ID, the `preprocessing/inspect.py` contains functionality for viewing and saving these images. The other files in the `preprocessing` directory contain 
individual purpose implementations (e.g., `preprocessing/distribution.py` plots the class distribution frequency of the dataset).

### Agriculture-Vision Dataset

This project makes use of the Agriculture-Vision dataset, containing aerial farmland images with one or multiple different anomaly segmentation masks.
Information about the dataset and its acquisition can be found at the [challenge website](https://www.agriculture-vision.com/contact-us), and for compatibility
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

## Model Information

There were three main deep neural network models constructed as part of this project: a semi-shallow single-network model [**L1**](https://github.com/amogh7joshi/farmland-anomalies/blob/master/model/light/light_network.py#L15),
 the deepest model, titled [**D1**](model/complex/architecture.py), and the successfully implemented model [**L2**](https://github.com/amogh7joshi/farmland-anomalies/blob/master/model/light/light_network.py#L112).
 
### Architecture

![architecture-l2](examples/architecture-l2.png)

Model **L2** (diagram generated using [Net2Vis](https://github.com/viscom-ulm/Net2Vis)) uses **Ensemble Learning** techniques, with two "sub-networks":

1. The top (and shallower) network, **L2-1**, which learns high-level image features.
2. The bottom (and deeper) network, **L2-2**, which learns deep spatial relations and features.

**L2-2** uses strided convolutions to pick up on features generally lost during downsampling, while **L2-1** uses pooling layers, 
to prevent gradient propagation issues which may arise. 

For specific details on the network architectures, see the [model](https://github.com/amogh7joshi/farmland-anomalies/tree/master/model) directory.

### Losses

To refine segmentation masks, multiple loss functions were used on a single loss instance. For example, a model was trained on an arbitrary loss *A* for 20 epochs,
then loss *B* for 20 more epochs, and finally loss *C* for 20 final epochs.

Primarily, dice loss and cross-entropy loss was used, however a third loss function titled **surface-channel** loss was developed, with the formula:

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{300}&space;\bg_white&space;\large&space;\mathrm{SCL}&space;=&space;\max\limits_{i\,\in\,&space;C}\displaystyle\iint_{I_i}||M(p)&space;-&space;p_T||^2&space;\,\mathrm{dA}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{300}&space;\bg_white&space;\large&space;\mathrm{SCL}&space;=&space;\max\limits_{i\,\in\,&space;C}\displaystyle\iint_{I_i}||M(p)&space;-&space;p_T||^2&space;\,\mathrm{dA}" title="\large \mathrm{SCL} = \max\limits_{i\,\in\, C}\displaystyle\iint_{I_i}||M(p) - p_T||^2 \,\mathrm{dA}" /></a>

This penalizes incorrect calculations heavily, and focuses on classifications over individual channels, allowing for the refinement of a prediction *as well as* segmentation.

## License and Contributions

![GitHub](https://img.shields.io/github/license/amogh7joshi/farmland-anomalies?logoColor=blue&style=flat-square)

All of the code in this repository is licensed under the MIT License, meaning you are free to work with it as you desire, but
this repository must be cited if you want to reuse the code. 

Although you are free to work with the project yourself, contributions will not be accepted to this repository. You are, however, welcome
to open an issue in the issues tab if you notice something that is broken. 


