# Hierarchical Scene Coordinate Classification and Regression for Visual Localization
This is the PyTorch implementation of our paper, a hierarchical scene coordinate prediction approach for one-shot RGB camera relocalization:

[Hierarchical Scene Coordinate Classification and Regression for Visual Localization](https://arxiv.org/abs/1909.06216), CVPR 2020    
Xiaotian Li, Shuzhe Wang, Yi Zhao, Jakob Verbeek, Juho Kannala    

## Setup
Python3 and the following packages are required:
```
cython
numpy
pytorch
opencv
tqdm
imgaug
```

It is recommended to use a conda environment:
1. Install anaconda or miniconda.
2. Create the environment: `conda env create -f environment.yml`.
3. Activate the environment: `conda activate hscnet`.

To run the evaluation script, you will need to build the cython module:

```bash
cd ./pnpransac
python setup.py build_ext --inplace
```

## Data

We currently support [7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/), [12-Scenes](https://graphics.stanford.edu/projects/reloc/), [Cambridge Landmarks](https://mi.eng.cam.ac.uk/projects/relocalisation/), and the three combined scenes which have been used in the paper. We will upload the code for the [Aachen Day-Night](https://www.visuallocalization.net/datasets/) dataset experiments.

You will need to download the datasets from the websites, and we provide a [data package](https://drive.google.com/file/d/1VnUbxYB6Akh8crv06lvDK60qg4lHXcRc/view?usp=sharing) which contains other necessary files for reproducing our results. Note that for the Cambridge Landmarks dataset, you will also need to rename the files according to the `train/test.txt` files and put them in the `train/test` folders. And the depth maps we used for this dataset are from [DSAC++](https://github.com/vislearn/LessMore). The provided label maps are obtained by running  [k-means](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans.html) hierarchically on the 3D points.




## Evaluation
The trained models for the main experiments in the paper can be downloaded [here](https://drive.google.com/drive/folders/1Io3ENgiwP6BfX4SlmSzIw_Uha1anWZrt?usp=sharing).

To evaluate on a scene from a dataset:

```bash
python eval.py \
        --model [hscnet|scrnet] \
        --dataset [7S|12S|Cambridge|i7S|i12S|i19S] \
        --scene scene_name \
        --checkpoint /path/to/saved/model/ \
        --data_path /path/to/data/
```

## Training 
You can train the hierarchical scene coordinate network or the baseline regression network by running the following command:

```bash
python train.py \
        --model [hscnet|scrnet] \
        --dataset [7S|12S|Cambridge|i7S|i12S|i19S] \
        --scene scene_name \ # not required for the combined scenes
        --n_iter number_of_training_iterations \
        --data_path /path/to/data/
```


## License

Copyright (c) 2020 AaltoVision.  
This code is released under the [MIT License](LICENSE).

## Acknowledgements

The PnP-RANSAC pose solver builds on [DSAC++](https://github.com/vislearn/LessMore). The sensor calibration file and the normalization translation files for the 7-Scenes dataset are from [DSAC](https://github.com/cvlab-dresden/DSAC). The rendered depth images for the Cambridge Landmarks dataset are from [DSAC++](https://github.com/vislearn/LessMore). 

## Citation

Please consider citing our paper if you find this code useful for your research:  

```
@inproceedings{li2020hscnet,
    title = {Hierarchical Scene Coordinate Classification and Regression for Visual Localization},
    author = {Li, Xiaotian and Wang, Shuzhe and Zhao, Yi and Verbeek, Jakob and Kannala, Juho},
    booktitle = {CVPR},
    year = {2020}
}
```


