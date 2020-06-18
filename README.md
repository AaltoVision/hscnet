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



## License

Copyright (c) 2020 AaltoVision.  
This code is released under the MIT License.

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


