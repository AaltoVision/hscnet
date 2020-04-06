## Abstract

Visual localization is critical to many applications in computer vision and robotics. To address single-image RGB localization, state-of-the-art feature-based methods match local descriptors between a query image and a pre-built 3D model. Recently, deep neural networks have been exploited to regress the mapping between raw pixels and 3D coordinates in the scene, and thus the matching is implicitly performed by the forward pass through the network. However, in a large and ambiguous environment, learning such a regression task directly can be difficult for a single network. In this work, we present a new hierarchical scene coordinate network to predict pixel scene coordinates in a coarse-to-fine manner from a single RGB image. The  network consists of  a series of output layers, each of them conditioned on the previous ones. The final output layer predicts the 3D coordinates and the others produce progressively finer discrete location labels. The proposed method outperforms the baseline regression-only network and allows us to train  compact models which scale robustly to large environments. It sets a new state-of-the-art for single-image RGB localization performance on  the 7-Scenes, 12-Scenes, Cambridge Landmarks  datasets, and three combined scenes. 
Moreover, for large-scale outdoor localization on the Aachen Day-Night dataset, we present a hybrid approach which outperforms existing scene coordinate regression methods, and reduces significantly the performance gap w.r.t. explicit feature matching methods.

## Paper

[arXiv preprint](https://arxiv.org/abs/1909.06216)

## Source Code
[GitHub repo](https://github.com/AaltoVision/hscnet)
(coming soon...)

## BibTeX Citation

```
@inproceedings{li2020hscnet,
    title = {Hierarchical Scene Coordinate Classification and Regression for Visual Localization},
    author = {Li, Xiaotian and Wang, Shuzhe and Zhao, Yi and Verbeek, Jakob and Kannala, Juho},
    booktitle = {CVPR},
    year = {2020}
}
```
