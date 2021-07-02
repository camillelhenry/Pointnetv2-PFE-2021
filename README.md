# PFE_2021 : PointNet++ implementation  

TO DO LIST : 

- Faire le READ_ME, avec objectif du PFE, comment utiliser les scripts et les adapter à ses propres bases
- Mettre en photo des résultats et des histogrammes de comparaison sur les meilleurs caractéristiques du nuage de points
- Mettre un lien vers le rapport de PFE et le poster 

## About the repo

This repo is an implementation for [PointNet](https://proceedings.neurips.cc/paper/2017/file/d8bf84be3800d12f74d8b05e9b89836f-Paper.pdf) in PyTorch, based on [yanx27](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) repo. 
It's a project realized for my end of study internship to obtain my engineering diploma at INSA (National Institute of Applied Sciences) of Strasbourg. This internship was carried out in the TRIO team of the ICube laboratory and was directed by Tania LANDES. For more information about the project, you can read my master thesis [put the link].

## About the project (Detection of openings by semantic segmentation of 3D indoor/outdoor point clouds: contribution of deep learning)

The goal of the project is to segment point clouds of buildings with the PointNet ++ neural network, and to focus on the results obtained on the classes of openings (windows and doors) in order to automate the Scan-to-BIM process. 

## Install 

The latest codes are tested on Window 10, CUDA 11.1, PyTorch 1.6 and Python 3.8. To run the codes, you also need some GPU devices (the one use for the project is NVIDIA GeForce GTX 1070 with 16 Go of dedicated memory. In order to use the GPU whith Python, you also need to install : 
- NVIDIA Driver for your GPU (you can find it in [NVIDIA's website](https://www.nvidia.com/Download/index.aspx?lang=en-us)
- CUDO Toolkit (you can find the CUDA Toolkit Archive [here](https://developer.nvidia.com/cuda-toolkit-archive). Be sure to check the CUDA Toolkit version that PyTorch currently supports. You can find that information on PyTorch's site.
- cuDNN library (navigate again in [NVIDIA's website](https://developer.nvidia.com/cudnn). Choose to download the version of cuDNN that corresponds to the PyTorch-supported version of the CUDA Toolkit that you downloaded in the last step.
- You can check if python is using your GPU devices with this code : 
```
import torch
torch.cuda.is_available()
```
