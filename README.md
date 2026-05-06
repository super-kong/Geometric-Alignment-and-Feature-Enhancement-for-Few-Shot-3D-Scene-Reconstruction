# Geometric Alignment and Feature Enhancement for Few-Shot 3D Scene Reconstruction with Gaussian Splatting

## About This Repository
This code is directly related to the manuscript submitted to **The Visual Computer**.
Please cite the corresponding paper if you use this code.

## Core Algorithm Principle
The core of our GeoFeat-GS framework integrates geometric awareness, adaptive feature learning, and geometric consistency constraints into 3D Gaussian Splatting (3DGS) to resolve sparse-view reconstruction ambiguity and instability. Key principles are as follows:
- **Geometry-aware 3DGS**: We initialize Gaussian primitives with higher density in geometrically complex regions (e.g., edges) and lower density in flat regions (via sparse-view rough 3D structure estimation). A geometry-aware loss adjusts update steps by local curvature, enhancing stability and accuracy under sparse observations.
- **MAFPN**: The Multi-stage Adaptive Feature Perception Network extracts robust sparse-view features via 3 stages: local feature extraction (lightweight CNN + attention), cross-view adaptive fusion, and global semantic enhancement (transformer). Its view-invariant outputs guide Gaussian optimization.
- **GCC-PSM**: The Geometric Consistency Constraint with Prior-Guided Score Matching ensures cross-view alignment by minimizing reprojection errors between Gaussians and MAFPN features. Prior-guided score matching leverages scene structure priors to reduce reconstruction ambiguity and stabilize optimization.

## Installation

Ubuntu 20.04, CUDA 11.6, Python 3.8.13, Pytorch 1.12.1+cu116 

``````
conda env create --file environment.yaml
conda activate IPSM_cuda116

pip install ./submodules/diff-gaussian-rasterization-confidence ./simple-knn
``````

## Pre-trained Models Preparation

```
mkdir pretrained_models
cd pretrained_models
```

Download [StableDiffusion-v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5), [StableDiffusionInpainting-v1.5](https://huggingface.co/runwayml/stable-diffusion-inpainting), [MiDaS](https://github.com/isl-org/MiDaS), [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) to ```./pretrained_models/```. (NOTE: Stable Diffusion V1.5 and Stable Diffusion Inpainting V1.5 cannot be downloaded from the original repo, but the same weight can be obtained from other clone repo.)

## Data Preparation

### LLFF

1. Download LLFF from [the official download link](https://drive.google.com/file/d/11PhkBXZZNYTD2emdG1awALlhCnkq7aN-/view?usp=drive_link).

2. Run COLMAP to obtain initial point clouds with sparse views:

   ```
   python tools/colmap_llff.py
   ```

3. Randomly select one image from sparse views and run BLIP to obtain its blip-based text results:

    ```
    python ./scripts/script_for_blip.py
    ```

4. The data format is supposed to be:

    ```
    |- <scene>
        |- 3_views
        |- images
        |- images_4
        |- images_8
        |- sparse
        |- blip_rst.txt
        |- poses_bounds.npy
        |- ...
    ```

### DTU

1. Download DTU dataset

   - Download the DTU dataset "Rectified (123 GB)" from the [official website](https://roboimagedata.compute.dtu.dk/?page_id=36/), and extract it.
   - Download masks (used for evaluation only) from [this link](https://drive.google.com/drive/folders/1OEmJcbP0XUVfG647mdYWxEy9yOw7L_Si?usp=drive_link) (backed up from RegNeRF).

2. Preprocess following [DNGaussian](https://github.com/Fictionarry/DNGaussian)

   - Poses: following [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting), run `convert.py` to get the poses and the undistorted images by COLMAP.
   - Render Path: following [LLFF](https://github.com/Fyusion/LLFF) to get the `poses_bounds.npy` from the COLMAP data. (Optional)

3. Run COLMAP to obtain initial point clouds with sparse views:

   ```
   python tools/colmap_dtu.py
   ```

4. Randomly select one image from sparse views and run BLIP to obtain its blip-based text results:

    ```
    python blip_script.py
    ```

5. The data format is supposed to be:

    ```
    |- <scene>
        |- 3_views
        |- images
        |- images_2
        |- images_4
        |- images_8
        |- mask
        |- sparse
        |- blip_rst.txt
        |- poses_bounds.npy
        |- ...
    ```

## Training & Rendering & Evaluating

Train & Render & Evaluate on the LLFF dataset with 3 views:

```
python ./scripts/script_for_llff.py
```

Train & Render & Evaluate on the DTU dataset with 3 views:

```
python ./scripts/script_for_dtu.py
```

## Acknowledgement

This code is developed on [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting), [FSGS](https://github.com/VITA-Group/FSGS), and [DNGaussian](https://github.com/Fictionarry/DNGaussian). Thanks for these great projects!