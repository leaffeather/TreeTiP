# *TreeTiP*: A *Tree* modeling method based on point clouds with the idea of the simulation of *T*ransport *i*n *P*lants

This name contains another meaning: The method always takes into account the position of the ***Tree TiP***(*i.e.* treetop), which makes the produced tree model more real.

**Note:** There is a common misspelling in the name of the skeleton extraction algorithm in my proposed papers and the correct should be "the Incomplete Simulation of the Tree Water and Nutrient Transport" and "ISTWNT" in short.

<center> ![TreeTiP](https://github.com/leaffeather/images/blob/main/TreeTiP.jpg?raw=true) </center>

## 1 Introduction
This work contains four parts:
- **Separating branches and leaves.** The implement is based on *k*d-tree and like diffusion process from a selected starting point. Only is the branch point cloud used to produce the subsequent tree model.
- **Skeleton extraction.** The implement is inspired by the ecological conclusion pointed out by Dr. Fan when he introduced his work [AdQSM](https://github.com/GuangpengFan/AdQSM) in a lecture that the trees tend to use the shortest path to transport water and nutrients for optimizing resource allocation. In order to improve the algorithm efficiency, a segmental iteration method and multi-threading technology are introduced into. Breakpoint connection and breaking loops are used to ensure the topological correctness of the initial tree skeleton.
- **Skeleton optimization.** The initial skeleton is usually inaccurate and the optimization scheme is designed for improving it. 
    - **Skeleton reconstruction at the stump.** Reconstruct the local skeleton at the stump in a way of using local points as the input of skeleton extraction algorithm which the criterion for layering is height.
    - **Bifurcation optimization.** According to an existed conclusion that the eigenvector corresponding to the maximum eigenvalue of the covariance matrix of the local branch point cloud describes the growth direction of the local branch, by solving an optimization problem, the bifurcation skeleton can tend to go through the center of the branches.
    - **Pruning twigs.**
    - **Skeleton smoothness.**
- **Tree modeling.** With radii calculated by least square circle fitting, an initial model can be produced. Users should observe it and find a height where radii of cylinders nearby look normal. Derive radii at other places in view of radii of models near the specified height according to some forestry principles. With the radii and skeleton, the final tree model is the intersection of every 3D convex hull expressing part of a branch.
 
It should be emphasized that **the produced model has an ideal pipeline shape**. Parameters that have already been able to be extracted: DBH, trunk volume, total volume, surface area, Branch Length(BL), Branch Chord Length(BCL), Branch Diameter(BD), Branch Height(BH), Branch Arc Height(BAH), Inclination Angle(IA), Azimuth, Axil Angle - Skeleton(AA), Branching Angle(BA)...


## 2 Environment

|            Item           |   Version  |
| :-----------------------: | :--------: |
|             OS            | Windows 11 |
|     Visual Studio 2019    |  16.11.17  |
|            PCL            |   1.11.1   |
| Boost (included with PCL) |    1.74    |
| Eigen (included with PCL) |      3     |
|  VTK (included with PCL)  |     8.2    |
|            CGAL           |    5.5.1   |

## 3 Instruction

Except manual compiling, you can also download the precompiled executable in **Releases**.

### 3.1 Requirements

1.  Microsoft Visual C++ x86 (and x64 when running on 64-bit Windows OS) runtimes (Especially **2019 redistributable**);
2.  A **.pcd** file of the input preprocessed tree point cloud (**Branches should be scanned as completely as possible**);
3.  PCL should be installed and relative paths in environment variables should be set correctly, or you should copy any **pcl_\*.dll** and **vtk\*.dll** to **bin\\**. It is suggested that the version of PCL should be the same as what are in Environment.

### 3.2 Usage
1. Delete all files in **data\\**;
2. Rename your **.pcd** tree point cloud as **test.pcd** and copy the renamed file to **data\\**;
3. Execute **Step0** to **Step5** in sequence. If you want to keep the same coordinate in the output, you should rename your .pcd file as **test_trans.pcd** and copy the renamed file to **data\\** and skip **Step0**. If you have a better method to separate branches and leaves, you can also rename your .pcd file as **test_trans_branch.pcd** and copy the renamed file to **data\\** and skip **Step1**. In **Step5**, you need specify a height manually. Please follow the prompt on the screen.
4. The produced tree skeleton is **data\\test_trans_branch_ske.obj**(or **data\\test_branch_ske.obj** if not executing **Step0**) and the produced tree model is **data\\test_trans_branch_model.obj**(or **data\\test_branch_model.obj** if not executing **Step0**).

**Note:** (1) Sometimes you might need to adjust a parameter for separating branches and leaves. **Right click** on **Step1** and click on **Edit**. Change the magnification `10` in  `-m 10`; (2) Information shown during **Step5** is only available without the window for model display closed; (3) You can edit these **.bat** files for more convenient operations.

### 3.3 About parameters

Parameters usually do not need to be adjusted. If you want to know the meanings of one or more of them, please refer to corresponding code annotations.

## 4 Limitations & Code maintenance instructions

1.  The treetop in the model is the centroid point of the local bin. The codes that need to be modified are at/after the part of skeleton extraction.
2.  The codes on measuring branch diameter (the diameter at the base of an offshoot) are erroneous.
3.  In bifurcation optimization, there are something unreasonable about the reclassified bin-s.
4.  The computer measurement of branch length and so on is different from the forestry measurement.
5.  Skeleton reconstruction at the stump does not work in rare cases.
6.  If the structure of a tree is complex and the density of its point cloud is uneven, some bifurcations in its produced model might be wrong.
7.  A better algorithm for separating branches and leaves should be applied. But according to features differing from the current, the surface reconstruction might need to be used to ensure uniform surface density at the same time. If not doing so, the step length would possibly be too large to produce a more accurate skeleton.

This work would not be released by myself as I devote myself to doing research on another method of tree modeling based on point clouds. If you find more bugs or want to submit some improvements, please contact me via [leaffeather@foxmail.com](mailto:leaffeather@foxmail.com) and remember to leave your Github address or name (or nickname or anonymity). If they should be useful indeed, I would publicly acknowledge here. Moreover, cooperation is welcomed.

## 5 Citations

If you find this repo useful in your research, please consider citing it and my other works:
```
@mastersthesis{YangMastersDissertation, 
author = {Yang, Jie}, 
title = {Tree skeleton extraction and modeling based on TLS data and their applications},
school = {College of Forestry, Nanjing Forestry University, China}, 
year = {2023},
month = {June}
}
```

```
@Article{f13101534,
AUTHOR = {Yang, Jie and Wen, Xiaorong and Wang, Qiulai and Ye, Jin-Sheng and Zhang, Yanli and Sun, Yuan},
TITLE = {A Novel Algorithm Based on Geometric Characteristics for Tree Branch Skeleton Extraction from LiDAR Point Cloud},
JOURNAL = {Forests},
VOLUME = {13},
YEAR = {2022},
NUMBER = {10},
ARTICLE-NUMBER = {1534},
URL = {https://www.mdpi.com/1999-4907/13/10/1534},
ISSN = {1999-4907},
DOI = {10.3390/f13101534}
}
```

```
@Article{rs14236097,
AUTHOR = {Yang, Jie and Wen, Xiaorong and Wang, Qiulai and Ye, Jin-Sheng and Zhang, Yanli and Sun, Yuan},
TITLE = {A Novel Scheme about Skeleton Optimization Designed for ISTTWN Algorithm},
JOURNAL = {Remote Sensing},
VOLUME = {14},
YEAR = {2022},
NUMBER = {23},
ARTICLE-NUMBER = {6097},
URL = {https://www.mdpi.com/2072-4292/14/23/6097},
ISSN = {2072-4292},
DOI = {10.3390/rs14236097}
}
```

## 6 An initiative
I think that it is urgent to quantify or standardize forestry measurement indices in a way of clear mathematical definitions, which can bring convenience to program and improve comparability and evaluability between models produced by different methods. If you are interested in or working on this, you can also contact me for further communication.
