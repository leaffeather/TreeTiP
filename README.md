# *TreeTiP*: A *Tree* modeling method based on point clouds with the idea of the simulation of *T*ransport *i*n *P*lants

This name contains another meaning: The method always takes into account the position of the ***Tree TiP***(*i.e.* treetop), which makes the produced tree model more real.
**Note:** There is a common misspelling in the name of the skeleton extraction algorithm in my proposed papers and the correct should be "the Incomplete Simulation of the Tree Water and Nutrient Transport" and "ISTWNT" in short.

## 1 Introduction
This work contains four parts:
- **Separating branches and leaves.** The implement is based on *k*d-tree and like diffusion process from a selected starting point. Only is the branch point cloud used to produce the subsequent tree model.
- **Skeleton extraction.** The implement is inspired by the ecological conclusion pointed out by Dr. Fan when he introduced his work [AdQSM](https://github.com/GuangpengFan/AdQSM) in a lecture that the trees tend to use the shortest path to transport water and nutrients for optimizing resource allocation. In order to improve the algorithm efficiency, a segmental iteration method and multi-threading technology are introduced into. Breakpoint connection and breaking loops are used to ensure the topological correctness of the initial tree skeleton.
- **Skeleton optimization.** The initial skeleton is usually inaccurate and the optimization scheme is designed for improving it. 
    - **Skeleton reconstruction at the stump.** Reconstruct the local skeleton at the stump in a way of using local points as the input of skeleton extraction algorithm which the criterion for layering is height.
    - **Bifurcation optimization.** According to an existed conclusion that the eigenvector corresponding to the maximum eigenvalue of the covariance matrix of the local branch point cloud describes the growth direction of the local branch, by solving an optimization problem, the bifurcation skeleton can tend to go through the center of the branches.
    - **Pruning twigs.**
    - **Skeleton smoothness.**
- **Tree modeling.** **Expected to be updated in mid June 2023.**
 
Parameters that have already been able to be extracted: DBH, trunk volume, total volume, surface area, Branch Length(BL), Branch Chord Length(BCL), Branch Diameter(BD), Branch Height(BH), Branch Arc Height(BAH), Inclination Angle(IA), Azimuth, Axil Angle - Skeleton(AA), Branching Angle(BA)...


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

**Expected to be updated in mid June 2023.**

## 4 Limitations & Code maintenance instructions

1.  The treetop in the model is the centroid point of the local bin. The codes that need to be modified are at/after the part of skeleton extraction.
2.  The codes on measuring branch diameter (the diameter at the base of an offshoot) are erroneous.
3.  In bifurcation optimization, there are something unreasonable about the reclassified bin-s.
4.  The computer measurement of branch length and so on is different from the forestry measurement.
5.  Skeleton reconstruction at the stump does not work in rare cases.
6.  If the structure of a tree is complex and the density of its point cloud is uneven, some bifurcations in its produced model might be wrong.
7.  A better algorithm for separating branches and leaves should be applied.

This work would not be released by myself as I devote myself to doing research on another method of tree modeling based on point clouds. If you find more bugs or want to submit some improvements, please contact me via [leaffeather@foxmail.com](mailto:leaffeather@foxmail.com) and remember to leave your Github address or name (or nickname or anonymity). If they should be useful indeed, I would publicly acknowledge here. Moreover, cooperation is welcomed.

## 5 Citations

If you find this repo useful in your research, please consider citing it and my other works:

**The detailed implement will release after passing the dissertation defense**

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
