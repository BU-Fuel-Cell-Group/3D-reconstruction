# 3D Reconstruction Analysis
## Welcome
Welcome to the BU Fuel Cell Group's 3D reconstruction analysis repository! This repository includes code used by lab members to analyze 3D reconstruction volumes, and has been applied with past projects including FIB/SEM reconstruction, uCT 3D reconstruction, and more.

## recon_processing_example.py
The code in this section is designed to function as a starting point for analyses of labeled 3D volumes. Please note that while this program has been largely optimized (numpy functions are used wherever possible), some functions can be quite slow, especially on very large datasets due to the nature of certain calculations (for example, tortuosity calculations require iterating through time steps). These could be rewritten to run in parallel for added efficiency.

The author has run this code largely on a computing cluster, due to the high memory demands of 3D reconstruction datasets. It is recommended that any workflows are first implemented on a small sample dataset.

The package "Skan" (Nunez-Iglesias et. al. 2018) is used for analysis of TPB structures.

## edge_filler.py
A common artifact in SEM images of porous samples are enhanced electron scattering at pore edges. This makes it hard to correctly segment the borders of pores, and can lead to voxel misidentification at their edges, which appear in the segmented volumes as thin stripes at pore edges. While this artifact has minimal impact on particle size and phase fraction calculations, it can have significant impacts on TPB quantification and cause significant over-estimation of TPB density.

To rectify this, the edge_filler.py program corrects for these mis-identified pixels by locating and re-assigning them. 
