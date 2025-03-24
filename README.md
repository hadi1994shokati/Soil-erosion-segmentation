# Description
This repository provides a foundational implementation for detecting soil erosion and deposition traces in high-resolution aerial imagery. Developed as part of our study, [Erosion-SAM: Semantic Segmentation of Soil Erosion by Water](https://doi.org/10.1016/j.catena.2025.108954) (CATENA, 2025), this code fine-tunes the Segment Anything Model (SAM) specifically for erosion analysis, introducing Erosion-SAM.

The dataset includes **405 manually segmented agricultural fields** from erosion-prone areas, covering bare cropland, vegetated cropland, and grassland. 
The implementation is developed in **Python** using the **PyTorch** framework.

# Requirements
To run the code, ensure you have the following dependencies installed:

* Python 3.8+
* PyTorch 2.0+
* NumPy
* OpenCV
* Additional libraries as listed in [requirements.txt](https://github.com/hadi1994shokati/Soil-erosion-segmentation/blob/main/requirements.txt)

# How to use the code?

In our paper, to address the variability in the sizes of agricultural fields, we considered two methods of pre-processing to standardize the image sizes for the model:

**1. Uniform Resizing:** This method involves resizing all field images to **256 × 256 pixels** before inputting them into the model.
* Run: [Resizing.py](https://github.com/hadi1994shokati/Soil-erosion-segmentation/blob/main/Resizing.py) script.

**2. Image Cropping:** In this approach, the training images are divided into multiple **256 × 256 image patches** with a step size of 256 pixels.
* Run: [cropping.py](https://github.com/hadi1994shokati/Soil-erosion-segmentation/blob/main/Cropping.py) script.


Note: The input images and their corresponding segmentation masks are subject to licensing restrictions. If you do not have access to a licensed dataset, please contact the [corresponding author](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/geowissenschaften/arbeitsgruppen/geographie/forschungsbereich/bodenkunde-und-geomorphologie/work-group/people-main-pages/doctoral-students/hadi-shokati/) at hadi.shokati@uni-tuebingen.de.


# Results
All fine-tuned Erosion-SAM models outperformed the baseline SAM, thanks to their adaptation to erosion-specific features. Below is an overview of the segmentation results comparing the baseline SAM with our fine-tuned models:
![image](https://github.com/user-attachments/assets/dbe73432-4690-42bc-9a17-f2a3c4eab6b9)
For detailed charts and additional visuals, read the [full paper](https://doi.org/10.1016/j.catena.2025.108954).

# How to Cite
If you use this code in your research, please cite our paper:

Hadi Shokati, Andreas Engelhardt, Kay Seufferheld, Ruhollah Taghizadeh-Mehrjardi, Peter Fiener, Hendrik P.A. Lensch, Thomas Scholten,
Erosion-SAM: Semantic segmentation of soil erosion by water,
CATENA,
Volume 254,
2025,
108954,
ISSN 0341-8162,
https://doi.org/10.1016/j.catena.2025.108954.


