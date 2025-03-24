# Soil-erosion-segmentation
This repository provides a foundational implementation for detecting soil erosion and deposition traces in high-resolution aerial imagery. Developed as part of our study, [Erosion-SAM: Semantic Segmentation of Soil Erosion by Water](https://doi.org/10.1016/j.catena.2025.108954) (CATENA, 2025), this code fine-tunes the Segment Anything Model (SAM) specifically for erosion analysis, introducing Erosion-SAM.

The data set comprised 405 manually segmented agricultural fields from erosion-prone areas for bare cropland, vegetated cropland, and grassland. 

The implementation is developed in Python and leverages the PyTorch framework.


# How to use the code?

In our paper, to address the variability in the sizes of agricultural fields, we considered two methods of pre-processing to standardize the image sizes for the model:
**1. Uniform Resizing:** This method involves resizing all field images to 256 × 256 pixels before inputting them into the model.
**2. Image Cropping:** In this approach, the training images are divided into multiple 256 × 256 image patches with a step size of 256 pixels.
   
After pre-processing, the results were compared to the baseline SAM. For execution:

To employ the Uniform Resizing approach, please run the [Resizing.py](https://github.com/hadi1994shokati/Soil-erosion-segmentation/blob/main/Resizing.py) script.

To utilize the Image Cropping approach, please run the [cropping.py](https://github.com/hadi1994shokati/Soil-erosion-segmentation/blob/main/Cropping.py) script.

Note: The input images and their corresponding segmentation masks are subject to licensing restrictions. If you do not have access to a licensed dataset, please contact the [corresponding author](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/geowissenschaften/arbeitsgruppen/geographie/forschungsbereich/bodenkunde-und-geomorphologie/work-group/people-main-pages/doctoral-students/hadi-shokati/) at hadi.shokati@uni-tuebingen.de.


# Results
All fine-tuned models outperformed the baseline SAM. An overview of soil erosion segmentation results with the baseline SAM and the fine-tuned models is shown below.
![image](https://github.com/user-attachments/assets/dbe73432-4690-42bc-9a17-f2a3c4eab6b9)


# References
Are you interested in using some of our code for your research and investigations? In that case please cite our paper:

Hadi Shokati, Andreas Engelhardt, Kay Seufferheld, Ruhollah Taghizadeh-Mehrjardi, Peter Fiener, Hendrik P.A. Lensch, Thomas Scholten,
Erosion-SAM: Semantic segmentation of soil erosion by water,
CATENA,
Volume 254,
2025,
108954,
ISSN 0341-8162,
https://doi.org/10.1016/j.catena.2025.108954.


