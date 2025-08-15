# Brain-Tumor-Segmentation-Using-U-Net-with-Streamlit-Web-Interface



Project Overview

This project implements an AI-powered web app for brain tumor segmentation using MRI images. Users can upload MRI scans and obtain segmentation masks highlighting tumor regions. The backend model is a U-Net, trained for precise pixel-level classification of brain tumors including Glioma, Meningioma, and Pituitary tumors.

Key Features

Upload MRI images (.jpg, .jpeg, .png).

Automatic tumor segmentation using a trained U-Net model.

Display results in two formats:

Original Image

Predicted Mask (Grayscale)

Technologies & Tools

Programming Language: Python

Web Framework: Streamlit

Deep Learning Framework: TensorFlow, Keras

Image Processing: OpenCV, Pillow (PIL)

Numerical Computation: NumPy

Machine Learning Model: U-Net (for segmentation)

Visualization: Matplotlib (optional for plotting masks)


Dataset

The model is trained using the Brain Tumor Segmentation Dataset by atikaakter11 on Kaggle. This dataset comprises MRI images labeled for brain tumor segmentation tasks.
https://www.kaggle.com/datasets/atikaakter11/brain-tumor-segmentation-dataset/data

Model Details

Architecture: U-Net

Input Size: 128 × 128 RGB images


Acknowledgements

U-Net architecture for medical image segmentation.

Streamlit for interactive web apps.

Open-source libraries: TensorFlow, Keras, NumPy, OpenCV, Pillow.
Output: Multi-class segmentation mask

Classes: Background, Glioma, Meningioma, Pituitary

Activation: Softmax

How It Works

Image is uploaded and resized to match model input.

Preprocessing: normalize pixel values to [0,1].

Model predicts segmentation mask for each pixel.

Outputs displayed as:

Grayscale mask for simplified visualization
