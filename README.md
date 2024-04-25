<br />
<p align="center">

  <h3 align="center">Acoustic Hackaton</h3>

  <p align="center">
    
  </p>
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
    </li>
    <li>
      <a href="#dataset-presentation">Dataset Presentation</a>
    </li>
    <li>
      <a href="#model-presentation">Models Presentation</a>
      <ul>
        <li><a href="#linear-regression">Built With</a></li>
        <li><a href="#k-nearest-neighbors-(knn)">Built With</a></li>
        <li><a href="#random-forest">Built With</a></li>
        <li><a href="#vggish-pretrained">Built With</a></li>
      </ul>
    </li>
  </ol>
</details>

## About The Project

This project aims to develop a method for human localization using room acoustics and multimodal analysis. The proposed method will utilize audio data to accurately estimate the position of individuals within a room.

### Built With
* [Jupyter Notebook](https://jupyter.org/)
* [Python](https://www.python.org/)
* [Pytorch](https://pytorch.org/)
* [Librosa](https://librosa.org/)

## Getting Started

This is the order in which each of the jupyter notebook need to be used in order to use our Models.

### How to use

1. Open the ```download_dataset.ipynb``` and run all the cells.
This step is to download the Dataset locally and access it easily.

2. You can choose one of the models and open it :
- ```linear_regression.ipynb```
- ```knn.ipynb```
- ```vgg.ipynb```
- ```random_forest.ipynb```
- ```VGGish_pretrained.ipynb``` - A voir

3.  (Optional) You can open ```visuals.ipynb``` and run all the cells to visualize the dataset
### Dataset Presentation

The dataset consists of three folders: Empty, Human 1, and Human 2.

- Empty: This folder contains recordings of an empty room, with no human presence.

- Human 1: This folder contains recordings of a single person, Human 1, in a room. The folder contains three files:
    - centroid.txt: This file contains the x, y, and z coordinates of the human over time.
    - deconvoled_trim.wav: This file contains the audio signal after it has been deconvolved from the room reverberation.
    - skeletons.txt: This file contains the skeletal data of the human, which is not used in this dataset.

- Human 2: This folder contains recordings of a single person, Human 2, in a room. The folder contains three files:
    - centroid.txt: This file contains the x, y, and z coordinates of the human over time.
    - deconvoled_trim.wav: This file contains the audio signal after it has been deconvolved from the room reverberation.
    - skeletons.txt: This file contains the skeletal data of the human, which is not used in this dataset.

### Models Presentation

#### Linear Regression

Linear regression is a statistical model that attempts to model the relationship between two or more variables by fitting a linear equation to observed data.

#### K-Nearest Neighbors (KNN)

KNN is a non-parametric algorithm for classification and regression. In classification, KNN classifies a new data point based on the majority class of its k nearest neighbors in the training data. In regression, KNN predicts the value of a new data point based on the average value of its k nearest neighbors in the training data.

#### Random Forest

Random forest is an ensemble learning method that combines multiple decision trees to make more accurate predictions. The final prediction is made by averaging the predictions of all the trees in the forest.

#### VGGish Pretrained

VGGish is a pre-trained convolutional neural network (CNN) model for audio feature extraction. It is trained on a massive dataset of audio clips with labels, and it is able to extract high-level features from audio signals.
