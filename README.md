# DeepFace-Classifier

## Overview

This project demonstrates a robust pipeline for facial feature extraction and recognition, leveraging advanced machine learning techniques. The solution is guided by the pre-trained FaceNet model and employs innovative strategies to tackle dataset biases and classification challenges.

## Features

**Face Detection:** Utilizes the pre-trained FaceNet model for accurate facial feature extraction.

**Classification:** Implements an SVM classifier for face recognition and clustering.

**Loss Function:** Employs Triplet Loss to ensure accurate feature embeddings in Euclidean space.

**Data Handling:** Includes data augmentation and balancing techniques to address uneven dataset distributions.

## Methodology

Face Detection: Features are extracted using the FaceNet model, a state-of-the-art neural network for face embeddings.

Embedding Optimization: Triplet Loss is applied to minimize intra-class distances while maximizing inter-class distances in embedding space.

## Dataset Challenges:

Uneven image distribution across classes in the LFW dataset.

Bias in SVM classifier toward majority classes.

## Solutions:

Data augmentation to enrich the dataset.

Balancing techniques to ensure uniform class representation.

## Results

Evaluation Metrics:

Confusion matrix for classifier performance assessment.

Visualization of feature embeddings in Euclidean space.

**End Results:
**
Example test image and its corresponding predicted match.
