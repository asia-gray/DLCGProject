# From Field to Canvas: Bridging Sports and Art with Deep Learning Pose Analysis

by Asia Gray, Miira Efrem, and Dina Cazun-Moreno

<img width="652" alt="Screenshot 2024-10-02 at 3 21 08 PM" src="https://github.com/user-attachments/assets/4f9369bc-8d90-4d27-903e-e03a09b12091">


This project is an Image Action Recognition system that uses the Stanford40 action dataset. It allows users to interact with the system through a Graphical User Interface (GUI) to recognize actions performed in images and we use VIT model as the base model. The Stanford40 dataset contains images of 40 different human action classes, such as "applauding", "fishing", "holding an umbrella" etc.

## Abstract
In this project, we aim to develop a robust deep-learning system capable of analyzing live sports images and correlating them with existing art pieces. Our approach integrates advanced techniques in pose estimation and posture analysis to identify similarities between the dynamic poses captured in sports imagery and the static compositions found in art.

## Introduction
Our primary goal is to create a model that accurately identifies and matches poses from live sports images with corresponding poses in an art image dataset. This will involve the extraction of key points in the body postures of athletes captured in sports images, followed by a comprehensive search for analogous poses within the art dataset.

The matching process will entail a thorough comparison of pose features, body proportions, and overall composition to identify the closest matches between the two domains. By leveraging deep learning methodologies, we anticipate achieving high levels of accuracy and precision in this matching process, facilitating the exploration of connections between athletic movement and artistic expression.

Ultimately, our proposed system has the potential to offer valuable insights into the shared elements of human movement across different visual mediums, bridging the gap between the worlds of sports and art through computational analysis and interpretation.

This project draws direct inspiration from the X account @ArtButSports. Note that this account does not use AI to generate any of its Sports to Arts matches.


##  Overview
The integration of art and athletics through computational analysis presents a unique challenge: accurately identifying and matching dynamic poses in sports images with their closest static counterparts in a diverse array of art pieces that don’t explicitly depict these specific sports actions. This research project leverages advanced deep learning techniques to address this challenge, developing a robust system that not only recognizes but also correlates poses across these visually distinct domains.

Our methodology involves a multi-stage process, beginning with the assembly of comprehensive datasets. We curated a collection of 15,000 live sports images across 100 different sports and an equivalent number of art images representing various styles and mediums. This rich dataset provides the foundation for our pose estimation and matching algorithms.

We employed a pre-trained Image Action Recognition model, which we adapted to identify key points and pose features in both sets of images, based on the 40 distinct poses it was trained on. Our matching algorithm, which computes similarities based on pose probabilities and spatial configurations, identifies the art piece that most closely mirrors the pose in a given sports image.

The results highlight the efficacy of our approach in bridging the gap between the dynamic movements captured in sports imagery and the expressive poses found in art.

<img width="595" alt="Screenshot 2024-10-02 at 3 25 03 PM" src="https://github.com/user-attachments/assets/9f9b1c9a-df97-4bb3-a74a-a77d66de3ee1">


## Acknowledgments

- The Stanford40 dataset was created and made publicly available by the Stanford Vision Lab.
- The VIT model used in this project were developed by [jeonsworld](https://github.com/jeonsworld/ViT-pytorch).
- Inspiration was directly drawn from the X (formerly known as Twitter) account: @ArtButSports







