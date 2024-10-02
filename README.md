# From Field to Canvas: Bridging Sports and Art with Deep Learning Pose Analysis


This project is an Image Action Recognition system that uses the Stanford40 action dataset. It allows users to interact with the system through a Graphical User Interface (GUI) to recognize actions performed in images and we use VIT model as the base model. The Stanford40 dataset contains images of 40 different human action classes, such as "applauding", "fishing", "holding an umbrella" etc.

##Abstract
In this project, we aim to develop a robust deep-learning system capable of analyzing live sports images and correlating them with existing art pieces. Our approach integrates advanced techniques in pose estimation and posture analysis to identify similarities between the dynamic poses captured in sports imagery and the static compositions found in art.

##Introduction
Our primary goal is to create a model that accurately identifies and matches poses from live sports images with corresponding poses in an art image dataset. This will involve the extraction of key points in the body postures of athletes captured in sports images, followed by a comprehensive search for analogous poses within the art dataset.

The matching process will entail a thorough comparison of pose features, body proportions, and overall composition to identify the closest matches between the two domains. By leveraging deep learning methodologies, we anticipate achieving high levels of accuracy and precision in this matching process, facilitating the exploration of connections between athletic movement and artistic expression.

Ultimately, our proposed system has the potential to offer valuable insights into the shared elements of human movement across different visual mediums, bridging the gap between the worlds of sports and art through computational analysis and interpretation.

This project draws direct inspiration from the X account @ArtButSports. Note that this account does not use AI to generate any of its Sports to Arts matches.

##Previous Work
Pose estimation has increasingly become a crucial component in understanding and interpreting human dynamics in images and videos. Through our preliminary research for background and ideas for this project, we explored the significant advancements in the field, particularly focusing on the application of deep learning techniques in pose estimation, which forms the foundation for our project aimed at matching sports images with art pieces based on pose similarity. 

We were recommended to read Michael Perez’s and Corey Toler-Franklin’s, “CNN-based action recognition and pose estimation for classifying animal behavior from videos: a survey,” which provided a comprehensive overview of CNN-based methods for action recognition and pose estimation, focusing on animal behaviors but offering valuable insights applicable to human pose estimation. Their survey highlights the evolution of pose estimation techniques and their increasing accuracy and robustness, driven by advancements in CNN architectures and training datasets. The adaptability of these methods to non-human subjects hints at their potential versatility and precision, which is critical for our application. While our model will mainly work with human subjects, especially for its inputs, the dataset that we utilize contains depictions of humans in addition to other living beings as well as inanimate objects.

The 2020 paper, “Action recognition using pose estimation with artificial 3D coordinates and CNN” by Jisu Kim and Deokwoo Lee introduces an innovative approach to action recognition by incorporating artificial 3D coordinates into a CNN framework. The researchers developed a method that maps 2D images into a 3D pose space, enhancing the accuracy of pose detection under varied and complex environments. This approach is particularly relevant to our project as it underscores the potential of using enhanced pose dimensionality (from 2D to 3D) to improve the matching accuracy between live sports actions and static art poses.

The integration of multi-source data is also beneficial for handling the complexity and variability of poses in live sports images, suggesting a pathway for robust identification of poses that correspond closely with historical art pieces. This also partially relates to our second class project, where we conducted data augmentation to better help the model predict the colorization of images. 

These studies collectively emphasize the importance of advanced pose estimation techniques and their application across various domains. Our project aims to utilize the findings of these articles to better inform the model we end up utilizing to refine the accuracy and precision of matching dynamic sports poses with art representations that could be either static or dynamic. 


##Overview
The integration of art and athletics through computational analysis presents a unique challenge: accurately identifying and matching dynamic poses in sports images with their closest static counterparts in a diverse array of art pieces that don’t explicitly depict these specific sports actions. This research project leverages advanced deep learning techniques to address this challenge, developing a robust system that not only recognizes but also correlates poses across these visually distinct domains.

Our methodology involves a multi-stage process, beginning with the assembly of comprehensive datasets. We curated a collection of 15,000 live sports images across 100 different sports and an equivalent number of art images representing various styles and mediums. This rich dataset provides the foundation for our pose estimation and matching algorithms.

We employed a pre-trained Image Action Recognition model, which we adapted to identify key points and pose features in both sets of images, based on the 40 distinct poses it was trained on. Our matching algorithm, which computes similarities based on pose probabilities and spatial configurations, identifies the art piece that most closely mirrors the pose in a given sports image.

The results highlight the efficacy of our approach in bridging the gap between the dynamic movements captured in sports imagery and the expressive poses found in art.

Technical Description
For our data collection, we began by assembling a diverse dataset of live sports images, encompassing 100 sports and poses. The dataset we used is the 100 Sports Image Classification dataset. It contains 15,000 images covering 100 different sports. Simultaneously, we compiled and explored existing databases containing paintings and other art pieces featuring human and non-human subjects in various poses to ensure comprehensive coverage. The art dataset we used is the Art Images: Drawing/Painting/Sculptures/Engravings dataset. It contains 15,000 images from five different types of art: Drawings and watercolors, Works of painting, Sculpture, Graphic Art, and Iconography.


## Acknowledgments

- The Stanford40 dataset was created and made publicly available by the Stanford Vision Lab.
- The VIT model used in this project were developed by [jeonsworld](https://github.com/jeonsworld/ViT-pytorch).





