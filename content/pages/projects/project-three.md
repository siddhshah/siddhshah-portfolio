---
type: ProjectLayout
title: Semantic Segmentation
colors: colors-a
date: '2024-12-20'
client: Awesome client
description: >-
  Uses a convolutional neural network to perform background-foreground
  separation on layered elements in varied-resolution media.
media:
  type: ImageBlock
  url: /images/img.png
  altText: Project image
featuredImage:
  type: ImageBlock
  url: /images/img.png
  altText: altText of the image
  caption: Caption of the image
  elementId: ''
---
This project involves semantic segmentation, a technique in computer vision or image-based machine learning where each pixel in an image is labeled or annotated.

I obtained a dataset of hundreds of traffic camera images in the suburbs of Chicago, IL, where each image had a corresponding set of annotations.

The goal of this neural network was to separate vehicles from their backgrounds, where performance was measured through Intersection over Union (IoU) metrics (a higher IoU \~ greater accuracy in vehicle detection).

These images had varied resolutions, perspectives, and densities of vehicles in different parts of the images. Some images had a greater amount of background elements (more trees, road signs, etc.) while some were more empty. This subtle variance posited the dataset as a good fit for the neural network's training loop.
