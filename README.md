# cs583_final_project_classify
Final project for CS 583 Computer Vision.
Image Classification using PyTorch.

## Overview
This uses a VGG 16 network architecture to classify images as either canine or feline.
The dataset used is a subset of Tiny ImageNet by grouping 11 original classes into two
new classes, canine and feline.

The classifier is implemented using PyTorch, and setup to run in Colab, by importing
code from this GitHub repo.

## Running the Code

The colab notebooks will download the original Tiny ImageNet dataset, setup the subset of it
with labels needed for this model, train the model, visualizing the loss and accuracy, then
finally run the model on the test portion of the dataset.

There are several colab notebooks to run different variants of training for the classifer.

* [cs583_final_project_classifier_from_scratch.ipynb](cs583_final_project_classifier_from_scratch.ipynb) - Trains the model from scratch.
* [cs583_final_project_classifier_transfer.ipynb](cs583_final_project_classifier_transfer.ipynb) - Uses transfer learning based on pretrained model weights from ImageNet.
* [cs583_final_project_classifier_custom_loss.ipynb](cs583_final_project_classifier_custom_loss.ipynb) - Trains the model from scratch using a custom implementation of cross entropy loss.

If you view the notebooks in GitHub they include an "Open in Colab" link that can be used to run them.

To change between the optimizer algorithm (i.e. SGD, ADAM, Adadelta) it is necessary to edit the block in the notebook
where the variable optimizer_ft is assigned. Any of the four optimizer setups documented in the paper are present but
commented out so they can be changed easily.

Each training run takes around 40 minutes to train for 30 epochs.

## Paper
The file [paper/Image_Classification_using_VGG_Network_and_Transfer_Learning.pdf](paper/Image_Classification_using_VGG_Network_and_Transfer_Learning.pdf) contains the report on this work.

## References
* K Simonyan, A Zisserman, Very Deep Convolutional Networks for Large-Scale Image Recognition arXiv preprint arXiv:1409.1556
* Y Ma, Tiny ImageNet Challenge  cs231n.stanford.edu, http://cs231n.stanford.edu/reports/2017/pdfs/935.pdf

For a complete set of references see the paper [Image Classification using VGG Network and Transfer Learning](paper/Image_Classification_using_VGG_Network_and_Transfer_Learning.pdf)
