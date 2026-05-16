# MATH 411 Equivariance Case Study

Code for the case study in my Math 411 project "Equivariance and Its Applications in Machine Learning." The experiment compares a standard convolutional network to a D4-equivariant network on foreground segmentation of the Oxford-IIIT Pet dataset.

## Setup

Download the [Oxford-IIIT Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) and place it at `data/oxford-iiit-pet/`. Then install dependencies:

```
pip install -r requirements.txt
```

## Usage

Train both models:

```
python train.py
```

Evaluate and print IoU and equivariance error:

```
python evaluate.py
```
