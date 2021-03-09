# Repository to hold code for deep learning based approach to genome annotation

* easel.py has keras-based model creation
* Model takes as input an image-like 2D tensor, which after DNA pre-processing will be a 128x128 normalized tensor with values that correspond with nucleotide value
* Input can either be an encoded long sequence (up to 16k bases) or multiple streams of nucleotide features
