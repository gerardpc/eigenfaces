# Eigenfaces

## Overview

This is a simple Python implementation of the Eigenfaces algorithm for face recognition. 
It is based on the paper [Eigenfaces for Recognition](http://www.face-rec.org/algorithms/pca/jcn.pdf) by Turk and 
Pentland, and follows the formulation given in a lecture by Steven Brunton from the University of Washington.

## Installation

This project uses [poetry](https://python-poetry.org/) for dependency management. To install the dependencies, run

```bash
pip install poetry
poetry install
```

## Usage

Run the script `eigenfaces_script.py` from the eigenfaces folder
to see an example of the algorithm in action. The script will load the
faces from the `data` folder, and then perform the following steps:

1. Calculate SVD of the data matrix (i.e. the matrix whose columns are the flattened images).
2. The eigenfaces are the left singular vectors of the data matrix, reshaped to the original image size
   (i.e., unflattened).
3. Project new faces onto the eigenfaces.

