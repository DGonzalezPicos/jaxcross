# jaxcross: High-Resolution Cross-Correlation Spectroscopy with JAX

Welcome to jaxcross! This is an _ongoing_ project to implement high-resolution cross-correlation spectroscopy using JAX. The goal is to provide a scalable and efficient framework for analyzing high-resolution spectroscopic data, including detrending and cross-correlation.
    

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)](https://www.python.org/downloads/)
![JAX Version](https://img.shields.io/badge/jax-0.2.19-blue)

## Introduction

jaxcross is designed to perform high-resolution cross-correlation spectroscopy using JAX. It is tailored to handle and analyze CRIRES+ time-series high-resolution spectra. The goal is to provide a scalable and efficient framework combining data processing analysis, eventually integrating with a retrieval framework to infer atmospheric properties from high-resolution spectra.

## Features

- Detrending with PCA or SysRem for time-series high-resolution spectra.
- Cross-correlation of atmospheric models with high-resolution spectra.
- Modular design for extensibility.

## Installation

To install the required dependencies and set up the environment, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/DGonzalezPicos/jaxcross.git
    cd jaxcross
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To use the tools provided in this project, you can follow the example below:

```python
import pathlib
from jaxcross import CRIRES

path = pathlib.Path("/home/dario/phd/pycrires/pycrires/product/obs_staring/")
files = sorted(path.glob("cr2res_obs_staring_extracted_*.fits"))

iOrder, iDet = 1,1
crires = CRIRES(files).read()

data = crires.order(iOrder).detector(iDet) 
print(data.flux.shape) # (100, 2048)

fig, ax = plt.subplots(5,1,figsize=(10,4))
data.imshow(ax=ax[0])

data.trim(20,20, ax=ax[1])
data.normalise(ax=ax[2])
data.imshow(ax=ax[2])
data.PCA(4, ax=ax[3])
data.gaussian_filter(15, ax=ax[4])
plt.show()
```