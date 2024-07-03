# jaxcross: High-Resolution Cross-Correlation Spectroscopy with JAX

Welcome to jaxcross! This is an ongoing project to implement high-resolution cross-correlation spectroscopy using JAX. The goal is to provide a scalable and efficient framework for analyzing high-resolution spectroscopic data, including detrending and cross-correlation.

[![Build Status](

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)](https://www.python.org/downloads/)
[![JAX Version](https://img.shields.io/badge/jax-0.2.19-blue)](


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
    git clone https://github.com/your-username/jaxcross.git
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
from jaxcross import CRIRES

crires = CRIRES(files).read()

