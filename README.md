# Self-Learning Restriction-Based Governance of Multi-Agent Systems
Code package for the PhD thesis of [Michael Oesterle](mailto:michaeloesterle01@gmail.com).

## Overview
This repository contains the code for the experiments described in Chapters 5-9 of the PhD thesis [Self-Learning Restriction-Based Governance of Multi-Agent Systems](thesis.pdf) by Michael Oesterle. Each chapter (i.e., subfolder) contains a Jupyter notebook `experiments.ipynb` with the actual experiments (or a folder `/experiments` if there are multiple experiments), while additional source files, data files and output files can be found in the folders `/src`, `/data`, and `/results`, respectively. Executing the notebooks reproduces the experimental results as reported in the thesis.


## Installation
You might want to create a virtual environment to keep the dependencies separated from other environments. With your target environment activated, execute
```console
$ pip install -r requirements.txt
```
to install the required dependencies.


## Usage
The experiments are implemented in the respective `experiments.ipynb` files. These files contain both the code and the instructions to reproduce the results shown in the thesis, making them self-explanatory and easy to follow.

Depending on the experiment, there are also sub-folders with source code (`/src`), data (`/data`) and results (`/results`; this can be text files and/or graphs).


## Citation
If you use this code or the thesis in your research, please cite it as
```bibtex
@phdthesis{oesterle-2024,
    author = {Michael Oesterle},
    title = {Self-Learning Restriction-Based Governance of Multi-Agent Systems},
    school = {University of Mannheim},
    address = {Mannheim, Germany},
    year = {2024}
}
```
or use the included [citation file](citation.cff).