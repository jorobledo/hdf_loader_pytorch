# hdf_pytorch

Code used for loading the Small Angle Neuron Scatterin virtual experiment dataset from hdf5 files in Pytorch. This repository is related to [Small Angle Neutron Scattering (SANS) virtual experiments at KWS-1](https://doi.org/10.5281/zenodo.10119316) (Zenodo database).

With small modifications regarding the hdf file structure, it can be used to load other hdf datasetsin `PyTorch`. 


## Installation

1 - Clone repository:

`git clone https://github.com/jorobledo/hdf_pytorch.git`

2 - Navigate to directory:

`cd hdf_pytorch`

3 - Install with default dependencies:

`pip install .` 

This hdf data loader is for `pytorch`, so you can fix dependencies to your specific environment.

## Use example

Check `/tests/test.py` for an example of how it is intended to be used in Pytorch with a [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).

