# hdf_pytorch

Code used for loading the Small Angle Neuron Scatterin virtual experiment dataset from hdf5 files in Pytorch. This repository is related to [Small Angle Neutron Scattering (SANS) virtual experiments at KWS-1]() (Zenodo database).

With small modifications regarding the hdf file structure, it can be used to load any hdf dataset. 

## Use example

Check `/tests/test.py` for an example of how it is intended to be used in Pytorch with a [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).
