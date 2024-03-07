from setuptools import setup

if __name__ == "__main__":
    setup(
    name='hdf_pytorch',
    version='1.0',
    description='A module to create a Pytorch Dataset from an hdf5 file.',
    author='Jose Robledo',
    author_email='j.robledo@fz-juelich.de',
    packages=['hdf_pytorch'],  #same as name
    install_requires=['torch', 'h5py'], #external packages as dependencies
    )