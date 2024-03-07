import h5py
from torch.utils.data import Dataset


class H5Dataset(Dataset):
    def __init__(self, h5_path, transforms=None):
        self.h5_file = h5py.File(h5_path, "r")
        self.transform = transforms

    def __getitem__(self, index):
        sample = self.h5_file["data"][index]
        if self.transform is not None:
            sample = self.transform(sample)
        return (
            sample,
            int(self.h5_file["target"][index]),
        )

    def __len__(self):
        return self.h5_file["target"].size


if __name__ == "__main__":
    print(
        """hdf reader adapted for Small Angle Neutron Scattering (SANS)
virtual experiments at KWS-1. To be used with Pytorch."""
    )
