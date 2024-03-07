from torch.utils.data import DataLoader
from hdf_pytorch import H5Dataset
import torchvision
import torch
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

batch_size = 5  # batch size
imsize_w = 180  # for resizing with
imsize_h = 180  # for resizing height
offset = 1.0  # for logarithmic tranfsormation

def test_load_dataset():
    # prepare transforms
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((imsize_w, imsize_h), antialias=True),
            torch.nn.ReLU(inplace=True),  # remove negative values if any
            torchvision.transforms.Lambda(lambda x: torch.log(x + offset)),
        ]
    )

    train_dataloader = DataLoader(
        H5Dataset("./example.h5", transforms=transforms),
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # use pytorch data loader
    for batch_idx, (data_, target_) in enumerate(train_dataloader):
        data_, target_ = data_.to(device), target_.to(device)
        print(batch_idx, data_.shape, target_)
        assert data_.shape == torch.Size([5, 1, 180, 180])
        
def test_plot():
    # See data being loaded
    plt.figure(figsize=(10, 10))
    with h5py.File("example.h5", "r") as hdf:
        for i, (data, target) in enumerate(zip(hdf["data"], hdf["target"])):
            plt.subplot(5, 5, i + 1)
            plt.imshow(np.log(np.where(data > 0, data, 0) + offset))
            plt.title(target)
            plt.axis("off")
            plt.savefig("./example_data.png", dpi=300)
    plt.tight_layout()
    
    assert os.path.isfile("./example_data.png")
