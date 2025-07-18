import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, random_split

import torchvision
from torchvision.transforms import v2

import os
import matplotlib.pyplot as plt

import json
from tqdm import tqdm
from PIL import Image

plt.style.use('dark_background')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Dataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform

        self.list_name_file = os.listdir(path)
        if 'coords.json' in self.list_name_file:
            self.list_name_file.remove('coords.json')

        self.len_dataset = len(self.list_name_file)

        with open(os.path.join(self.path, 'coords.json'), 'r') as f:
            self.dict_coords = json.load(f)

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, index):
        name_file = self.list_name_file[index]
        path_img = os.path.join(self.path, name_file)

        img = Image.open(path_img)
        coord = self.dict_coords[name_file]

        if self.transform:
            img = self.transform(img)
            coord = torch.tensor(coord, dtype=torch.float32)

        return img, coord


transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.5, ), std=(0.5, ))
    ]
)

dataset = Dataset(path=r"C:\Users\STARLINECOMP\PycharmProjects\Pytorch\content\dataset", transform=transform)


train_set, val_set, test_set = random_split(dataset, [0.7, 0.1, 0.2])
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

class MyModel(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.flatten = nn.Flatten()

        self.liner = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, out)
        )

    def forward(self, inp):
        x = self.conv(inp)
        x = self.flatten(x)
        out = self.liner(x)
        return out

param_model = torch.load(r'C:\Users\STARLINECOMP\PycharmProjects\Pytorch\model_state_dict_11.pt')

model = MyModel(1, 2).to(device)
loss_model = nn.MSELoss()

model.load_state_dict(param_model['model'])

model.eval()
with torch.no_grad():
    total_mae = 0
    for x, targets in test_loader:
        x = x.to(device)
        targets = targets.to(device)

        pred = model(x)
        loss = loss_model(pred, targets)

        total_mae += torch.sum(torch.abs(pred - targets)).item()

    mean_val_mae = total_mae / (len(val_set) * 2)

print(f'test_mae={mean_val_mae:.4f}')

