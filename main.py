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

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
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

model = MyModel(1, 2).to(device)
loss_model = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10)

EPOCHS = 50
train_mae = []
val_mae= []
lr_list = []
best_mae = None
count = 0

for epoch in range(EPOCHS):

    model.train()
    total_mae = 0
    train_loop = tqdm(train_loader, leave=False)
    for x, targets in train_loop:
        x = x.to(device)
        targets = targets.to(device)

        pred = model(x)
        loss = loss_model(pred, targets)

        opt.zero_grad()
        loss.backward()

        opt.step()

        total_mae += torch.sum(torch.abs(pred - targets)).item()

        train_loop.set_description(f'EPOCH [{epoch+1}/{EPOCHS}], train_mae{total_mae:.4f}')

    mean_train_mae = total_mae / (len(train_set) * 2)
    train_mae.append(mean_train_mae)

    model.eval()
    with torch.no_grad():
        total_mae = 0
        for x, targets in val_loader:
            x = x.to(device)
            targets = targets.to(device)

            pred = model(x)
            loss = loss_model(pred, targets)

            total_mae += torch.sum(torch.abs(pred - targets)).item()

        mean_val_mae = total_mae / (len(val_set) * 2)
        val_mae.append(mean_val_mae)

    lr_scheduler.step(mean_val_mae)
    lr = lr_scheduler._last_lr[0]
    lr_list.append(lr)

    print(f'Epoch [{epoch+1}/{EPOCHS}], train_mae{mean_train_mae:.4f}, val_mae={mean_val_mae:.4f}, lr={lr:.4f}')

    if best_mae is None:
        best_mae = total_mae

    if total_mae < best_mae:
        best_mae = total_mae
        count = 0

        checkpoint = {
            'model': model.state_dict(),
            'opt': opt.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'EPOCHS': EPOCHS,
            'save_epochs': epoch
        }
        torch.save(checkpoint, f'model_state_dict_{epoch+1}.pt')
        print(f'на {epoch + 1} эпохе модель сохранила значение mae {mean_val_mae:.4f}')

    else:
        count += 1

    if count > 10:
        print(f'на {epoch} эпохе обучение остановилось значение mae {mean_val_mae:.4f}')
        break


plt.plot(train_mae)
plt.plot(val_mae)
plt.legend(['train_mae', 'val_mae'])
plt.show()
