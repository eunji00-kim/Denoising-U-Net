import os
import numpy as np
import torch

from glob import glob
from torch.utils.data import Dataset, DataLoader


def normalize(image, min_range, max_range):
    image = (image - min_range) / (max_range - min_range)

    return image


class ct_dataset(Dataset):
    def __init__(self, data_path, min_range, max_range, augmentation=False):
        self.files = sorted(glob(os.path.join(data_path, '*.npz')))
        self.min_range = min_range
        self.max_range = max_range
        self.augmentation = augmentation
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        self.path = self.files[idx]
        self.data = np.load(self.path)

        input_image = self.data['input']
        target_image = self.data['target']

        if self.augmentation:
            seed = np.random.randint(43)
            random_state = np.random.RandomState(seed)

            if random_state.rand() < 0.5:
                input_image = np.fliplr(input_image)
                target_image = np.fliplr(target_image)

        self.input = normalize(input_image, self.min_range, self.max_range)
        self.target = normalize(target_image, self.min_range, self.max_range)

        self.input = torch.tensor(self.input.astype(np.float32))
        self.target = torch.tensor(self.target)

        self.input = self.input.unsqueeze(0)
        self.target = self.target.unsqueeze(0)

        return (self.input, self.target)


# check if the code works well

# if __name__ == "__main__":
     
#      ct_dataset = ct_dataset(data_path='npz_dataset/train_dataset', min_range=-1024.0, max_range=3072.0, augmentation=True)
#      print(len(ct_dataset))
#      dataset = DataLoader(ct_dataset, shuffle=False, batch_size=4)
#      print(len(dataset))
#      for data in dataset:
#         input = data[0]
#         target = data[1]
#         print(input.shape, target.shape)
