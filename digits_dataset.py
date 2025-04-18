import torch
from torch.utils.data import Dataset

class DigitsDataset(Dataset):
    def __init__(self, filename):
        self.data = []

        with open(filename, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip().split(',')
                line = [float(num) for num in line]
                self.data.append(line)

        self.X = [row[:-1] for row in self.data]
        self.Y = [row[-1] for row in self.data]

        self.classes = list(set(self.Y))

    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float64)
        y = torch.tensor(self.Y[idx], dtype=torch.long)

        return x, y

        

        
        
