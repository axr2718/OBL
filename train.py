from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn

def train(model: nn.Module,
          train_dataset: Dataset,
          criterion: nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          batch_size: int):
    
    device = next(model.parameters()).device
    
    dataloader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=8,)
                            #pin_memory=True)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, y in dataloader:
            x = x.to(device=device,) #non_blocking=True)
            #print(x)
            y = y.to(device=device,) #non_blocking=True)

            optimizer.zero_grad()

            output = model(x)
            loss = criterion(output, y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * y.size(0)

        print(epoch_loss / len(train_dataset))

    return model
