from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)

def train(model: nn.Module,
          train_dataset: Dataset,
          criterion: nn.Module,
          optimizer: torch.optim.Optimizer,
          gmin: float,
          epochs: int,
          batch_size: int):
    
    model.train()
    
    device = next(model.parameters()).device
    
    dataloader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=8,
                            pin_memory=True)

    gradmax = float('inf')
    
    for epoch in range(epochs):
        def closure():
            optimizer.zero_grad()
            y_hat = model(x)

            loss = criterion(y_hat, y)
            loss += (1e-4 / (2*len(train_dataset))) * sum(torch.sum(p * p) for p in model.parameters())
            loss += (1e-4 / (len(train_dataset))) * sum(torch.log(torch.cosh(2.3099 * p)).sum() for p in model.parameters())
            loss.backward()

            return loss
        
        epoch_loss = 0.0

        for x, y in dataloader:
            x = x.to(device=device, non_blocking=True)
            y = y.to(device=device, non_blocking=True)

            #output = model(x)
            loss = optimizer.step(closure)

            #loss.backward()
            #optimizer.step()

            epoch_loss += loss.item() * y.size(0)
        
        with torch.no_grad():
            gradmax = max([p.grad.abs().max() for p in model.parameters()])

        epoch_loss /= len(train_dataset)
        print(f'Epoch: {epoch} | Loss: {epoch_loss:.4f} | GRADMAX: {gradmax:.4f}')

        if (gradmax <= gmin):
            print(f'GRADMAX achieved!')
            break

    return model
