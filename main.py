from mlp import MLP
import torchvision
from torchvision.transforms import v2
import torch
import torch.nn as nn
from train import train

if __name__ == '__main__':

    seed = 42
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    transform = torchvision.transforms.Compose([v2.ToImage(),
                                                v2.ToDtype(torch.uint8, scale=True),
                                                v2.ToDtype(torch.float32, scale=True),
                                                v2.Normalize(mean=(0.5,), std=(0.5,))])


    train_dataset = torchvision.datasets.FashionMNIST(root='./data', 
                                                      train=True,
                                                      transform=transform,
                                                      download=False)

    test_dataset = torchvision.datasets.FashionMNIST(root='./data',
                                                     train=False,
                                                     transform=transform,
                                                     download=False)
    
    input_dim = train_dataset[0][0].reshape(-1).size(0)
    output_dim = len(train_dataset.classes)
    
    model = MLP(input_dim=input_dim, output_dim=output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3, momentum=0.5, fused=True)

    trained_model = train(model=model,
                          train_dataset=train_dataset,
                          criterion=criterion,
                          optimizer=optimizer,
                          epochs=20,
                          batch_size=128)