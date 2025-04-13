import torch.optim.sgd
from mlp import MLP
import torchvision
from torchvision.transforms import v2
import torch
import torch.nn as nn
from train import train
from config import Config
from test import test

if __name__ == '__main__':

    config = Config()

    seed = 42
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    transform = torchvision.transforms.Compose([v2.ToImage(),
                                                v2.ToDtype(torch.uint8, scale=True),
                                                v2.ToDtype(torch.float32, scale=True),
                                                v2.Normalize(mean=(0.5,), std=(0.5,))])


    train_dataset = torchvision.datasets.MNIST(root='./data', 
                                                      train=True,
                                                      transform=transform,
                                                      download=False)

    test_dataset = torchvision.datasets.MNIST(root='./data',
                                                     train=False,
                                                     transform=transform,
                                                     download=False)
    
    input_dim = train_dataset[0][0].reshape(-1).size(0)
    output_dim = len(train_dataset.classes)
    
    model = MLP(input_dim=input_dim, output_dim=output_dim)
    model.to(device=device)
    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn='strong_wolfe')

    trained_model = train(model=model,
                          train_dataset=train_dataset,
                          criterion=criterion,
                          optimizer=optimizer,
                          gmin=config.gmin,
                          epochs=config.epochs,
                          batch_size=len(train_dataset))
    
    print("Testing the trained model on training data.")
    metrics = test(model=model,
                test_dataset=train_dataset,
                device=device)
    acc = metrics['accuracy']
    precision = metrics['precision']
    recall = metrics['recall']
    f1 = metrics['f1']
    print(f'Accuracy: {acc * 100:.4f} | Precision {precision* 100:.4f} | Recall {recall* 100:.4f} | F1: {f1* 100:.4f}')

    
    print("Testing the trained model on testing data.")
    metrics = test(model=model,
                test_dataset=test_dataset,
                device=device)
    acc = metrics['accuracy']
    precision = metrics['precision']
    recall = metrics['recall']
    f1 = metrics['f1']
    print(f'Accuracy: {acc * 100:.4f} | Precision {precision* 100:.4f} | Recall {recall* 100:.4f} | F1: {f1* 100:.4f}')

    torch.save(model.state_dict(), './mnist-mlp.pt')
