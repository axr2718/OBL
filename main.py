from mlp import MLP
import torchvision
from torchvision.transforms import v2
import torch
import torch.nn as nn
from train import train
from config import Config
from test import test
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':

    config = Config()

    seed = 42
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'gmin is {config.gmin}')

    transform = torchvision.transforms.Compose([v2.ToImage(),
                                                v2.ToDtype(torch.uint8, scale=True),
                                                v2.ToDtype(torch.float64, scale=True),
                                                v2.Normalize(mean=(0.5,), std=(0.5,))])


    train_dataset = torchvision.datasets.CIFAR10(root='./data', 
                                                      train=True,
                                                      transform=transform,
                                                      download=True)

    test_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                     train=False,
                                                     transform=transform,
                                                     download=True)
    
    input_dim = train_dataset[0][0].reshape(-1).size(0)
    output_dim = len(train_dataset.classes)
    hidden_units = config.hidden_units
    
    model = MLP(input_dim=input_dim, hidden_units=hidden_units, output_dim=output_dim)
    model.to(device=device)
    #model.double()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn='strong_wolfe')

    print(f'Training model with {hidden_units} hidden units.')
    trained_model = train(model=model,
                          train_dataset=train_dataset,
                          criterion=criterion,
                          optimizer=optimizer,
                          gmin=config.gmin,
                          epochs=config.epochs,
                          batch_size=len(train_dataset))
    
    print(f'Testing the trained model with {hidden_units} hidden units on training data.')
    metrics = test(model=model,
                test_dataset=train_dataset,
                device=device)
    acc = metrics['accuracy']
    precision = metrics['precision']
    recall = metrics['recall']
    f1 = metrics['f1']
    print(f'Accuracy: {acc * 100:.4f} | Precision {precision* 100:.4f} | Recall {recall* 100:.4f} | F1: {f1* 100:.4f}')

    
    print(f'Testing the trained model with {hidden_units} hidden units on testing data.')
    metrics = test(model=model,
                test_dataset=test_dataset,
                device=device)
    acc = metrics['accuracy']
    precision = metrics['precision']
    recall = metrics['recall']
    f1 = metrics['f1']
    print(f'Accuracy: {acc * 100:.4f} | Precision {precision* 100:.4f} | Recall {recall* 100:.4f} | F1: {f1* 100:.4f}')

    torch.save(model.state_dict(), './mnist-mlp-5.pt')

    print(f'Training bogus model with {10} hidden units.')
    bogus_model = MLP(input_dim=input_dim, hidden_units=10, output_dim=output_dim)
    bogus_model.to(device=device)
    bogus_criterion = nn.CrossEntropyLoss()
    bogus_optimizer = torch.optim.LBFGS(bogus_model.parameters(), line_search_fn='strong_wolfe')

    with torch.no_grad():
        bogus_model.fc1.weight.data[:5, :] = model.fc1.weight.data
        
        bogus_model.fc2.weight.data[:, :5] = model.fc2.weight.data

    bogus_trained_model = train(model=bogus_model,
                                train_dataset=train_dataset,
                                criterion=bogus_criterion,
                                optimizer=bogus_optimizer,
                                gmin=config.gmin,
                                epochs=config.epochs,
                                batch_size=len(train_dataset))

    print(f'Testing the trained model with {10} hidden units on training data.')
    metrics = test(model=bogus_model,
                test_dataset=train_dataset,
                device=device)
    acc = metrics['accuracy']
    precision = metrics['precision']
    recall = metrics['recall']
    f1 = metrics['f1']
    print(f'Accuracy: {acc * 100:.4f} | Precision {precision* 100:.4f} | Recall {recall* 100:.4f} | F1: {f1* 100:.4f}')

    
    print(f'Testing the trained model with {10} hidden units on testing data.')
    metrics = test(model=bogus_model,
                test_dataset=test_dataset,
                device=device)
    acc = metrics['accuracy']
    precision = metrics['precision']
    recall = metrics['recall']
    f1 = metrics['f1']
    print(f'Accuracy: {acc * 100:.4f} | Precision {precision* 100:.4f} | Recall {recall* 100:.4f} | F1: {f1* 100:.4f}')

    torch.save(bogus_model.state_dict(), './mnist-mlp-10.pt')
        