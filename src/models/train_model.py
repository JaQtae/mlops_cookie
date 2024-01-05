import sys

import torch
#from src.data.load_dataset import mnist
from model import MyAwesomeModel, mytest, mytrain

import matplotlib.pyplot as plt
import numpy as np

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
def train(lr: float = 1e-3, num_epochs: int = 50, batch_size: int = 64, plot: bool = False) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(f"Learning Rate: {lr}")
    print(f"Batch Size: {batch_size}")

    
    # TODO: Implement training loop here
    model = MyAwesomeModel()
    #train_set, _ = mnist(batch_size)
    train_set = torch.load("data/processed/trainloader.pt")
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model_out, train_losses = mytrain(model, train_set, criterion, optimizer, epochs=num_epochs)
    
    if plot:
        plt.plot(np.arange(1, num_epochs+1), train_losses, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')

        plt.legend()
        plotname = f'TrainLossPlot_{num_epochs}_{lr:.0e}'
        print(f'Saving plot as: {plotname}.png')
        plt.savefig(f'{plotname}.png')
        plt.show()

    torch.save(model_out.state_dict(), f"models/trained_model_{num_epochs}_{lr:.0e}.pt")
    print(f'Model saved as models/trained_model_{num_epochs}_{lr:.0e}.pt')
    

if __name__ == "__main__":
    train()