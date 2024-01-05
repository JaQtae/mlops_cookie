from pathlib import Path

import click
import numpy as np
import torch
from model import MyAwesomeModel

@click.group()
def cli():
    """Command line interface."""
    pass

@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")

    model = MyAwesomeModel()
    state_dict = torch.load("../../models/" + model_checkpoint)
    model.load_state_dict(state_dict)
    
    test_set = torch.load("../../data/processed/testloader.pt")

    model.eval()
    accuracies = []
    with torch.no_grad():
        for images, labels in test_set:
            images = images.unsqueeze(1)
            # images = images.view(images.shape[0], -1)
            ps = model(images)
            # ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            # print(f'Accuracy: {accuracy.item() * 100}%')
            accuracies.append(accuracy)
    print("Estimate of accuracy: ", np.mean(accuracies))
    
cli.add_command(evaluate)
    
if __name__ == "__main__":
    cli()