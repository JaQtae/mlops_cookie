import torch
from torch.utils.data import DataLoader, TensorDataset
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def mnist(batch_size: int = 64):
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    
    train_images = torch.load(os.path.join(project_root, 'data', 'processed', 'processed_images_train.pt'))
    train_labels = torch.load(os.path.join(project_root, 'data', 'processed', 'processed_target_train.pt'))
    
    test_images = torch.load(os.path.join(project_root, 'data', 'processed', 'test_images.pt')).unsqueeze(1)
    test_labels = torch.load(os.path.join(project_root, 'data', 'processed', 'test_target.pt'))
    
    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, test_dataloader