import torch
from torchvision import transforms
import os

from torch.utils.data import TensorDataset, DataLoader

def process_mnist(input_folder: str, output_folder: str) -> None:
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0,), (1,))])
    
    train_imgs = []
    train_labels = []
    for i in range(5):
        train_imgs.append(torch.load(f'{input_folder}/train_images_{i}.pt'))
        train_labels.append(torch.load(f'{input_folder}/train_target_{i}.pt'))
    
    test_images = torch.load(f"{output_folder}/test_images.pt")
    test_labels = torch.load(f"{output_folder}/test_target.pt")
       
    train_images = torch.concatenate(train_imgs, dim=0).unsqueeze(1) # [5000, 28, 28] x 5 --> [25000, 28, 28]
    train_labels = torch.concatenate(train_labels, dim=0)
    
    train_dataset = TensorDataset(train_images, train_labels)
    
    test_dataset = TensorDataset(test_images, test_labels)
    
    train_images.transform = transform
    train_dataset.transform = transform
        
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Save the processed tensor
    output_file_path_images = os.path.join(output_folder, 'processed_images_train.pt')
    output_file_path_labels = os.path.join(output_folder, 'processed_target_train.pt')
    
    output_file_path_train_dataloader = os.path.join(output_folder, 'trainloader.pt')
    output_file_path_test_dataloader = os.path.join(output_folder, 'testloader.pt')

    torch.save(train_images, output_file_path_images)
    torch.save(train_labels, output_file_path_labels)
    torch.save(train_dataloader, output_file_path_train_dataloader)
    torch.save(test_dataloader, output_file_path_test_dataloader)
    
if __name__ == "__main__":
    raw_data_folder = '../../data/raw'
    processed_data_folder = '../../data/processed'

    process_mnist(raw_data_folder, processed_data_folder)