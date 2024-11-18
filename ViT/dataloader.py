from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

# Load the ImageNet 1K dataset
# dataset = load_dataset('imagenet-1k', trust_remote_code=True)

# Define a custom dataset class
class ImageNetDataset:
    def __init__(self, dataset, device, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample['image']
        label = sample['label']

        # to make sure there are 3 channels
        image = image.convert("RGB")

        # transforming to correct dimensions
        if self.transform:
            image = self.transform(image)
        
        # image.to(self.device)
        # label.to(self.device)

        return image, label



if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = load_dataset('imagenet-1k', trust_remote_code=True)
    train_dataset = ImageNetDataset(dataset['train'], transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1)

    # Example usage
    for images, labels in train_loader:
        print(images.shape, labels.shape)
