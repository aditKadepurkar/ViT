from ViT.vit import ViT
from ViT.dataloader import ImageNetDataset
import torch
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader

def train():
    model = ViT()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    # load data(imagenet 1k dataset)
    dataset = load_dataset('imagenet-1k', trust_remote_code=True)
    train_dataset = ImageNetDataset(dataset['train'], transform=transform)
    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1)

    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print("Starting training")

    for epoch in range(10):
        for idx, (images, labels) in enumerate(dataloader):

            # forward
            logits = model(images)
            
            # loss calculation
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
            
            # backpropagate the loss!!
            loss.backward()

            # logging
            if idx % 1 == 0:
                print(f"Epoch {epoch}, Iteration {idx}, Loss {loss.item()}")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
    
    # save
    print("Saving model to model/model.pth")
    torch.save(model.state_dict(), 'model/model.pth')

if __name__ == "__main__":
    train()
