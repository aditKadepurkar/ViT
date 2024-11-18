from ViT.vit import ViT
from ViT.dataloader import ImageNetDataset
import torch
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers.optimization import get_scheduler

def train():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    model = ViT()
    model.to(device=device)
    model.train()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # load data(imagenet 1k dataset)
    # ok now it is loading pokemon classification dataset 'keremberke/pokemon-classification
    dataset = load_dataset('imagenet-1k', trust_remote_code=True)
    train_dataset = ImageNetDataset(dataset['train'], device=device, transform=transform)
    dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=32, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * 100
    )


    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:e}")
    print("Starting training")

    with tqdm(total=100, desc="Epochs", unit="epoch") as pbar1:
        for epoch in range(1, 101):
            loss_total = 0
            with tqdm(total=len(dataloader), desc=f"Epoch {epoch}", unit="batch", leave=True) as pbar:
                for idx, (images, labels) in enumerate(dataloader):

                    # forward
                    images = images.to(device)
                    logits = model(images)

                    labels = labels.to(device)

                    # loss calculation
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                    
                    # backpropagate the loss!!
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

                    loss_total += loss.item()

                    # logging
                    pbar.set_postfix(loss=loss.item())
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    pbar.update(1)

            pbar.write(f"Loss: {loss_total / len(dataloader)}")
            pbar1.update(1)

    # save
    print("Saving model to model/model.pth")
    torch.save(model.state_dict(), 'model/model.pth')

if __name__ == "__main__":
    train()
