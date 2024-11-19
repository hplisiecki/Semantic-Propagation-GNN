import torch
from tqdm import tqdm
import wandb
import numpy as np

def train_model(model, optimizer, criterion, train_dataloader, val_dataloader, num_epochs, device,
                save_dir = None, use_wandb =False, scheduler = None):

    # To store losses and correlations
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_loss = np.inf

    for epoch in range(num_epochs):
        ### Training Phase ###
        model.train()
        total_train_loss = 0

        for data in tqdm(train_dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            labels = data.y

            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()
            total_train_loss += loss.item() * data.num_graphs

        avg_train_loss = total_train_loss / len(train_dataloader.dataset)
        train_losses.append(avg_train_loss)

        ### Validation Phase ###
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data in val_dataloader:
                data = data.to(device)
                outputs = model(data)
                labels = data.y
                loss = criterion(outputs, labels)
                total_val_loss += loss.item() * data.num_graphs

                # Compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = total_val_loss / len(val_dataloader.dataset)
        val_losses.append(avg_val_loss)
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            if save_dir is not None:
                torch.save(model.state_dict(), save_dir)

        if use_wandb:
            wandb_dict = {'train_loss': avg_train_loss,
                        'val_loss': avg_val_loss,
                        'val_accuracy': val_accuracy,
                        'best_val_loss': best_loss,}
            if scheduler:
                wandb_dict['learning_rate'] : scheduler.get_last_lr()[0]
            wandb.log(wandb_dict)

        ### Print Epoch Results ###
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'Validation Accuracy: {val_accuracy:.4f}')
        print('-' * 30)