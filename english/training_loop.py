import torch
from scipy.stats import pearsonr
from tqdm import tqdm
import wandb
import numpy as np

def train_model(model, optimizer, criterion, train_dataloader, val_dataloader, num_epochs, metric_names, device,
                save_dir = None, use_wandb =False, scheduler = None):

    # To store losses and correlations
    train_losses = []
    val_losses = []
    val_correlations = {metric: [] for metric in metric_names}
    best_corr = -1

    for epoch in range(num_epochs):
        ### Training Phase ###
        model.train()
        total_train_loss = 0

        for data in tqdm(train_dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            outputs = model(data)  # Outputs are logits of size [batch_size, num_classes]
            preds = torch.cat([outputs[metric] for metric in metric_names], dim=1)
            valence = data.v.unsqueeze(1)
            arousal = data.a.unsqueeze(1)
            labels = torch.cat([valence, arousal], dim=1)

            loss = criterion(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()

            total_train_loss += loss.item() * data.num_graphs  # Multiply by batch size

        avg_train_loss = total_train_loss / len(train_dataloader.dataset)
        train_losses.append(avg_train_loss)

        ### Validation Phase ###
        model.eval()
        total_val_loss = 0
        all_preds = {metric: [] for metric in metric_names}
        all_labels = {metric: [] for metric in metric_names}

        with torch.no_grad():
            for data in val_dataloader:
                data = data.to(device)
                outputs = model(data)
                preds = torch.cat([outputs[metric] for metric in metric_names], dim=1)
                valence = data.v.unsqueeze(1)
                arousal = data.a.unsqueeze(1)
                labels = torch.cat([valence, arousal], dim=1)

                loss = criterion(preds, labels)

                total_val_loss += loss.item() * data.num_graphs

                # Collect predictions and labels for correlation calculation
                for idx, metric in enumerate(metric_names):
                    all_preds[metric].extend(outputs[metric].squeeze().tolist())
                    all_labels[metric].extend(labels[:, idx].tolist())

        avg_val_loss = total_val_loss / len(val_dataloader.dataset)
        val_losses.append(avg_val_loss)
        # Calculate correlation coefficients
        epoch_correlations = {}
        for metric in metric_names:
            pred_values = all_preds[metric]
            true_values = all_labels[metric]
            if len(set(pred_values)) > 1 and len(set(true_values)) > 1:
                corr, _ = pearsonr(pred_values, true_values)
            else:
                corr = 0.0  # Undefined correlation
            epoch_correlations[metric] = corr
            val_correlations[metric].append(corr)

        mean_corr = np.mean([epoch_correlations[metric] for metric in metric_names])

        if mean_corr > best_corr:
            best_corr = mean_corr
            if save_dir is not None:
                torch.save(model.state_dict(), f'{save_dir}')

        if use_wandb:
            takeout_dict = {'train_loss': avg_train_loss, 'val_loss': avg_val_loss}
            for metric in metric_names:
                takeout_dict[f'val_{metric}'] = epoch_correlations[metric]

            takeout_dict['best_corr'] = best_corr

            takeout_dict['mean_corr'] = mean_corr
            if scheduler:
                takeout_dict['learning_rate'] = scheduler.get_last_lr()[0]

            wandb.log(takeout_dict)


        ### Print Epoch Results ###
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        for metric in metric_names:
            print(f'Validation Correlation ({metric}): {epoch_correlations[metric]:.4f}')
        print('-' * 30)
