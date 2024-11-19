from tqdm import tqdm
import torch
import numpy as np
import wandb
def training_loop(model, optimizer, scheduler, epochs, train_dataloader,
          val_dataloader, criterion, device, save_dir,
                  use_wandb = False):

    best_corr_total = 0
    for epoch_num in range(epochs):
        total_loss_train = 0

        for dataloader_pack in tqdm(train_dataloader):

            input_mask_packs = sum(
                [[train_input['input_ids'].squeeze(1).to(device), train_input['attention_mask'].to(device)] for train_input in
                 dataloader_pack[:-1]], [])

            train_label = dataloader_pack[-1].to(device)

            outputs = model(*input_mask_packs)
            del input_mask_packs

            batch_loss = criterion(outputs.float(), train_label.view(-1,1).float())
            del train_label

            total_loss_train += batch_loss.item()

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
            scheduler.step()

        total_loss_val = 0
        correlations = []
        with torch.no_grad():
            for dataloader_pack in val_dataloader:
                input_mask_packs = sum(
                    [[train_input['input_ids'].squeeze(1).to(device), train_input['attention_mask'].to(device)] for train_input in
                     dataloader_pack[:-1]], [])

                val_label = dataloader_pack[-1].to(device)

                outputs = model(*input_mask_packs)
                del input_mask_packs

                batch_loss = criterion(outputs.float(), val_label.view(-1,1).float())
                del val_label

                total_loss_val += batch_loss.item()

                outputs_detached = outputs.cpu().detach().view(-1).numpy()
                labels_np = dataloader_pack[-1].numpy()

                correlations.append(np.corrcoef(outputs_detached, labels_np)[0,1])

            total_corr = np.mean(correlations)
            if best_corr_total < total_corr / len(val_dataloader):
                best_corr_total = total_corr / len(val_dataloader)
                torch.save(model.state_dict(), save_dir)
                print('Saved model')

        if use_wandb:
            wandb_dict = {"loss": total_loss_train / len(train_dataloader), "lr": scheduler.get_last_lr()[0],
                          "epoch": epoch_num, "val_loss": total_loss_val/ len(val_dataloader)}

            wandb_dict["corr_variable"] = total_corr / len(val_dataloader)
            if epoch_num % 2 == 0:
                wandb.log(wandb_dict)

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_dataloader): .10f} \
                | Val Loss: {total_loss_val / len(val_dataloader): .10f} | Val Corr: {total_corr / len(val_dataloader): .10f}')
