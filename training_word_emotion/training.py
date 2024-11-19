import wandb
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModel
import torch
import pandas as pd
from training_word_emotion.training_loop import training_loop
from training_word_emotion.dataset_and_model import Dataset, BertRegression
from training_word_emotion.utils import load_data, check_max_token_length, set_seed
from transformers import logging
logging.set_verbosity_error()

###############################################################################
"""
English word norm bert training script 
"""
###############################################################################
###############################################################################
# HYPERPARAMETERS
#################################

set_seed()

emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness','surprise', 'trust']
for emotion in emotions:
    # empty torch
    torch.cuda.empty_cache()

    hidden_dim = 768
    dropout = 0.1
    warmup_steps = 600
    save_dir = f'models/{emotion}'

    model_dir = "nghuyong/ernie-2.0-en"

    model_name = ['bert']

    model_initialization = [AutoModel.from_pretrained(model_dir)]

    epochs = 100
    batch_size = 500
    learning_rate = 5e-5
    eps = 1e-8
    weight_decay = 0.3
    amsgrad = True
    betas = (0.9, 0.999)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    ###############################################################################
    # DATA LOADING
    #################################

    df_train, df_test, df_val = load_data(emotion)

    ###############################################################################
    # INITIALIZATION
    #################################
    # TOKENIZERS
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    max_len = check_max_token_length(tokenizer, emotion)

    # MODEL
    model = BertRegression(model_name, model_initialization, emotion, dropout, hidden_dim)

    # DATALOADERS
    train, val = Dataset(tokenizer, df_train, max_len), Dataset(tokenizer, df_val, max_len)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=learning_rate, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, betas=betas)


    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=len(train_dataloader) * epochs)

    ###############################################################################
    # TRAINING
    #################################

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    wandb.init(project="emolex_prediction", entity="hubertp", name = emotion)
    wandb.watch(model, log_freq=5)

    training_loop(model, optimizer, scheduler, epochs, train_dataloader, val_dataloader, criterion,
                  device, save_dir, use_wandb = True)