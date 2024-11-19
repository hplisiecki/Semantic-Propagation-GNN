import wandb
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModel
import torch
import pandas as pd
from training_word_emotion.training_loop import training_loop
from training_word_emotion.dataset_and_model import Dataset, BertRegression
from training_word_emotion.utils import load_data, check_max_token_length, set_seed
from transformers import logging
import numpy as np
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

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    max_len = check_max_token_length(tokenizer, emotion)

    # MODEL
    model = BertRegression(model_name, model_initialization, emotion, dropout, hidden_dim)


    test = Dataset(tokenizer, df_test, max_len)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    criterion = torch.nn.MSELoss()

    model.load_state_dict(torch.load(save_dir))

    model = model.to(device)
    criterion = criterion.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_dataloader:
            input_mask_packs = sum(
                [[train_input['input_ids'].squeeze(1).to(device), train_input['attention_mask'].to(device)] for train_input in
                 batch[:-1]], [])
            outputs = model(*input_mask_packs)

            predictions.append(outputs.cpu().detach().view(-1).numpy())
    df_test['predictions'] = np.concatenate(predictions)

    # print results
    print(f"Results for {emotion}")
    # pearson correlation import
    from scipy.stats import pearsonr
    print(f"Pearson correlation: {pearsonr(df_test['predictions'], df_test['score'])}")


