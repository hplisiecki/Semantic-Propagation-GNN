import pandas as pd
import re

def remove_emojis_and_symbols(text):
    # Unicode ranges for emojis, symbols, and non-printing characters
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        "\U0001F680-\U0001F6FF"  # Transport & Map symbols
        "\U0001F700-\U0001F77F"  # Alchemical Symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed characters
        "\u200d"  # Zero-width joiner
        "\u200c"  # Zero-width non-joiner
        "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)

train = pd.read_csv(r'data/polish/train_set.csv')
test = pd.read_csv(r'data/polish/test_set.csv')
val = pd.read_csv(r'data/polish/val_set.csv')


train['text'] = train['text'].apply(lambda x: remove_emojis_and_symbols(x))
train['text'] = train['text'].apply(lambda x: x.replace('_users_', 'użytkownicy'))
train['text'] = train['text'].apply(lambda x: x.replace('_link_', 'odnośnik'))

test['text'] = test['text'].apply(lambda x: remove_emojis_and_symbols(x))
test['text'] = test['text'].apply(lambda x: x.replace('_users_', 'użytkownicy'))
test['text'] = test['text'].apply(lambda x: x.replace('_link_', 'odnośnik'))

val['text'] = val['text'].apply(lambda x: remove_emojis_and_symbols(x))
val['text'] = val['text'].apply(lambda x: x.replace('_users_', 'użytkownicy'))
val['text'] = val['text'].apply(lambda x: x.replace('_link_', 'odnośnik'))

train.to_csv(r'D:\GitHub\bias_free_modeling\data\polish\train_set_prepared.csv', index=False)
test.to_csv(r'D:\GitHub\bias_free_modeling\data\polish\test_set_prepared.csv', index=False)
val.to_csv(r'D:\GitHub\bias_free_modeling\data\polish\val_set_prepared.csv', index=False)

