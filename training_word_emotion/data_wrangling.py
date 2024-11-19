import pandas as pd

# load a tab separated txt  file
df = pd.read_csv(r'D:\GitHub\bias_free_modeling\data\emolex\NRC-Emotion-Intensity-Lexicon\NRC-Emotion-Intensity-Lexicon-v1.txt', sep='\t', header=None)

df.columns = ['word', 'emotion', 'score']

df['word'].unique()

for emotion in df['emotion'].unique():
    temp = df[df['emotion'] == emotion].sort_values(by=['score'], ascending=False)
    # split with sklearn into train test eval
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(temp, test_size=0.4, random_state=42)
    test, val = train_test_split(test, test_size=0.5, random_state=42)
    # save
    train.to_csv(f'data/emolex/train_{emotion}.csv', index=False)
    test.to_csv(f'data/emolex/test_{emotion}.csv', index=False)
    val.to_csv(f'data/emolex/val_{emotion}.csv', index=False)
    # print sizes
    print(f'{emotion} train size: {len(train)}')
    print(f'{emotion} test size: {len(test)}')
    print(f'{emotion} val size: {len(val)}')
