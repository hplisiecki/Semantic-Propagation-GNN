import numpy as np
import pandas as pd
from stargazer.stargazer import Stargazer, LineLocation
import statsmodels.api as sm
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from matplotlib.lines import Line2D
from tqdm import tqdm
from polish.regressions.sentences import neutral_sentences, political_sentences
from polish.regressions.utils import predict_text

scores = pd.read_excel('data/polish/politicians_without_nonmentioned.xlsx')


# Compute valence scores for

names = scores['Politician'].to_list()

# raw names

results = predict_text(names)
scores['raw_names'] = results['norm_Valence_M']

# neutral sentences
filled_neutral_sentences = []
for name in names:
    filled_neutral_sentences.append([sentence.replace('[Name]', name) for sentence in neutral_sentences])

results = []
for politician_sentences in tqdm(filled_neutral_sentences):
    poltician_result = predict_text(politician_sentences)
    results.append(np.mean(poltician_result['norm_Valence_M']))

scores['neutral_sentences'] = results

# political sentences
filled_political_sentences = []
for name in names:
    filled_political_sentences.append([sentence.replace('[Name]', name) for sentence in political_sentences])

results = []
for politician_sentences in tqdm(filled_political_sentences):
    poltician_result = predict_text(politician_sentences)
    results.append(np.mean(poltician_result['norm_Valence_M']))

scores['political_sentences'] = results

# save
scores.to_csv('data/polish/politicians_without_nonmentioned.csv', index=False)


## Prepare variables for regression
# load
scores = pd.read_csv('data/polish/politicians_without_nonmentioned.csv')

id = list(scores['Politician'])
valence_only_names = scores['raw_names'].to_numpy() * 100 # rebased from 0-1 tp 0-100
valence_neutral_sentences = scores['neutral_sentences'].to_numpy() * 100
valence_political_sentences = scores['political_sentences'].to_numpy() * 100

trust = scores['Trust'].to_numpy()
gender = scores['Gender'].to_numpy()
weights = scores['m in training'].to_numpy()

# Create factors table from party list and replece ZP (most common party in this set), with intercept
party = scores['Party']
party_factor = pd.get_dummies(party)
party_factor.insert(0, 'intercept', True)
party_factor = party_factor.drop("ZP", axis=1)


# Set font to Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']



variable_names = list(party_factor.columns) + ["gender"]
X = np.column_stack((party_factor,gender))
X = pd.DataFrame(X, columns=variable_names)


print('Only names:',
    'mean =', valence_only_names.mean().round(1),
    'standard deviation =', valence_only_names.std().round(2)
)

print('Neutral sentences:',
    'mean =', valence_neutral_sentences.mean().round(1),
    'standard deviation =', valence_neutral_sentences.std().round(2)
)

print('Political sentences:',
    'mean =', valence_political_sentences.mean().round(1),
    'standard deviation =', valence_political_sentences.std().round(2)
)


### Raw names regression

model1 = sm.WLS(valence_only_names, X, weights=weights).fit()

### Neutral sentences regression

model2 = sm.WLS(valence_neutral_sentences, X, weights=weights).fit()

### Political sentences regression

model3 = sm.WLS(valence_political_sentences, X, weights=weights).fit()


# Summary
summary = Stargazer([model1, model2, model3])

summary.dependent_variable_name(("Valence of:"))
summary.show_confidence_intervals(False)
summary.covariate_order(variable_names)
summary.custom_columns(['raw names','neutral sentences', 'political sentences'], [1,1,1])

print(summary.render_latex())


## Permutation test
np.random.seed(2137)

p_valence_only_names = deepcopy(valence_only_names)
p_valence_neutral_sentences = deepcopy(valence_neutral_sentences)
p_valence_political_sentences = deepcopy(valence_political_sentences)

pv1 = []
pv2 = []
pv3 = []

n_perm = 100000

for i in tqdm(range(n_perm)):
    np.random.shuffle(p_valence_only_names)
    np.random.shuffle(p_valence_neutral_sentences)
    np.random.shuffle(p_valence_political_sentences)

    p_model1 = sm.WLS(p_valence_only_names, X, weights=weights).fit()
    p_model2 = sm.WLS(p_valence_neutral_sentences, X, weights=weights).fit()
    p_model3 = sm.WLS(p_valence_political_sentences, X, weights=weights).fit()

    pv1.append(p_model1.rsquared)
    pv2.append(p_model2.rsquared)
    pv3.append(p_model3.rsquared)


p_valuep_model1 = len([x for x in pv1 if x > model1.rsquared])/n_perm
p_valuep_model2 = len([x for x in pv2 if x > model2.rsquared])/n_perm
p_valuep_model3 = len([x for x in pv3 if x > model3.rsquared])/n_perm


print("Raw names p-value:", p_valuep_model1)
print("Neutral sentences p-value:", p_valuep_model2)
print("Political sentences p-value:", p_valuep_model3)



# Equivalence testing
import pandas as pd
import statsmodels.api as sm
import numpy as np

scores = pd.read_csv('data/polish/politicians_without_nonmentioned.csv')

names_dict = {'3D' : 5.73,
              'K' : 6.15,
              'KO' : 5.83,
              'Left': 3.03,
              'gender': 9.77,
              'ZP' : 0}

neut_sent_dict = {'3D' : 9.09,
              'K' : 8.37,
              'KO' : 3.71,
              'Left': 5.46,
              'gender': 10.10,
               'ZP' : 0}

polit_sent_dict = {'3D' : 2.72,
              'K' : 2.56,
              'KO' : 2.3,
              'Left': 2.51,
              'gender': 1.88,
                     'ZP' : 0}

scores['raw_names'] = scores['raw_names'].to_numpy() * 100 # rebased from 0-1 tp 0-100
scores['neutral_sentences'] = scores['neutral_sentences'].to_numpy() * 100
scores['political_sentences'] = scores['political_sentences'].to_numpy() * 100

scores['debiased_raw_names'] = scores['raw_names'] - scores['Party'].map(names_dict) - scores['Gender'] * names_dict['gender']
scores['debiased_neutral_sentences'] = scores['neutral_sentences'] - scores['Party'].map(neut_sent_dict) - scores['Gender'] * neut_sent_dict['gender']
scores['debiased_political_sentences'] = scores['political_sentences'] - scores['Party'].map(polit_sent_dict) - scores['Gender'] * polit_sent_dict['gender']

scores['bias_raw_names'] = scores['Party'].map(names_dict) + scores['Gender'] * names_dict['gender']
scores['bias_neutral_sentences'] = scores['Party'].map(neut_sent_dict) + scores['Gender'] * neut_sent_dict['gender']
scores['bias_political_sentences'] = scores['Party'].map(polit_sent_dict) + scores['Gender'] * polit_sent_dict['gender']

id = list(scores['Politician'])


valence_only_names = scores['debiased_raw_names'].to_numpy()# rebased from 0-1 tp 0-100
valence_neutral_sentences = scores['debiased_neutral_sentences'].to_numpy()
valence_political_sentences = scores['debiased_political_sentences'].to_numpy()

# concat
valence_stimuli = np.concatenate([valence_only_names, valence_neutral_sentences, valence_political_sentences])

weights = scores['m in training'].to_numpy()

weights_all = np.concatenate([weights, weights, weights])


variable_names = ['Intercept', 'Bias', 'Neutral Sentences', 'Political Sentences']
intercept = np.ones(len(valence_only_names))
X1 = scores['bias_raw_names'].to_numpy().reshape(-1,1)
dummy_1 = np.zeros(len(X1))
dummy_2 = np.zeros(len(X1))
X1 = np.column_stack((intercept, X1, dummy_1, dummy_2))
X1 = pd.DataFrame(X1, columns=variable_names)


X2 = scores['bias_neutral_sentences'].to_numpy().reshape(-1,1)
dummy_1 = np.ones(len(X2))
dummy_2 = np.zeros(len(X2))
X2 = np.column_stack((intercept, X2, dummy_1, dummy_2))
X2 = pd.DataFrame(X2, columns=variable_names)


X3 = scores['bias_political_sentences'].to_numpy().reshape(-1,1)
dummy_1 = np.zeros(len(X3))
dummy_2 = np.ones(len(X3))
# append intercept
X3 = np.column_stack((intercept, X3, dummy_1, dummy_2))
X3 = pd.DataFrame(X3, columns=variable_names)


# concat all three dataframes
X = pd.concat([X1, X2, X3])


model = sm.WLS(valence_stimuli, X, weights=weights_all).fit()


from stargazer.stargazer import Stargazer, LineLocation

summary = Stargazer([model])

print(summary.render_latex())

## Permutation test
import numpy as np
from copy import deepcopy
from tqdm import tqdm
np.random.seed(2137)

p_valence_stimuli = deepcopy(valence_stimuli)


pv = []

n_perm = 100000

for i in tqdm(range(n_perm)):
    np.random.shuffle(p_valence_stimuli)

    p_model = sm.WLS(p_valence_stimuli, X, weights=weights_all).fit()

    pv.append(float(p_model.summary().tables[1].data[2][1]))


b = float(model.summary().tables[1].data[2][1])


p_valuep_model = len([x for x in pv if x < b])/n_perm



print("Equivalence test p-value:", p_valuep_model)

# Equivalence test p-value: 0.01213




####### Prediction Difference Analysis
import pandas as pd
import statsmodels.api as sm
import numpy as np


scores = pd.read_csv('data/polish/politicians_without_nonmentioned.csv')
id = list(scores['Politician'])

valence_only_names = scores['Valence_M'].to_numpy() * 100 # rebased from 0-1 tp 0-100

nsr = pd.read_csv(r'data/testing_bias/neutral_sentences_results.csv')
valence_neutral_sentences = []
for name in id:
    filtered_df = nsr.loc[nsr['Politician'] == name]
    valence_neutral_sentences.append(filtered_df['Valence_M'].mean()  * 100 ) # rebased from 0-1 to 0-100
valence_neutral_sentences = np.array(valence_neutral_sentences)

psr = pd.read_csv('data/testing_bias/political_sentences_results.csv')
valence_political_sentences = []
for name in id:
    filtered_df = psr.loc[psr['Politician'] == name]
    valence_political_sentences.append(filtered_df['Valence_M'].mean()  * 100 ) # rebased from 0-1 to 0-100
valence_political_sentences = np.array(valence_political_sentences)


valence_only_names_unbiased = scores['raw_names'].to_numpy() * 100 # rebased from 0-1 tp 0-100
valence_neutral_sentences_unbiased = scores['neutral_sentences'].to_numpy() * 100
valence_political_sentences_unbiased = scores['political_sentences'].to_numpy() * 100


valence_only_names_diff = valence_only_names - valence_only_names_unbiased
valence_neutral_sentences_diff = valence_neutral_sentences - valence_neutral_sentences_unbiased
valence_political_sentences_diff = valence_political_sentences - valence_political_sentences_unbiased

# concat
valence_stimuli_diff = np.concatenate([valence_only_names_diff, valence_neutral_sentences_diff, valence_political_sentences_diff])


weights = scores['m in training'].to_numpy()

weights_all = np.concatenate([weights, weights, weights])

names_dict = {'3D' : 5.73,
              'K' : 6.15,
              'KO' : 5.83,
              'Left': 3.03,
              'gender': 9.77,
              'ZP' : 0}

neut_sent_dict = {'3D' : 9.09,
              'K' : 8.37,
              'KO' : 3.71,
              'Left': 5.46,
              'gender': 10.10,
               'ZP' : 0}

polit_sent_dict = {'3D' : 2.72,
              'K' : 2.56,
              'KO' : 2.3,
              'Left': 2.51,
              'gender': 1.88,
                     'ZP' : 0}

scores['bias_raw_names'] = scores['Party'].map(names_dict) + scores['Gender'] * names_dict['gender']
scores['bias_neutral_sentences'] = scores['Party'].map(neut_sent_dict) + scores['Gender'] * neut_sent_dict['gender']
scores['bias_political_sentences'] = scores['Party'].map(polit_sent_dict) + scores['Gender'] * polit_sent_dict['gender']

variable_names = ['Intercept', 'Bias', 'Neutral Sentences', 'Political Sentences']
intercept = np.ones(len(valence_only_names))
X1 = scores['bias_raw_names'].to_numpy().reshape(-1,1)
dummy_1 = np.zeros(len(X1))
dummy_2 = np.zeros(len(X1))
X1 = np.column_stack((intercept, X1, dummy_1, dummy_2))
X1 = pd.DataFrame(X1, columns=variable_names)

X2 = scores['bias_neutral_sentences'].to_numpy().reshape(-1,1)
dummy_1 = np.ones(len(X2))
dummy_2 = np.zeros(len(X2))
X2 = np.column_stack((intercept, X2, dummy_1, dummy_2))
X2 = pd.DataFrame(X2, columns=variable_names)


X3 = scores['bias_political_sentences'].to_numpy().reshape(-1,1)
dummy_1 = np.zeros(len(X3))
dummy_2 = np.ones(len(X3))
# append intercept
X3 = np.column_stack((intercept, X3, dummy_1, dummy_2))
X3 = pd.DataFrame(X3, columns=variable_names)



# concat all three dataframes
X = pd.concat([X1, X2, X3])


model = sm.WLS(valence_stimuli_diff, X, weights=weights_all).fit()


from stargazer.stargazer import Stargazer, LineLocation

summary = Stargazer([model])

print(summary.render_latex())


## Permutation test
import numpy as np
from copy import deepcopy
from tqdm import tqdm
np.random.seed(2137)

p_valence_stimuli = deepcopy(valence_stimuli_diff)


pv = []

n_perm = 100000

for i in tqdm(range(n_perm)):
    np.random.shuffle(p_valence_stimuli)

    p_model = sm.WLS(p_valence_stimuli, X, weights=weights_all).fit()

    pv.append(float(p_model.summary().tables[1].data[2][1]))


b = float(model.summary().tables[1].data[2][1])


p_valuep_model = len([x for x in pv if x > b])/n_perm



print("Equivalence test p-value:", p_valuep_model)

# Equivalence test p-value: 0.00276
