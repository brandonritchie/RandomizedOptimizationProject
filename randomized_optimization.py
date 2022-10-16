#%%
import mlrose_hiive as mlhive
import pandas as pd
import matplotlib.pyplot as plt
import random
from mlrose_hiive import genetic_alg as gen
from mlrose_hiive import random_hill_climb as rhc
from mlrose_hiive import mimic 
from mlrose_hiive import simulated_annealing as sa
from pytictoc import TicToc
t = TicToc()

# %%
## ANN Section
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
# Data Read
data_raw = pd.read_csv('https://raw.githubusercontent.com/brandonritchie/SupervisedLearningProject1/master/Train.csv')

# Data cleaning
data_raw = pd.get_dummies(data_raw, columns = ['job_type','marital','education','default', 'prev_campaign_outcome'])

# Transformation of date time: https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/
data_raw['sin_time'] = np.sin(2*np.pi*data_raw.day_of_month/365)
data_raw['cos_time'] = np.cos(2*np.pi*data_raw.day_of_month/365)

data_raw['housing_loan'] = np.where(data_raw['housing_loan'] == 'yes', 1,0)
data_raw['personal_loan'] = np.where(data_raw['personal_loan'] == 'yes', 1,0)

data_raw = data_raw.fillna(data_raw.mean())

data_cleaned = data_raw.drop(columns=['communication_type', 'day_of_month', 'month', 'id','days_since_prev_campaign_contact'])

# Downsample to balanced data
sub = data_cleaned.loc[data_cleaned.term_deposit_subscribed == 1]
nsub = data_cleaned.loc[data_cleaned.term_deposit_subscribed == 0].head(len(sub))
data_cleaned = pd.concat([sub,nsub])

X = data_cleaned.drop(['term_deposit_subscribed'], axis = 1)
y = data_cleaned[['term_deposit_subscribed']]

X_train1,X_test1,y_train,y_test = train_test_split(X,y,test_size=0.2)
X_train2, X_val1, y_train, y_val = train_test_split(X_train1,y_train,test_size=0.3)
y_train = y_train.reset_index().drop('index', axis = 1)
y_test = y_test.reset_index().drop('index', axis = 1)
sc_train = StandardScaler()
sc_val = StandardScaler()
sc_test = StandardScaler()
X_train = pd.DataFrame(sc_train.fit_transform(X_train2.values), columns = X_train2.columns)
X_val = pd.DataFrame(sc_train.fit_transform(X_val1.values), columns = X_val1.columns)
X_test = pd.DataFrame(sc_test.fit_transform(X_test1.values), columns = X_test1.columns)
#%%
from mlrose_hiive import NeuralNetwork as ann
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score

r_l = list(range(1,15,2))

restart_num = []
time_l = []
precision_l = []
accuracy_l = []

for s in r_l:
    hill = ann(activation='identity',
        hidden_nodes=[2,20],
        max_iters=500,
        algorithm = 'random_hill_climb',
        early_stopping=True,
        random_state = 17,
        restarts = s,
        max_attempts=120)

    t.tic()
    hill.fit(X_train,y_train)
    clock_time = t.tocvalue()
    t.toc()

    preds = hill.predict(X_val)
    precision = precision_score(y_val,preds)
    accuracy = accuracy_score(y_val,preds)
    restart_num.append(s)

    time_l.append(clock_time)
    precision_l.append(precision)
    accuracy_l.append(accuracy)

hill_data = pd.DataFrame({
    'Restart':restart_num,
    'time':time_l,
    'precision':precision_l,
    'accuracy':accuracy_l
})
hill_data.to_csv('ann_hillClimb.csv')

#%%
# RHC Learning Curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
def plot_learning_curve(estimator, X, y, ax=None, ylim=(0.5, 1.01), cv=None, n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5)):

    train_sizes, train_scores, test_scores = \
        learning_curve(estimator, X, y, cv=cv,scoring = 'accuracy', n_jobs=n_jobs, train_sizes=train_sizes)
              
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    # Plot learning curve
    ax.xlabel("Training examples")
    ax.ylabel("Accuracy")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    ax.title('Learning Curve Random Hill Climb')
    ax.legend(loc="best")

    return plt

fig = plt
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

estimator = hill = ann(activation='identity',
        hidden_nodes=[2,20],
        max_iters=500,
        algorithm = 'random_hill_climb',
        early_stopping=True,
        random_state = 17,
        restarts = 1)
plot_learning_curve(estimator, X_train, y_train, ax = fig, cv=cv, train_sizes=np.linspace(.1, 1.0, 5))

# %%
# ANN Simulated Annealing
names = ['GeomDecay','ExpDecay','ArithDecay']
schedule_l = [mlhive.GeomDecay,mlhive.ExpDecay,mlhive.ArithDecay]

decay_meth_l = []

time_l = []
precision_l = []
accuracy_l = []
index = 0
for i in schedule_l:

    sim_anneal = ann(activation='identity',
        hidden_nodes=[2,20],
        max_iters=500,
        early_stopping=True,
        schedule = schedule_l,
        restarts = 15)

    t.tic()
    sim_anneal.fit(X_train,y_train)
    clock_time = t.tocvalue()
    t.toc()

    pred = sim_anneal.predict(X_val)

    precision = precision_score(y_val,pred)
    accuracy = accuracy_score(y_val,pred)

    decay_meth_l.append(names[index])
    time_l.append(clock_time)
    precision_l.append(precision)
    accuracy_l.append(accuracy)
    index += 1

sim_anneal_data = pd.DataFrame({
    'DecayMethod':decay_meth_l,
    'Time':time_l,
    'Precision':precision_l,
    'Accuracy':accuracy_l
})

sim_anneal_data.to_csv('ann_simAnneal.csv')

#%%
# Simulated Annealing learning curve
def plot_learning_curve(estimator, X, y, ax=None, ylim=(0.5, 1.01), cv=None, n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5)):

    train_sizes, train_scores, test_scores = \
        learning_curve(estimator, X, y, cv=cv,scoring = 'accuracy', n_jobs=n_jobs, train_sizes=train_sizes)
              
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    # Plot learning curve
    ax.xlabel("Training examples")
    ax.ylabel("Accuracy")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    ax.title('Learning Curve Simulated Annealing - Geometric Temperature Decay')
    ax.legend(loc="best")

    return plt

fig = plt
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

estimator = ann(activation='identity',
        hidden_nodes=[2,20],
        max_iters=500,
        early_stopping=True,
        schedule = mlhive.GeomDecay,
        restarts = 10)
plot_learning_curve(estimator, X_train, y_train, ax = fig, cv=cv, train_sizes=np.linspace(.1, 1.0, 5))

# %%
# ANN Genetic Algorithm
population = [5,10,15,20,30]
mutation_rate = [.01,.05,.1,.15,.2]

mutation_rate_l = []
population_list = []

time_list = []
precision_list = []
accuracy_list = []

for pop in population:
    for mutation in mutation_rate:
        print(pop)
        print(mutation)
        gen_a = ann(activation='identity',
            hidden_nodes=[2,20],
            max_iters=500,
            algorithm='genetic_alg',
            pop_size=pop,
            mutation_prob=mutation,
            early_stopping=True,
            random_state = 17)
        
        t.tic()
        gen_a.fit(X_train,y_train)
        clock = t.tocvalue

        pred = gen_a.predict(X_val)

        precision = precision_score(y_val,pred)
        accuracy = accuracy_score(y_val,pred)

        population_list.append(pop)
        mutation_rate_l.append(mutation)

        time_list.append(clock_time)
        precision_list.append(precision)
        accuracy_list.append(accuracy)

ga_data = pd.DataFrame({
    'PopulationSize':population_list,
    'MutationRate':mutation_rate_l,
    'Time':time_list,
    'Precision':precision_list,
    'Accuracy':accuracy_list
})
ga_data.to_csv('ann_GeneticAlgorithm.csv')
# %%
def plot_learning_curve(estimator, X, y, ax=None, ylim=(0.5, 1.01), cv=None, n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5)):

    train_sizes, train_scores, test_scores = \
        learning_curve(estimator, X, y, cv=cv,scoring = 'accuracy', n_jobs=n_jobs, train_sizes=train_sizes)
              
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    # Plot learning curve
    ax.xlabel("Training examples")
    ax.ylabel("Accuracy")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    ax.title('Learning Curve Genetic Algorithm')
    ax.legend(loc="best")

    return plt

fig = plt
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

estimator = ann(activation='identity',
            hidden_nodes=[2,20],
            max_iters=500,
            algorithm='genetic_alg',
            pop_size=20,
            mutation_prob=.01,
            early_stopping=True,
            random_state = 17)
plot_learning_curve(estimator, X_train, y_train, ax = fig, cv=cv, train_sizes=np.linspace(.1, 1.0, 5))

# %%

#%%
## Travelling Salesman
# Sample size
sample_size = [10,20,30,40,50]

time = []
state = []
fit = []
curve = []
name = []
length = []
problem = []

for ss in sample_size:
    print(f'Sample: {ss}')
    x = random.sample(range(1,100), ss)
    y = random.sample(range(1,100), ss)
    coords = list(zip(x,y))

    traveler = mlhive.TravellingSales(coords=coords)
    travel_fit = mlhive.TSPOpt(length=len(x),fitness_fn=traveler,maximize=True)

    # Random Hill Climb
    t.tic()
    rhc_state, rhc_fit,rhc_curve = rhc(travel_fit,
                                random_state = 17,
                                curve=True)
    rhc_sec = t.tocvalue()
    t.toc()
    time.append(rhc_sec)
    state.append(rhc_state)
    fit.append(rhc_fit)
    curve.append(rhc_curve)
    name.append('RHC')
    # Genetic
    t.tic()
    gen_state, gen_fit, gen_curve = gen(travel_fit,
                                random_state = 17,
                                curve=True)
    gen_sec = t.tocvalue()
    t.toc()
    time.append(gen_sec)
    state.append(gen_state)
    fit.append(gen_fit)
    curve.append(gen_curve)
    name.append('GENETIC')
    # MIMIC
    t.tic()
    mimic_state,mimic_fit,mimic_curve = mimic(travel_fit,
                                random_state = 17,
                                curve=True)
    mim_sec = t.tocvalue()
    t.toc()
    time.append(mim_sec)
    state.append(mimic_state)
    fit.append(mimic_fit)
    curve.append(mimic_curve)
    name.append('MIMIC')
    #Simulated Annealing
    t.tic()
    sa_state,sa_fit,sa_curve = sa(travel_fit,
                                random_state = 17,
                                curve=True)
    sim_sec = t.tocvalue()
    t.toc()
    time.append(sim_sec)
    state.append(sa_state)
    fit.append(sa_fit)
    curve.append(sa_curve)
    name.append('SIMANNEAL')

    for i in range(4):
        length.append(ss)
        problem.append('TravellingSales')

travel_df = pd.DataFrame({
    'Problem':problem,
    'Algorithm':name,
    'Length':length,
    'Fitness':fit,
    'Time':time,
    'State':state,
    'Curve':curve})

travel_df.to_csv('travel_stats.csv')

algo_l = []
length_l = []
fit_l = []
iters_l = []
for i in range(len(travel_df)):
    algo = travel_df.iloc[i,1]
    length = travel_df.iloc[i,2]
    curve = travel_df.iloc[i,6]
    for x in curve:
        fit_val = x[0]
        iter = x[1]

        algo_l.append(algo)
        length_l.append(length)
        fit_l.append(fit_val)
        iters_l.append(iter)

travel_iters = pd.DataFrame({
    'Algorithm': algo_l,
    'Samples':length_l,
    'Score':fit_l,
    'Iterations':iters_l
})

travel_iters.to_csv('travel_iters.csv')
# %%
## Four Peaks
sample_size = [10,20,30,40,50]

time = []
state = []
fit = []
curve = []
name = []
length = []
problem = []

for ss in sample_size:
    print(ss)
    four_peak = mlhive.FourPeaks()
    peaks_fit = mlhive.DiscreteOpt(length=ss,fitness_fn=four_peak,maximize=True)
    # Random Hill Climb
    t.tic()
    rhc_state, rhc_fit,rhc_curve = rhc(peaks_fit,
                                random_state = 17,
                                curve=True)
    rhc_sec = t.tocvalue()
    t.toc()
    time.append(rhc_sec)
    state.append(rhc_state)
    fit.append(rhc_fit)
    curve.append(rhc_curve)
    name.append('RHC')
    # Genetic
    t.tic()
    gen_state, gen_fit, gen_curve = gen(peaks_fit,
                                random_state = 17,
                                curve=True)
    gen_sec = t.tocvalue()
    t.toc()
    time.append(gen_sec)
    state.append(gen_state)
    fit.append(gen_fit)
    curve.append(gen_curve)
    name.append('GENETIC')
    # MIMIC
    t.tic()
    mimic_state,mimic_fit,mimic_curve = mimic(peaks_fit,
                                random_state = 17,
                                curve=True)
    mim_sec = t.tocvalue()
    t.toc()
    time.append(mim_sec)
    state.append(mimic_state)
    fit.append(mimic_fit)
    curve.append(mimic_curve)
    name.append('MIMIC')
    #Simulated Annealing
    t.tic()
    sa_state,sa_fit,sa_curve = sa(peaks_fit,
                                random_state = 17,
                                curve=True)
    sim_sec = t.tocvalue()
    t.toc()
    time.append(sim_sec)
    state.append(sa_state)
    fit.append(sa_fit)
    curve.append(sa_curve)
    name.append('SIMANNEAL')

    for i in range(4):
        length.append(ss)
        problem.append('FourPeaks')

fourpeaks_df = pd.DataFrame({
    'Problem':problem,
    'Algorithm':name,
    'Length':length,
    'Fitness':fit,
    'Time':time,
    'State':state,
    'Curve':curve})

fourpeaks_df.to_csv('fourpeaks_stats.csv')

algo_l = []
length_l = []
fit_l = []
iters_l = []
for i in range(len(fourpeaks_df)):
    algo = fourpeaks_df.iloc[i,1]
    length = fourpeaks_df.iloc[i,2]
    curve = fourpeaks_df.iloc[i,6]
    for x in curve:
        fit_val = x[0]
        iter = x[1]

        algo_l.append(algo)
        length_l.append(length)
        fit_l.append(fit_val)
        iters_l.append(iter)

fourpeaks_iters = pd.DataFrame({
    'Algorithm': algo_l,
    'Samples':length_l,
    'Score':fit_l,
    'Iterations':iters_l
})

fourpeaks_iters.to_csv('fourpeaks_iters.csv')
# %%
## One Max
sample_size = [10,20,30,40,50]

time = []
state = []
fit = []
curve = []
name = []
length = []
problem = []

for ss in sample_size:
    one_max = mlhive.OneMax()
    one_max_fit = mlhive.DiscreteOpt(length = ss,fitness_fn=one_max)
    # Random Hill Climb
    t.tic()
    rhc_state, rhc_fit,rhc_curve = rhc(one_max_fit,
                                random_state = 17,
                                curve=True)
    rhc_sec = t.tocvalue()
    t.toc()
    time.append(rhc_sec)
    state.append(rhc_state)
    fit.append(rhc_fit)
    curve.append(rhc_curve)
    name.append('RHC')
    # Genetic
    t.tic()
    gen_state, gen_fit, gen_curve = gen(one_max_fit,
                                random_state = 17,
                                curve=True)
    gen_sec = t.tocvalue()
    t.toc()
    time.append(gen_sec)
    state.append(gen_state)
    fit.append(gen_fit)
    curve.append(gen_curve)
    name.append('GENETIC')
    # MIMIC
    t.tic()
    mimic_state,mimic_fit,mimic_curve = mimic(one_max_fit,
                                random_state = 17,
                                curve=True)
    mim_sec = t.tocvalue()
    t.toc()
    time.append(mim_sec)
    state.append(mimic_state)
    fit.append(mimic_fit)
    curve.append(mimic_curve)
    name.append('MIMIC')
    #Simulated Annealing
    t.tic()
    sa_state,sa_fit,sa_curve = sa(one_max_fit,
                                random_state = 17,
                                curve=True)
    sim_sec = t.tocvalue()
    t.toc()
    time.append(sim_sec)
    state.append(sa_state)
    fit.append(sa_fit)
    curve.append(sa_curve)
    name.append('SIMANNEAL')

    for i in range(4):
        length.append(ss)
        problem.append('OneMax')

onemax_df = pd.DataFrame({
    'Problem':problem,
    'Algorithm':name,
    'Length':length,
    'Fitness':fit,
    'Time':time,
    'State':state,
    'Curve':curve})

onemax_df.to_csv('onemax_stats.csv')

algo_l = []
length_l = []
fit_l = []
iters_l = []
for i in range(len(onemax_df)):
    algo = onemax_df.iloc[i,1]
    length = onemax_df.iloc[i,2]
    curve = onemax_df.iloc[i,6]
    for x in curve:
        fit_val = x[0]
        iter = x[1]

        algo_l.append(algo)
        length_l.append(length)
        fit_l.append(fit_val)
        iters_l.append(iter)

onemax_iters = pd.DataFrame({
    'Algorithm': algo_l,
    'Samples':length_l,
    'Score':fit_l,
    'Iterations':iters_l
})

onemax_iters.to_csv('onemax_iters.csv')
