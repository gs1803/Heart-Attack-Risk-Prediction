# Guhan Sivakumar (GDS200002)
# Nandini Elangovan (NKE200002)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report, precision_recall_curve
from scikitplot.metrics import plot_precision_recall
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from itertools import product
from neural_network import NeuralNetwork

df = pd.read_csv("https://raw.githubusercontent.com/gs1803/CS4375-Datasets/main/heart_attack_prediction_dataset.csv")
df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
df = df.drop(['Patient ID', 'Blood Pressure', 'Country', 'Continent'], axis=1)
df = pd.get_dummies(df, dtype=int)
df = df.rename(columns={'Sex_Female': 'Gender', 'Hemisphere_Northern Hemisphere': 'Hemisphere'})
df = df.drop(['Sex_Male', 'Hemisphere_Southern Hemisphere'], axis=1)

df = df[['Age', 'Gender', 'Diet_Average', 'Diet_Healthy', 'Diet_Unhealthy',
         'Hemisphere', 'Cholesterol', 'Systolic', 'Diastolic', 'Heart Rate',
         'Diabetes','Family History', 'Smoking', 'Alcohol Consumption',
         'Previous Heart Problems', 'Medication Use', 'Stress Level',
         'Sedentary Hours Per Day', 'Income', 'BMI', 'Triglycerides',
         'Physical Activity Days Per Week', 'Sleep Hours Per Day', 'Heart Attack Risk']]

ncols = len(df.columns)
df_corr = df.corr()
fig = sns.heatmap(df_corr, cmap='coolwarm')
fig.invert_yaxis()
plt.show()

scaler = StandardScaler()
clean_df = pd.DataFrame(scaler.fit_transform(df.iloc[:, 0:(ncols - 1)]))
X = clean_df
y = df.iloc[:, (ncols - 1)]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)


# Testing the ensemble methods and getting the class probabilities
params = {'max_depth': [5, 7, 9],
          'n_estimators': [50, 100, 200]
          }
rf = RandomForestClassifier(random_state=7)
# grid_search = GridSearchCV(rf, params, cv=10, scoring='accuracy', return_train_score=False)
# grid_search.fit(X, y)
# print(grid_search.best_params_)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
pred_probas_rf = rf.predict_proba(X_test)
print(classification_report(y_test, pred_rf, zero_division=False))
print(accuracy_score(y_test, pred_rf))
skplt.metrics.plot_confusion_matrix(y_test, pred_rf)
skplt.metrics.plot_roc(y_test, pred_probas_rf)
plot_precision_recall(y_test, pred_probas_rf)
plt.show()

ada = AdaBoostClassifier(random_state=7)
params =  {'n_estimators': [50, 100, 150, 200]}
# grid_search1 = GridSearchCV(ada, params, cv=10, scoring='accuracy', return_train_score=False)
# grid_search1.fit(X, y)
# print(grid_search1.best_params_)
ada.fit(X_train, y_train)
pred_ada = ada.predict(X_test)
pred_probas_ada = ada.predict_proba(X_test)
print(classification_report(y_test, pred_ada))
print(accuracy_score(y_test, pred_ada))
skplt.metrics.plot_confusion_matrix(y_test, pred_ada)
skplt.metrics.plot_roc(y_test, pred_probas_ada)
plot_precision_recall(y_test, pred_probas_ada)
plt.show()

bg = BaggingClassifier(random_state=7)
params =  {'n_estimators': [50, 100, 150, 200]}
# grid_search2 = GridSearchCV(tote_bag, params, cv=10, scoring='accuracy', return_train_score=False)
# grid_search2.fit(X, y)
# print(grid_search2.best_params_)
bg.fit(X_train, y_train)
pred_bg = bg.predict(X_test)
pred_probas_bg = bg.predict_proba(X_test)
print(classification_report(y_test, pred_bg))
print(accuracy_score(y_test, pred_bg))

skplt.metrics.plot_confusion_matrix(y_test, pred_bg)
skplt.metrics.plot_roc(y_test, pred_probas_bg)
plot_precision_recall(y_test, pred_probas_bg)
plt.show()

pred_proba_rf = rf.predict_proba(X)
pred_proba_ada = ada.predict_proba(X)
pred_proba_bg = bg.predict_proba(X)

# Making the new dataframe
rf_pred_0 = []
ada_pred_0 = []
bg_pred_0 = []

for i, _ in enumerate(pred_proba_rf):
    rf_pred_0.append(pred_proba_rf[i][0])

for j, _ in enumerate(pred_proba_ada):
    ada_pred_0.append(pred_proba_ada[j][0])

for k, _ in enumerate(pred_proba_bg):
    bg_pred_0.append(pred_proba_bg[k][0])

mod_X = X
mod_X['rf_proba_0'] = rf_pred_0
mod_X['ada_proba_0'] = ada_pred_0
mod_X['bg_proba_0'] = bg_pred_0

# Testing the ANN on the original dataset
nn_obj_og = NeuralNetwork(activation_func='relu', hidden_layer_sizes=[20,]*3,
                          initialization='random', epochs=50, optimizer='adam', 
                          learning_rate=0.01, lr_decay_type='time', alpha=0.01,
                          random_state=18)

X_og = clean_df.values
y_og = df.iloc[:, (ncols - 1)].values.reshape(-1, 1)

X_og_train, X_og_test, y_og_train, y_og_test = train_test_split(X_og, y_og, test_size=0.3, random_state=7)

fit_og = nn_obj_og.nn_train(X_og_train, y_og_train)
y_og_train_pred = nn_obj_og.nn_predict(X_og_train)
y_og_test_pred = nn_obj_og.nn_predict(X_og_test)

train_accuracy_og = accuracy_score(y_og_train, y_og_train_pred)
test_accuracy_og = accuracy_score(y_og_test, y_og_test_pred)

print("")
print(f"Train Accuracy: {train_accuracy_og}")
print(f"Test Accuracy: {test_accuracy_og}")

print(classification_report(y_og_test, y_og_test_pred))
pred_probas_og = nn_obj_og.nn_predict_proba(X_og_test)

skplt.metrics.plot_confusion_matrix(y_og_test, y_og_test_pred)
skplt.metrics.plot_roc(y_og_test, pred_probas_og)
plot_precision_recall(y_og_test, pred_probas_og)
plt.show()
nn_obj_og.plots()

nn_obj_mod = NeuralNetwork(activation_func='relu', hidden_layer_sizes=[20,]*3,
                           initialization='random', epochs=50, optimizer='adam', 
                           learning_rate=0.01, lr_decay_type='time', alpha=0.01,
                           random_state=18)
X_mod = mod_X.values
y_mod = df.iloc[:, (ncols - 1)].values.reshape(-1, 1)

X_mod_train, X_mod_test, y_mod_train, y_mod_test = train_test_split(X_mod, y_mod, test_size=0.3, random_state=7)

fit_mod = nn_obj_mod.nn_train(X_mod_train, y_mod_train)

y_mod_train_pred = nn_obj_mod.nn_predict(X_mod_train)
y_mod_test_pred = nn_obj_mod.nn_predict(X_mod_test)

train_accuracy_mod = accuracy_score(y_mod_train, y_mod_train_pred)
test_accuracy_mod = accuracy_score(y_mod_test, y_mod_test_pred)

print("")
print(f"Train Accuracy: {train_accuracy_mod}")
print(f"Test Accuracy: {test_accuracy_mod}")
print(classification_report(y_mod_test, y_mod_test_pred))

pred_probas_mod = nn_obj_mod.nn_predict_proba(X_mod_test)

skplt.metrics.plot_confusion_matrix(y_mod_test, y_mod_test_pred)
skplt.metrics.plot_roc(y_mod_test, pred_probas_mod)
plot_precision_recall(y_mod_test, pred_probas_mod)
plt.show()
nn_obj_mod.plots()

# activation_funcs = ['sigmoid', 'tanh', 'relu', 'linear']
# hidden_layer_sizes = [[20, 20, 20], [50, 50, 50]]
# initializations = ['xavier', 'he', 'random']
# epochs = [50, 100]
# optimizers = ['adam', 'sgd']
# learning_rates = [0.01, 0.05]
# decay_types = ['step', 'exponential', 'time']
# alphas = [0.01]
# regularizations = ['l1', 'l2']

# output = []

# kf = KFold(n_splits=4, shuffle=True, random_state=7)

# for af, hls, i, e, o, lr, dt, ral, r in product(activation_funcs, hidden_layer_sizes, initializations, 
#                                                 epochs, optimizers, learning_rates, decay_types,
#                                                 alphas, regularizations):
#     nn_test = NeuralNetwork(activation_func=af, hidden_layer_sizes=hls, 
#                             initialization=i, epochs=e, optimizer=o, 
#                             learning_rate=lr, decay_type=dt, lr_decrease=0.5, 
#                             lr_epoch_drop=5, alpha=ral, regularization=r)
#     accuracy_train_values = []
#     accuracy_test_values = []

#     for train_index, val_index in kf.split(X_mod):
#         X_train_fold, X_val_fold = X_mod[train_index], X_mod[val_index]
#         y_train_fold, y_val_fold = y_mod[train_index], y_mod[val_index]

#         fit = nn_test.nn_train(X_train_fold, y_train_fold, verbose=False)
#         y_train_pred = nn_test.nn_predict(X_train_fold)
#         y_test_pred = nn_test.nn_predict(X_val_fold)
        
#         accuracy_train = accuracy_score(y_train_fold, y_train_pred)
#         accuracy_train_values.append(accuracy_train)

#         accuracy_test = accuracy_score(y_val_fold, y_test_pred)
#         accuracy_test_values.append(accuracy_test)

#     average_train_accuracy = np.mean(accuracy_train_values)
#     average_test_accuracy = np.mean(accuracy_test_values)

#     output_dict = {'Parameters': nn_test.get_params(), 'Train/Test Split': '75-25', 
#                    'Dataset Size': '(8764, 26)', 'Train Accuracy': average_train_accuracy,
#                    'Test Accuracy': average_test_accuracy}
#     output.append(output_dict)
# out_df = pd.DataFrame(output)
# out_df = out_df.sort_values(by=['Test Accuracy'], ascending=False)
# out_df.to_csv("data/final_out.csv", index=False)

# Dictionaries of the best params
best_params_s = {'activation_func': 'sigmoid', 
                 'hidden_layer_sizes': [20, 20, 20], 
                 'initialization': 'he', 
                 'epochs': 100, 
                 'optimizer': 'adam', 
                 'learning_rate': 0.05, 
                 'lr_decay_type': 'time', 
                 'lr_decrease': 0.5, 
                 'lr_epoch_drop': 5, 
                 'alpha': 0.01, 
                 'regularization': 'l2', 
                 'tolerance': 0.0001, 
                 'tolerance_counter': 10, 
                 'beta_1': 0.9, 
                 'beta_2': 0.999, 
                 'epsilon': 1e-08, 
                 'random_state': 18}

best_params_t = {'activation_func': 'tanh', 
                 'hidden_layer_sizes': [20, 20, 20], 
                 'initialization': 'he', 
                 'epochs': 50, 
                 'optimizer': 'sgd', 
                 'learning_rate': 0.01, 
                 'lr_decay_type': 'time', 
                 'lr_decrease': 0.5, 
                 'lr_epoch_drop': 5, 
                 'alpha': 0.01, 
                 'regularization': 'l2', 
                 'tolerance': 0.0001, 
                 'tolerance_counter': 10, 
                 'beta_1': 0.9, 
                 'beta_2': 0.999, 
                 'epsilon': 1e-08, 
                 'random_state': 18}

best_params_r = {'activation_func': 'relu', 
                 'hidden_layer_sizes': [50, 50, 50], 
                 'initialization': 'random', 
                 'epochs': 100, 
                 'optimizer': 'adam', 
                 'learning_rate': 0.01, 
                 'lr_decay_type': 'time', 
                 'lr_decrease': 0.5, 
                 'lr_epoch_drop': 5, 
                 'alpha': 0.01, 
                 'regularization': 'l1', 
                 'tolerance': 0.0001, 
                 'tolerance_counter': 10, 
                 'beta_1': 0.9, 
                 'beta_2': 0.999, 
                 'epsilon': 1e-08, 
                 'random_state': 18}

# Running KFold on the best parameters
kf = KFold(n_splits=4, shuffle=True, random_state=7)

nn_final_s = NeuralNetwork()
nn_final_s.set_params(best_params_s)
accuracy_train_values_s = []
accuracy_test_values_s = []

for train_index, test_index in kf.split(X_mod):
    X_train_fold, X_test_fold = X_mod[train_index], X_mod[test_index]
    y_train_fold, y_test_fold = y_mod[train_index], y_mod[test_index]

    fit = nn_final_s.nn_train(X_train_fold, y_train_fold, verbose=False)
    y_train_pred = nn_final_s.nn_predict(X_train_fold)
    y_test_pred = nn_final_s.nn_predict(X_test_fold)
    
    accuracy_train = accuracy_score(y_train_fold, y_train_pred)
    accuracy_train_values_s.append(accuracy_train)

    accuracy_test = accuracy_score(y_test_fold, y_test_pred)
    accuracy_test_values_s.append(accuracy_test)

    print(classification_report(y_test_fold, y_test_pred))
    pred_proba_s = nn_obj_mod.nn_predict_proba(X_test_fold)

    skplt.metrics.plot_confusion_matrix(y_test_fold, y_test_pred)
    skplt.metrics.plot_roc(y_test_fold, pred_proba_s)
    plot_precision_recall(y_test_fold, pred_proba_s)
    plt.show()

    nn_final_s.plots()

average_train_accuracy_s = np.mean(accuracy_train_values_s)
average_test_accuracy_s = np.mean(accuracy_test_values_s)

print(average_test_accuracy_s)
print(average_test_accuracy_s)

nn_final_t = NeuralNetwork()
nn_final_t.set_params(best_params_t)
accuracy_train_values_t = []
accuracy_test_values_t = []

for train_index, test_index in kf.split(X_mod):
    X_train_fold, X_test_fold = X_mod[train_index], X_mod[test_index]
    y_train_fold, y_test_fold = y_mod[train_index], y_mod[test_index]

    fit = nn_final_t.nn_train(X_train_fold, y_train_fold, verbose=False)
    y_train_pred = nn_final_t.nn_predict(X_train_fold)
    y_test_pred = nn_final_t.nn_predict(X_test_fold)
    
    accuracy_train = accuracy_score(y_train_fold, y_train_pred)
    accuracy_train_values_t.append(accuracy_train)

    accuracy_test = accuracy_score(y_test_fold, y_test_pred)
    accuracy_test_values_t.append(accuracy_test)

    print(classification_report(y_test_fold, y_test_pred))
    pred_proba_t = nn_obj_mod.nn_predict_proba(X_test_fold)

    skplt.metrics.plot_confusion_matrix(y_test_fold, y_test_pred)
    skplt.metrics.plot_roc(y_test_fold, pred_proba_t)
    plot_precision_recall(y_test_fold, pred_proba_t)
    plt.show()

    nn_final_t.plots()

average_train_accuracy_t = np.mean(accuracy_train_values_t)
average_test_accuracy_t = np.mean(accuracy_test_values_t)

print(average_train_accuracy_t)
print(average_test_accuracy_t)

nn_final_r = NeuralNetwork()
nn_final_r.set_params(best_params_r)
accuracy_train_values_r = []
accuracy_test_values_r = []
kf = KFold(n_splits=4, shuffle=True, random_state=7)

for train_index, test_index in kf.split(X_mod):
    X_train_fold, X_test_fold = X_mod[train_index], X_mod[test_index]
    y_train_fold, y_test_fold = y_mod[train_index], y_mod[test_index]

    fit = nn_final_r.nn_train(X_train_fold, y_train_fold, verbose=False)
    y_train_pred = nn_final_r.nn_predict(X_train_fold)
    y_test_pred = nn_final_r.nn_predict(X_test_fold)
    
    accuracy_train = accuracy_score(y_train_fold, y_train_pred)
    accuracy_train_values_r.append(accuracy_train)

    accuracy_test = accuracy_score(y_test_fold, y_test_pred)
    accuracy_test_values_r.append(accuracy_test)

    print(classification_report(y_test_fold, y_test_pred))
    pred_proba_r = nn_obj_mod.nn_predict_proba(X_test_fold)

    skplt.metrics.plot_confusion_matrix(y_test_fold, y_test_pred)
    skplt.metrics.plot_roc(y_test_fold, pred_proba_r)
    plot_precision_recall(y_test_fold, pred_proba_r)
    plt.show()

    nn_final_r.plots()

average_train_accuracy_r = np.mean(accuracy_train_values_r)
average_test_accuracy_r = np.mean(accuracy_test_values_r)

print(average_train_accuracy_r)
print(average_test_accuracy_r)
