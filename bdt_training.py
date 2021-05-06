from matplotlib import pyplot as plt
import uproot
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from bdt_input import *


###training variables
training_columns = ['N_jets', 's33', 'mass_jet1','mass_jet2','dRmin_mu_jet','ptrel_mu_jet','csv_jet1','csv_jet2','dRmin_mu_jet_scaled', 'st']


###build training sets
sig_df = tree.pandas.df(training_columns).sample(n = 10000)
mc_df = tree_mc.pandas.df(training_columns).sample(n = 10000)
bkg_df = pd.concat([tree_bkg2.pandas.df(training_columns).sample(n = 1250), tree_bkg3.pandas.df(training_columns).sample(n = 1250), tree_bkg4.pandas.df(training_columns).sample(n = 1250), tree_bkg5.pandas.df(training_columns).sample(n = 1250), tree_bkg6.pandas.df(training_columns).sample(n = 1250), tree_bkg8.pandas.df(training_columns).sample(n = 1250)], copy=True, ignore_index=True)
test_df = pd.concat([tree.pandas.df(training_columns).sample(n = 5000), tree_mc.pandas.df(training_columns).sample(n = 5000)], copy=True, ignore_index=True)


###select classifier
bdt = XGBClassifier(max_depth=9)



bkg_df['catagory'] = 0  # Use 0 for background
mc_df['catagory'] = 1  # Use 1 for signal
sig_df['catagory'] = 1 
test_df['catagory'] = 1

###training with k-fold
kf = KFold(n_splits=10, random_state=42, shuffle=True)
for train_index, test_index in kf.split(bkg_df):
    sig_df_train, sig_df_test = sig_df.iloc[train_index], sig_df.iloc[test_index]
    mc_df_train, mc_df_test = mc_df.iloc[train_index], mc_df.iloc[test_index]
    test_df_train, test_df_test = test_df.iloc[train_index], test_df.iloc[test_index]
    bkg_df_train, bkg_df_test = bkg_df.iloc[train_index], bkg_df.iloc[test_index]
    training_data = pd.concat([sig_df_train, bkg_df_train], copy=True, ignore_index=True)
    bdt.fit(training_data[training_columns], training_data['catagory']) 

###show the importances of the variables
print bdt.feature_importances_


mc_vali = mc_df_test
bkg_vali = bkg_df_test
sig_vali = sig_df_test




for df in [mc_vali, bkg_vali, sig_vali, training_data]:
    df['BDT'] = bdt.predict_proba(df[training_columns])[:,1]


def plot_comparision(var, mc_df, bkg_df, sig_df):
    _, bins, _ = plt.hist(mc_df[var], histtype='step', label='TTbar', bins=60, normed=False)
    _, bins, _ = plt.hist(bkg_df[var], histtype='step', label='WJet', bins=60, normed=False)
    _, bins, _ = plt.hist(sig_df[var], histtype='step', label='Signal', bins=60, normed=False) 
    plt.xlabel(var)
    plt.xlim(bins[0], bins[-1])
    plt.legend(loc='best')
    plt.show()

def plot_mass(mc_df, bkg_df, plot, **kwargs):
    counts, bins, _ = plt.hist(mc_df[plot],label='Signal', bins=50, histtype='step', **kwargs)
    counts, bins, _ = plt.hist(bkg_df[plot], label='WJet', bins=50, histtype='step', **kwargs)
    plt.xlabel(plot)
    plt.xlim(bins[0], bins[-1])
    plt.legend(loc='best')
    plt.show()

def plot_roc(bdt, training_data, training_columns, label=None):
    y_score = bdt.predict_proba(training_data[training_columns])[:,1]
    fpr, tpr, thresholds = roc_curve(training_data['catagory'], y_score)
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--', label='Randomly guess')
    plt.plot(fpr, tpr, label='ROC curve')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def plot_significance(bdt, training_data, training_columns, label=None):
    y_score = bdt.predict_proba(training_data[training_columns])[:,1]
    fpr, tpr, thresholds = roc_curve(training_data['catagory'], y_score)

    n_sig = 10000
    n_bkg = 10000
    S = n_sig*tpr
    B = n_bkg*fpr
    metric = S/np.sqrt(S+B)

    plt.plot(thresholds, metric, label=label)
    plt.xlabel('BDT cut value')
    plt.ylabel('$\\frac{S}{\\sqrt{S+B}}$')
    plt.xlim(0, 1.0)

    optimal_cut = thresholds[np.argmax(metric)]
    plt.axvline(optimal_cut, color='black', linestyle='--')
    print optimal_cut
    plt.show() 




#BDT score
plot_comparision('BDT', mc_vali, bkg_vali, sig_vali)
#Best Cut
plot_significance(bdt, training_data, training_columns)



