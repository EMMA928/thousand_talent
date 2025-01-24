#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import pandas as pd
import os
base_directory = '/cluster/work/lawecon/Work/lixiang'  # Adjust this to your home directory path
dataprocDirectory = os.path.join(base_directory, 'thousand_talent/data/proc')
datarawDirectory = os.path.join(base_directory, 'thousand_talent/data/raw')
# %%


balanced_dataset_merged = pd.read_csv(os.path.join(dataprocDirectory,
                                                   'balanced_panel_never_treated_control_group_log.csv'))


# %%


import pickle as pk


# %%


features_lists = pk.load(open(os.path.join(dataprocDirectory,'FE_dict_log.pk'), 'rb'))


# %%


import sys
features_lists = pk.load(open(os.path.join(dataprocDirectory,'FE_dict_log.pk'), 'rb'))
feature_num = int(sys.argv[1])
features = [f[feature_num] for f in features_lists]


# %%


def partial_out_FE(features,suffix):
    from sklearn.feature_extraction import DictVectorizer
    v = DictVectorizer(sparse=True)
    FE = v.fit_transform(features)
    
    from sklearn.linear_model import LinearRegression
    
    def partial_out_FE(y_name):
        y_vals = cohort_df[y_name].values
        reg = LinearRegression(fit_intercept=False).fit(FE, y_vals)
        predicted_y_vals = reg.predict(FE)
        residual = y_vals - predicted_y_vals
        return f"residualized_{y_name}",residual
    
    list_of_X = []
    relative_periods = cohort_df.relative_period.unique()
    for rp in relative_periods:
        list_of_X.append(f"relative_period{rp}Xtreated")
    list_of_X.append("postXtreated")
    list_of_X.append(f"post0to3Xtreated")
    if cohort < 2016:
        list_of_X.append(f"post4to8Xtreated")
        
    list_of_X.extend(['num_docs','log_1plus_num_docs',
                      'sum_citescore','log_1plus_sum_citescore',
                      'if_top10','if_top50',
                      'if_lower50','frac_count_docs'])
        
    from p_tqdm import p_map,p_umap
    X_residuals = p_umap(partial_out_FE,list_of_X,num_cpus=15)
    for colname,vals in X_residuals:
        cohort_df[colname+'_'+suffix] = vals

new_folder = os.path.join(dataprocDirectory, f"log_1plus_transformation")

# %%
for cohort in [2011,2012,2013,2015,2016,2017]:
    cohort_df = balanced_dataset_merged[balanced_dataset_merged.cohort==cohort]
    cohort_features = [features[i] for i in cohort_df.index]
    for FE,suffix in [(cohort_features,'FE0')]:
        partial_out_FE(FE,suffix)
    cohort_df.to_csv(os.path.join(new_folder,
                                  f'balanced_panel_never_treated_control_group_residualized_{cohort}_log.csv'),index=False)


# %%


