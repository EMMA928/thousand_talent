

import pandas as pd
import os
import sys


base_directory = '/cluster/work/lawecon/Work/lixiang'  # Adjust this to your home directory path
dataprocDirectory = os.path.join(base_directory, 'thousand_talent/data/proc')
datarawDirectory = os.path.join(base_directory, 'thousand_talent/data/raw')


# %%


balanced_dataset_merged = pd.read_csv(os.path.join(dataprocDirectory,
                                                   'balanced_panel_never_treated_control_group.csv'))


# %%


import pickle as pk


# %%


features_lists = pk.load(open(os.path.join(dataprocDirectory,'FE_dict.pk'), 'rb'))


# %%

feature_num = int(sys.argv[1])
features = [f[feature_num]
             for f in features_lists]


# %%


def partial_out_FE(features,suffix):
    from sklearn.feature_extraction import DictVectorizer
    v = DictVectorizer(sparse=True)
    FE = v.fit_transform(features)
    
    from sklearn.linear_model import LinearRegression
    
    def partial_out_FE(y_name):
        y_vals = balanced_dataset_merged[y_name].values
        reg = LinearRegression(fit_intercept=False).fit(FE, y_vals)
        predicted_y_vals = reg.predict(FE)
        residual = y_vals - predicted_y_vals
        return f"residualized_{y_name}",residual
    
    list_of_X = []
    relative_periods = balanced_dataset_merged.relative_period.unique()
    for rp in relative_periods:
        list_of_X.append(f"relative_period{rp}Xtreated")
    list_of_X.append("postXtreated")
    list_of_X.append("post0to3Xtreated")
    list_of_X.append("post4to8Xtreated")
    
    list_of_X.extend(['num_docs','ihs_num_docs',
                      'sum_citescore','ihs_sum_citescore',
                      'if_top10','if_top50',
                      'if_lower50','frac_count_docs'])

    # add cohort by relative time dummy
        
    from p_tqdm import p_map,p_umap
    X_residuals = p_umap(partial_out_FE,list_of_X,num_cpus=15)
    for colname,vals in X_residuals:
        balanced_dataset_merged.loc[:, colname+'_'+suffix] = vals





new_folder = os.path.join(dataprocDirectory, f"ihs_transformation")
os.makedirs(new_folder, exist_ok=True)

# Save the residualized outcomes to the new folder
for FE, suffix in [(features, f'FE{feature_num}')]:
    partial_out_FE(FE, suffix)



# Save the residualized dataset into the new folder
output_file_path = os.path.join(new_folder,
                                f'balanced_panel_never_treated_control_group_residualized_FE{feature_num}.csv')
balanced_dataset_merged.to_csv(output_file_path, index=False)
