import pandas as pd
import os
base_directory = '/cluster/work/lawecon/Work/lixiang'  # Adjust this to your home directory path
dataprocDirectory = os.path.join(base_directory, 'thousand_talent/data/proc')
datarawDirectory = os.path.join(base_directory, 'thousand_talent/data/raw')




balanced_dataset_merged = pd.read_csv(os.path.join(dataprocDirectory,
                                                   'balanced_panel_never_treated_control_group.csv'))



balanced_dataset_merged.loc[balanced_dataset_merged.before_first_year==1,'afid_masked_unique'] = "['before_first_year']"





balanced_dataset_merged['num_af_weight'] = 1/balanced_dataset_merged['afid_masked_nunique_filtered']


balanced_dataset_merged['af_list'] = balanced_dataset_merged.afid_masked_unique_filtered.map(lambda x: x.replace("[",'').replace("]",'').split(' '))


def return_features(i):
    relative_period,cohort,min_year,weight,aff_index,authid,subfield_most_frequent = balanced_dataset_merged.loc[i,['relative_period','cohort','min_year','num_af_weight','af_list','authid','subfield_most_frequent']].values
    # FE set 0: weight X afid X relative_period X cohort + authid X cohort + subfield_most_frequent X relative_period X cohort
    features0 = {str(weight)+'_'+af+'_'+str(relative_period)+'_'+str(cohort):1 for af in aff_index}
    features0[str(authid)+'_'+str(cohort)] = 1
    features0[subfield_most_frequent+'_'+str(relative_period)+'_'+str(cohort)] = 1
    # FE set 1: afid X relative_period X cohort:weight + authid X cohort + subfield_most_frequent X relative_period X cohort
    features1 = {af+'_'+str(relative_period)+'_'+str(cohort):weight for af in aff_index}
    features1[str(authid)+'_'+str(cohort)] = 1
    features1[subfield_most_frequent+'_'+str(relative_period)+'_'+str(cohort)] = 1
    # FE set 2: weight X afid X relative_period X cohort + authid X cohort + subfield_most_frequent X relative_period X cohort + min_year X relative_period X cohort
    features2 = {str(weight)+'_'+af+'_'+str(relative_period)+'_'+str(cohort):1 for af in aff_index}
    features2[str(authid)+'_'+str(cohort)] = 1
    features2[subfield_most_frequent+'_'+str(relative_period)+'_'+str(cohort)] = 1 
    features2[str(min_year)+'_'+str(relative_period)+'_'+str(cohort)] = 1
    # FE set 3: min_year X weight X afid X relative_period X cohort + authid X cohort + min_year X subfield_most_frequent X relative_period X cohort
    features3 = {str(min_year)+'_'+str(weight)+'_'+af+'_'+str(relative_period)+'_'+str(cohort):1 for af in aff_index}
    features3[str(authid)+'_'+str(cohort)] = 1
    features3[str(min_year)+'_'+subfield_most_frequent+'_'+str(relative_period)+'_'+str(cohort)] = 1
    # FE set 4: min_year X afid X relative_period X cohort:weight + authid X cohort + min_year X subfield_most_frequent X relative_period X cohort
    features4 = {str(min_year)+'_'+af+'_'+str(relative_period)+'_'+str(cohort):weight for af in aff_index}
    features4[str(authid)+'_'+str(cohort)] = 1
    features4[str(min_year)+'_'+subfield_most_frequent+'_'+str(relative_period)+'_'+str(cohort)] = 1 
    # FE set 5: afid X relative_period X cohort:weight + authid X cohort + subfield_most_frequent X relative_period X cohort + min_year X relative_period X cohort
    features5 = {af+'_'+str(relative_period)+'_'+str(cohort):weight for af in aff_index}
    features5[str(authid)+'_'+str(cohort)] = 1
    features5[subfield_most_frequent+'_'+str(relative_period)+'_'+str(cohort)] = 1 
    features5[str(min_year)+'_'+str(relative_period)+'_'+str(cohort)] = 1
    return [features0,features1,features2,features3,features4,features5]




from p_tqdm import p_map,p_umap




features = p_map(return_features,balanced_dataset_merged.index,num_cpus=15)



import pickle as pk



pk.dump(features,open(os.path.join(dataprocDirectory,'FE_dict.pk'), 'wb'))




