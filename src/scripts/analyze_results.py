import os
import glob
import re
import argparse
import pickle
import numpy as np
import pandas as pd
import copy
from src.utils.utils import process_json_files, sort_and_remove_empty

PICKLE_METRICS = [ 
            'te_avg_acc', 'te_unf_acc', 'te_wga', 
        ]

DISCARDED_GROUPS={
    'Waterbirds' : [[0], [1], [2], [3]],
    'CelebA' : [[0], [1], [2], [3]],
    'MetaShift' : [[0], [1], [2], [3]],
    'MultiNLI': [[0], [1], [2], [3], [4], [5]],
    'CivilComments': [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]],
    'CompositionalNICOpp': [[]],
    'CelebAMultiAttr': [
                        [1], [2], [3], [4], [5], [6], [7], [8], [9], 
                        [10], [11], [12], [13], [14], [15], [16], [17], [18], [19],
                        [21], [22], [24], [25], [26], [28], [29], [31]
                         ],
}

# For the non-compositional shift case with original splits
# DISCARDED_GROUPS={
#     'Waterbirds' : [[]],
#     'CelebA' : [[]],
#     'MetaShift' : [[]],
#     'NICOpp': [[]],
#     'MultiNLI': [[]],
#     'CivilComments': [[]],
#     'CelebAMultiAttr': [[]]
# }


METHOD_LATEX={
    'ERM': 'ERM',
    'GroupDRO': 'G-DRO',
    'LA_Group': 'LC',
    'LA_Cond': 'sLA',
    'IRM': 'IRM',
    'VREx': 'VREx',
    'Fishr': 'Fishr',
    'Mixup': 'Mixup',
    'CRM': 'CRM',
    'CRM_Empirical_Prior': '$B^\star$ + Emp Prior',
    'CRM_Naive_Uniform_Prior': '$\hat{B}$ + Unf Prior',
    'CRM_Naive_Empirical_Prior': '$\hat{B}$ + Emp Prior'
}

COL_RENAME_DICT= {
    'method': 'Method',
    'te_avg_acc': 'Average Acc',
    'te_unf_acc': 'Balanced Acc',
    'te_micro_f1_unf': 'Macro F1',
    'te_wga': 'Worst Group Acc'
}

def rank_agg_scenario(dataset_df):

    disc_scenarios= dataset_df['Discarded Group'].unique()
    methods= dataset_df['method'].unique()

    #Add a seed column
    seed_df= []
    for scenario in disc_scenarios:
        for method in methods:
            total_seeds= ((dataset_df['Discarded Group'] == scenario) & (dataset_df['method'] == method)).sum()
            for idx in range(total_seeds):
                seed_df.append( idx )
    seed_df= pd.Series(seed_df)
    dataset_df['seed']= seed_df
    num_seeds= len(dataset_df['seed'].unique())

    #Remove rows corresponding to CRM ablation methods
    corr_idx= ~dataset_df['method'].isin(['CRM_Naive_Uniform_Prior', 'CRM_Naive_Empirical_Prior']) 
    # corr_idx= ~dataset_df['method'].isin(['CRM_Empirical_Prior', 'CRM_Naive_Uniform_Prior', 'CRM_Naive_Empirical_Prior']) 
    dataset_df= dataset_df[corr_idx]

    #Conditioned on discarded group and seed, find rank of each method
    rank_df=[]    
    for scenario in disc_scenarios:
        for seed in range(num_seeds):
            print(scenario, seed)
            curr_df= dataset_df[(dataset_df['Discarded Group'] == scenario) & (dataset_df['seed'] == seed)]
            for metric in PICKLE_METRICS:
                curr_df[metric + '_rank'] = curr_df.fillna(0.0)[metric].rank(method='dense', ascending=False).astype(int)
            curr_df= curr_df.drop(PICKLE_METRICS, axis=1)
            rank_df.append(curr_df)
    rank_df= pd.concat(rank_df, axis=0)

    #Take mean over discarded scenarios and random seeds to get average rank
    rank_df= rank_df.drop(['Discarded Group', 'seed'], axis=1)
    grouped_df= rank_df.groupby(['method'])
    rank_df = pd.DataFrame( { key :  grouped_df.apply(lambda x: x[key+'_rank'].mean()) for key in PICKLE_METRICS } )
    
    #Generate the final table
    res_dataset=[]
    methods= dataset_df['method'].unique()
    for method in methods:
        res_method={}
        res_method['method']= METHOD_LATEX[method]
        for key in PICKLE_METRICS:
            res_method[key] = rank_df[key][method]
        res_dataset.append(res_method)
    res_dataset= pd.DataFrame(res_dataset)    

    return res_dataset

def analyze_agg_scenario(dataset_df):

    disc_scenarios= dataset_df['Discarded Group'].unique()
    methods= dataset_df['method'].unique()

    #Add a seed column
    seed_df= []
    for scenario in disc_scenarios:
        for method in methods:
            total_seeds= ((dataset_df['Discarded Group'] == scenario) & (dataset_df['method'] == method)).sum()
            for idx in range(total_seeds):
                seed_df.append( idx )
    seed_df= pd.Series(seed_df)
    dataset_df['seed']= seed_df

    #Aggregate with min operator on the Discarded Group column
    dataset_df= dataset_df.drop(['Discarded Group'], axis=1)
    grouped_df= dataset_df.groupby(['seed', 'method'])
    agg_df= pd.DataFrame( { key :  grouped_df.apply(lambda x: x[key].mean()) for key in PICKLE_METRICS } )
    agg_df= agg_df.reset_index()

    #Compute mean (standard error) over the random seeds
    agg_df= agg_df.drop(['seed'], axis=1)
    grouped_df = agg_df.groupby(['method'])
    mean_df= pd.DataFrame( { key :  grouped_df.apply(lambda x: 100 * x[key].mean()) for key in PICKLE_METRICS } ).round(1)
    sem_df=  pd.DataFrame( { key :  grouped_df.apply(lambda x: 100 * x[key].sem()) for key in PICKLE_METRICS } ).round(1)    

    #Generate the final table
    res_dataset=[]
    for method in methods:
        res_method={}
        res_method['method']= METHOD_LATEX[method]
        for key in PICKLE_METRICS:
            res_method[key] = '$ ' + str( mean_df[key][method]) + ' $' + ' \small{$( ' + str( sem_df[key][method]) + ')$}' 
        res_dataset.append(res_method)
    res_dataset= pd.DataFrame(res_dataset)

    return res_dataset

def analyze_per_scenario(dataset_df):
    disc_scenarios= dataset_df['Discarded Group'].unique()
    methods= dataset_df['method'].unique()
    
    grouped_df = dataset_df.groupby(['Discarded Group', 'method'])
    mean_df= pd.DataFrame( { key :  grouped_df.apply(lambda x: 100 * x[key].mean()) for key in PICKLE_METRICS } ).round(1)
    sem_df=  pd.DataFrame( { key :  grouped_df.apply(lambda x: 100 * x[key].sem()) for key in PICKLE_METRICS } ).round(1)

    res_dataset=[]
    for scenario in disc_scenarios:
        for method in methods:
            res_method={}
            res_method['Discarded Group']= ''
            # res_method['Discarded Group']= scenario
            res_method['method']= METHOD_LATEX[method]
            for key in PICKLE_METRICS:
                res_method[key] = '$ ' + str(mean_df[key][scenario, method]) + ' $' + ' $( ' +  str(sem_df[key][scenario, method]) + ')$'
                # res_method[key] = '$ ' + str(mean_df[key][scenario, method]) + ' $' + ' \small{$( ' +  str(sem_df[key][scenario, method]) + ')$}'
            res_dataset.append(res_method)
    res_dataset= pd.DataFrame(res_dataset)
    return res_dataset

def get_full_dataframe(args, dataset):

    final_df= []
    for scenario_idx, add_disc_groups in enumerate(DISCARDED_GROUPS[dataset]):
        
        add_disc_gr_save_path=''
        if add_disc_groups:
            for group in add_disc_groups:
                add_disc_gr_save_path+= '_' + str(group)
                
        for algorithm in args.algorithms:
            dir_path = os.path.join(
                args.dir, algorithm, dataset,
                f'group_labels_{args.group_labels}', 
                'add_discarded_groups' + add_disc_gr_save_path,
                'va_' + args.selection_criterion
                )
            
            pattern= os.path.join(dir_path, f'results_hpcomb_*_seed*_{args.selection_criterion}.csv')
            pickle_files= glob.glob(pattern)
            assert len(pickle_files) > 0            
            for file in pickle_files:
                curr_df= pd.read_csv(file)
                curr_df.insert(0, 'Discarded Group', scenario_idx)
                final_df.append(curr_df)
    
    return pd.concat(final_df, axis=0).reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read results')
    parser.add_argument('--dir')
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--algorithms', nargs='+')
    parser.add_argument('--group_labels', default='yes')
    parser.add_argument('--selection_criterion', default='unf_acc')
    args = parser.parse_args()

    GROUP_METRICS= []
    PICKLE_METRICS= GROUP_METRICS + PICKLE_METRICS

    agg_scenario_df=[]
    for dataset in args.datasets:
        print('Dataset: ', dataset)
        dataset_df= get_full_dataframe(args, dataset)

        #Results for each discarded group scenario
        res_dataset= analyze_per_scenario(copy.deepcopy(dataset_df))
        res_dataset= res_dataset.rename(COL_RENAME_DICT, axis='columns')
        print('Results for all scenarios')
        print(res_dataset.to_latex(escape=False, index=False))

        #Results post aggregations across different discarded group scenarios
        res_dataset= analyze_agg_scenario(copy.deepcopy(dataset_df))
        if dataset == 'CompositionalNICOpp':
            res_dataset.insert(0, 'Dataset', 'NICOpp')
        else:
            res_dataset.insert(0, 'Dataset', dataset)
        agg_scenario_df.append(res_dataset)

    agg_scenario_df= pd.concat(agg_scenario_df, axis=0)
    agg_scenario_df= agg_scenario_df.rename(COL_RENAME_DICT, axis='columns')
    print('Results aggregated over all scenarios')
    print(agg_scenario_df.to_latex(escape=False, index=False))
