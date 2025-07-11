import os

TOTAL_SEEDS=3
# dir= '/checkpoint/divyatmahajan/CRM_ICLR/No-Finetune/'
dir= '/checkpoint/divyatmahajan/CRM_ICLR/Finetune/'
SEL_METRICS= ['unf_acc']
# SEL_METRICS= ['wga', 'unf_acc']
# DATASETS= ['Waterbirds', 'CelebA', 'MetaShift', 'NICOpp', 'CompositionalNICOpp', 'MultiNLI', 'CivilComments']
DATASETS= ['Waterbirds', 'CelebA', 'MetaShift', 'CompositionalNICOpp', 'MultiNLI', 'CivilComments']
# DATASETS= ['Waterbirds', 'CelebA', 'MetaShift', 'CompositionalNICOpp']
# ALGORITHMS= ['IRM']
# ALGORITHMS= ['IRM', 'VREx', 'Mixup']
# ALGORITHMS= ['ERM', 'GroupDRO', 'LA_Group', 'LA_Cond', 'CRM']
ALGORITHMS= ['ERM', 'GroupDRO', 'LA_Group', 'LA_Cond', 'IRM', 'VREx', 'Mixup', 'CRM']
DISCARDED_GROUPS={
    'Waterbirds' : [[], [0], [1], [2], [3]],
    'CelebA' : [[], [0], [1], [2], [3]],
    'MetaShift' : [[], [0], [1], [2], [3]],
    'CompositionalNICOpp': [[]],
    # 'Waterbirds' : [[0], [1], [2], [3]],
    # 'CelebA' : [[0], [1], [2], [3]],
    # 'MetaShift' : [[0], [1], [2], [3]],
    # 'Waterbirds' : [[]],
    # 'CelebA' : [[]],
    # 'MetaShift' : [[]],
    # 'NICOpp': [[]],
    # 'CompositionalNICOpp': [[], [1], [2], [3], [4], [5], [6]],
    'MultiNLI': [[], [0], [1], [2], [3], [4], [5]],
    'CivilComments': [[], [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]],
}

DATASETS= ['CelebAMultiAttr']
ALGORITHMS= ['ERM', 'GroupDRO', 'LA_Group', 'LA_Cond', 'IRM', 'VREx', 'Mixup', 'CRM_Multi_Attr']
DISCARDED_GROUPS={
    'CelebAMultiAttr': [
                        [], [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], 
                        [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], 
                        [21], [22], [24], [25], [26], [28], [29], [31]
                        ]
}


def check_logs(curr_path, sel_metric):
    count=0
    for _, _, f_list in os.walk(curr_path):
        for fname in f_list:
            if sel_metric + '_f1_complete.csv' in fname:
                count+=1
    if count != TOTAL_SEEDS:
        print('Error: Found only ', count, 'files for the case of model selection with group balanced acc for ', curr_path)                

def check_logs_htune(curr_path):
    count=0
    for root, _, f_list in os.walk(curr_path):
        if root != curr_path:
            continue
        for fname in f_list:
            if sel_metric + '.pt' in fname:
                count+=1
    if count != 5:
        print('Error: Found only ', count, 'files for the case of model selection with group balanced acc for ', curr_path)                

for dataset in DATASETS:
    for algorithm in ALGORITHMS:
        for discarded_group in DISCARDED_GROUPS[dataset]:
            add_disc_gr_save_path=''
            if len(discarded_group):
                for group in discarded_group:
                    add_disc_gr_save_path+= '_' + str(group)
                    for sel_metric in SEL_METRICS:                       
                        curr_path = os.path.join(
                        dir, algorithm, dataset,
                        'group_labels_yes', 
                        'add_discarded_groups' + add_disc_gr_save_path,
                        'va_' + sel_metric,
                        ''
                        )
                        check_logs(curr_path, sel_metric)
                        
                        # curr_path= os.path.join(
                        #     dir, algorithm, dataset,
                        #     'group_labels_yes',
                        #     'add_discarded_groups' + add_disc_gr_save_path,
                        #     ''
                        # )
                        # check_logs_htune(curr_path)
            else:
                for sel_metric in SEL_METRICS:                       
                    curr_path = os.path.join(
                    dir, algorithm, dataset,
                    'group_labels_yes', 
                    'add_discarded_groups' + add_disc_gr_save_path,
                    'va_' + sel_metric,
                    ''
                    )
                    check_logs(curr_path, sel_metric)

                    # curr_path= os.path.join(
                    #     dir, algorithm, dataset,
                    #     'group_labels_yes',
                    #     'add_discarded_groups' + add_disc_gr_save_path,
                    #     ''
                    # )
                    # check_logs_htune(curr_path)
