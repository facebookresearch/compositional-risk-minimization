# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import argparse

ALGORITHMS= 'ERM GroupDRO LA_Group LA_Cond IRM VREx Mixup CRM'
DATASETS= 'Waterbirds CelebA MetaShift'
DISCARDED_GROUPS= [''] + [str(i) for i in range(4)]

# ALGORITHMS= 'ERM GroupDRO LA_Group LA_Cond IRM VREx Mixup CRM '
# DATASETS= 'MultiNLI'
# DISCARDED_GROUPS= [''] + [str(i) for i in range(6)]

# ALGORITHMS= 'ERM GroupDRO LA_Group LA_Cond IRM VREx Mixup CRM '
# DATASETS= 'CivilComments'
# DISCARDED_GROUPS= [''] + [str(i) for i in range(16)]

#No need to specify additionally discarded groups for NICOpp as the original train split has missing groups
# ALGORITHMS= 'ERM GroupDRO LA_Group LA_Cond IRM VREx Mixup CRM '
# DATASETS= 'CompositionalNICOpp'
# DISCARDED_GROUPS=['']

#To repoduce results for the original NICOpp (Table 14) where the validation set still has samples from the missing groups in train set
# ALGORITHMS= 'ERM GroupDRO LA_Group LA_Cond IRM VREx Mixup CRM '
# DATASETS= 'NICOpp'
# DISCARDED_GROUPS=['']

# ALGORITHMS= 'ERM GroupDRO LA_Group LA_Cond IRM VREx Mixup CRM_Multi_Attr'
# DATASETS= 'CelebAMultiAttr'
# DISCARDED_GROUPS= [''] + [str(i) for i in range(32)]

parser = argparse.ArgumentParser(description='')
parser.add_argument('--case', type=str, help='Choose from htune/train_unf/eval_unf/quick_run')
parser.add_argument('--finetune', action= 'store_true', help='Finetune the pretrained backbone or not')
parser.add_argument('--slurm_dir', type=str, default="",help='Directory to save the slurm logs.')
parser.add_argument('--partition', type=str, default='', help='Slurm partition to use')
parser.add_argument('--out_dir', type=str, help='Directory to save the results')
args = parser.parse_args() 

if args.case == 'htune':
    base_script= 'python -m src.main --data_path data --case train  --num_hparams_combs 5 --num_seeds 1 --slurm_partition ' + args.partition  + ' --slurm_dir ' + args.slurm_dir + ' --datasets ' + DATASETS + ' --algorithms ' + ALGORITHMS 
elif args.case == 'train_unf':
    base_script= 'python -m src.main --data_path data --case train --best_hparams_comb_selection_metric va_unf_acc --num_seeds 3 --slurm_partition ' + args.partition + ' --slurm_dir ' + args.slurm_dir + ' --datasets ' + DATASETS + ' --algorithms ' + ALGORITHMS
elif args.case == 'eval_unf':
    base_script= 'python -m src.main --data_path data --case eval --best_hparams_comb_selection_metric va_unf_acc --num_seeds 3 --slurm_partition ' + args.partition +  ' --slurm_dir ' + args.slurm_dir + ' --datasets ' + DATASETS + ' --algorithms ' + ALGORITHMS
elif args.case == 'quick_run':
    base_script= 'python -m src.main --data_path data --case train --quick_run --seed 1000 --slurm_partition ' + args.partition  + ' --slurm_dir ' + args.slurm_dir + ' --datasets ' + DATASETS + ' --algorithms ' + ALGORITHMS
else:
    raise AssertionError('Valid case not provided')

if args.finetune:
    out_dir= os.path.join(args.out_dir, 'Finetune')
    base_script+= ' --out_dir ' + out_dir
else:
    out_dir= os.path.join(args.out_dir, 'No-Finetune')
    base_script+=  ' --out_dir ' + out_dir + ' --precompute_features '

for group in DISCARDED_GROUPS:
    script= base_script + ' --add_discarded_group ' + group
    print(script)
    os.system(script)