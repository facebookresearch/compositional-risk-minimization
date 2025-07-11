import os
import argparse
import torch
from torch.nn.functional import cross_entropy
import numpy as np
import pandas as pd
import pickle
from src.utils.hparams_registry import get_hparams
from src.utils.utils import load_checkpoint, save_checkpoint, set_all_seeds, is_degenerate
from src.utils.utils import report_stats, get_metrics, Iter, prepare_out_dir, find_best_hparams_comb
from src.data.datasets import get_loaders
from src.models.algorithms import get_algorithm, ERM
from src.models.networks import get_network, get_optim, TwinNets

SEL_METRIC= ['unf_acc']

class ValidationHelper:
    def __init__(self, patience: int =2000, min_delta: float=1e-4):
        """
        Inputs:
            patience: Number of epochs to wait before early stopping
            min_delta: Minimum change in validation loss to be considered as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.best_epoch = -1
        
    def save_model(self, validation_loss, epoch):
        """
        Inputs:
            validation_loss: Validation loss of the current epoch
            epoch: Current epoch
        """
        if validation_loss < (self.min_validation_loss + self.min_delta):
            self.min_validation_loss = validation_loss
            self.best_epoch = epoch
            self.counter= 0
            return True
        return False
    
    def early_stop(self, validation_loss):        
        """
        Inputs:
            validation_loss: Validation loss of the current epoch
        """
        if validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def get_log_prior_prob(prior_count: torch.Tensor) -> torch.Tensor:
    """
        Compute the log prior probabilities for each group based on their counts in the training set.
        Set the discarded group probabilties to a small number (equivalently the prob. to a large neg number)
    """
    prior_prob= prior_count/torch.sum(prior_count)
    return torch.log(prior_prob.clip(min=1e-8))

def train(args):
    
    set_all_seeds(args['seed'])
    hparams = get_hparams(args)
    prepare_out_dir(hparams)

    loaders = get_loaders(hparams)
    #Sanity Check
    for g_idx in loaders['discarded_groups']:
        if loaders['group_stats_tr'][g_idx] != 0:
            raise AssertionError("One of the dropped groups does not have zero count in training data")

    net = get_network(hparams)        
    optim = get_optim(hparams, net)
    algorithm = get_algorithm(hparams, net, optim)
    step_st = load_checkpoint(hparams, algorithm) if args['resume'] else 0

    val_helper={}
    for sel_metric in SEL_METRIC:
        val_helper[sel_metric]= ValidationHelper()

    if hparams['algorithm_name'] in ['CRM', 'CRM_Multi_Attr']:
        prior_prob= torch.Tensor( loaders['group_stats_tr'] ).to(hparams['device'])
        algorithm.train_log_prior_prob = get_log_prior_prob(prior_prob)
 
    print('Training starts..')
    tr_loader_iterator = Iter(loaders['tr'])
    for step in range(step_st, hparams['num_step']):
        batch = tr_loader_iterator.next()
        algorithm.update(batch)
    
        if step % hparams['checkpoint_freq'] == 0:
            stats= report_stats(hparams, algorithm, loaders, step, num_y= hparams['num_y'])
            for sel_metric in SEL_METRIC:
                val_loss= -1 * stats['va_' + sel_metric]
                if val_helper[sel_metric].save_model(val_loss, step):
                    print('Improvement in Validation Metric: ', sel_metric)
                    save_checkpoint(hparams, algorithm, step, sel_metric= sel_metric)

    # Final evaluation on the test set with the best model
    evaluate(args)

def evaluate(args):
    "Evaluate the model on the test set with the best model."
    set_all_seeds(args['seed'])
    hparams = get_hparams(args)
    loaders = get_loaders(hparams)
    #Sanity Check
    for g_idx in loaders['discarded_groups']:
        if loaders['group_stats_tr'][g_idx] != 0:
            raise AssertionError("One of the dropped groups does not have zero count in training data")
        
    net = get_network(hparams)
    optim = get_optim(hparams, net)
    algorithm = get_algorithm(hparams, net, optim)
    
    if hparams['algorithm_name'] in ['CRM', 'CRM_Multi_Attr']:
        prior_prob= torch.Tensor( loaders['group_stats_tr'] ).to(hparams['device'])
        algorithm.train_log_prior_prob = get_log_prior_prob(prior_prob)

    #If doing hparam selection, then look at multiple selection metrics. Else only corresponding to the input selection metric.
    if args['best_hparams_comb_selection_metric']:
        SEL_METRICS_INFERENCE= [args['best_hparams_comb_selection_metric'].replace('va_','')]
    else:
        SEL_METRICS_INFERENCE= SEL_METRIC
        
    for sel_metric in SEL_METRICS_INFERENCE:
        step_st = load_checkpoint(hparams, algorithm, sel_metric= sel_metric)
        final_stats=[]
        SPLITS= ['te']
        if hparams['algorithm_name'] in ['CRM', 'CRM_Multi_Attr']:
            #Adaptation to compositional shift via extrapolated biases
            algorithm.get_crm_extrapolate_bias(loaders['tr'], discarded_groups= loaders['discarded_groups'])

            #Naive Estimator & Empirical Estimate q(z)
            stats={'method': 'CRM_Naive_Empirical_Prior'}
            for split in SPLITS:
                log_prior_prob= get_log_prior_prob( torch.Tensor( loaders['group_stats_' + split] ).to(hparams['device']) )
                ys, y_hats, ms = algorithm.evaluate(loaders[split], log_prior_prob= log_prior_prob, extrapolate= False)
                metrics = get_metrics(hparams, ys, y_hats, ms, num_y= hparams['num_y'])        
                for metric, value in metrics.items():
                    stats[f'{split}_{metric}'] = value
            final_stats.append(stats)

            #Naive Estimator & Uniform q(z)
            stats={'method': 'CRM_Naive_Uniform_Prior'}
            for split in SPLITS:
                log_prior_prob= torch.zeros_like( algorithm.train_log_prior_prob )
                ys, y_hats, ms = algorithm.evaluate(loaders[split], log_prior_prob= log_prior_prob, extrapolate= False)                
                metrics = get_metrics(hparams, ys, y_hats, ms, num_y= hparams['num_y'])                
                for metric, value in metrics.items():
                    stats[f'{split}_{metric}'] = value
            final_stats.append(stats)

            #CRM Estimator & Empirical Estimate q(z)
            stats={'method': 'CRM_Empirical_Prior'}
            for split in SPLITS:
                log_prior_prob= get_log_prior_prob( torch.Tensor( loaders['group_stats_' + split] ).to(hparams['device']) )
                ys, y_hats, ms = algorithm.evaluate(loaders[split], log_prior_prob= log_prior_prob, extrapolate= True)
                metrics = get_metrics(hparams, ys, y_hats, ms, num_y= hparams['num_y'])        
                for metric, value in metrics.items():
                    stats[f'{split}_{metric}'] = value
            final_stats.append(stats)

            #CRM Estimator & Uniform q(z)
            stats={'method': 'CRM'}
            for split in SPLITS:
                log_prior_prob= torch.zeros_like( algorithm.train_log_prior_prob )
                ys, y_hats, ms = algorithm.evaluate(loaders[split], log_prior_prob= log_prior_prob, extrapolate= True)                
                metrics = get_metrics(hparams, ys, y_hats, ms, num_y= hparams['num_y'])                
                for metric, value in metrics.items():
                    stats[f'{split}_{metric}'] = value
            final_stats.append(stats)
            
        else:
            stats={'method': hparams['algorithm_name']}
            for split in SPLITS:
                ys, y_hats, ms = algorithm.evaluate(loaders[split])
                metrics = get_metrics(hparams, ys, y_hats, ms, num_y= hparams['num_y'])
                for metric, value in metrics.items():
                    stats[f'{split}_{metric}'] = value
            final_stats.append(stats)

        final_stats= pd.DataFrame(final_stats)
        print('Final Results with Model Selection w.r.t ', sel_metric)
        print(final_stats)
        final_stats.to_csv(hparams['results_file'][:-5] + '_sel_metric_' + str(sel_metric) +  '.csv', float_format='%.2f', index=False)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--slurm_partition', type=str, default=None,
                        help='Partition to run the jobs on. If None, runs locally.')
    parser.add_argument('--slurm_dir', type=str, default="",
                        help='Directory to save the slurm logs.')
    parser.add_argument('--out_dir', default='./out',
                        help='Directory to save the results')
    parser.add_argument('--case', type=str, choices=['train', 'eval'], required=True, 
                        help='Choose from train/eval. If the model has already been trained, then choose the eval flag for computing evals on the test dataset with the best model.' +
                             'Note that even the train option would compute the evals on test dataset with the best model at the end of training loop.')
    parser.add_argument('--data_path', default='datasets/',
                        help='Path to the datasets directory. The datasets should be downloaded and preprocessed before running this script.')
    parser.add_argument('--datasets', nargs='+', default=['Waterbirds'], 
                        help='List of datasets to run the experiments on. Choose from: ' +
                             'Waterbirds, CelebA, CelebAMultiAttr, MultiNLI, CivilComments, MetaShift, CompositionalNICOpp, NICOpp, CelebAMultiAttr')
    parser.add_argument('--group_labels', default='yes',
                        help='Choose "yes" for ground-truth'
                             'Choose "no" if the user has inferreed group labels via some oher method and input a .pt file')
    parser.add_argument('--algorithms', nargs='+', default=['CRM'],
                        help='List of algorithms to run the experiments on. Choose from: ' +
                             'ERM, GroupDRO, LA_Group, LA_Cond, IRM, VREx, Mixup, CRM, CRM_Multi_Attr')
    parser.add_argument('--quick_run', action='store_true',
                        help='Run with a single hparam combination to debug')
    parser.add_argument('--precompute_features', action="store_true", 
                        help='Linear probing case, the model uses the precompute features from the backbone network.')
    parser.add_argument('--add_discarded_groups', nargs='*', default=[], 
                        help='List of additional groups to be discarded from the training set' +
                             'The groups should be specified as integers starting from 0' +
                             'For example, if the dataset has 4 groups and you want to discard group 2, then specify --add_discarded_groups 2')
    parser.add_argument('--resume', action='store_true', 
                        help='Resume training from the last checkpoint.')
    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument('--hparams_comb', type=int, 
                        help='Specify a single hparams_comb to run.')
    group1.add_argument('--num_hparams_combs', type=int, 
                        help='Specify the number of hparams_combs to run for hparam turning.')
    group1.add_argument('--best_hparams_comb_selection_metric', type=str,
                        choices=['va_wga', 'va_unf_acc', 'va_avg_acc', 'flip_rate'], 
                        help='Specify metric for determining best hp_comb.',)
    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument('--seed', type=int, help='Specify a single seed to run.')
    group2.add_argument('--num_seeds', type=int, help='Specify the number of seeds to run for each hparams_comb.')
    args = parser.parse_args()

    def get_hparams_combs(args, out_dir, dataset=''):
        # Determine which hparams_combs to run:
        # the best one wrt a metric, a range, or a specified one.
        if args.best_hparams_comb_selection_metric is not None:
            # HP selection is done at seed=0.
            # the best HP is then ran for multiple seeds
            print(out_dir)
            best_hp = find_best_hparams_comb(out_dir, args.best_hparams_comb_selection_metric,seed=0)
            print(f'best_hp for {out_dir} is {best_hp}.')
            return [best_hp]
        elif args.num_hparams_combs is not None:
            return range(args.num_hparams_combs)
        else:
            return [args.hparams_comb]

    def get_seeds(args):
        # Determine which seeds to run:
        # a range or a specified one
        if args.num_seeds is not None:
            return range(args.num_seeds)
        else:
            return [args.seed]

    def get_full_out_dir(args, algorithm, dataset, seed):
        if args.group_labels in ['yes', 'no']:
            folder_name = 'group_labels_' + args.group_labels
            gl = args.group_labels
        elif '.pt' in args.group_labels:
            folder_name = 'group_labels_inferred'
            gl = args.group_labels    
        add_disc_gr_save_path=''
        for group in args.add_discarded_groups:
            add_disc_gr_save_path+= '_' + str(group)
        out_dir = os.path.join(args.out_dir, algorithm, dataset, folder_name, 'add_discarded_groups' + add_disc_gr_save_path)
        return out_dir, gl

    # creating a list of experiments
    args_list = []
    for dataset in args.datasets:
        for algorithm in args.algorithms:
            for seed in get_seeds(args):
                out_dir, gl = get_full_out_dir(args, algorithm, dataset, seed)
                for hparams_comb in get_hparams_combs(args, out_dir, dataset=dataset):
                    if args.best_hparams_comb_selection_metric:           
                        out_dir = os.path.join(out_dir, args.best_hparams_comb_selection_metric)
                        
                    args_list += [{'data_path': args.data_path,
                                   'dataset': dataset,
                                   'group_labels': gl,
                                   'algorithm': algorithm,
                                   'hparams_comb': hparams_comb,
                                   'seed': seed,
                                   'out_dir': out_dir,
                                   'quick_run': args.quick_run,
                                   'resume': args.resume,
                                   'add_discarded_groups': args.add_discarded_groups,
                                   'precompute_features': args.precompute_features,
                                   'best_hparams_comb_selection_metric': args.best_hparams_comb_selection_metric
                                   }]

    run_case = {'train': train, 'eval': evaluate}[args.case]
    print(f'About to launch {len(args_list)} jobs..')
    # running locally
    if args.slurm_partition is None:
        for args_ in args_list:
            run_case(args_)

    # running on cluster
    else:
        import submitit
        executor = submitit.SlurmExecutor(folder=args.slurm_dir)
        executor.update_parameters(
            time= 20 * 60,
            gpus_per_node=1,
            array_parallelism=512,
            cpus_per_task=4,
            constraint='volta32gb',
            partition=args.slurm_partition)
        executor.map_array(run_case, args_list)
