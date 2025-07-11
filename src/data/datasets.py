import os
from typing import List
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, SequentialSampler, Sampler
from torch.utils.data import RandomSampler, WeightedRandomSampler
from torchvision import transforms, datasets
import torchvision.transforms.functional as F
from transformers import BertTokenizer

def sanity_check_loader(loader):
    group_stats={}
    y, m = [], []
    for idx, batch in enumerate(loader):
        _, _, y_, m_ = batch
        y.append(y_)
        m.append(m_)
    
    y = torch.cat(y)
    m = torch.cat(m)
    g= 2*y + m
    for idx in range(g.shape[0]):
        key= g[idx].item()
        if key in group_stats.keys():
            group_stats[key]+=1
        else:
            group_stats[key]=1

    print(group_stats)
    return

def check_discarded_groups(group_counts, add_discarded_groups):
    for g_idx in add_discarded_groups:
        if group_counts[int(g_idx)] != 0:
            raise AssertionError("Some of the discared groups still have non-zero counts in the dataset")

def get_loaders(hparams):

    Dset = {'Waterbirds': Waterbirds if not hparams['precompute_features'] else FeatWaterbirds,
            'CelebA': CelebA if not hparams['precompute_features'] else FeatCelebA,
            'CelebAMultiAttr': CelebAMultiAttr if not hparams['precompute_features'] else FeatCelebAMultiAttr,
            'MultiNLI': MultiNLI if not hparams['precompute_features'] else FeatMultiNLI,
            'CivilComments': CivilComments if not hparams['precompute_features'] else FeatCivilComments,
            'Simple_2d': FeatSimple2d,
            'SynAED_2': FeatSynAED,
            'SynAED_10': FeatSynAED,
            'SynAED_20': FeatSynAED,
            'SynAED_30': FeatSynAED,
            'SynAED_40': FeatSynAED,
            'SynAED_50': FeatSynAED,
            'SynGenAED_2': FeatSynAED,
            'SynGenAED_7': FeatSynAED,
            'SynGenAED_8': FeatSynAED,
            'SynGenAED_9': FeatSynAED,
            'SynGenAED_10': FeatSynAED,
            'SynGenAED_11': FeatSynAED,
            'MetaShift': MetaShift if not hparams['precompute_features'] else FeatMetaShift,
            'NICOpp': NICOpp if not hparams['precompute_features'] else FeatNICOpp,
            'CompositionalNICOpp': NICOpp if not hparams['precompute_features'] else FeatNICOpp}[hparams['dataset_name']]
    
    data_path = hparams['data_path']
    # subg= 1
    subg = hparams['algorithm_name'] == 'SUBG'
    gl = hparams['group_labels']
    add_discarded_groups= hparams['add_discarded_groups']
    bs = hparams['batch_size']

    #To generate a large set of randomly discarded groups for CompositionalNICOpp
    if hparams['dataset_name'] == 'CompositionalNICOpp' and len(add_discarded_groups):
        #Special structure where the argument add_discarded_group does not directly denote the groups to be discarded, rather it denotes which scenario of dropped groups to load based on pre-processing for NICOpp.
        assert len(add_discarded_groups) == 1
        add_discarded_groups= np.load('data/nicopp_dropped_group_exp/discarded_group_scenario_' + add_discarded_groups[0] + '.npy').tolist()
        
    if ( 'SynAED' in hparams['dataset_name'] or 'SynGenAED' in hparams['dataset_name'] )  and len(add_discarded_groups):
        #Special structure where the argument add_discarded_group does not directly denote the groups to be discarded, rather it denotes which scenario of dropped groups to load based on pre-processing for NICOpp.
        assert len(add_discarded_groups) == 1
        add_discarded_groups= np.load( data_path + '/syn_aed_dep/seed_' + str(hparams['seed']) + '_discarded_group_scenario_' + add_discarded_groups[0] + '.npy').tolist()        

    tr = Dset(data_path, split='tr', num_attr= hparams['num_m'], num_labels= hparams['num_y'], group_labels=gl, subg=subg, add_discarded_groups= add_discarded_groups)
    #For the case of compositional NICOpp, we have to remove group from validation that were absent in training
    if hparams['dataset_name'] == 'CompositionalNICOpp':
        va = Dset(data_path, split='va', num_attr= hparams['num_m'], num_labels= hparams['num_y'], group_labels=gl, subg=False, add_discarded_groups= tr.discarded_groups)        
    else:
        va = Dset(data_path, split='va', num_attr= hparams['num_m'], num_labels= hparams['num_y'], group_labels=gl, subg=False, add_discarded_groups= add_discarded_groups)
    te = Dset(data_path, split='te', num_attr= hparams['num_m'], num_labels= hparams['num_y'], group_labels='yes', subg=False, add_discarded_groups= [])

    if hparams['algorithm_name'] == 'RWG':
        tr_w = tr.weights_g
    elif hparams['balanced_batch']:
        tr_w = tr.weights_y
    else:
        tr_w = None

    #Sanity check to see if the discarded groups are dropped from the training and validation datasets
    check_discarded_groups(tr.group_sizes, add_discarded_groups)
    check_discarded_groups(va.group_sizes, add_discarded_groups)

    return {'tr': MyDataLoader(hparams, tr, bs, tr_w, True),
            'va': MyDataLoader(hparams, va, 16*bs, None, False),
            'te': MyDataLoader(hparams, te, 16*bs, None, False),
            # 'tr_': MyDataLoader(hparams, tr_, bs, None, False),
            'group_stats_tr': tr.group_sizes,
            'group_stats_va': va.group_sizes,
            'group_stats_te': te.group_sizes,
            'discarded_groups': tr.discarded_groups
            }


# ############################################################################
# ############################### Data Loader ################################
# ############################################################################


class MyDataLoader:
    def __init__(self, hparams, dataset, batch_size, weights, shuffle):

        if weights is not None:
            sampler = WeightedRandomSampler(
                weights, num_samples=len(dataset), replacement=True)
        elif shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        batch_size if batch_size != -1 else len(dataset)
        if hparams['precompute_features'] or 'Color' in hparams['dataset_name']:
            sampler = FastBatchSampler(sampler, batch_size)
            batch_size = None

        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=hparams['num_workers'])

        self.n_examples = len(dataset.y)
        if isinstance(dataset.y, list) or isinstance(dataset.y, np.ndarray):
            self.y = torch.LongTensor(dataset.y)
        else:
            self.y = dataset.y.long()

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


class FastBatchSampler(Sampler):
    def __init__(self, sampler, batch_size):
        self.sampler = sampler
        self.batch_size = batch_size
    
    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch 
    
    def __len__(self):
        return (len(self.sampler) // self.batch_size +
                (len(self.sampler) % self.batch_size > 0))

# ############################################################################
# ################################ Datasets ##################################
# ############################################################################


class BaseGroupDataset:
    def __init__(
                self, 
                root: str,
                split: str,
                metadata, 
                transform,
                num_attr: int,
                num_labels: int,                
                group_labels: str, 
                subg: int, 
                add_discarded_groups: List[str]
            ):
        """
        Args:
            root: Directory where the data is stored
            split: Train/Val/Test split of the dataset
            metadata: CSV file containing the dataset
            transform: Transformation to be applied to the input image/text data
            num_attr: Total number of spurious attributes in the dataset
            num_labels: Total number of class labels in the dataset
            group_labels: Either use inferred group labels or the true group labels
            subg: Either perform subgroup balancing or don't perform subgroup balancing
            add_discarded_groups: A list containing group indices to be dropped out during training & validation
        """

        self.num_attr= num_attr
        self.num_labels= num_labels

        df = pd.read_csv(metadata)
        # dataset is ImagenetBG
        if 'backgrounds_challenge' in metadata:
            df['split'] = df['split'].replace(
                {'train': 0, 'val': 1, 'mixed_rand': 2})
        df = df[df["split"] == ({"tr": 0, "va": 1, "te": 2}[split])]
        self.transform_ = transform
        self.idx = list(range(len(df)))
        self.x = df["filename"].astype(str).map(
            lambda x: os.path.join(root, x)).tolist()
        self.y = df["y"].tolist()

        if group_labels == 'yes':
            self.a = df["a"].tolist()
        elif group_labels == 'no':
            self.a = [0] * len(df["a"].tolist()) 
        else:
            assert split in ['tr', 'va']
            self.a = torch.load(group_labels)[split].cpu()

        #Convert self.x, self.y and self.a to be np.arrays to make operations like count groups and drop groups simpler
        self.x= np.array(self.x)
        self.y= np.array(self.y)
        self.a= np.array(self.a)

        self.discarded_groups= []
        self._count_groups()

        if subg:
            self.subg()

        if len(add_discarded_groups):
            self.drop_group(add_discarded_groups)

        print('Statistics: ', len(self.x), self.num_labels, self.num_attr,  np.sum( np.array(self.group_sizes) !=0)  ) 
        print('Group Stats: ', self.group_sizes)


    def _count_groups(self):
        self.weights_g, self.weights_y = [], []
        if isinstance(self.y, list) and isinstance(self.a, list):
            raise AssertionError("Expected self.y and self.a to be of the type numpy array")
        self.g= self.num_attr * self.y + self.a

        self.group_sizes = [0] * self.num_attr * self.num_labels
        self.class_sizes = [0] * self.num_labels

        for i in self.idx:
            self.group_sizes[int(self.num_attr * self.y[i] + self.a[i])] += 1
            self.class_sizes[int(self.y[i])] += 1

        for i in self.idx:
            self.weights_g.append(len(self) / self.group_sizes[
                int(self.num_attr * self.y[i] + self.a[i])])
            self.weights_y.append(len(self) / self.class_sizes[int(self.y[i])])

        #Maintain a list of groups that are not present
        self.discarded_groups= []
        for curr_y in range(self.num_labels):
            for curr_a in range(self.num_attr):
                curr_g= int(self.num_attr * curr_y + curr_a )
                if self.group_sizes[curr_g] == 0:
                    self.discarded_groups.append(curr_g)                    

        #Sanity Check
        group_counts_curr= np.array(self.group_sizes)
        group_counts_curr= np.delete(group_counts_curr, group_counts_curr==0)
        _, group_counts_alt= np.unique(self.g, return_counts=True)
        if not np.array_equal( group_counts_curr, group_counts_alt):
            raise AssertionError("Error in the count groups function")


    def subg(self):
        perm = torch.randperm(len(self)).tolist()
        min_size = min(list(self.group_sizes))

        counts_g = [0] * self.num_attr * self.num_labels
        new_idx = []
        for p in perm:
            y, a = self.y[self.idx[p]], self.a[self.idx[p]]
            if counts_g[int(self.num_attr * y + a)] < min_size:
                counts_g[int(self.num_attr * y + a)] += 1
                new_idx.append(self.idx[p])
        self.idx = new_idx
        self._count_groups()

    def drop_group(self, add_discarded_groups):
        """Drop groups during training to create compositional distribtion shift."""
        
        for g_idx in add_discarded_groups:
            if isinstance(self.g, list):
                raise AssertionError("Expected self.g to be of type tensor array")
            indices= self.g != int(g_idx)
            self.x = self.x[indices]
            self.y= self.y[indices]
            self.a= self.a[indices]
            self.g = self.g[indices]

        self.idx= list(range(len(self.y)))
        self._count_groups()
        
        return
    
    def __getitem__(self, index):
        if isinstance(index, list):  # Feat data
            i = torch.LongTensor(self.idx)[index]
            x = self.x[i]
            y = self.y[i]
            a = self.a[i]
        else:  # non-Feat data
            i = self.idx[index]
            x = self.transform(self.x[i])
            y = torch.tensor(self.y[i], dtype=torch.long)
            a = torch.tensor(self.a[i], dtype=torch.long)

        return i, x, y, a

    def __len__(self):
        return len(self.idx)


# ##############################################################################
# ########################### Featurized Versions ##############################
# ##############################################################################


class BaseFeatDataset(BaseGroupDataset):
    def __init__(self, 
                 data_path: str, 
                 split: str, 
                 num_attr: int,
                 num_labels: int,
                 group_labels: int, 
                 subg: int, 
                 dataset_name: str, 
                 add_discarded_groups: List[str]
                ):
        """
        Args:
            data_path: Directory where the data is stored
            split: Train/Val/Test split of the dataset
            num_attr: Total number of spurious attributes in the dataset
            num_labels: Total number of class labels in the dataset
            group_labels: Either use inferred group labels or the true group labels
            subg: Either perform subgroup balancing or don't perform subgroup balancing
            dataset_name: Name of the dataset
            add_discarded_groups: A list containing group indices to be dropped out during training & validation       
        """
        self.num_attr= num_attr
        self.num_labels= num_labels

        pt = torch.load(os.path.join(
            data_path, dataset_name, "features.pt"))

        self.x = pt[split]["x"].float()
        self.y = pt[split]["y"].squeeze().long()
        self.idx = list(range(len(self.x)))

        if group_labels == 'yes':
            self.a = pt[split]["m"].squeeze().long()
        elif group_labels == 'no':
            self.a = 0 * self.y
        else:
            self.a = torch.load(group_labels)[split].cpu()
        self._count_groups()

        if subg:
            self.subg()

        if len(add_discarded_groups):
            self.drop_group(add_discarded_groups)
        
        print('Statistics: ', self.x[self.idx].shape, self.num_labels, self.num_attr)
        print('Group Stats: ',self.group_sizes)

    def transform(self, x):
        # no transform cause x is already in feature sapce
        return x

class FeatWaterbirds(BaseFeatDataset):
    def __init__(self, data_path, split, num_attr, num_labels, group_labels, subg, add_discarded_groups):
        super().__init__(
            data_path, split, num_attr, num_labels, group_labels, subg, 'waterbirds', add_discarded_groups= add_discarded_groups)


class FeatCelebA(BaseFeatDataset):
    def __init__(self, data_path, split, num_attr, num_labels, group_labels, subg, add_discarded_groups):
        super().__init__(
            data_path, split, num_attr, num_labels, group_labels, subg, 'celeba', add_discarded_groups= add_discarded_groups)

class FeatCelebAMultiAttr(BaseFeatDataset):
    def __init__(self, data_path, split, num_attr, num_labels, group_labels, subg, add_discarded_groups):
        super().__init__(
            data_path, split, num_attr, num_labels, group_labels, subg, 'celeba_multiattr', add_discarded_groups= add_discarded_groups)

class FeatMultiNLI(BaseFeatDataset):
    def __init__(self, data_path, split, num_attr, num_labels, group_labels, subg, add_discarded_groups):
        super().__init__(
            data_path, split, num_attr, num_labels, group_labels, subg, 'multinli', add_discarded_groups= add_discarded_groups)


class FeatCivilComments(BaseFeatDataset):
    def __init__(self, data_path, split, num_attr, num_labels, group_labels, subg, add_discarded_groups):
        super().__init__(
            data_path, split, num_attr, num_labels, group_labels, subg, 'civilcomments', add_discarded_groups= add_discarded_groups)


class FeatMetaShift(BaseFeatDataset):
    def __init__(self, data_path, split, num_attr, num_labels, group_labels, subg, add_discarded_groups):
        super().__init__(
            data_path, split, num_attr, num_labels, group_labels, subg, 'metashift', add_discarded_groups= add_discarded_groups)

class FeatNICOpp(BaseFeatDataset):
    def __init__(self, data_path, split, num_attr, num_labels, group_labels, subg, add_discarded_groups):
        super().__init__(
            data_path, split, num_attr, num_labels, group_labels, subg, 'nicopp', add_discarded_groups= add_discarded_groups)

class FeatSynAED(BaseFeatDataset):
    def __init__(self, data_path, split, num_attr, num_labels, group_labels, subg, add_discarded_groups):
        super().__init__(
            data_path, split, num_attr, num_labels, group_labels, subg, 'syn_aed_dep', add_discarded_groups= add_discarded_groups)

class FeatSimple2d(BaseFeatDataset):
    def __init__(self, data_path, split, num_attr, num_labels, group_labels, subg, add_discarded_groups):
        super().__init__(
            data_path, split, num_attr, num_labels, group_labels, subg, 'simple_2d', add_discarded_groups= add_discarded_groups)


# ############################################################################
# ######################### Non-Featurized Versions ##########################
# ############################################################################


class Waterbirds(BaseGroupDataset):
    def __init__(self, data_path, split, num_attr, num_labels, group_labels, subg, add_discarded_groups):
        root = os.path.join(
            data_path,
            "waterbirds",
            "waterbird_complete95_forest2water2")
        metadata = os.path.join(
            data_path,
            "waterbirds",
            "metadata_waterbirds.csv")
        transform = transforms.Compose([
            transforms.Resize((int(224 * (256 / 224)),
                               int(224 * (256 / 224)),)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        super().__init__(root, split, metadata, transform,
                         num_attr, num_labels,
                         group_labels, subg, add_discarded_groups= add_discarded_groups)

    def transform(self, x):
        return self.transform_(Image.open(x).convert("RGB"))


class CelebA(BaseGroupDataset):
    def __init__(self, data_path, split, num_attr, num_labels, group_labels, subg, add_discarded_groups):
        root = os.path.join(
            data_path,
            "celeba", "img_align_celeba")
        metadata = os.path.join(
            data_path,
            "celeba",
            "metadata_celeba.csv")
        transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        super().__init__(root, split, metadata, transform,
                         num_attr, num_labels,
                         group_labels, subg, add_discarded_groups= add_discarded_groups)

    def transform(self, x):
        return self.transform_(Image.open(x).convert("RGB"))

class CelebAMultiAttr(BaseGroupDataset):
    def __init__(self, data_path, split, num_attr, num_labels, group_labels, subg, add_discarded_groups):
        root = os.path.join(
            data_path,
            "celeba", "img_align_celeba")
        metadata = os.path.join(
            data_path,
            "celeba_multiattr",
            "metadata_celeba.csv")
        transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        super().__init__(root, split, metadata, transform,
                         num_attr, num_labels,
                         group_labels, subg, add_discarded_groups= add_discarded_groups)

    def transform(self, x):
        return self.transform_(Image.open(x).convert("RGB"))


class MultiNLI(BaseGroupDataset):

    def __init__(self, data_path, split, num_attr, num_labels, group_labels, subg, add_discarded_groups):
        root = os.path.join(
            data_path,
            "multinli",
            "glue_data", "MNLI")
        metadata = os.path.join(
            data_path,
            "multinli",
            "metadata_multinli.csv")

        self.features = []
        for feature_file in [
                "cached_train_bert-base-uncased_128_mnli",
                "cached_dev_bert-base-uncased_128_mnli",
                "cached_dev_bert-base-uncased_128_mnli-mm"]:
            features = torch.load(os.path.join(root, feature_file))
            self.features += features

        self.input_ids = torch.tensor(
            [f.input_ids for f in self.features], dtype=torch.long)
        self.input_masks = torch.tensor(
            [f.input_mask for f in self.features], dtype=torch.long)
        self.segment_ids = torch.tensor(
            [f.segment_ids for f in self.features], dtype=torch.long)
        self.label_ids = torch.tensor(
            [f.label_id for f in self.features], dtype=torch.long)
        self.x_array = torch.stack(
            (self.input_ids, self.input_masks, self.segment_ids), dim=2)
        super().__init__("", split, metadata, self.transform,
                         num_attr, num_labels,
                         group_labels, subg, add_discarded_groups= add_discarded_groups)

    def transform(self, i):
        return self.x_array[int(i)]


class CivilComments(BaseGroupDataset):

    def __init__(self, data_path, split, num_attr, num_labels, group_labels, subg, add_discarded_groups, grains="fine"):
        text = pd.read_csv(os.path.join(
            data_path,
            "civilcomments",
            "civilcomments_{}.csv".format(grains)))
        metadata = os.path.join(
            data_path,
            "civilcomments",
            "metadata_civilcomments_{}.csv".format(grains))

        self.text_array = list(text["comment_text"])
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        super().__init__("", split, metadata, self.transform,
                         num_attr, num_labels,
                         group_labels, subg, add_discarded_groups= add_discarded_groups)

    def transform(self, i):
        text = self.text_array[int(i)]
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=220,
            return_tensors="pt",)

        if len(tokens) == 3:
            return torch.squeeze(
                torch.stack((
                    tokens["input_ids"],
                    tokens["attention_mask"],
                    tokens["token_type_ids"]), dim=2), dim=0)
        else:
            return torch.squeeze(
                torch.stack((
                    tokens["input_ids"],
                    tokens["attention_mask"]), dim=2), dim=0)

class NICOpp(BaseGroupDataset):
    def __init__(self, data_path, split, num_attr, num_labels, group_labels, subg, add_discarded_groups):
        metadata = os.path.join(
            data_path,
            "nicopp",
            "metadata.csv")

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        super().__init__("", split, metadata, transform,
                         num_attr, num_labels,
                         group_labels, subg, add_discarded_groups= add_discarded_groups)

    def transform(self, x):
        return self.transform_(Image.open(x).convert("RGB"))

class MetaShift(BaseGroupDataset):
    def __init__(self, data_path, split, num_attr, num_labels, group_labels, subg, add_discarded_groups):
        metadata = os.path.join(
            data_path,
            "metashift",
            "metadata_metashift.csv")

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        super().__init__("", split, metadata, transform,
                         num_attr, num_labels,
                         group_labels, subg, add_discarded_groups= add_discarded_groups)

    def transform(self, x):
        return self.transform_(Image.open(x).convert("RGB"))
