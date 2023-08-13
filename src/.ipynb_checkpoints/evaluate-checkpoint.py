import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import torch
from config import model_name
from torch.utils.data import Dataset, DataLoader
from os import path
import sys
import pandas as pd
from ast import literal_eval
import importlib
from multiprocessing import Pool
from transformers import AutoModel, AutoTokenizer
from utils.utils import latest_checkpoint
import os
import datetime

try:
    Model = getattr(importlib.import_module(f"model.{model_name}"), model_name)
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
    print(config)
except AttributeError:
    print(f"{model_name} not included!")
    exit()
    
device = torch.device(config.device if torch.cuda.is_available() else "cpu")


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def value2rank(d):
    values = list(d.values())
    ranks = [sorted(values, reverse=True).index(x) for x in values]
    return {k: ranks[i] + 1 for i, k in enumerate(d.keys())}


def calculate_single_user_metric(pair):
    try:
        auc = roc_auc_score(*pair)
        mrr = mrr_score(*pair)
        ndcg5 = ndcg_score(*pair, 5)
        ndcg10 = ndcg_score(*pair, 10)
        return [auc, mrr, ndcg5, ndcg10]
    except ValueError:
        return [np.nan] * 4
    
class NewsDataset(Dataset):
    """
    Load news for evaluation.
    """
    def __init__(self, news_path=None, news_df = None):
        super(NewsDataset, self).__init__()
        if news_path:
            self.news_parsed = pd.read_csv(
                news_path,
                usecols=['newsId'] + config.dataset_attributes['news'],
                converters={
                    attribute: literal_eval
                    for attribute in set(config.dataset_attributes['news']) & set([
                        'catId'
                    ])
                }
                )
        if news_df is not None:
            self.news_parsed = news_df
        
        # print(self.news_parsed.info())
        self.tokenizer = AutoTokenizer.from_pretrained(config.encoder_pretrained_path)
        self.news2dict = self.news_parsed.to_dict('index')
        # print(self.news2dict)
        for key1 in self.news2dict.keys():
            for key2 in self.news2dict[key1].keys():
                # print(key2)
                if key2 == 'title':
                    self.news2dict[key1][key2] = self.tokenizer(
                self.news2dict[key1][key2], padding='max_length', truncation=True, max_length = config.num_words_title, return_tensors="pt")
                    for k in self.news2dict[key1][key2].keys():
                        self.news2dict[key1][key2][k] = self.news2dict[key1][key2][k][0]
                        assert len(self.news2dict[key1][key2][k]) == config.num_words_title
                elif key2 == 'sapo':
                    self.news2dict[key1][key2] = self.tokenizer(
                self.news2dict[key1][key2], padding='max_length', truncation=True, max_length = config.num_words_abstract, return_tensors="pt")
                    for k in self.news2dict[key1][key2].keys():
                        self.news2dict[key1][key2][k] = self.news2dict[key1][key2][k][0]
                        assert len(self.news2dict[key1][key2][k]) == config.num_words_abstract
                elif key2 == 'catId':
                    self.news2dict[key1][key2] = torch.tensor(
                self.news2dict[key1][key2])
                else:
                    # print('oke')
                    self.news2dict[key1][key2] = str(self.news2dict[key1][key2])

    def __len__(self):
        return len(self.news_parsed)

    def __getitem__(self, idx):
        item = self.news2dict[idx]
        return item
    
class UserDataset(Dataset):
    """
    Load users for evaluation, duplicated rows will be dropped
    """
    def __init__(self, behaviors_path = None, user_df = None):
        super(UserDataset, self).__init__()
        if behaviors_path is not None:
            self.behaviors = pd.read_csv(behaviors_path, converters={
                    attribute: literal_eval
                    for attribute in set([
                        'clicked_news', 'click_0_itemIDs'
                    ])
                })
        if user_df is not None: 
            self.behaviors = user_df
        # self.behaviors_parsed.clicked_news.fillna(' ', inplace=True)
        # self.behaviors.drop_duplicates(inplace=True)
        
        

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        try:
            item = {
                "user":
                torch.tensor(int(row.user)),
                "clicked_news_string":
                " ".join([str(i) for i in row.clicked_news]),
                "clicked_news":
                [int(i) for i in row.clicked_news[-config.num_clicked_news_a_user:]]
            }
        except:
            print(row.user)
        
        item['clicked_news_length'] = len(item["clicked_news"])
        repeated_times = config.num_clicked_news_a_user - len(
            item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"] = item["clicked_news"] + [0] * repeated_times
        item["clicked_news"] = torch.tensor(item["clicked_news"])

        return item['user'], item['clicked_news']
    
class BehaviorsDataset(Dataset):
    """
    Load behaviors for evaluation, (user, time) pair as session
    """
    def __init__(self, behaviors_path):
        super(BehaviorsDataset, self).__init__()
        self.behaviors = pd.read_csv(behaviors_path, converters={
                attribute: literal_eval
                for attribute in set([
                    'clicked_news', 'click_0_itemIDs', 'candidate_news'
                ])
            })
        # self.behaviors.clicked_news.fillna(' ', inplace=True)
        # self.behaviors.impressions = self.behaviors.impressions.str.split()

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        candidate_news = list(map(str, row.candidate_news))
        candidate_news = ' '.join(candidate_news)
#         print('can', candidate_news)
        item = {
            # "impression_id": row.impression_id,
            "user": int(row.user),
            "time": str(row['dt']),
            "clicked_news_string": " ".join([str(i) for i in row.clicked_news]),
            "impressions": candidate_news,
            "clicked": row.clicked
        }
        return item

@torch.no_grad()
def evaluate(model, directory, num_workers = None, max_count=sys.maxsize):
    """
    Evaluate model on target directory.
    Args:
        model: model to be evaluated
        directory: the directory that contains two files (behaviors.tsv, news_parsed.tsv)
        num_workers: processes number for calculating metrics
    Returns:
        AUC
        MRR
        nDCG@5
        nDCG@10
    """
    news_dataset = NewsDataset(path.join(directory, 'segment_news_data.csv'))
    news_dataloader = DataLoader(news_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=False)
    
#     batch = next(iter(news_dataloader))
#     print('batch', batch)

    news2vector = {}
    for minibatch in tqdm(news_dataloader,
                          desc="Calculating vectors for news"):
        news_ids = minibatch['newsId']
        # if any(id not in news2vector for id in news_ids):
        news_vector = model.get_personalized_content_vector(minibatch)
        for id, vector in zip(news_ids, news_vector):
            if id not in news2vector:
                news2vector[str(id)] = vector.unsqueeze(0)
    
    padding_all = {
            'catId': torch.tensor([0]),
            'title': {'input_ids': torch.tensor([0] * config.num_words_title).unsqueeze(0), 'token_type_ids': torch.tensor([0] * config.num_words_title).unsqueeze(0), 'attention_mask': torch.tensor([0] * config.num_words_title).unsqueeze(0)},
            'sapo': {'input_ids': torch.tensor([0] * config.num_words_abstract).unsqueeze(0), 'token_type_ids': torch.tensor([0] * config.num_words_abstract).unsqueeze(0), 'attention_mask': torch.tensor([0] * config.num_words_abstract).unsqueeze(0)}
        }

    padding = {
        k: v
        for k, v in padding_all.items()
        if k in config.dataset_attributes['news']
    }
    news2vector['0'] = model.get_personalized_content_vector(padding)
    print('padded', news2vector['0'].shape)
    

    user_dataset = UserDataset(path.join(directory, 'val/behaviors_data.csv'))
    user_dataloader = DataLoader(user_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=False)
    
    # print(news2vector)
    user2vector = {}
    for minibatch in tqdm(user_dataloader,
                          desc="Calculating vectors for users"):
        user_strings = [str(u_id.item()) for u_id in minibatch[0]]
        if any(user_string not in user2vector for user_string in user_strings):
#         print('mini', minibatch)
        # print(len(minibatch["clicked_news"])) 
            clicked_news_vector = torch.stack([
                torch.stack([news2vector[str(x.item())].to(device) for x in news_list],
                            dim=0) for news_list in minibatch[1]
            ],
                                              dim=0)
            clicked_news_vector = clicked_news_vector.squeeze(2)

            user_vector = model.get_predicted_nextarticle_vector(clicked_news_vector)
            for user, vector in zip(user_strings, user_vector):
                user2vector[user] = vector
    
    behaviors_dataset = BehaviorsDataset(path.join(directory, 'val/behaviors_data.csv'))
    behaviors_dataloader = DataLoader(behaviors_dataset,
                                      batch_size=1,
                                      shuffle=False)

    count = 0

    tasks = []

    for minibatch in tqdm(behaviors_dataloader,
                          desc="Calculating probabilities"):
#         try:
        count += 1
        if count == max_count:
            break
#         print(minibatch['clicked'][0].split())
#         print(minibatch['impressions'][0].split())
        filtered_lists = [(int(c), id_) for c, id_ in zip(minibatch['clicked'][0].split(), minibatch['impressions'][0].split()) if id_ in news2vector]
        y_true, candi_ids = zip(*filtered_lists)
        y_true = list(y_true)
        candi_ids = list(candi_ids)    
        
        candidate_news_vector = torch.stack(
            [news2vector[x] for x in candi_ids], dim=1)

#         print('candidate', candidate_news_vector.shape)
        user_vector = user2vector[str(minibatch['user'][0].item())]
#         print('user', user_vector.shape)
        click_probability = model.get_prediction(candidate_news_vector,
                                                 user_vector)

        y_pred = click_probability.tolist()
        
#         print(y_pred)
#         print(y_true)
        tasks.append((y_true, y_pred))
#         except:
#             print('encode of news id in impression not exist')

    with Pool(processes=num_workers) as pool:
        results = pool.map(calculate_single_user_metric, tasks)

    aucs, mrrs, ndcg5s, ndcg10s = np.array(results).T
    return np.nanmean(aucs), np.nanmean(mrrs), np.nanmean(ndcg5s), np.nanmean(
        ndcg10s)


if __name__ == "__main__":
#     model = Model(config).to(device)
#     model.eval()
    checkpoint_dir = os.path.join(f'./checkpoint/{model_name}/')
    checkpoint_path = latest_checkpoint(checkpoint_dir)
    print(checkpoint_path)
#     if checkpoint_path is not None:
#         print(f"Load saved parameters in {checkpoint_path}")
#         checkpoint = torch.load(checkpoint_path)
#         epoch = checkpoint['epoch']
#         model.load_state_dict(checkpoint['model_state_dict'])
#         model.eval()
#     val_auc, val_mrr, val_ndcg5, val_ndcg10 = evaluate(model, '../data_news/weekly_data/')
#     print(f"validation AUC: {val_auc:.4f}, validation MRR: {val_mrr:.4f}, validation nDCG@5: {val_ndcg5:.4f}, validation nDCG@10: {val_ndcg10:.4f}")