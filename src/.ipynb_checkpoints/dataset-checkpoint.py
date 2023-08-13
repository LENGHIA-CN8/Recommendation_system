from torch.utils.data import Dataset, DataLoader
import pandas as pd
from ast import literal_eval
from os import path
import numpy as np
from config import model_name
import importlib
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
tqdm.pandas()

try:
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
    print(config)
except AttributeError:
    print(f"{model_name} not included!")
    exit()


class BaseDataset(Dataset):
    def __init__(self, behaviors_path, news_path):
        super(BaseDataset, self).__init__()
        assert all(attribute in [
            'catId', 'title', 'sapo'
        ] for attribute in config.dataset_attributes['news'])
        assert all(attribute in ['user', 'clicked_news', 'click_0_itemIDs', 'candidate_news'
        ] for attribute in config.dataset_attributes['record'])
        
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        self.behaviors_parsed = pd.read_csv(behaviors_path, converters={
                attribute: literal_eval
                for attribute in set([
                    'clicked_news', 'click_0_itemIDs', 'candidate_news'
                ])
            })
        
        self.news_parsed = pd.read_csv(
            news_path,
            index_col='newsId',
            usecols=['newsId'] + config.dataset_attributes['news'],
            converters={
                attribute: literal_eval
                for attribute in set(config.dataset_attributes['news']) & set([
                    'catId'
                ])
            }
            )
        self.news_id2int = {x: i for i, x in enumerate(self.news_parsed.index)}
        self.news2dict = self.news_parsed.to_dict('index')
        # print(self.news2dict)
        for key1 in self.news2dict.keys():
            for key2 in self.news2dict[key1].keys():
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
                else:
                    self.news2dict[key1][key2] = torch.tensor(
                self.news2dict[key1][key2])
        
        padding_all = {
            'catId': torch.tensor(0),
            'title': {'input_ids': torch.tensor([0] * config.num_words_title), 'token_type_ids': torch.tensor([0] * config.num_words_title), 'attention_mask': torch.tensor([0] * config.num_words_title)},
            'sapo': {'input_ids': torch.tensor([0] * config.num_words_abstract), 'token_type_ids': torch.tensor([0] * config.num_words_abstract), 'attention_mask': torch.tensor([0] * config.num_words_abstract)}
        }
        # for key in padding_all.keys():
        #     padding_all[key] = torch.tensor(padding_all[key])

        self.padding = {
            k: v
            for k, v in padding_all.items()
            if k in config.dataset_attributes['news']
        }

    def __len__(self):
        return len(self.behaviors_parsed)

    def __getitem__(self, idx):
        item = {}
        row = self.behaviors_parsed.iloc[idx]
        # print(row)
        if 'user' in config.dataset_attributes['record']:
            item['user'] = row.user
        
        item["clicked"] = torch.tensor(list(map(int, row.clicked.split())))
        
#         candidate_news = str(row.candidate_news) + ' ' + ' '.join([str(idnew) for idnew in row.click_0_itemIDs])
        candidate_news = list(map(str, row.candidate_news))
        candidate_news = ' '.join(candidate_news)
        item["candidate_news"] = [
            self.news2dict[int(x)] if self.news2dict.get(int(x)) else self.padding
            for x in candidate_news.split()
        ]
        
        item["clicked_news"] = [
            self.news2dict[int(x)] 
            for x in row.clicked_news[-config.num_clicked_news_a_user:]
        ]
        if 'clicked_news_length' in config.dataset_attributes['record']:
            item['clicked_news_length'] = len(item["clicked_news"])
        repeated_times = config.num_clicked_news_a_user - \
            len(item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"] = item["clicked_news"] + [self.padding
                                ] * repeated_times
        

#         print(len(item["candidate_news"]))
#         print(len(item["clicked_news"]))
#         print(len(item["clicked"]))
        
        return item
    
if __name__ == '__main__':
    train_data = BaseDataset('../data_news/31-7-2023/train/behaviors_data.csv', '../data_news/31-7-2023/segment_news_data.csv')
    dataloader = DataLoader(train_data,
                   batch_size=config.batch_size,
                   shuffle=True,
                   )
    i = 0 
    for s, out_data in tqdm(enumerate(dataloader)):
        i += 1
        print('---------------------------')
#         out_data = next(iter(dataloader)
#         print('candidate_news', out_data['candidate_news'])
        print(out_data.keys())
        print('clicked', out_data['clicked'])
        print(len(out_data['clicked']))
#         print(out_data['clicked_news'][-1]['title']['input_ids'].shape)
#         print(len(out_data['clicked_news']))
        if i == 1:
            break     
