from infrastructure.SQL import MySQL, sql_config
from infrastructure.hbase import HbaseNewsEncode, encode_hbase_config
import logging
import time
from datetime import datetime
from typing import List, Dict
from utils.utils import segment_text, preprocess_text, latest_checkpoint
from evaluate import NewsDataset, DataLoader
import traceback
from pandas import DataFrame
import torch
import importlib
from config import model_name
import numpy as np
import threading
import os
from tqdm import tqdm

# from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(module)s: %(message)s')
logger = logging.getLogger(__name__)

sql_connect = MySQL(sql_config)
print(sql_connect)
        
global run_periodical
run_periodical = True

try:
    Model = getattr(importlib.import_module(f"model.{model_name}"), model_name)
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
    print(config)
except AttributeError:
    print(f"{model_name} not included!")
    exit()
    
device = torch.device(config.device if torch.cuda.is_available() else "cpu")

class NewsEncoder:
    def __init__(self, sql_connect, model):
        self.sql_connect = sql_connect
        self.model = model
        
    def encode_clicked_items(self,
                                news_df: DataFrame, 
                                use_cuda_text: bool = True ) -> Dict[int, dict] :
        """Encode news to vectors

        Args:
            item_ids (list): list id of news

        Returns:
            dict: new_vector, new_infor
        """
        # create data set
        news_dataset = NewsDataset(
            news_df=news_df
        )
        
        # create data loader
        news_dataloader = DataLoader(
            dataset=news_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0
        )
        # if use_cuda_text:
        #     self.model = self.model.to("cuda:0")
            
        # load model => encode news
        news_scoring = []
        popular_news_scores = []
        news2vector = {}
        # Iterate over the data loader
        with torch.no_grad():
            for batch in news_dataloader:
                # Access the processed batch
                # print('batch', batch)
                news_ids = batch['newsId']
                # news score
                new_vector_tensor = self.model.get_personalized_content_vector(batch)
                new_vector_numpy = new_vector_tensor.cpu().detach().numpy()
#                 logger.info(f"New_vector_tensor shape: {new_vector_tensor.shape}")
               
                for id, vector in zip(news_ids, new_vector_numpy):                   
                    news2vector[str(id)] = { 
                        "encode_vec" : np.expand_dims(vector, 0) 
                    }
        
        padding_all = {
            'catId': torch.tensor([0]),
            'title': {'input_ids': torch.tensor([0] * config.num_words_title).unsqueeze(0), 'token_type_ids': torch.tensor([0] * config.num_words_title).unsqueeze(0), 'attention_mask': torch.tensor([0] * config.num_words_title).unsqueeze(0)},
            'sapo': {'input_ids': torch.tensor([0] * config.num_words_abstract).unsqueeze(0), 'token_type_ids': torch.tensor([0] * config.num_words_abstract).unsqueeze(0), 'attention_mask': torch.tensor([0] * config.num_words_abstract).unsqueeze(0)}
        }
        new_vector_tensor = self.model.get_personalized_content_vector(padding_all)
        new_vector_numpy = new_vector_tensor.cpu().detach().numpy()
        news2vector['0'] = {
           "encode_vec" : new_vector_numpy
        }
        # add news infor: title, publicdate, cate, subcate
        # rsl = {}
        # for idx, news_id in enumerate(item_ids):
        #     rsl[news_id] = {
        #         "encode_vec": news_scoring[idx],
        #         "popular_score": popular_news_scores[idx],
        #     }
        return news2vector
        
    def push_encode_items(self, all_history_items_lst: List[int], 
                             check_exist_keys = False, 
                             return_rsl = False) -> None:
        '''
            Push list items to db
            input: list item id list(int)
        '''
        # get non exist keys
        if check_exist_keys:
            hbase_client = HbaseNewsEncode(encode_hbase_config)
            not_exist_keys_list = hbase_client.get_non_exist_keys(encode_hbase_config['table_name'], 
                                                               all_history_items_lst)
            hbase_client.close()
            all_history_items_lst = not_exist_keys_list
        
        results = {}
        
        # encode and push
        batch_size: int = 30
        
        for start_index in tqdm(range(0, len(all_history_items_lst), batch_size)):
            hbase_client = HbaseNewsEncode(encode_hbase_config)
            news_df = self.sql_connect.get_news_by_ids(all_history_items_lst[start_index: start_index+batch_size])
            news_df[['title', 'sapo']] = news_df[['title', 'sapo']].applymap(preprocess_text)
            news_df[['title', 'sapo']] = news_df[['title', 'sapo']].applymap(segment_text)
#             logger.info(f'News_df need to be encoded: \n {news_df.head()}')
            try:
                rsl = self.encode_clicked_items(news_df)
                if rsl: # neu ton tai tsl
#                     logger.info('update newsvector to Hbase feature store')
                    if return_rsl:
                        results.update(rsl)                    
                hbase_client.put_encode_news(encode_hbase_config['table_name'], rsl)
#                 hbase_client.scan_by_multi_key_encode_news()
            except: 
                error= str(traceback.format_exc())
                logger.info("ERRORTRACKING PUSH CANDIDATES: "+error)
            hbase_client.close()

        if return_rsl:
            return results
    
    def encode_news(self):
        """
            periodically 3 minutes to encode new_items
        """
        logger.info(f'run_peri: {run_periodical}')
        while run_periodical:
            st_time = datetime.now()
            logger.info(f'Time to start encoding: {st_time}')
            result_sql = []        

            result_sql.extend(self.sql_connect.get_news_by_times('10 MINUTE'))
            print(result_sql)
            news_ids = [int(row_token[0]) for row_token in result_sql]
            time_now_str = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            logger.info(f"Total {len(news_ids)} New ids of items: {news_ids} at {time_now_str}")

            if len(news_ids) != 0:
                self.push_encode_items(news_ids, check_exist_keys=True)
                
            try:
                time.sleep(2*60 - int(time.time()- time_) )
            except:
                time.sleep(2*60)
    
    def encode_n_days_ago_new_items(self, n_days):
        '''
             encode n days ago news_items
        '''
        start_time = datetime.now()

        logger.info(f"Time start to encode {n_days} days ago news_items: {start_time} ")
        result_sql = []
        
        result_sql.extend(self.sql_connect.get_news_by_times(f'{n_days} DAY'))
        news_ids = [int(row_token[0]) for row_token in result_sql]
        logger.info(f"3 day before News ids: {news_ids}")
        if len(news_ids) != 0:
            self.push_encode_items(news_ids, check_exist_keys=False)

def run_encode_news(): 
    
    model = Model(config).to(device)
    checkpoint_dir = os.path.join('./checkpoint', model_name)
    checkpoint_path = latest_checkpoint(checkpoint_dir, serving=True)
    print(checkpoint_path)
    if checkpoint_path is not None:
        print(f"Load saved parameters in {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    newsencoder =  NewsEncoder(sql_connect, model)
    
    newsencoder.encode_n_days_ago_new_items(3)
    ## encode news_items
    newsencoder.encode_news()

if __name__ == "__main__":       
    while True:
        run_periodical = True
        logger.info("Start encode news phase")
#         run_encode_news()
        # encode news
        t1 = threading.Thread(target=run_encode_news)
        t1.start()
        time.sleep(300)
        run_periodical = False
        # torch.cuda.empty_cache()
        # continue
        t1.join()


