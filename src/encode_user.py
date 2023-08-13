from infrastructure.SQL import MySQL, sql_config
from infrastructure.hbase import HbaseUserInteract, user_interact_hbase_config, HbaseNewsEncode, encode_hbase_config, HbaseUsersEncode, user_encode_hbase_config
from infrastructure.Aerospike import Aerospike
import logging
import time
from datetime import datetime
from typing import List, Dict
from utils.utils import segment_text, preprocess_text, latest_checkpoint
from utils.customer_logging import KafkaLogging
from evaluate import UserDataset, DataLoader
import traceback
import torch
import importlib
from config import model_name, InfraConfig
import numpy as np
import threading
from encode_news import NewsEncoder
import pandas as pd
from tqdm import tqdm
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(module)s: %(message)s')
logger = logging.getLogger(__name__)

global run_periodical
run_periodical = True

try:
    Model = getattr(importlib.import_module(f"model.{model_name}"), model_name)
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
    print(config)
except AttributeError:
    print(f"{model_name} not included!")
    exit()

infra_config = InfraConfig()
sql_connect = MySQL(sql_config)
device = torch.device(config.device if torch.cuda.is_available() else "cpu")
  
class UsersEncoder:
    def __init__(self, sql_connect, model, infra_config):
        # The code `self.sql_connect = sql_connect` assigns the SQL connection object to the
        # `sql_connect` attribute of the `UsersEncoder` class. This allows the `UsersEncoder` class to
        # access the SQL connection and perform SQL operations.
        self.infra_config = infra_config
        self.sql_connect = sql_connect
        self.model = model
        self.news_encoder = NewsEncoder(self.sql_connect, self.model)
        cluster = [("172.26.49.69", 3500), ("172.26.49.70", 3500), ("172.26.49.71", 3500), ("172.26.49.72", 3500)]
        self.aero_client = Aerospike(cluster, debug=False)
    
    def encode_by_user_ids(self, 
                           user_ids: List[int]) -> Dict[int, dict]: 
        """ Encode user by user_ids

        Args:
            user_ids (List[int]): _description_
            
        Returns:
            dict: user_vector, user_infor: clicked news_ids
        """
        news_encode_hbase = HbaseNewsEncode(encode_hbase_config)
        hbase_client = HbaseUserInteract(user_interact_hbase_config)
#         user_ids = user_ids[:10]
        # get history
        user_history, all_history_news = hbase_client.get_history_user(user_interact_hbase_config['table_name'], user_ids, config.num_clicked_news_a_user) # dict[int, list], list[int]
        
        
        hbase_client.close()
        if len(user_history) == 0:
            time.sleep(5)
            logger.info('no user history queried from log hbase')
        else:
            logger.info(f'extracted user history from log hbase')
            logger.info(f'{len(all_history_news)} extracted all history id of users need encoded')
            logger.info(f'some history id will be encoded {all_history_news[:10]}')
            self.news_encoder.push_encode_items(all_history_news, check_exist_keys=True)
        
        user_history = pd.DataFrame(list(user_history.items()), columns=['user', 'clicked_news'])
        user_history['clicked_news'] = user_history['clicked_news'].apply(lambda x: news_encode_hbase.get_exist_keys(encode_hbase_config['table_name'], x))
        
        logger.info(f'user_history_df \n: {user_history.head()}')
        user_dataset = UserDataset(
            user_df=user_history,
        )
        
        user_dataloader = DataLoader(
                    dataset=user_dataset,
                    batch_size=32,
                    shuffle=False,
                    num_workers=0
                )
        
        user2vector = {}
        for minibatch in tqdm(user_dataloader,
                            desc="Calculating vectors for users"):
            total_batch = []
            user_strings = minibatch[0].tolist()
            for news_list in minibatch[1]:
                news_list = news_list.tolist()
                news2vector = news_encode_hbase.scan_by_multi_key_encode_news(encode_hbase_config['table_name'], news_list)
                all_f = []
                for x in news_list:
                    try:
                        all_f.append(torch.tensor(news2vector[x]['encode_vec']).to(device))
                    except:
                        print(f'{x} is in news2vector', list(news2vector.keys()).get(x))
                        
                new_list_fea = torch.stack(all_f,
                            dim=0)
                total_batch.append(new_list_fea)
            clicked_news_vector = torch.stack(total_batch,
                                            dim=0)
            clicked_news_vector = clicked_news_vector.squeeze(2)
            
            user_vector = self.model.get_predicted_nextarticle_vector(clicked_news_vector)
            for user, vector, clicked_news in zip(user_strings, user_vector, minibatch[1]):
                if user not in user2vector:
                    user2vector[user] = {
                        "encode_vec": vector.cpu().detach().numpy(),
                        "clicked_news": clicked_news.tolist()
                    }

        news_encode_hbase.close()
        
        return user2vector
        
    def push_bz_encoded_users(self, user_ids: List[int]) -> None:
        """ Push bz encoded users to Hbase

        Args:
            user_ids (List[int])
        """
        # batch size encode
        batch_size: int = 1000
        for start_index in tqdm(range(0, len(user_ids), batch_size), desc="Encode batch size users - 1k: "):            
            try:
                # encode and push to db
                rsl = self.encode_by_user_ids(user_ids[start_index:start_index+batch_size])
                if rsl: # rsl != None
                    logger.info('User encoded complete ... ')
                    self.get_score_recommend(user_feature=rsl)
#                     logger.info('Saving to user feature hbase')
#                     hbase_user_client = HbaseUsersEncode(user_encode_hbase_config)
#                     hbase_user_client.put_encode_users(table_name=user_encode_hbase_config['table_name'],
#                                                        data=rsl)
#                     hbase_user_client.close()
                    logger.info('Done calcul recommend list for users')
                    time.sleep(5)
                
            except: 
                error= str(traceback.format_exc())
                logger.info("ERRORTRACKING PUSH ENCODED Users: "+error)
            
    def encode_frequent_active_users(self, num_days: int = 3) -> None:
        """ Encode users who have active since 3 days ago 

        Args:
            num_days (int): _description_
        """    
        ## encode history of users which actived in 3 days ago
        user_ids = self.aero_client.get_active_user_time(num_days)
        user_ids = [ x for x in user_ids if x.isdigit() ] # clean user
        user_ids = list(map(int, user_ids)) # list int
        logger.info(f"encode frequent active user: {user_ids[:10]}")
        logger.info(f"Total user ids active need to be encoded: {len(user_ids)}")
        time.sleep(30)
        try:
            # encode and push to db
            self.push_bz_encoded_users(list(set(user_ids[:100])))
#             self.get_score_recommend(list(set(user_ids)))
        except: 
            error= str(traceback.format_exc())
            logger.info("ERRORTRACKING ENCODE FAUsers: "+error)
                
    def encode_interact_users_kafka(self):
        """Encode users who have interacted with news (captured by Kafka)
        """
        customer_log = KafkaLogging(self.infra_config)
        while run_periodical:
            interact_user_ids = []
            while True:
                logs = customer_log.next_batch_log(10, 3)
                for log in logs:
                    user_id = log['user_id']
                    interact_user_ids.append(int(user_id))

                if len(set(interact_user_ids)) >  10: 
                    break
            try:
                # update user encode 
                time_now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                logger.info(f"ENCODE INTERACT USERS (KAFKA) - NUM USERS: {len(set(interact_user_ids))} - AT: {time_now}")
                self.push_bz_encoded_users(user_ids=list(set(interact_user_ids)))

                # update recommend list
#                 self.get_score_recommend(list(set(interact_user_ids)))
            except: 
                error= str(traceback.format_exc())
                logger.info("ERRORTRACKING ENCODE INTERACT users: "+error)
                
    def get_score_recommend(self, user_ids=None, user_feature=None, return_rsl = False):
        ## news feature
        logger.info('Start calculate recommend list !!!!')
        result_sql = self.sql_connect.get_news_by_times('3 DAY')
        news_ids = [int(row_token[0]) for row_token in result_sql]
#         print(f'Total {len(news_ids)} news ids from 3 days ago up to now')
#         self.news_encoder.push_encode_items(news_ids, check_exist_keys=True)
        news_encode_hbase = HbaseNewsEncode(encode_hbase_config)
        news2vector = news_encode_hbase.scan_by_multi_key_encode_news(encode_hbase_config['table_name'], news_ids)
        news_list_fea = torch.stack([torch.tensor(news2vector[x]['encode_vec']).to(device) for x in news2vector.keys()],
                        dim=0)
        news_encode_hbase.close()
        
        ## user feature
        hbase_user_client = HbaseUsersEncode(user_encode_hbase_config)
        if user_feature is None:
            user_feature = hbase_user_client.scan_by_multi_key_encode_users(table_name=user_encode_hbase_config['table_name'],
                                           list_user=user_ids)
        
        print('Calculate recommend list for users: ', list(user_feature.keys())[:10])
        results = []
        ## rec
        for u_id in user_feature.keys():
            score = self.model.get_prediction(news_list_fea.squeeze(1), torch.tensor(user_feature[u_id]['encode_vec']).to(device)).detach().cpu().tolist()
            combined_list = list(zip(news_ids, score))

            # Sort the list of tuples based on the second element (score) in descending order
            sorted_list = sorted(combined_list, key=lambda x: x[1], reverse=True)

            # Print or store the sorted list of tuples
            user_feature[u_id]['recommend_list'] = sorted_list[:1000]
            
            results.append({u_id : sorted_list[:1000]})
        
        print('final save', list(user_feature.keys())[:10])
        for u_id in user_feature.keys():
            hbase_user_client.put_encode_users(table_name=user_encode_hbase_config['table_name'], data={u_id: user_feature[u_id]})
#         print(hbase_user_client.scan_by_multi_key_encode_users(table_name=user_encode_hbase_config['table_name'],
#                                            list_user=[6818491211246072110])[6818491211246072110]['recommend_list'])
        hbase_user_client.close()
        
        if return_rsl:
            return results

def run_encode_user():
    model = Model(config).to(device)
    checkpoint_dir = os.path.join('./checkpoint', model_name)
    checkpoint_path = latest_checkpoint(checkpoint_dir, serving=True)
    if checkpoint_path is not None:
        print(f"Load saved parameters in {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    
    userencoder = UsersEncoder(sql_connect, model, infra_config)
    
#     userencoder.get_score_recommend([6818491211246072110, 1540006981906398468])
    userencoder.encode_frequent_active_users(num_days=3)
    userencoder.encode_interact_users_kafka()
#     userencoder.push_bz_encoded_users(user_ids = [6818491211246072110, 1540006981906398468])

if __name__ == "__main__":
    while True:
        run_periodical = True
        print('-'*100)
        logger.info("Start encode user phase")
        run_encode_user()
        # encode news
        t1 = threading.Thread(target=run_encode_user)
        t1.start()
        print('-'*100)
        print('Round encoded finish, Come to next round !!!')
        time.sleep(300)
        run_periodical = False
        t1.join()


