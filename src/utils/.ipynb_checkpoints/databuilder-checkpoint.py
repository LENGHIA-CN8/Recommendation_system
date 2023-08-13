import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)
sys.path.append('./')
print(sys.path)

import importlib
from data.preprocess_log import get_spark_df
from config import model_name
import logging
import pandas as pd
from .utils import preprocess_text, convert_string_to_list, segment_text, take_k_clicked_0_news, filter_not_exist, shuffle_list
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, DataFrame
from infrastructure.SQL import MySQL, sql_config
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import datetime

tqdm.pandas()

logger = logging.getLogger(__name__)

try:
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
    print(config)
except AttributeError:
    print(f"{model_name} not included!")
    exit()
    
class DataBuilder:
    def __init__(self, day_interval = 7, data_folder_path=None, spark_session=None):
        self.day_interval = day_interval
        self.selects = ['dt','guid','itemsBox', 'domain','clickOrView','date']
        self.data_folder_path = data_folder_path
        self.spark_session = spark_session
        self.sql_client = MySQL(sql_config)
        self.end = datetime.datetime.now().strftime("%Y-%m-%d")
        current_date = datetime.datetime.now()
        seven_days_ago = current_date - datetime.timedelta(days=day_interval)
        # Format the date in "YYYY-MM-DD" format
        self.std = seven_days_ago.strftime("%Y-%m-%d")
        
    def filter_post_no_info(self, list_news_ids:list, news_id_in_sql: dict) -> list:
        """

        Args:
            list_news_id (list): _description_

        Returns:
            list: _description_
        """
#         news_info_dict = self.sql_client.get_news_by_ids(list_news_ids) # news have info
        post_have_info = [id_ for id_ in list_news_ids if news_id_in_sql.get(id_) is not None]
#         if len(post_have_info) != len(list_news_ids):
#             logger.info('filtered')
        return post_have_info #list(int)

    def process_log_hdfs(self, date_time_indexes):
        total_df = get_spark_df(date_time_indexes, self.spark_session, selects=self.selects)
        print('Showing ...')
        print(total_df.printSchema())
        df_to_csv = total_df.toPandas()
        print(df_to_csv.head(5))
        df_to_csv.drop('guid', axis = 1, inplace=True)
        df_to_csv = df_to_csv[df_to_csv['click_0_itemIDs'].apply(lambda x: len(x) > config.k_candidate_train - 1)].reset_index(drop=True)
        
        logger.info('Saving log ...')
        os.makedirs(self.data_folder_path, exist_ok=True)
        df_to_csv.to_csv(f'{self.data_folder_path}/log.csv', index = False)
        
        ## Save id_new to query SQL
        ids_merged_list = list(set(df_to_csv['candidate_news'].unique().tolist() + df_to_csv['click_0_itemIDs'].explode().unique().tolist() + df_to_csv['clicked_news'].explode().unique().tolist()))
        logger.info(f'Total news: {len(ids_merged_list)}')
        with open(f'{self.data_folder_path}/id_news.txt', 'w') as file:
            file.writelines('\n'.join(str(x) for x in ids_merged_list))
        
        return df_to_csv
    
    def process_news_data_sql(self, return_rsl=None):
        ids_merged_list = []
        with open(os.path.join(self.data_folder_path,'id_news.txt'), 'r') as file:
            for line in file:
                ids_merged_list.append(int(line))
    
        print('Total ids in ids_news.txt: ', len(ids_merged_list))
        batch_size = 100  # Specify the batch size

        # Initialize an empty DataFrame to store the results
        all_records = pd.DataFrame()

        # Split the news IDs into batches
        batches = [ids_merged_list[i:i + batch_size] for i in range(0, len(ids_merged_list), batch_size)]

    #     print(batches[:10])
        # Iterate over each batch and fetch the records
        for batch in tqdm(batches):
            batch_records = self.sql_client.get_news_by_ids(batch)
            all_records = pd.concat([all_records, batch_records], ignore_index=True)

        print('Total rows after query from SQL:', len(all_records))
        print(all_records.head())
        print('Saving...')

        news_data_path = os.path.join(self.data_folder_path,'news_data.csv')
        all_records.to_csv(news_data_path, index=False)
        
        news_id_dict = {id_: "" for id_ in all_records['newsId']}
        
        if return_rsl:
            return news_id_dict
    
    def process_log_raw_data(self):
        ## process news_id 
        news_id_in_sql = self.process_news_data_sql(return_rsl=True) 
        
        news_df = pd.read_csv(os.path.join(self.data_folder_path, 'news_data.csv'))
        logger.info(f'Total news crawled from SQL {len(news_df)}')
        print(news_df.head())
        print('Segment data')
        news_df[['title', 'sapo']] = news_df[['title', 'sapo']].applymap(preprocess_text)
        news_df[['title', 'sapo']] = news_df[['title', 'sapo']].applymap(segment_text)
        print(news_df.head())
        logger.info('Saving segment_news_data')
        news_df.to_csv(os.path.join(self.data_folder_path, 'segment_news_data.csv'), index = False)
        
        ## process log
        behavior_df = pd.read_csv(os.path.join(self.data_folder_path, 'log.csv'))
        print(behavior_df.head())
        print(len(behavior_df))
        behavior_df[['clicked_news', 'click_0_itemIDs']] = behavior_df[['clicked_news', 'click_0_itemIDs']].applymap(convert_string_to_list)
        behavior_df['clicked_news'] = behavior_df['clicked_news'].progress_apply(lambda x: self.filter_post_no_info(x, news_id_in_sql))
        behavior_df['click_0_itemIDs'] = behavior_df['click_0_itemIDs'].progress_apply(lambda x: self.filter_post_no_info(x, news_id_in_sql))
        behavior_df = behavior_df[behavior_df['click_0_itemIDs'].apply(lambda x: len(x) > config.k_candidate_train - 1)].reset_index(drop=True)
        filter_item = behavior_df['candidate_news'].progress_apply(lambda x: False if news_id_in_sql.get(x) is None else True)
        behavior_df = behavior_df[filter_item].reset_index(drop=True)
        behavior_df = behavior_df[behavior_df['user'].apply(lambda x: x.isdigit())].reset_index(drop=True)
        print(behavior_df.head())
        print(len(behavior_df))

        logger.info('Split data')
        train_df, val_df = train_test_split(behavior_df, test_size = 0.2, random_state=42)
        test_df, val_df = train_test_split(val_df, test_size = 0.5, random_state=42)
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        
        train_df['clicked'] = '1 ' + '0 '*(config.k_candidate_train-1)
        train_df['click_0_itemIDs'] = train_df['click_0_itemIDs'].apply(lambda x: take_k_clicked_0_news(x, config.k_candidate_train-1))
        val_df['clicked'] = val_df['click_0_itemIDs'].apply(lambda x: '1 ' + '0 ' * len(x))
        test_df['clicked'] = test_df['click_0_itemIDs'].apply(lambda x: '1 ' + '0 ' * len(x))
        
        train_df = train_df.progress_apply(shuffle_list, axis=1)
        val_df = val_df.progress_apply(shuffle_list, axis=1)
        test_df = test_df.progress_apply(shuffle_list, axis=1)

        logger.info('Saving behaviors_data')
        
        train_path = os.path.join(self.data_folder_path, 'train')
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        train_df.to_csv(train_path + '/behaviors_data.csv', index = False)

        val_path = os.path.join(self.data_folder_path, 'val')
        if not os.path.exists(val_path):
            os.makedirs(val_path)
        val_df.to_csv(val_path + '/behaviors_data.csv', index = False)
        
        test_path = os.path.join(self.data_folder_path, 'test')
        if not os.path.exists(test_path):
            os.makedirs(test_path)
        val_df.to_csv(test_path + '/behaviors_data.csv', index = False)
    
    def build_data(self):
        ### query log info
        logger.info(f'Crawling log from {self.std} to {self.end}')
        date_time_indexes = pd.date_range(self.std, self.end)
        self.process_log_hdfs(date_time_indexes)
        self.process_log_raw_data()
        
if __name__ == "__main__":
    CONF: SparkConf = SparkConf().setAppName("Save log data").setMaster("local").set('spark.executor.memory', '20gb').set("spark.driver.memory", "10g")
    SPARK_CONTEXT: SparkContext = SparkContext(conf=CONF)
    SPARK_CONTEXT.setLogLevel("WARN")
    SPARK_SESSION: SparkSession = SparkSession(sparkContext=SPARK_CONTEXT)
        
    databuilder = DataBuilder(day_interval=7, data_folder_path=config.WEEKLY_DATA_DIR, spark_session=SPARK_SESSION) 
    ### query log info
#     logger.info(f'Crawling log from {std} to {end}')
#     date_time_indexes = pd.date_range(std, end)
#     databuilder.process_log_hdfs(date_time_indexes)
    databuilder.build_data()
    SPARK_SESSION.stop()
    print('Hello')