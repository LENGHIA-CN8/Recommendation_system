import sys
sys.path.append('./')
print(sys.path)

import os
import pandas as pd
from utils.utils import preprocess_text, convert_string_to_list, segment_text, take_k_clicked_0_news, filter_not_exist
import logging
from sklearn.model_selection import train_test_split
from config import model_name
import importlib
import time
import argparse



try:
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
    print(config)
except AttributeError:
    print(f"{model_name} not included!")
    exit()
    
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(module)s:%(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    current_time_local = time.localtime()
    year = current_time_local.tm_year
    month = current_time_local.tm_mon
    day = current_time_local.tm_mday
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_folder_path', type=str, help = 'path to the data fold')
    
    args = parser.parse_args()
    
    data_folder_path = args.data_folder_path
    ## process news
    print(os.path.join(data_folder_path, 'news_data.csv'))
    news_df = pd.read_csv(os.path.join(data_folder_path, 'news_data.csv'))
    logger.info(f'Total news crawled from SQL {len(news_df)}')
    print(news_df.head())
    print(news_df.info())
    news_df[['title', 'sapo']] = news_df[['title', 'sapo']].applymap(preprocess_text)
    news_df[['title', 'sapo']] = news_df[['title', 'sapo']].applymap(segment_text)
    print(news_df.head())
    logger.info('Saving segment_news_data')
    news_df.to_csv(os.path.join(data_folder_path, 'segment_news_data.csv'), index = False)
    
    ## process log
    behavior_df = pd.read_csv(os.path.join(data_folder_path, 'log.csv'))
    logger.info(f'Total log crawled {len(behavior_df)}')
    behavior_df[['clicked_news', 'click_0_itemIDs']] = behavior_df[['clicked_news', 'click_0_itemIDs']].applymap(convert_string_to_list)
    filtered_rows = behavior_df.apply(filter_not_exist, axis=1, news_df=news_df)
    behavior_df = behavior_df[filtered_rows]
    logger.info(f'Total log after filter {len(behavior_df)}')
    
    logger.info('Split data')
    train_df, val_df = train_test_split(behavior_df, test_size = 0.1, random_state=42)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    train_df['clicked'] = '1 ' + '0 '*(config.k_candidate_train-1)
    train_df['click_0_itemIDs'] = train_df['click_0_itemIDs'].apply(lambda x: take_k_clicked_0_news(x, config.k_candidate_train-1))
    val_df['clicked'] = val_df['click_0_itemIDs'].apply(lambda x: '1 ' + '0 ' * len(x))

    logger.info('Saving behaviors_data')
    
    train_path = os.path.join(data_folder_path, 'train')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    train_df.to_csv(train_path + '/behaviors_data.csv', index = False)
    
    val_path = os.path.join(data_folder_path, 'val')
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    val_df.to_csv(val_path + '/behaviors_data.csv', index = False)
    