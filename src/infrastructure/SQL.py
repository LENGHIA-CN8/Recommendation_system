import sys
sys.path.append('./')

import mysql.connector
import pandas as pd
import datetime
from tqdm import tqdm
import time
import os
import argparse



sql_config = {
  'user': 'cafef_model_experiment',
  'password': 'tLOfVcNjFQ4vMHkS78IB', #tLOfVcNjFQ4vMHkS78IB
  'host': '192.168.23.191',
  'database': 'news',
  'raise_on_warnings': True
}

class MySQL:
    def __init__(self, config) -> None:
        self.config = config

    def get_connector(self):
        # print('config arg: ',self.config)
        connector = mysql.connector.connect(**self.config)
        return connector
    
    def close_connector(self):
        self.connector.close()
        
    def get_news_by_times(self, interval_time):
        """
        get new newsid by interval time
        """
        self.connector = self.get_connector()
        cursor = self.connector.cursor()
        query = "describe news.news_resource"
        cursor.execute(query)
        records = cursor.fetchall()
        print(records)
        query = (f"SELECT newsId \
            FROM news_resource \
            WHERE sourceNews = 'cafef' \
            AND  publishDate >= DATE_SUB(NOW(), INTERVAL {interval_time}) ")
        cursor.execute(query)
        records = cursor.fetchall()
        # records = pd.DataFrame(records, columns=['newsId', 'publishDate', 'catId', 'title', 'sapo', 'content'])
        self.connector.close()
        cursor.close()
        return records

    def get_new_newsid(self, start_time, end_time):
        """
        get new newsid
        """
        self.connector = self.get_connector()
        cursor = self.connector.cursor()
        query = "describe news.news_resource"
        cursor.execute(query)
        records = cursor.fetchall()
        print(records)
        query = ("SELECT newsId, publishDate, catId, title, sapo\
                FROM news.news_resource WHERE publishDate BETWEEN %s AND %s\
                ")
        cursor.execute(query, (start_time, end_time))
        records = cursor.fetchall()
        records = pd.DataFrame(records, columns=['newsId', 'publishDate', 'catId', 'title', 'sapo'])
        self.connector.close()
        cursor.close()
        return records
    
    def get_news_by_ids(self, news_ids):
        """
        Get news by IDs
        """
        self.connector = self.get_connector()
        cursor = self.connector.cursor()
#         query = "describe news.news_resource"
#         cursor.execute(query)
#         records = cursor.fetchall()
#         print(records)

        query = """
            SELECT newsId, sourceNews, publishDate, catId, title, sapo
            FROM news.news_resource
            WHERE newsId IN ({}) and sourceNews = 'CafeF' 
        """
        placeholders = ', '.join([str(id) for id in news_ids])
        query = query.format(placeholders)
        cursor.execute(query)
        records = cursor.fetchall()
        records = pd.DataFrame(records, columns=['newsId', 'sourceNews', 'publishDate', 'catId', 'title', 'sapo'])

        self.connector.close()
        cursor.close()

        return records
    
def encode_utf8(x):
    if isinstance(x, str):
        return x.encode('utf-8').decode('utf-8', 'ignore')
    else:
        return x

# Assuming records is your DataFrame
# records_encoded = records.applymap(encode_utf8)
 
if __name__ == "__main__":
    
    sql_connect = MySQL(sql_config)
    ids_merged_list = []
    
#     parser = argparse.ArgumentParser()
    
#     parser.add_argument('data_folder_path', type=str, help = 'path to the data fold')
    
#     args = parser.parse_args()
    
#     data_folder_path = args.data_folder_path
    
#     with open(os.path.join(data_folder_path,'id_news.txt'), 'r') as file:
#         for line in file:
#             ids_merged_list.append(int(line))
    
#     print('Total ids in ids_news.txt: ', len(ids_merged_list))
#     batch_size = 100  # Specify the batch size
    
#     # Initialize an empty DataFrame to store the results
#     all_records = pd.DataFrame()
    
#     # Split the news IDs into batches
#     batches = [ids_merged_list[i:i + batch_size] for i in range(0, len(ids_merged_list), batch_size)]
    
# #     print(batches[:10])
#     # Iterate over each batch and fetch the records
#     for batch in tqdm(batches):
#         batch_records = sql_connect.get_news_by_ids(batch)
#         all_records = pd.concat([all_records, batch_records], ignore_index=True)
    
#     print('Total rows:', len(all_records))
#     print(all_records.head())
    
#     print('Saving...')
    
#     news_data_path = os.path.join(data_folder_path,'news_data.csv')
#     all_records.to_csv(news_data_path, index=False)
      
#     t = pd.read_csv(news_data_path)
#     print(t.head())
    
    news_id_list = [20130304105356794]
    for ids in news_id_list:
        print(ids)
        records = sql_connect.get_news_by_ids([ids])
        print(records.empty)
        print('Total rows: ', len(records))
        print(records.head())
        print('\n')