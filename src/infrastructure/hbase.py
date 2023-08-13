import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)

import hashlib
import logging 
import random
import happybase
from thriftpy2.transport.base import TTransportException
from happybase import Connection
from config import InfraConfig
from tqdm import tqdm
import pickle
import ast
import numpy as np
from collections import OrderedDict


logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(module)s: %(message)s')
logger = logging.getLogger(__name__)

class HBase:
    def __init__(self, table_sep=':', retries=3, new_format=True, **config):
        self.config = config
        self.retries = retries
        self.table_sep = table_sep
        self.connection = self._create_connection_pool(config['host'], config['port'], new_format)
            
    def _create_connection_pool(self, host, port, new_format):
        if new_format:
            return Connection(
                        host=host,
                        port=port,
                        transport='framed',
                        protocol='compact',
                        timeout=5000,
        )
        else:
            return happybase.ConnectionPool(
                        size=self.config['num_connections'],
                        host=host,
                        port=port,
                        timeout=5000,
                        table_prefix=self.config['namespace'],
                        table_prefix_separator=self.table_sep
                    )
    
    def close(self):
        self.connection.close()
        
    def get_pool(self):
        """Select a connection pool randomly from pools.
        """

        if len(self.pools) == 0:
            return None

        if len(self.pools) == 1:
            return self.pools[0]

        i = random.randint(0, len(self.pools) - 1)

        return self.pools[i]
    
    def row_base(self, table_name, row, columns=None,
                 timestamp=None, include_timestamp=False):
        """Return values of row with specific column family.
        """
        table = self.connection.table(table_name)
        row_data = table.row(
            row=row,
            columns=columns)
        
        return row_data
        
    def scan_base(self, table_name, row_start=None,
                  row_stop=None, row_prefix=None, columns=None,
                  filter=None, timestamp=None, include_timestamp=False,
                  batch_size=30, scan_batching=None, limit=None,
                  sorted_columns=False, reverse=False):
        """Return values of rows with specific row index.
        """
        table = self.connection.table(table_name)
        records = [record for record in table.scan(
            row_start=row_start,
            row_stop=row_stop,
            row_prefix=row_prefix,
            columns=columns,
            filter=filter,
            timestamp=timestamp,
            include_timestamp=include_timestamp,
            batch_size=batch_size,
            scan_batching=scan_batching,
            limit=limit,
            sorted_columns=sorted_columns,
            reverse=reverse
        )]

        return records
    
    def read_batch_by_keys(self, list_key, table, columns=None) -> dict():
        """
        Args:
            list_key (list): list salt key
            table: table hbase
        Returns:
            list: result which read from hbase
        """        
        batch_size: int = 30
        rsl = []

        for start_index in range(0, len(list_key), batch_size):
#             end_index: int = min(start_index+batch_size, len(list_key))
            tmp = table.rows(list_key[start_index:start_index+batch_size], columns=columns)
            rsl.extend(tmp)

        return rsl
    
    def scan_by_multi_key(self, table_name, list_key, is_salt = True, columns=None, include_timestamp=True):
        table = self.connection.table(table_name)
        results = {}
        keys = []
        # map salt key
        for key in tqdm(list_key, desc="GET user interation in Hbase: "):
            key = int(key)
            if is_salt == True:
                try:
                    key_salt = get_rowkey(key)
                except:
                    error= str(traceback.format_exc())
                    logger.info(f"ERRORTRACKING - HBPool GET ROWKEY BY BATCH, KEY - {key}: "+error)
                    continue
            else:
                key_salt = key
            keys.append(key_salt)
            
        # add batch keys
        data = self.read_batch_by_keys(keys, table, columns=[b'cf:positive'])
        ###
        for i in data:
            # get clicked news for user in interact post table
            try: 
                results[i[0].decode()[4:]] = filter_his(ast.literal_eval(i[1][b'cf:positive'].decode()))
            except:
#                 print(i[0])
#                 print(i[1].keys())
                logger.info("Something wrongs in user log hbase may be don't have cf:positive field")
                continue
        """
        return dict(str(id): cf:pos)
        """
        return results

class HbaseNewsEncode(HBase):
    def __init__(self, config):
        super().__init__(new_format=True,**config)
        
    def get_non_exist_keys(self, table_name, 
                           list_key, 
                           is_salt = True):
        '''
            return keys dont exist in table - list(int)
        '''
        table = self.connection.table(table_name)
        
        if is_salt:
            salt_list_key = [get_rowkey(int(key)) for key in list_key]
        else:
            salt_list_key = [str(key) for key in list_key]
        
        hbase_rsl = self.read_batch_by_keys(salt_list_key, table) # batch
        
        exist_keys = [int(key[0][4:]) for key in hbase_rsl]
        if len(exist_keys) == len(list_key):
            logger.info('All news_id key exist in Hbase news')
        list_key = list(map(int, list_key))
        non_exsit_keys = list(set(list_key).difference(set(exist_keys)))
        
        return non_exsit_keys
    
    def get_exist_keys(self, table_name, 
                           list_key, 
                           is_salt = True):
        
        table = self.connection.table(table_name)
        
        if is_salt:
            salt_list_key = [get_rowkey(int(key)) for key in list_key]
        else:
            salt_list_key = [str(key) for key in list_key]
        
        hbase_rsl = self.read_batch_by_keys(salt_list_key, table) # batch
        
        exist_keys = [int(key[0][4:]) for key in hbase_rsl]
        
        return exist_keys
    
    def put_encode_news(self, table_name, 
                        data, 
                        is_salt = True):
        ''' write data to hbase by batch
            input: data - dict(id: {"encode_vec": ,"info": })
            
        '''
        table = self.connection.table(table_name)
        batch = table.batch(batch_size=500)
        # check keys not in db
        data_keys = list(map(int,data.keys())) 
        
        for key in tqdm(data_keys):
            if is_salt is True:
                key_salt = get_rowkey(key) 
            else:
                key_salt = str(key)
            batch.put(key_salt, {b"cf:data": pickle.dumps(data[str(key)])})
        
        batch.send()
        
    def scan_by_multi_key_encode_news(self, table_name: str, 
                                      list_key: list, 
                                      get_infor: bool = True,
                                      is_salt:bool = True):
        ''' read data from hbase
            type- result: {
                id: {
                    'encode_vec': np.array,
                    'infor': dict {
                        'title':
                        'public_date':
                        'sub category': 
                    }
                }
            }
        '''
        table = self.connection.table(table_name)
        results = {}
        map_saltkey_key = dict()
        keys = []

        for key in tqdm(list_key, desc="GET encoded news vector in Hbase: "):
#             if int(str(key)[:4]) <= 2015:
#                 continue
            
            if is_salt == True:
                key_salt = get_rowkey(key) 
            else:
                key_salt = key
            keys.append(key_salt)
            map_saltkey_key[key_salt] = key 
        
        data = self.read_batch_by_keys(keys, table) # batch
    
        for k,v in data:
            key = map_saltkey_key[k]
            result = pickle.loads(v[b'cf:data'])
            try:
                if result["encode_vec"].shape[0] != 0:
                    if get_infor:
                        results[key] = {"encode_vec": result["encode_vec"]}
                    else:
                        results[key] = result["encode_vec"]
            except:
                print('Resulttttttt')
                print(result)
                    
        return results

class HbaseUserInteract(HBase):
    def __init__(self, config):
        super().__init__(new_format=True,**config)
    
    def get_history_user(self, user_interact_table, user_ids, num_clicked_news_a_user):
        '''
        input: 
                user_ids (List) int
        output: 
                dict() - key: str(user_id), value: list(int) history, 
                list(set()) - id of item => to get infor
        '''
        user_ids = list(map(int, user_ids))

        data = self.scan_by_multi_key(user_interact_table, user_ids) #dict: key - user, value: history 
        
        result = {}
        all_clicked_items = []
        for k, v in data.items():
            ## sort by timestamp
            items = v #sorted(v, reverse=True) # clicked_item, list - int / descending 
            if len(items) == 0:
                continue        
            else:
                # rmv items which have len(item) < 16:
                len_items = [len(str(items)) for item in items]
                rmv_items_indices = np.where(np.array(len_items) < 16)
                items = np.delete(items, rmv_items_indices)
                clicked_items = list(OrderedDict.fromkeys(items.tolist())) # rmv duplicate val, dont lose the order of org list

                if len(clicked_items) >= 2:
                    ## max log = 50
                    result[k] = clicked_items[:num_clicked_news_a_user] #clicked_items
                    all_clicked_items.extend(clicked_items[:num_clicked_news_a_user])
                    
        return result, list(set(all_clicked_items))

class HbaseUsersEncode(HBase):
    def __init__(self, config):
        super().__init__(new_format=True,**config)
        
    def put_encode_users(self, table_name, 
                        data, 
                        is_salt = True):
        ''' write data to hbase by batch
            input: data - dict(id: {"encode_vec":})
            
        '''
        logger.info(f'Saving {len(data.keys())} user feature')
        table = self.connection.table(table_name)
        batch = table.batch(batch_size=500)
        # check keys not in db
        data_keys = list(map(int,data.keys())) 
        
        print(f'Saving user {data_keys[:10]} ...' )
        for key in tqdm(data_keys):
            if is_salt is True:
                key_salt = get_rowkey(key) 
            else:
                key_salt = str(key)
            batch.put(key_salt, {b"cf:data": pickle.dumps(data[key])})
            
        batch.send()
        
    def scan_by_multi_key_encode_users(self, table_name, list_user: list, is_salt: bool = True):
        
        table = self.connection.table(table_name)
        results = {}
        map_saltkey_key = dict()
        keys = []
        
        print(f'Scan user id: {list_user[:10]}')
        for key in tqdm(list_user, desc="GET encoded user vectors in Hbase: "):
#             if int(str(key)[:4]) <= 2015:
#                 continue
            
            if is_salt == True:
                key_salt = get_rowkey(key) 
            else:
                key_salt = key
            keys.append(key_salt)
            map_saltkey_key[key_salt] = key 
        
        data = self.read_batch_by_keys(keys, table)
        
        for k,v in data:
            key = map_saltkey_key[k]
            result = pickle.loads(v[b'cf:data']) 
            results[key] = result
                    
        return results
        
def filter_his(list_history):
    """_summary_

    Args:
        list_history (list): neg list or pos list

    Returns:
        list: after filter invalid candidates
    """
    list_history = [can for can in list_history if len(str(can)) >= 8]
    return list_history

def get_rowkey(x) -> str:
    mode: int = x % 1000
    row_key: str = f"{mode:03d}_{x}"
    row_key = row_key.encode("utf-8")
    
    return row_key

infra_config = InfraConfig()

user_interact_hbase_config = {
    'host': infra_config.user_interact_hbase_host,  # HBase host
    'port': 9090,
    'num_connections': 1,  # Number of connection pool size
    'namespace': infra_config.hbase_namespace,  # Namespace for HBase tables
    'table_name': infra_config.user_interact_table
}

encode_hbase_config = {
    'host': infra_config.encode_hbase_host,  # HBase host
    'port': 9090,
    'num_connections': 1,  # Number of connection pool size
    'namespace': infra_config.hbase_namespace,  # Namespace for HBase tables
    'table_name': infra_config.encode_news_table
}

user_encode_hbase_config = {
    'host': infra_config.encode_hbase_host,  # HBase host
    'port': 9090,
    'num_connections': 1,  # Number of connection pool size
    'namespace': infra_config.hbase_namespace,  # Namespace for HBase tables
    'table_name': infra_config.encode_users_table
}


if __name__ == '__main__':
    hbase = HBase(new_format=True, **user_interact_hbase_config)
    news_encode_hbase = HbaseNewsEncode(encode_hbase_config)
    hbase_user_client = HbaseUsersEncode(user_encode_hbase_config)
    
#     user_id = [6818491211246072110, 1540006981906398468, 4356342122112614399]
#     print(hbase_user_client.scan_by_multi_key_encode_users(user_encode_hbase_config['table_name'], user_id).keys())
    
    list_news = [20190328124114492, 188230725141806343, 188230802085813026, 188230805084046052]
    print('extract', list_news)
    print('Non exist', news_encode_hbase.get_non_exist_keys(encode_hbase_config['table_name'], list_news))
    print('Exist', news_encode_hbase.get_exist_keys(encode_hbase_config['table_name'], list_news))
    print([x for x in news_encode_hbase.scan_by_multi_key_encode_news(encode_hbase_config['table_name'], list_news).keys()])
    
#     b_key = get_rowkey(20130304105356794)
#     print(b_key)
#     print(news_encode_hbase.row_base(encode_hbase_config['table_name'], b_key))
#     print(news_encode_hbase.scan_by_multi_key_encode_news(encode_hbase_config['table_name'], [20130304105356794]))

#     user_id = [6818491211246072110]
#     row_key = get_rowkey(user_id[0])    
#     column_to_retrieve = [b'cf']
#     result = hbase.scan_by_multi_key(user_interact_hbase_config['table_name'], user_id, columns=column_to_retrieve, include_timestamp=True)
#     print(result)
    
    hbase.close()
    news_encode_hbase.close()
    hbase_user_client.close()
#     print('res', result[b'cf:user'])
#     print(result[b'cf:positive'])
#     print(result[b'cf:negative'])

