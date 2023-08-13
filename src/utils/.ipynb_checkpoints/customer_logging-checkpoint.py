import sys
sys.path.append('./')

from infrastructure.kafka import KafkaProcessor
from config import InfraConfig
import time

class KafkaLogging:
    """ Process log for each a customer
    """
#     TIME_POINT_START = None
    def __init__(self, config, domain='cafef.vn'):
        self.domain = domain
        self.TIME_POINT_START = None
        # kafka_config = {
        #         "group_id": 'soha_web_din_per_ctr',
        #         "module_type" : 'consumer',
        #         "bootstrap_servers" : config.KAFKA_CLUSTER_BOOTSTRAP_SERVERS,
        #         "auto_offset_reset" :"earliest"
        # }
        self.kafka_repo = KafkaProcessor(topics = config.KAFKA_CLUSTER_TOPICS,
                                         module_type = 'consumer',
                                         bootstrap_servers = config.KAFKA_CLUSTER_BOOTSTRAP_SERVERS,
                                         auto_offset_reset = "earliest"
                                        )
#         print(self.kafka_repo)
        self._remake_kafka_consumer()
        
    def _check_log(self, log):
        """ check conditions
        Args:
            log (dict): converted log
        Return:
            bool
        """
#         if self.TIME_POINT_START is not None and log['time'] < self.TIME_POINT_START:
#             return False
        if log['domain'] != self.domain :
            return False
        if log['action'] != '1' :
            return False
        if not log['user_id'].isdigit():
            return False
        if log['user_id'] == '' or log['user_id'] == None or log['user_id'] == '-1':
            return False
        
        return True
        
    def _remake_kafka_consumer(self):
        self.kafka_consumer = self.kafka_repo.get_consumer(offset=0)
        
    def _convert_kafka_raw_log(self, raw):
        """ convert raw log to dictionary
        Args:
            raw : log raw format 
        Return:
            result (dict): target log dict format
        """
        result = {}
        log = raw.decode('utf-8').strip().split('\t')
#         print(log)
        result['time'] = log[0]
        result['user_id'] = log[2]
        result['domain'] = log[4]
        result['action'] = log[9]
        
        return result
    
    def next_batch_log(self, batch_size, limit_time):
        """ Private function get kafka log by batch and limit
        Args:
            batch_size (int): maximum number of logs per each time
            limit_time (int): time limit per each time
            check_function (func): function check log
        Return:
            result (list): list of logs 
        """
        result = []
        t = time.time()
        while len(result) < batch_size:
            data = next(self.kafka_consumer)
            log = []
            try:
                log = self._convert_kafka_raw_log(data.value)
            except:
                # error= str(traceback.format_exc())
                # mess = "ERRORTRACKING KAFKA LOG : "+ error
                # print(mess)
                # print(data.value)
                continue  
            if self._check_log(log):
                print(log)
                time.sleep(2)
                result.append(log)
            if time.time() - t >= limit_time:
                break
        if len(result):
            result = sorted(result, key=lambda x: x['time'], reverse=False)
            self.TIME_POINT_START = result[0]['time']
        return result

    
if __name__ == "__main__":
    config = InfraConfig()
    customer_log = KafkaLogging(config)
    print(customer_log.next_batch_log(10, 10000))
#     for message in customer_log.kafka_consumer:
# #         res = process_kafka_log(message.value)
# #         print(res)
#         print(message.value)
#         time.sleep(3)