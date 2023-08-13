import numpy as np
from kafka import KafkaConsumer, KafkaProducer, TopicPartition

class KafkaProcessor:
    """ Module interaction with kafka database
    """

    def __init__(self, topics, module_type='consumer', **config):
        """ 
        Args:
            module_type (str): 'consumer' or 'producer', default 'consumer'
            Parameters, refer to 
                https://kafka-python.readthedocs.io/en/master/apidoc/KafkaConsumer.html#kafka.KafkaConsumer 
                https://kafka-python.readthedocs.io/en/master/apidoc/KafkaProducer.html
        """
        if module_type.lower() == 'consumer':
            # self.consumer_config = copy.copy(self.DEFAULT_CONFIG_CONSUMER)
            self.consumer_config = {}
            self.consumer_config.update(config)
            self.topics = topics
            print(self.consumer_config)
            print(self.topics)
        else: 
            self.producer_config = {}
            self.producer_config.update(config)
    def _reset_consumer(self, offset):
        """ private method to return consumer was reset to offset
        Args:
            offset (int): offset number
        """
        kafka_consumer = KafkaConsumer(**self.consumer_config)
        topic_partitions = [TopicPartition(topic , 0) for topic in [self.topics]]
        kafka_consumer.assign(topic_partitions)
        if not isinstance(offset, (int, np.int8, np.int16, np.int32, np.int64)):
            raise Exception("type offset must be integer, can't {}!".format(type(offset)))
        kafka_consumer.seek(topic_partitions[0], offset)
        return kafka_consumer

    def get_consumer(self, offset='auto'):
        """ Create consumer from offset
        Args:
            offset (str or int): consumer will be started from this offset, default 'auto' -> continue offset
        """
        if offset == 'auto':
            kafka_consumer = KafkaConsumer( self.topics, **self.consumer_config)
            
        else:
            kafka_consumer = self._reset_consumer(offset)
        return kafka_consumer

    def get_producer(self):
        kafka_producer = KafkaProducer(**self.producer_config)
        return kafka_producer
