import os

model_name = os.environ['MODEL_NAME'] if 'MODEL_NAME' in os.environ else 'CHAMELEON'
# Currently included model
assert model_name in [
    'CHAMELEON'
]


class BaseConfig():
    """
    General configurations appiled to all models
    """
    num_epochs = 8
    num_batches_show_loss = 100  # Number of batchs to show loss
    num_steps_show_loss = 1000
    # Number of batchs to check metrics on validation dataset
    num_batches_validate = 1000
    batch_size = 8
    learning_rate = 0.0001
    num_workers = 4  # Number of workers for data loading
    num_clicked_news_a_user = 10  # Number of sampled click history for each user
    num_words_title = 30
    num_words_abstract = 60
    # word_freq_threshold = 1
    # entity_freq_threshold = 2
    # entity_confidence_threshold = 0.5
    negative_sampling_ratio = 2  # K
    dropout_probability = 0.2
    # Modify the following by the output of `src/dataprocess.py`
    num_words = 1 + 70975
    num_categories = 1 + 274
    num_entities = 1 + 12957
    num_users = 1 + 50000
    word_embedding_dim = 300
    category_embedding_dim = 100
    # Modify the following only if you use another dataset
    entity_embedding_dim = 100
    # For additive attention
    query_vector_dim = 200

class InfraConfig():
    """
    General configurations applied to infra
    """
    user_interact_table = 'cafef:user_interact_post'
    encode_news_table = 'cafef:EncodeNews'
    encode_users_table = 'cafef:ScoreForEncodeUsers'
    
    user_interact_hbase_host = 'thrift.notinews.ml.adt.internal'  # HBase host
    encode_hbase_host = 'thrift.data-collection.ml.adt.internal'
    
    hbase_port = 9090,
    hbase_connections = 1,  # Number of connection pool size
    hbase_namespace = 'cafef'  # Namespace for HBase tables
    
    KAFKA_CLUSTER_TOPICS = 'rt-tracking-links'
    KAFKA_CLUSTER_BOOTSTRAP_SERVERS = ['failover.cloud.kafka.adt:9092']
    
class CHAMELEONConfig(BaseConfig, InfraConfig):
    dataset_attributes = {
        "news": ['catId', 'title', 'sapo'],
        "record": []
    }
    news_embedding_dim = 768
    k_candidate_train = 4
    encoder_pretrained_path = 'vinai/phobert-base'
    device = 'cuda:1'
    WEEKLY_DATA_DIR = '../data_news/weekly_data'
