import sys
sys.path.append('./')

import os
import importlib
import ast
import re
import logging
from datetime import datetime
import pandas as pd
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import expr, to_timestamp, concat_ws, col, flatten, first, collect_list
from pyspark.sql.types import ArrayType, LongType, StringType, IntegerType
from config import model_name
import time

try:
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
    print(config)
except AttributeError:
    print(f"{model_name} not included!")
    exit()
    
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(module)s:%(message)s')
logger = logging.getLogger(__name__)
    
os.environ['PYSPARK_PYTHON'] = '/data2/nghiatl/anaconda3/envs/SEO/bin/python'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/data2/nghiatl/anaconda3/envs/SEO/bin/python'

def get_single_spark_df(dt, SPARK_SESSION, base_filter=None, selects=None):
    day_str = str(dt)
    domain = 'cafef'
    base_filter=((F.col('domain')== f'{domain}.vn') & (F.col('guid')!='-1') & (F.col('itemsBox').isNotNull()))
    
    try:
        df: DataFrame = SPARK_SESSION.read.parquet(f"hdfs://10.3.71.86:8020/Data/Logging/AdTrackingLinks/{day_str}/*.parquet").where(base_filter).withColumn('date', F.lit(dt))
    except:
        df: DataFrame = SPARK_SESSION.read.parquet(f"hdfs://10.3.71.234:8020/Data/Logging/AdTrackingLinks/{day_str}/*.parquet").where(base_filter).withColumn('date', F.lit(dt))    
    if selects != None:
        df = df.select(selects)

    return df
    
def get_spark_df(date_time_indexes, SPARK_SESSION, filters=None, selects=None):

    df = get_single_spark_df(dt=date_time_indexes[0].date(), SPARK_SESSION=SPARK_SESSION, selects=selects)
    df = process_df(df)
    for i in range(1, len(date_time_indexes)):
        df_dt = get_single_spark_df(dt=date_time_indexes[i].date(), SPARK_SESSION=SPARK_SESSION, selects=selects)
        df_dt = process_df(df_dt)
        df = df.union(df_dt)
    return df

def convert_candidate_news(candidate_news):
    candidate_news = re.sub(r'[\]\[]','',candidate_news).split(',')
    candidate_news = [element.strip() for element in candidate_news[:4]]
    
    return ' '.join(candidate_news)

def split_list(item_list):
    convert_list = [l[0] for l in item_list]
    
    return convert_list

def extract_item_ids(items_box):
    item_ids = items_box.split(",")
    item_ids = [int(item_id.split("-")[-1].strip()) for item_id in item_ids if item_id.split("-")[-1].strip() != '']
    # print(item_ids)
    return item_ids

def process_df(filtered_df= None):
    extract_item_ids_udf = F.udf(extract_item_ids, ArrayType(LongType()))
    filtered_df = filtered_df.withColumn("itemIDs", extract_item_ids_udf("itemsBox"))
    filtered_df_click_0 = filtered_df.filter(filtered_df.clickOrView == 0)
    filtered_df_click_1 = filtered_df.filter(filtered_df.clickOrView == 1)
    # # Define a window partitioned by guid and sorted by dt
    window = Window.partitionBy("guid").orderBy("dt")

    # # Create a column containing the sorted list of itemIDs for each guid
    grouped_df = filtered_df_click_1.withColumn("sorted_itemIDs", F.collect_list("itemIDs").over(window))
    
    split_list_df = F.udf(split_list, ArrayType(LongType()))
    grouped_df = grouped_df.withColumn("sorted_itemIDs", split_list_df("sorted_itemIDs"))
    grouped_df = grouped_df.filter(F.size(grouped_df['sorted_itemIDs']) > 1)
    # Calculate the maximum length of sorted_itemIDs for each guid
    max_length_df = grouped_df.groupBy("guid").agg(F.max(F.size("sorted_itemIDs")).alias("max_length"))

    # Join with the original DataFrame to filter out rows with other lengths
    grouped_df = grouped_df.join(max_length_df, ["guid"]).filter(F.size("sorted_itemIDs") == F.col("max_length"))

    # Drop the max_length column if desired
    grouped_df = grouped_df.drop("max_length")

    # Extract the last element of sorted_itemIDs
    last_element = F.expr("sorted_itemIDs[size(sorted_itemIDs) - 1]")

    # Add the last element to a new column called candidate_news
    grouped_df = grouped_df.withColumn("candidate_news", last_element)

    grouped_df = grouped_df.withColumn("sorted_itemIDs", F.expr("slice(sorted_itemIDs, 1, size(sorted_itemIDs) - 1)"))
    
    grouped_df = grouped_df.withColumnRenamed("sorted_itemIDs", "clicked_news")
    
    new_column_names = ['click_0_' + col for col in filtered_df_click_0.columns]
    df_renamed_click_0 = filtered_df_click_0.toDF(*new_column_names)
    
    # Perform the join operation with the additional condition on dt
    df_updated = grouped_df.join(df_renamed_click_0, (grouped_df["guid"] == df_renamed_click_0["click_0_guid"]), "left")

    # Convert the string columns to timestamp
    df_updated = df_updated.withColumn('dt', col('dt').cast('timestamp'))
    df_updated = df_updated.withColumn('click_0_dt', col('click_0_dt').cast('timestamp'))

    # Filter the DataFrame based on the condition
    df_updated = df_updated.filter(df_updated.dt < df_updated.click_0_dt)
    df_updated = df_updated.filter(
        expr("dt + interval 1 hour >= click_0_dt")
    )

    # Group by 'guid' and concatenate 'click_0_itemIDs'
    # df_updated = df_updated.groupBy('guid').agg(concat_ws(',', collect_list('click_0_itemIDs')).alias('candidate_news'), first('click_0_dt').alias('click_0_dt'), first('dt').alias('dt'))
    df_updated = df_updated.groupBy('guid').agg(first('dt').alias('dt'), first('guid').alias('user'), first('clicked_news').alias('clicked_news'), collect_list('click_0_itemIDs').alias('click_0_itemIDs'), first('candidate_news').alias('candidate_news'))
    df_updated = df_updated.withColumn('click_0_itemIDs', flatten(col('click_0_itemIDs')))    
    df_updated = df_updated.withColumn("dt", col("dt").cast("string"))
    
    return df_updated


if __name__ == '__main__':
    CONF: SparkConf = SparkConf().setAppName("Save log data").setMaster("local").set('spark.executor.memory', '20gb').set("spark.driver.memory", "10g")
    SPARK_CONTEXT: SparkContext = SparkContext(conf=CONF)
    SPARK_CONTEXT.setLogLevel("WARN")
    SPARK_SESSION: SparkSession = SparkSession(sparkContext=SPARK_CONTEXT)
        
    current_time_local = time.localtime()
    year = current_time_local.tm_year
    month = current_time_local.tm_mon
    day = current_time_local.tm_mday
    selects=['dt','guid','itemsBox', 'domain','clickOrView','date']
    
    std = '2023-07-10'
    end = '2023-07-10'
    
    ### query log info
    logger.info(f'Crawling log from {std} to {end}')
    date_time_indexes = pd.date_range(std, end)
    total_df = get_spark_df(date_time_indexes, SPARK_SESSION, selects=selects)
    print('Showing ...')
    print(total_df.printSchema())
    df_to_csv = total_df.toPandas()
    print(df_to_csv.head(5))
    df_to_csv.drop('guid', axis =1, inplace=True)
    df_to_csv = df_to_csv[df_to_csv['click_0_itemIDs'].apply(lambda x: len(x) > config.k_candidate_train - 1)].reset_index(drop=True)
    SPARK_SESSION.stop()
    
    ### Save log user
    logger.info(f'Log crawled ... \n {df_to_csv.head()}')
    logger.info(f'Total logs crawled: {len(df_to_csv)}')
    logger.info('Saving log ...')
    os.makedirs(f'../data_news/{day}-{month}-{year}/', exist_ok=True)
    df_to_csv.to_csv(f'../data_news/{day}-{month}-{year}/log.csv', index = False)
    
    ## Save id_new to query SQL
    ids_merged_list = list(set(df_to_csv['candidate_news'].unique().tolist() + df_to_csv['click_0_itemIDs'].explode().unique().tolist() + df_to_csv['clicked_news'].explode().unique().tolist()))
    logger.info(f'Total news: {len(ids_merged_list)}')
    with open(f'../data_news/{day}-{month}-{year}/id_news.txt', 'w') as file:
        file.writelines('\n'.join(str(x) for x in ids_merged_list))