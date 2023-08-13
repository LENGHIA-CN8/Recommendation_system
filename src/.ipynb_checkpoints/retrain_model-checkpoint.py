from utils.databuilder import DataBuilder
from utils.utils import update_phase
from datetime import datetime
import time  
import logging
import schedule
import traceback
from config import model_name
from train import train
import importlib
import os
import pandas as pd
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, DataFrame

logger = logging.getLogger(__name__)
try:
    Model = getattr(importlib.import_module(f"model.{model_name}"), model_name)
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
    print(config)
except AttributeError:
    print(f"{model_name} not included!")
    exit()

def rmv_ckpt(folder_path):
    pass

def rmv_file(folder_path):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Iterate over the files and remove them one by one
    for file in files:
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Removed: {file_path}")

def build_data_func():    
    # build data
    ## remove built data
    rmv_file(os.path.join(config.WEEKLY_DATA_DIR, "train"))
    rmv_file(os.path.join(config.WEEKLY_DATA_DIR, "val"))
    rmv_file(os.path.join(config.WEEKLY_DATA_DIR, "test"))
    
    CONF: SparkConf = SparkConf().setAppName("Save log data").setMaster("local").set('spark.executor.memory', '20gb').set("spark.driver.memory", "10g")
    SPARK_CONTEXT: SparkContext = SparkContext(conf=CONF)
    SPARK_CONTEXT.setLogLevel("WARN")
    SPARK_SESSION: SparkSession = SparkSession(sparkContext=SPARK_CONTEXT)

    logger.info("Start build weekly data")
    
    databuilder = DataBuilder(day_interval=7, data_folder_path=config.WEEKLY_DATA_DIR, spark_session=SPARK_SESSION)
    databuilder.build_data()
    
    logger.info("Finish build weekly data")
    SPARK_SESSION.stop()
    
def build_retrain_model() -> None:
    file_path = "./checkpoint/CHAMELEON/training.txt"
    update_phase(file_path, 'True')
    retries = 3
    for i in range(retries):
        try:
            # build data
            build_data_func()
            break
        except:
            error= str(traceback.format_exc())
            logger.info(f"ERRORTRACKING - BUILD DATA : "+error)
            continue
        
    # train model
    print(config.WEEKLY_DATA_DIR)
    train(config.WEEKLY_DATA_DIR)
    update_phase(file_path, 'False')
    
if __name__ == "__main__":
    build_retrain_model()
    schedule.every().day.at("23:00", timezone('Asia/Ho_Chi_Minh')).do(build_retrain_model).tag("run_retrain_model", 1)
    while True:
        schedule.run_pending()
        time.sleep(1)
