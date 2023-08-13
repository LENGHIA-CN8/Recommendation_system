import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, BackgroundTasks
import yaml
import logging
import telegram
from infrastructure.hbase import HbaseUsersEncode, user_encode_hbase_config
from infrastructure.SQL import MySQL, sql_config
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(module)s: %(message)s')
logger = logging.getLogger(__name__)

sql_connect = MySQL(sql_config)


app = FastAPI()

class Textmessage(BaseModel):
    user_id: int
        
        
@app.get("/")
def main():
    return {"message": "Welcome!"}

@app.get("/recommend")
async def recommend(user_id:int, seed_post_id:int):
    user_id = int(user_id)
    user_ids = [user_id]
    logger.info(f'Rec for user_id: {user_ids}')
    hbase_user_client = HbaseUsersEncode(user_encode_hbase_config)
    res = hbase_user_client.scan_by_multi_key_encode_users(user_encode_hbase_config['table_name'], user_ids)
    hbase_user_client.close()
    
    if len(res) == 0 or 'recommend_list' not in res[user_id].keys():
        list_news_ids = sql_connect.get_news_by_times('1 DAY')
        list_news_ids = [id_[0] for id_ in list_news_ids]
        list_news_ids = list_news_ids[::-1]
        news_df_not_order = sql_connect.get_news_by_ids(list_news_ids)
        news_df = pd.DataFrame()
        for id_ in list_news_ids:
            row = news_df_not_order[news_df_not_order["newsId"] == id_]
            news_df = pd.concat([news_df, row], ignore_index=True)
        list_rec = {
        "message":"Recommend successfully",
        "recommend": [ {"id" : str(news_df.iloc[id_]['newsId']), "score": 1} for id_ in range(len(news_df)) if int(news_df.iloc[id_]['newsId']) != seed_post_id]
    }
        return list_rec
    
#     print(res[user_id])
    list_rec = {
        "message":"Recommend successfully",
        "recommend": [ {"id" : str(post[0]), "score": post[1]} for post in res[user_id]['recommend_list'][:100] if int(post[0]) != seed_post_id]
    }
    
    return list_rec

@app.get("/debug")
async def recommend(user_id:int, seed_post_id:int):
    user_id = int(user_id)
    user_ids = [user_id]
    logger.info(f'Rec for user_id: {user_ids}')
    hbase_user_client = HbaseUsersEncode(user_encode_hbase_config)
    res = hbase_user_client.scan_by_multi_key_encode_users(user_encode_hbase_config['table_name'], user_ids)
    hbase_user_client.close()
    
#     print(res[user_id]['clicked_news'])
    if len(res) == 0 or 'recommend_list' not in res[user_id].keys():
        list_news_ids = sql_connect.get_news_by_times('1 DAY')
        list_news_ids = [id_[0] for id_ in list_news_ids]
        print(list_news_ids)
        list_news_ids = list_news_ids[::-1]
        logger.info(f"List ids rec: {list_news_ids[:10]}")
        news_df_not_order = sql_connect.get_news_by_ids(list_news_ids)
        news_df = pd.DataFrame()
        for id_ in list_news_ids:
            row = news_df_not_order[news_df_not_order["newsId"] == id_]
            news_df = pd.concat([news_df, row], ignore_index=True)
        news_df['score'] = [1] * len(news_df)
        news_df.rename(columns={'newsId':'id'}, inplace=True)
        news_df = news_df[news_df['id'] != str(seed_post_id)]
        news_df['id'] = news_df['id'].astype(str)
        news_df = news_df.to_dict(orient='records')
        user_df = {
        "message":"Recommend successfully",
        "recommend": news_df
        }
        return user_df
    
    list_rec = res[user_id]['recommend_list'][:100]
    list_news_ids = [ids[0] for ids in res[user_id]['recommend_list'][:100]]
    list_score = [ids[1] for ids in res[user_id]['recommend_list'][:100]]
    
    logger.info(f"List ids rec: {list_news_ids}")
    logger.info(f"List ids score: {list_score}")
    news_df_not_order = sql_connect.get_news_by_ids(list_news_ids)
    news_df = pd.DataFrame()
    # Create a boolean mask based on list_id
    for id_ in list_news_ids:
        row = news_df_not_order[news_df_not_order["newsId"] == id_]
        news_df = pd.concat([news_df, row], ignore_index=True)
        
    news_df['score'] = list_score
    news_df.rename(columns={'newsId':'id'}, inplace=True)
    news_df = news_df[news_df['id'] != str(seed_post_id)]
    news_df['id'] = news_df['id'].astype(str)
    print(news_df.head())
    news_df = news_df.to_dict(orient='records')
    
    clicked_news = [i for i in res[user_id]['clicked_news'][-3:] if i != 0]
    clicked_news = sql_connect.get_news_by_ids(clicked_news)
    clicked_news.rename(columns={'newsId':'id'}, inplace=True)
    clicked_news['id'] = clicked_news['id'].astype(str)
    user_df = {
        "message":"Recommend successfully",
        "clicked items": clicked_news.to_dict(orient='records'),
        "recommend": news_df
    }
    
    return user_df


if __name__ == "__main__":
    uvicorn.run(app, host="10.5.1.230", port=8003)