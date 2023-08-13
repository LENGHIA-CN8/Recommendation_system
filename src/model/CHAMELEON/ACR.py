import sys
sys.path.append('../../')
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
# from model.general.attention.additive import AdditiveAttention
from transformers import AutoModel, AutoTokenizer
from config import model_name
import importlib

try:
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
except AttributeError:
    print(f"{model_name} not included!")
    exit()
device = torch.device(config.device if torch.cuda.is_available() else "cpu")

def freeze(model):  
    for param in model.parameters():
        param.requires_grad = False

    return model


class ACRModule(torch.nn.Module):
    def __init__(self, config):
        super(ACRModule, self).__init__()
        self.config = config
        self.PhoBERT = AutoModel.from_pretrained(config.encoder_pretrained_path)
        self.PhoBERT = freeze(self.PhoBERT)
        self.acr_fc = nn.Linear(1537, 768)
        
    def forward(self, news):
        # print(news)
        # print(type(news))
        title_news_vector = self.PhoBERT(news['title']['input_ids'].to(device), token_type_ids=None, attention_mask = news['title']['attention_mask'].to(device)).pooler_output
        
        sapo_news_vector = self.PhoBERT(news['sapo']['input_ids'].to(device), token_type_ids=None, attention_mask = news['sapo']['attention_mask'].to(device)).pooler_output
        
        category_news_vector = news['catId'].unsqueeze(-1).to(device)
        
        # print(sapo_news_vector.shape)
        # print(title_news_vector.shape)
        # print(category_news_vector.shape)
        
        final_news_vector = torch.cat((category_news_vector, title_news_vector, sapo_news_vector), 1)
        final_news_vector = self.acr_fc(final_news_vector)
        # print('final vec', final_news_vector.shape)
        return final_news_vector
    
if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    
    text = 'tôi không thích'
    text = tokenizer(text, padding='max_length', truncation=True, max_length = 256, return_tensors="pt")
    news_encoder = ACRModule(config).to(device)
    # total_params = sum(p.numel() for p in news_encoder.parameters())
    # print(f"{total_params:,} total parameters.")
    # total_trainable_params = sum(p.numel() for p in news_encoder.parameters() if p.requires_grad)
    # print(f"{total_trainable_params:,} training parameters.")
    
    print('text', text)
    print(news_encoder)
    for key in text:
        text[key] = text[key].to(device)
    output = news_encoder(text)
    print(output)
    

