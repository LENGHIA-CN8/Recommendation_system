import torch
import torch.nn as nn
import torch.nn.functional as F
# from model.general.attention.additive import AdditiveAttention
from transformers import AutoModel, AutoTokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NewsEncoder(torch.nn.Module):
    def __init__(self, config):
        super(NewsEncoder, self).__init__()
        self.config = config
        self.PhoBERT = AutoModel.from_pretrained(config.encoder_pretrained_path)

    def forward(self, news):
        final_news_vector = self.PhoBERT(news['input_ids'],token_type_ids=None, attention_mask = news['attention_mask'])
        return final_news_vector.pooler_output
    
if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    
    text = 'tôi không thích'
    text = tokenizer(text, padding='max_length', truncation=True, max_length = 256, return_tensors="pt")
    config = {'encoder_pretrained_path': 'vinai/phobert-base'}
    news_encoder = NewsEncoder(config).to(device)
    
    print('text', text)
    print(news_encoder)
    for key in text:
        text[key] = text[key].to(device)
    output = news_encoder(text)
    print(output)
    print(output.pooler_output.shape)
    

