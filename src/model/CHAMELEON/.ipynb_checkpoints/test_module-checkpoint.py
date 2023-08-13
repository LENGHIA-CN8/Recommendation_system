import sys
sys.path.append('../../')
from config import CHAMELEONConfig
from model.CHAMELEON import CHAMELEON
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    print('go')
    config = CHAMELEONConfig()
    model = CHAMELEON(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    candidate_news = [
        { "text" : ['tôi kh thích', 'mùa giải này khó khăn đấy', 'harry kane sẽ về tay ai']} * 3
    ]
    print(candidate_news)
    print(model)
    