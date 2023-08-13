import torch
import torch.nn as nn
from model.CHAMELEON.ACR import ACRModule
from model.CHAMELEON.NAR import NARModule
from model.CHAMELEON.CAR import CARModule
from model.general.click_predictor.dot_product import DotProductClickPredictor
from model.general.click_predictor.cosine_sim import CosineSim


class CHAMELEON(torch.nn.Module):
    """
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """
    def __init__(self, config):
        super(CHAMELEON, self).__init__()
        self.config = config
        self.news_encoder = ACRModule(config)
        self.user_encoder = NARModule(config)
        self.car = CARModule(config)
        # self.renceny_embedd = nn.Embedding(config.num_bins, config.recency_embedding_size)
        self.click_predictor = CosineSim()

    def forward(self, candidate_news, clicked_news):
        """
        Args:
            candidate_news:
                [
                    {
                        "title+abstract+category": batch_size * text,
                    } * (1 + K)
                ]
            clicked_news:
                [
                    {
                        "title+abstract+category": batch_size * text,
                    } * num_clicked_news_a_user
                ]
        Returns:
            click_probability: batch_size
        """
        # batch_size, 1 + K, news_embedding_dim
        candidate_news_vector = torch.stack(
            [self.get_personalized_content_vector(x) for x in candidate_news], dim=1)
        # print('candidate_news_vector', candidate_news_vector.shape)
        
        # batch_size, num_clicked_news_a_user, news_embedding_dim
        clicked_news_vector = torch.stack(
            [self.get_personalized_content_vector(x) for x in clicked_news], dim=1)
        # print('clicked_news_vector', clicked_news_vector.shape)
        
        # batch_size, news_embedding_dim
        user_vector = self.user_encoder(clicked_news_vector)
        # print('user_vector', user_vector.shape)
        
        # batch_size, 1 + K
        click_probability = self.click_predictor(candidate_news_vector,
                                                 user_vector)
        return click_probability 

    def get_article_content_vector(self, news):
        """
        Args:
            news:
                {
                    "title+abstract+category": batch_size * text
                }
        Returns:
            (shape) batch_size, news_embedding_dim
        """
        # batch_size, news_embedding_dim
        return self.news_encoder(news)
    
    def get_personalized_content_vector(self, news):
        """
        Args:
            news:
                {
                    "title+abstract+category": batch_size * text
                }
        Returns:
            (shape) batch_size, news_embedding_dim
        """
        
        article_content_embedd = self.get_article_content_vector(news)
        
        return self.car(article_content_embedd)

    def get_predicted_nextarticle_vector(self, clicked_news_vector):
        """
        Args:
            clicked_news_vector: batch_size, num_clicked_news_a_user, news_embedding_dim
        Returns:
            (shape) batch_size, news_embedding_dim
        """
        # batch_size, news_embedding_dim
        return self.user_encoder(clicked_news_vector)

    def get_prediction(self, news_vector, user_vector):
        """
        Args:
            news_vector: candidate_size, news_embedding_dim
            user_vector: news_embedding_dim
        Returns:
            click_probability: candidate_size
        """
        if len(news_vector.shape) == 2:
            news_vector = news_vector.unsqueeze(dim=0)
        # candidate_size
        return self.click_predictor(
            news_vector,
            user_vector.unsqueeze(dim=0)).squeeze(dim=0)



