import torch
import torch.nn as nn

class CARModule(nn.Module):
    def __init__(self, config = None):
        super(CARModule, self).__init__()
        self.FC = nn.Linear(768, 768)

    def forward(self, news_vector):
        """
        Args:
            clicked_news_vector: batch_size, num_news, news_embedding_dim
        Returns:
            (shape) batch_size, num_news, news_embedding_dim_1
        """
        user_personalized = self.FC(news_vector)
        return user_personalized
    
if __name__ == '__main__':
    input_demo = torch.randn((5, 10, 1000), device = 'cuda:0')
    print(input_demo.shape)
    CAR_model = CARModule().to('cuda:0')
    out_demo = CAR_model(input_demo)
    print(out_demo.shape)