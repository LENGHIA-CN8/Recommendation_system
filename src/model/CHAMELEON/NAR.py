# import sys
# sys.path.append('../')
import torch
import torch.nn as nn

class NARModule(nn.Module):
    def __init__(self, config = None):
        super(NARModule, self).__init__()
        self.lstm = nn.LSTM(input_size=768, hidden_size=255, num_layers= 1, batch_first=True)
        self.sr = nn.Sequential(
            nn.Linear(255, 768),
            nn.ReLU()
        )

    def forward(self, user_personalized_clicked_news_vector):
        """
        Args:
            user_personalized_clicked_news_vector: batch_size, num_clicked_news_a_user, news_embedding_dim
        Returns:
            (shape) batch_size, news_embedding_dim
        """
        # user_personalized_embedd = sel.car(clicked_news_vector)
        out, _ = self.lstm(user_personalized_clicked_news_vector)
        hidden_output = out[:,-1,:].contiguous()
        predicted_next_embedd = self.sr(hidden_output)
        return predicted_next_embedd
    
if __name__ == '__main__':
    input_demo = torch.randn((5, 10, 768), device = 'cuda:0')
    print(input_demo.shape)
    NAR_model = NARModule().to('cuda:0')
    out_demo = NAR_model(input_demo)
    print(out_demo.shape)
