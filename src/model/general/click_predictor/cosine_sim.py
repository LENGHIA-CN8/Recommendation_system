import torch


class CosineSim(torch.nn.Module):
    def __init__(self):
        super(CosineSim, self).__init__()

    def forward(self, candidate_news_vector, user_vector):
        """
        Args:
            candidate_news_vector: batch_size, candidate_size, X
            user_vector: batch_size, X
        Returns:
            (shape): batch_size
        """
        # batch_size, candidate_size
        probability = torch.bmm(candidate_news_vector,
                                user_vector.unsqueeze(dim=-1)).squeeze(dim=-1)
        
        norm_candidate = torch.norm(candidate_news_vector, p=2, dim=-1)
        norm_user = torch.norm(user_vector, p=2, dim=-1)
        
        # Add an additional dimension to norm_user for broadcasting
        norm_user = norm_user.unsqueeze(dim=-1)

        # Compute cosine similarity
        cosine_similarity = probability / (norm_candidate * norm_user)


        return cosine_similarity
