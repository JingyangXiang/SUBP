import torch
import torch.nn.functional as F


def BlockAngularRedundancy(weight: torch.Tensor, mask: torch.Tensor, alpha: float) -> torch.Tensor:
    # get l1 norm
    score_l1_norm = weight.abs().sum(dim=-1)
    score_l1_norm_normalize = score_l1_norm / score_l1_norm.sum(dim=-1, keepdim=True)

    # get cos similarity (we think zero blocks are vertical)
    weight_normalize = F.normalize(weight * mask, dim=-1)
    score_cos_similarity = torch.einsum('ijk,ink ->ijn', weight_normalize, weight_normalize).abs().sum(dim=-1)
    score_cos_similarity_normalize = score_cos_similarity / score_cos_similarity.sum(dim=-1, keepdim=True)
    score_cos_similarity_normalize = torch.where(score_cos_similarity_normalize == 0, 1, score_cos_similarity_normalize)

    # compute score
    score = score_l1_norm_normalize - alpha * score_cos_similarity_normalize
    return score
