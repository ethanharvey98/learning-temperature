# PyTorch
import torch

class SINCERELoss(torch.nn.Module):
    def __init__(self, temperature_param=0.07) -> None:
        super().__init__()
        self.temperature_param = temperature_param

    def forward(self, embeds: torch.Tensor, labels: torch.tensor):
        """Supervised InfoNCE REvisited loss with cosine distance

        Args:
            embeds (torch.Tensor): (B, D) embeddings of B images normalized over D dimension.
            labels (torch.tensor): (B,) integer class labels.

        Returns:
            torch.Tensor: Scalar loss.
        """
        # calculate logits (activations) for each embeddings pair (B, B)
        # using matrix multiply instead of cosine distance function for ~10x cost reduction
        logits = embeds @ embeds.T
        logits /= torch.nn.functional.softplus(self.temperature_param)
        # determine which logits are between embeds of the same label (B, B)
        same_label = labels.unsqueeze(0) == labels.unsqueeze(1)

        # masking with -inf to get zeros in the summation for the softmax denominator
        denom_activations = torch.full_like(logits, float("-inf"))
        denom_activations[~same_label] = logits[~same_label]
        # get logsumexp of the logits between embeds of different labels for each row (B,)
        base_denom_row = torch.logsumexp(denom_activations, dim=0)
        # reshape to be (B, B) with row values equivalent, to be masked later
        base_denom = base_denom_row.unsqueeze(1).repeat((1, len(base_denom_row)))

        # get mask for numerator terms by removing comparisons between an image and itself (B, B)
        in_numer = same_label
        in_numer[torch.eye(in_numer.shape[0], dtype=bool)] = False
        # delete same_label so don't need to copy for in_numer
        del same_label
        # count numerator terms for averaging (B,)
        numer_count = in_numer.sum(dim=0)
        # numerator activations with others zeroed (B, B)
        numer_logits = torch.zeros_like(logits)
        numer_logits[in_numer] = logits[in_numer]

        # construct denominator term for each numerator via logsumexp over a stack (B, B)
        log_denom = torch.zeros_like(logits)
        log_denom[in_numer] = torch.stack(
            (numer_logits[in_numer], base_denom[in_numer]), dim=0).logsumexp(dim=0)

        # cross entropy loss of each positive pair with the logsumexp of the negative classes (B, B)
        # entries not in numerator set to 0
        ce = -1 * (numer_logits - log_denom)
        # take average over rows with entry count then average over batch
        loss = torch.sum(ce / numer_count) / ce.shape[0]
        return loss


class KappaELBoLoss(torch.nn.Module):
    
    def __init__(self, kappa, sigma_param, criterion=torch.nn.CrossEntropyLoss()):
        
        super().__init__()
        
        self.criterion = criterion
        self.kappa = kappa
        self.sigma_param = sigma_param

    def forward(self, labels, logits, params, N=1):
        
        nll = self.criterion(logits, labels)
        
        loc_diff_norm = (params**2).sum()
        tau_star = (loc_diff_norm/len(params)) + torch.nn.functional.softplus(self.sigma_param)**2
        term1 = (torch.nn.functional.softplus(self.sigma_param)**2/tau_star) * len(params)
        term2 = (1/tau_star) * loc_diff_norm
        term3 = (len(params) * torch.log(tau_star)) - (len(params) * torch.log(torch.nn.functional.softplus(self.sigma_param)**2))
        kl = (1/2) * (term1 + term2 - len(params) + term3)
        
        return {'kl': kl, 'tau_star': tau_star, 'loss': nll + (1/self.kappa) * (1/N) * kl, 'nll': nll}    
