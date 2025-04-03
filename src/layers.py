# PyTorch
import torch

class VariationalLinear(torch.nn.Module):
    def __init__(self, layer, sigma_param, use_posterior=False):
        super().__init__()
        self.layer = layer
        self.sigma_param = sigma_param
        self.use_posterior = use_posterior
        
    def forward(self, x):
        if self.training or self.use_posterior:
            return torch.nn.functional.linear(
                x,
                self.variational_weight,
                self.variational_bias,
            )
            
        return self.layer(x)    
    
    @property
    def variational_weight(self):
        return self.layer.weight + torch.nn.functional.softplus(self.sigma_param) * torch.randn_like(self.layer.weight).to(self.layer.weight.device)
            
    @property
    def variational_bias(self):
        return self.layer.bias + torch.nn.functional.softplus(self.sigma_param) * torch.randn_like(self.layer.bias).to(self.layer.bias.device) if self.layer.bias is not None else None