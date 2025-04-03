# PyTorch
import torch
# Importing our custom module(s)
import layers

def add_variational_layers(module, sigma_param):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            setattr(module, name, layers.VariationalLinear(child, sigma_param))
        else:
            add_variational_layers(child, sigma_param)
            
def use_posterior(self, flag):
    for child in self.modules():
        if isinstance(child, layers.VariationalLinear):
            child.use_posterior = flag
            
def flatten_params(model, excluded_params=['lengthscale_param', 'noise_param', 'outputscale_param', 'sigma_param']):
    return torch.cat([param.view(-1) for name, param in model.named_parameters() if param.requires_grad and name not in excluded_params])
