def initialize_torch_weights_apply_fn(module):
    import torch.nn
    if hasattr(module, 'weight') and isinstance(module.weight, torch.nn.parameter.Parameter):
        torch.nn.init.xavier_uniform_(module.weight)
