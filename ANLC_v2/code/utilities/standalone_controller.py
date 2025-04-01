import torch
import torch.nn as nn

class ExtractedController(nn.Module):
    def __init__(self, state_dim, control_dim, hidden_dims, activations, 
                 use_saturation=False, ctrl_sat=1.0, beta_sfpl=1.0):
        super(ExtractedController, self).__init__()
        
        # control layers
        self.ctrl_layers = nn.ModuleList()
        self.ctrl_activs = activations
        
        if len(hidden_dims) > 0:
            self.ctrl_layers.append(nn.Linear(state_dim, hidden_dims[0]))

            for i in range(len(hidden_dims)-1):
                self.ctrl_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                
            self.ctrl_layers.append(nn.Linear(hidden_dims[-1], control_dim))
        else:
            # implementing this currently
            self.ctrl_layers.append(nn.Linear(state_dim, control_dim))
        
        # Saturation parameters
        self.use_saturation = use_saturation
        self.ctrl_sat = ctrl_sat
        self.beta_sfpl = beta_sfpl
    
    def forward(self, x):
        sfpl = torch.nn.Softplus(beta=self.beta_sfpl, threshold=50)
        
        # Apply control network
        u = x.detach().clone()
        for idx in range(len(self.ctrl_layers)):
            uhat = self.ctrl_layers[idx](u)
            # activation
            if idx < len(self.ctrl_activs):
                if self.ctrl_activs[idx] == 'tanh':
                    u = torch.tanh(uhat)
                elif self.ctrl_activs[idx] == 'pow2':
                    u = torch.pow(uhat, 2)
                elif self.ctrl_activs[idx] == 'sfpl':
                    u = sfpl(uhat)
                elif self.ctrl_activs[idx] == 'linear':
                    u = uhat
                else:
                    raise ValueError(f'Not Implemented Activation Function {self.ctrl_activs[idx]}.')
            else:
                u = uhat
                
        # Apply saturation if needed
        if self.use_saturation:
            u = torch.tensor(self.ctrl_sat) * torch.tanh(u)
            
        return u


def extract_controller_from_model(full_model, state_dim, control_dim):
    # Create the extracted controller with the same architecture
    hidden_dims = [] # for linear controller
    
    if full_model.use_lin_ctr:
        # Extract from linear controller
        controller = ExtractedController(
            state_dim=state_dim,
            control_dim=control_dim,
            hidden_dims=hidden_dims,  # No hidden layers for linear controller
            activations=['linear'],
            use_saturation=full_model.use_saturation,
            ctrl_sat=full_model.ctrl_sat,
            beta_sfpl=full_model.beta_sfpl
        )
        
        # Copy the parameters from the linear controller
        controller.ctrl_layers[0].weight.data = full_model.control.weight.data.clone()
        controller.ctrl_layers[0].bias.data = full_model.control.bias.data.clone()
    else:
        # Extract from nonlinear controller
        # Get the architecture from ctrl_layers
        if len(full_model.ctrl_layers) > 1:
            for i in range(len(full_model.ctrl_layers) - 1):
                hidden_dims.append(full_model.ctrl_layers[i].out_features)
        
        controller = ExtractedController(
            state_dim=state_dim,
            control_dim=control_dim,
            hidden_dims=hidden_dims,
            activations=full_model.ctrl_activs,
            use_saturation=full_model.use_saturation,
            ctrl_sat=full_model.ctrl_sat,
            beta_sfpl=full_model.beta_sfpl
        )
        
        # Copy parameters
        for i, layer in enumerate(full_model.ctrl_layers):
            controller.ctrl_layers[i].weight.data = layer.weight.data.clone()
            controller.ctrl_layers[i].bias.data = layer.bias.data.clone()
    
    return controller
