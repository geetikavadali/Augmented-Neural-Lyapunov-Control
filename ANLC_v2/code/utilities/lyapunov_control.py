import torch 

def extract_lyapunov_branch(model):
    """Extract the Lyapunov branch from the trained model."""
    lyapunov_branch = torch.nn.ModuleList()
    
    for idx, layer in enumerate(model.layers):
        # Create a new linear layer with the same dimensions
        extracted_layer = torch.nn.Linear(
            layer.weight.shape[1],
            layer.weight.shape[0],
            bias=model.lyap_bias[idx]
        )
        
        # Copy the weights
        extracted_layer.weight.data = layer.weight.data.clone()
        
        # Copy the bias if it exists
        if model.lyap_bias[idx]:
            extracted_layer.bias.data = layer.bias.data.clone()
            
        lyapunov_branch.append(extracted_layer)
    
    return lyapunov_branch

def extract_nonlinear_control_branch(model):
    """Extract the nonlinear control branch from the trained model."""
    control_branch = torch.nn.ModuleList()
    
    for idx, layer in enumerate(model.ctrl_layers):
        # Create a new linear layer with the same dimensions
        extracted_layer = torch.nn.Linear(
            layer.weight.shape[1],
            layer.weight.shape[0],
            bias=model.ctrl_bias[idx]
        )
        
        # Copy the weights
        extracted_layer.weight.data = layer.weight.data.clone()
        
        # Copy the bias if it exists
        if model.ctrl_bias[idx]:
            extracted_layer.bias.data = layer.bias.data.clone()
            
        control_branch.append(extracted_layer)
    
    return control_branch

def extract_linear_control(model):
    """Extract the linear control component from the trained model."""
    linear_control = torch.nn.Linear(
        model.control.weight.shape[1],
        model.control.weight.shape[0],
        bias=model.lin_contr_bias
    )
    
    # Copy the weights and bias
    linear_control.weight.data = model.control.weight.data.clone()
    if model.lin_contr_bias:
        linear_control.bias.data = model.control.bias.data.clone()
    
    return linear_control

# Usage example
def analyze_trained_model(trained_model):
    # Extract branches
    lyapunov_branch = extract_lyapunov_branch(trained_model)
    nonlinear_control_branch = extract_nonlinear_control_branch(trained_model)
    linear_control = extract_linear_control(trained_model)
    
    # Analysis example: Print architecture details
    print("Lyapunov Branch Architecture:")
    for idx, layer in enumerate(lyapunov_branch):
        print(f"  Layer {idx+1}: Input {layer.weight.shape[1]}, Output {layer.weight.shape[0]}")
        print(f"  Activation: {trained_model.activs[idx]}")
    
    print("\nNonlinear Control Branch Architecture:")
    for idx, layer in enumerate(nonlinear_control_branch):
        print(f"  Layer {idx+1}: Input {layer.weight.shape[1]}, Output {layer.weight.shape[0]}")
        print(f"  Activation: {trained_model.ctrl_activs[idx]}")
    
    print("\nLinear Control Component:")
    print(f"  Input {linear_control.weight.shape[1]}, Output {linear_control.weight.shape[0]}")
    
    # Return the extracted components for further analysis
    torch.save(lyapunov_branch, os.path.join(final_dir_run, "lyapunov_weights.pt"))
    torch.save(nonlinear_control_branch, os.path.join(final_dir_run, "nonlinear_control_weights.pt"))
  
    return {
        "lyapunov_branch": lyapunov_branch,
        "nonlinear_control_branch": nonlinear_control_branch,
        "linear_control": linear_control
    }

# You could also create a forward function to test each branch individually
def lyapunov_forward(model, x):
    """Test forward pass through only the Lyapunov branch."""
    lyap_branch = extract_lyapunov_branch(model)
    
    out = x
    for idx, layer in enumerate(lyap_branch):
        out = layer(out)
        
        # Apply the activation function
        if model.activs[idx] == 'relu':
            out = torch.nn.functional.relu(out)
        elif model.activs[idx] == 'tanh':
            out = torch.tanh(out)
        elif model.activs[idx] == 'sigmoid':
            out = torch.sigmoid(out)
        # Add other activations as needed
    
    return out

def control_forward(model, x):
    """Test forward pass through the control branches (both nonlinear and linear)."""
    nonlinear_control = extract_nonlinear_control_branch(model)
    linear_control = extract_linear_control(model)
    
    # Nonlinear control branch
    nonlin_out = x
    for idx, layer in enumerate(nonlinear_control):
        nonlin_out = layer(nonlin_out)
        
        # Apply the activation function
        if idx < len(model.ctrl_activs):
            if model.ctrl_activs[idx] == 'relu':
                nonlin_out = torch.nn.functional.relu(nonlin_out)
            elif model.ctrl_activs[idx] == 'tanh':
                nonlin_out = torch.tanh(nonlin_out)
            elif model.ctrl_activs[idx] == 'sigmoid':
                nonlin_out = torch.sigmoid(nonlin_out)
            # Add other activations as needed
    
    # Linear control branch
    lin_out = linear_control(x)
    
    # Combine according to your model's logic
    if model.use_lin_ctr:
        combined_out = nonlin_out + lin_out
    else:
        combined_out = nonlin_out
    
    # Apply saturation if needed
    if model.use_saturation and model.ctrl_sat is not None:
        combined_out = torch.clamp(combined_out, -model.ctrl_sat, model.ctrl_sat)
    
    return combined_out
