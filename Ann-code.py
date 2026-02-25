import numpy as np

def tanh(x):
    """Hyperbolic Tangent activation function."""
    return np.tanh(x)

def run_ann_assignment():
    # 1. Define inputs (using the same inputs from the example: 0.05, 0.10)
    inputs = np.array([0.05, 0.10])
    
    # 2. Choose weights randomly from interval [-0.5, 0.5]
    # We need weights for:
    # Input to Hidden (H3, H4): 4 weights (w13, w23, w14, w24)
    # Hidden to Output (O1, O2): 4 weights (w31, w41, w32, w42)
    weights_input_hidden = np.random.uniform(-0.5, 0.5, (2, 2))
    weights_hidden_output = np.random.uniform(-0.5, 0.5, (2, 2))
    
    # 3. Biases b1, b2 = 0.5, 0.7 respectively
    bias_hidden = 0.5
    bias_output = 0.7
    
    print("--- ANN Assignment Task ---")
    print(f"Inputs: {inputs}")
    print(f"Random Weights (Input to Hidden):\n{weights_input_hidden}")
    print(f"Random Weights (Hidden to Output):\n{weights_hidden_output}")
    print(f"Biases: b1={bias_hidden}, b2={bias_output}")
    print("-" * 30)
    
    # Forward Pass: Input to Hidden Layer
    # net_h = inputs * weights + bias
    net_hidden = np.dot(inputs, weights_input_hidden) + bias_hidden
    out_hidden = tanh(net_hidden)
    
    print(f"Hidden Layer Net Inputs: {net_hidden}")
    print(f"Hidden Layer Outputs (after tanh): {out_hidden}")
    
    # Forward Pass: Hidden to Output Layer
    net_output = np.dot(out_hidden, weights_hidden_output) + bias_output
    out_output = tanh(net_output)
    
    print("-" * 30)
    print(f"Output Layer Net Inputs: {net_output}")
    print(f"Final Network Output: {out_output}")

if __name__ == "__main__":
    # Set seed for reproducibility if needed, or leave random as requested
    # np.random.seed(42) 
    run_ann_assignment()
