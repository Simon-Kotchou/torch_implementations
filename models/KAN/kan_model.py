import torch
import torch.nn as nn

def bspline_basis(x, degree, knots):
    if degree == 0:
        return (knots[:-1] <= x) & (x < knots[1:])
    
    numerator_left = (x - knots[:-degree]) * bspline_basis(x, degree-1, knots[:-1])
    denominator_left = knots[degree:-1] - knots[:-degree+1]
    term_left = numerator_left / denominator_left
    
    numerator_right = (knots[degree+1:] - x) * bspline_basis(x, degree-1, knots[1:])
    denominator_right = knots[degree+1:] - knots[1:-degree]
    term_right = numerator_right / denominator_right
    
    return term_left + term_right

class BSpline(nn.Module):
    def __init__(self, num_bases, degree, domain=(0, 1)):
        super(BSpline, self).__init__()
        self.num_bases = num_bases
        self.degree = degree
        self.domain = domain
        self.coefficients = nn.Parameter(torch.randn(num_bases))
        self.knots = torch.linspace(domain[0], domain[1], num_bases + degree + 1)
        
    def forward(self, x):
        x = (x - self.domain[0]) / (self.domain[1] - self.domain[0])  # Normalize input to [0, 1]
        basis_values = bspline_basis(x, self.degree, self.knots)
        output = torch.matmul(basis_values, self.coefficients)
        return output

class KAN(nn.Module):
    def __init__(self, input_dim, hidden_widths, num_bases, degree, domain=(0, 1), sparsity_reg=1e-5):
        super(KAN, self).__init__()
        self.input_dim = input_dim
        self.hidden_widths = hidden_widths
        self.num_layers = len(hidden_widths)
        self.domain = domain
        self.sparsity_reg = sparsity_reg
        
        self.edge_activations = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                layer_input_dim = input_dim
            else:
                layer_input_dim = hidden_widths[i-1]
            
            layer_output_dim = hidden_widths[i]
            layer_activations = nn.ModuleList([BSpline(num_bases, degree, domain) for _ in range(layer_input_dim * layer_output_dim)])
            self.edge_activations.append(layer_activations)
        
        self.pruning_mask = nn.ParameterList([nn.Parameter(torch.ones(hidden_widths[i]), requires_grad=False) for i in range(self.num_layers)])
        
    def forward(self, x):
        activations = x
        for layer_idx, layer_activations in enumerate(self.edge_activations):
            layer_outputs = []
            for i in range(activations.shape[1]):
                for j in range(self.hidden_widths[layer_idx]):
                    edge_idx = i * self.hidden_widths[layer_idx] + j
                    output = layer_activations[edge_idx](activations[:, i]) * self.pruning_mask[layer_idx][j]
                    layer_outputs.append(output)
            activations = torch.stack(layer_outputs, dim=1)
            activations = activations.view(activations.shape[0], -1, self.hidden_widths[layer_idx])
            activations = torch.sum(activations, dim=1)
        return activations.squeeze()
    
    def sparsity_loss(self):
        sparsity_loss = 0
        for layer_activations in self.edge_activations:
            for activation_func in layer_activations:
                sparsity_loss += torch.mean(torch.abs(activation_func.coefficients))
        return sparsity_loss
    
    def prune_edges(self, threshold):
        for layer_activations in self.edge_activations:
            for activation_func in layer_activations:
                activation_func.coefficients.data = torch.where(torch.abs(activation_func.coefficients) > threshold,
                                                                activation_func.coefficients,
                                                                torch.zeros_like(activation_func.coefficients))
    
    def prune_nodes(self, threshold):
        for i in range(self.num_layers):
            self.pruning_mask[i].data = (torch.abs(self.pruning_mask[i]) > threshold).float()

def train(model, dataloader, criterion, optimizer, num_epochs, sparsity_reg):
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets) + sparsity_reg * model.sparsity_loss()
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")