import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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

def train(model, dataloader, criterion, optimizer, num_epochs, sparsity_reg, base_schedule, degree_schedule):
    for epoch in range(num_epochs):
        # Update the number of bases and degree according to the schedule
        if epoch in base_schedule:
            model.update_bases(base_schedule[epoch])
        if epoch in degree_schedule:
            model.update_degree(degree_schedule[epoch])
        
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets) + sparsity_reg * model.sparsity_loss()
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

def evaluate(model, criterion, dataloader):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_samples = 0
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
        avg_loss = total_loss / total_samples
    return avg_loss

def visualize_edge_activations(model):
    for layer_idx, layer_activations in enumerate(model.edge_activations):
        for edge_idx, activation_func in enumerate(layer_activations):
            x = torch.linspace(model.domain[0], model.domain[1], 100)
            y = activation_func(x)
            plt.plot(x.detach().numpy(), y.detach().numpy(), label=f"Layer {layer_idx}, Edge {edge_idx}")
    plt.legend()
    plt.show()

def analyze_edge_importance(model):
    edge_importance = []
    for layer_activations in model.edge_activations:
        for activation_func in layer_activations:
            importance = torch.mean(torch.abs(activation_func.coefficients)).item()
            edge_importance.append(importance)
    return edge_importance

class LinearKAN(nn.Module):
    def __init__(self, in_features, out_features, num_bases, degree):
        super(LinearKAN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_functions = nn.ModuleList([BSpline(num_bases, degree) for _ in range(in_features * out_features)])
        
    def forward(self, x):
        outputs = []
        for i in range(self.out_features):
            output = 0
            for j in range(self.in_features):
                idx = i * self.in_features + j
                output += self.activation_functions[idx](x[:, j])
            outputs.append(output)
        return torch.stack(outputs, dim=1)