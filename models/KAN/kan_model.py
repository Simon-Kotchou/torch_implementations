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
    
class ConvolutionalKAN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_bases, degree):
        super(ConvolutionalKAN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation_functions = nn.ModuleList([BSpline(num_bases, degree) for _ in range(in_channels * out_channels * kernel_size * kernel_size)])
        
    def forward(self, x):
        outputs = []
        for i in range(self.out_channels):
            output = torch.zeros(x.shape[0], x.shape[2] - self.kernel_size + 1, x.shape[3] - self.kernel_size + 1)
            for j in range(self.in_channels):
                for k in range(self.kernel_size):
                    for l in range(self.kernel_size):
                        idx = (i * self.in_channels + j) * self.kernel_size * self.kernel_size + k * self.kernel_size + l
                        output += self.activation_functions[idx](x[:, j, k:k+x.shape[2]-self.kernel_size+1, l:l+x.shape[3]-self.kernel_size+1])
            outputs.append(output)
        return torch.stack(outputs, dim=1)
    

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_bases, degree, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = ConvolutionalKAN(in_channels, out_channels, kernel_size=3, num_bases=num_bases, degree=degree, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = ConvolutionalKAN(out_channels, out_channels, kernel_size=3, num_bases=num_bases, degree=degree, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                ConvolutionalKAN(in_channels, out_channels, kernel_size=1, num_bases=num_bases, degree=degree, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        identity = self.shortcut(identity)
        
        out += identity
        out = self.relu(out)
        
        return out

class ResNet18KAN(nn.Module):
    def __init__(self, num_classes, num_bases, degree):
        super(ResNet18KAN, self).__init__()
        self.in_channels = 64
        
        self.conv1 = ConvolutionalKAN(3, 64, kernel_size=7, num_bases=num_bases, degree=degree, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(BasicBlock, 64, 2, num_bases, degree, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, num_bases, degree, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, num_bases, degree, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, num_bases, degree, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = LinearKAN(512, num_classes, num_bases=num_bases, degree=degree)
        
    def _make_layer(self, block, out_channels, num_blocks, num_bases, degree, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, num_bases, degree, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out