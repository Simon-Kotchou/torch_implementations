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
