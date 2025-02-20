import torch
import math
from typing import Callable, Tuple, List, Optional, Dict, Any

class DormandPrinceSolver:
    """
    Dormand-Prince (DOPRI5) adaptive step size ODE solver for rectified flow.
    
    This is a fifth-order Runge-Kutta method with error estimation and adaptive
    step size control. It's more efficient than fixed-step methods for
    most image generation applications.
    """
    def __init__(
        self,
        velocity_fn: Callable,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        safety_factor: float = 0.9,
        min_step_size: float = 1e-5,
        max_step_size: float = 0.1,
        max_steps: int = 1000,
        device: torch.device = None
    ):
        """
        Initialize the Dormand-Prince ODE solver.
        
        Args:
            velocity_fn: Function that computes the velocity field v(t, x)
            rtol: Relative tolerance for error control
            atol: Absolute tolerance for error control
            safety_factor: Safety factor for step size adjustment (0.8-0.9 recommended)
            min_step_size: Minimum allowed step size
            max_step_size: Maximum allowed step size
            max_steps: Maximum number of steps allowed
            device: Torch device to use
        """
        self.velocity_fn = velocity_fn
        self.rtol = rtol
        self.atol = atol
        self.safety_factor = safety_factor
        self.min_step_size = min_step_size
        self.max_step_size = max_step_size
        self.max_steps = max_steps
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # DOPRI5 Butcher tableau coefficients
        self.a = torch.tensor([
            [0, 0, 0, 0, 0, 0],
            [1/5, 0, 0, 0, 0, 0],
            [3/40, 9/40, 0, 0, 0, 0],
            [44/45, -56/15, 32/9, 0, 0, 0],
            [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
        ], device=self.device)
        
        self.c = torch.tensor([0, 1/5, 3/10, 4/5, 8/9, 1, 1], device=self.device)
        
        # 5th order weights
        self.b5 = torch.tensor([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0], device=self.device)
        
        # 4th order weights (for error estimation)
        self.b4 = torch.tensor([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40], device=self.device)
        
        # Error estimation coefficients
        self.e = self.b5 - self.b4

    def _rk_step(
        self, 
        t: torch.Tensor, 
        y: torch.Tensor, 
        h: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute a single Runge-Kutta step with error estimate.
        
        Args:
            t: Current time (batch_size tensor)
            y: Current state (batch_size x state_dim tensor)
            h: Step size
            
        Returns:
            Tuple of (next state, error estimate, stages)
        """
        batch_size = y.shape[0]
        k = []
        
        # Compute the six intermediate stages
        for i in range(6):
            ti = t + self.c[i] * h
            yi = y.clone()
            
            # Add contributions from previous stages
            for j in range(i):
                yi = yi + h * self.a[i, j] * k[j]
                
            ki = self.velocity_fn(yi, ti)
            k.append(ki)
            
        # Compute final stage
        ti = t + self.c[6] * h
        yi = y.clone()
        for j in range(6):
            yi = yi + h * self.a[6, j] * k[j]
        k6 = self.velocity_fn(yi, ti)
        k.append(k6)
        
        # Compute 5th order solution
        y_next = y.clone()
        for i in range(7):
            y_next = y_next + h * self.b5[i] * k[i]
            
        # Compute error estimate (difference between 5th and 4th order solutions)
        error = torch.zeros_like(y)
        for i in range(7):
            error = error + h * self.e[i] * k[i]
            
        return y_next, error, k

    def _estimate_error_norm(
        self, 
        error: torch.Tensor, 
        y0: torch.Tensor, 
        y1: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate normalized error based on relative and absolute tolerance.
        
        Args:
            error: Raw error estimate
            y0: Current state
            y1: Next state
            
        Returns:
            Normalized error (scalar)
        """
        # Scale error based on tolerances
        scale = self.atol + self.rtol * torch.max(y0.abs(), y1.abs())
        error_norm = torch.sqrt(torch.mean(torch.pow(error / scale, 2)))
        return error_norm

    def _adapt_step_size(
        self, 
        h: float, 
        error_norm: torch.Tensor
    ) -> float:
        """
        Adapt step size based on error estimate.
        
        Args:
            h: Current step size
            error_norm: Normalized error estimate
            
        Returns:
            New step size
        """
        if error_norm == 0:
            return self.max_step_size
        
        if torch.isnan(error_norm) or torch.isinf(error_norm):
            return self.min_step_size
        
        # Fifth order method - use power 1/5 for error scaling
        exponent = -1.0 / 5.0
        factor = self.safety_factor * torch.pow(error_norm, exponent)
        
        # Restrict step size change to avoid oscillations
        factor = torch.clamp(factor, min=0.1, max=10.0)
        
        new_h = h * factor
        new_h = torch.clamp(torch.tensor(new_h), min=self.min_step_size, max=self.max_step_size).item()
        
        return new_h

    def integrate(
        self,
        y0: torch.Tensor,
        t_span: torch.Tensor,
        record_intermediate: bool = True,
        record_steps: int = 100
    ) -> Dict[str, Any]:
        """
        Integrate ODE from t_span[0] to t_span[1] with adaptive stepping.
        
        Args:
            y0: Initial state (batch_size x state_dim tensor)
            t_span: Time span [t0, t1] as tensor of shape [2]
            record_intermediate: Whether to record intermediate states
            record_steps: Number of intermediate states to record
                
        Returns:
            Dictionary containing:
                - 'y1': Final state
                - 'trajectory': Dict with 'ts' and 'ys' for intermediate states
                - 'nfe': Number of function evaluations
                - 'n_steps': Number of steps taken
                - 'status': 'success' or 'max_steps_reached'
        """
        t0, t1 = t_span[0].item(), t_span[1].item()
        assert t1 > t0, "End time must be greater than start time"
        
        # Initialize
        t = torch.ones(y0.shape[0], device=self.device) * t0
        y = y0.clone()
        
        # Initial step size estimation (conservative)
        h = torch.clamp(torch.tensor((t1 - t0) / 100), 
                      min=self.min_step_size, max=self.max_step_size).item()
        
        # For tracking
        n_steps = 0
        n_func_evals = 0
        status = 'success'
        
        # For recording trajectory
        if record_intermediate:
            record_times = torch.linspace(t0, t1, record_steps, device=self.device)
            trajectory = {
                'ts': [t0],
                'ys': [y0.detach().cpu()]
            }
            next_record_idx = 1
        
        # Main integration loop
        while t[0].item() < t1 and n_steps < self.max_steps:
            n_steps += 1
            
            # Ensure we don't overshoot the final time
            h = min(h, t1 - t[0].item())
            
            # RK step with error estimation
            y_new, error, stages = self._rk_step(t, y, h)
            n_func_evals += len(stages)
            
            # Calculate error norm
            error_norm = self._estimate_error_norm(error, y, y_new)
            
            # Step is accepted if error is within tolerance
            if error_norm <= 1.0:
                t = t + h
                y = y_new
                
                # Record intermediate state if needed
                if record_intermediate:
                    while next_record_idx < len(record_times) and t[0].item() >= record_times[next_record_idx].item():
                        trajectory['ts'].append(record_times[next_record_idx].item())
                        trajectory['ys'].append(y.detach().cpu())
                        next_record_idx += 1
            
            # Compute next step size
            h = self._adapt_step_size(h, error_norm)
            
            # Debug info
            if n_steps % 10 == 0:
                print(f"t = {t[0].item():.4f}, h = {h:.6f}, error = {error_norm.item():.6e}, " 
                      f"steps = {n_steps}, nfe = {n_func_evals}")
        
        # Check if we reached max steps
        if n_steps >= self.max_steps:
            status = 'max_steps_reached'
            
        # Ensure we record the final state
        if record_intermediate and t[0].item() < t1:
            trajectory['ts'].append(t1)
            
            # Need to take one more step to reach exactly t1
            final_h = t1 - t[0].item()
            t_final = torch.ones(y0.shape[0], device=self.device) * t1
            velocity = self.velocity_fn(y, t)
            y_final = y + final_h * velocity
            trajectory['ys'].append(y_final.detach().cpu())
            n_func_evals += 1
            y = y_final
        
        # Prepare return value
        result = {
            'y1': y,
            'nfe': n_func_evals,
            'n_steps': n_steps,
            'status': status
        }
        
        if record_intermediate:
            # Convert lists to tensors for easier processing
            trajectory['ts'] = torch.tensor(trajectory['ts'], device=self.device)
            trajectory['ys'] = torch.stack(trajectory['ys'], dim=0)
            result['trajectory'] = trajectory
            
        return result
