import torch
import torch.nn as nn
import numpy as np

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. System & AS Rule Functions
# ==========================================

def simplified_as_rule(h, g, lam, c1, c2):
    """
    Implements the simplified AS rule using exponential approximation.
    s* = arg min {c1*exp(-c2*h_i) + lam*g_i} for i in {0, 1, ..., L}
    h_0 = 0, g_0 = 0 (virtual antenna for zero transmit power)
    """
    batch_size, L = h.shape
    
    # Virtual antenna i=0: h_0 = 0, g_0 = 0
    # metric_0 = c1 * exp(0) + lam * 0 = c1
    metric_0 = torch.full((batch_size, 1), c1, device=h.device)
    
    # Metrics for antennas i=1 to L
    metric_L = c1 * torch.exp(-c2 * h) + lam.unsqueeze(1) * g
    
    # Concatenate and find the index of the minimum metric
    metrics = torch.cat([metric_0, metric_L], dim=1)
    s_star = torch.argmin(metrics, dim=1)
    return s_star

def compute_average_interference(lam, E_s_linear, mu_h, mu_g, L, c1, c2, num_samples=10000):
    """
    Simulates average interference for a given lambda using Monte Carlo.
    """
    # Generate Rayleigh fading channel power gains (Exponential distribution)
    h = torch.empty(num_samples, L).exponential_(1.0 / mu_h)
    g = torch.empty(num_samples, L).exponential_(1.0 / mu_g)
    
    lam_tensor = torch.full((num_samples,), lam)
    s_star = simplified_as_rule(h, g, lam_tensor, c1, c2)
    
    # Extract g_{s*}. If s* == 0, g_{s*} = 0
    g_selected = torch.zeros(num_samples)
    mask = s_star > 0
    g_selected[mask] = g[mask, s_star[mask] - 1]
    
    I_avg = E_s_linear * torch.mean(g_selected).item()
    return I_avg

def find_lambda_line_search(I_ave_linear, E_s_linear, mu_h, mu_g, L, c1, c2, tol=1e-3, max_iter=50):
    """
    Finds optimal lambda using binary search such that I_avg <= I_ave.
    """
    # Check if unconstrained transmission is already below threshold
    if compute_average_interference(0.0, E_s_linear, mu_h, mu_g, L, c1, c2) <= I_ave_linear:
        return 0.0
        
    lam_low = 0.0
    lam_high = 100.0 # Initial upper bound
    
    # Expand upper bound if necessary
    while compute_average_interference(lam_high, E_s_linear, mu_h, mu_g, L, c1, c2) > I_ave_linear:
        lam_high *= 2.0
        
    # Binary Search
    for _ in range(max_iter):
        lam_mid = (lam_low + lam_high) / 2.0
        I_mid = compute_average_interference(lam_mid, E_s_linear, mu_h, mu_g, L, c1, c2)
        
        if abs(I_mid - I_ave_linear) < tol:
            break
        elif I_mid > I_ave_linear:
            lam_low = lam_mid
        else:
            lam_high = lam_mid
            
    return (lam_low + lam_high) / 2.0

# ==========================================
# 2. Deep Neural Network Models
# ==========================================

class ClassificationDNN(nn.Module):
    """
    Part A: DNN for Classification
    Input: 2L + 1 (h, g, I_ave)
    Output: L + 1 (Antenna index probabilities)
    """
    def __init__(self, L):
        super(ClassificationDNN, self).__init__()
        input_dim = 2 * L + 1
        output_dim = L + 1
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Linear(48, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.Linear(12, output_dim)
            # Note: CrossEntropyLoss applies Softmax internally
        )

    def forward(self, x):
        return self.net(x)


class RegressionDNN(nn.Module):
    """
    Part B: DNN for Regression
    Input: 6 features (E_s, I_ave, M, L, mu_h_sigma2, mu_g_sigma2)
    Output: 1 (Predicted Lambda)
    """
    def __init__(self):
        super(RegressionDNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 48),
            nn.ReLU(),
            nn.Linear(48, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, 1) # Linear activation for regression
        )

    def forward(self, x):
        return self.net(x)