import torch
import numpy as np
import matplotlib.pyplot as plt
from modules import RegressionDNN, simplified_as_rule, find_lambda_line_search

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

def calculate_sep(h, c1, c2):
    """Calculates instantaneous SEP based on exponential approximation."""
    return c1 * torch.exp(-c2 * h)

def generate_plots():
    print("Generating Evaluation Plots...")
    # Load Model
    reg_model = RegressionDNN()
    reg_model.load_state_dict(torch.load("regression_model.pth"))
    reg_model.eval()

    # Base Parameters
    M = 4
    c1, c2 = 0.5, 0.6
    mu_g, mu_h = 7.0, 9.0
    I_ave_db_range = np.arange(-20, 21, 2)
    num_eval_samples = 10000

    results = {}

    for E_s_db in [25, 30]:
        for L in [2, 4]:
            if L == 2 and E_s_db == 25: continue # Skip combination not needed for plots
            
            E_s_linear = 10**((E_s_db - 30) / 10)
            sep_ml, int_ml = [], []
            sep_opt, int_opt = [], []

            for I_ave_db in I_ave_db_range:
                I_ave_linear = 10**(I_ave_db/10)
                
                # 1. Conventional Approach (True Lambda)
                lam_opt = find_lambda_line_search(I_ave_linear, E_s_linear, mu_h, mu_g, L, c1, c2)
                
                # 2. ML Approach (Predicted Lambda)
                input_features = torch.tensor([[E_s_db, I_ave_db, M, L, mu_h, mu_g]], dtype=torch.float32)
                lam_pred = reg_model(input_features).item()
                lam_pred = max(0, lam_pred) # Lambda must be non-negative

                # Simulate Channel
                h = torch.empty(num_eval_samples, L).exponential_(1.0 / mu_h)
                g = torch.empty(num_eval_samples, L).exponential_(1.0 / mu_g)
                
                # Apply rules
                s_star_opt = simplified_as_rule(h, g, torch.full((num_eval_samples,), lam_opt), c1, c2)
                s_star_ml = simplified_as_rule(h, g, torch.full((num_eval_samples,), lam_pred), c1, c2)

                # Calculate SEP & Interference (ML)
                h_sel_ml = torch.zeros(num_eval_samples)
                g_sel_ml = torch.zeros(num_eval_samples)
                mask_ml = s_star_ml > 0
                h_sel_ml[mask_ml] = h[mask_ml, s_star_ml[mask_ml] - 1]
                g_sel_ml[mask_ml] = g[mask_ml, s_star_ml[mask_ml] - 1]
                
                sep_ml.append(torch.mean(calculate_sep(h_sel_ml, c1, c2)).item())
                int_ml.append(E_s_linear * torch.mean(g_sel_ml).item())
                
                # Calculate SEP & Interference (Opt)
                h_sel_opt = torch.zeros(num_eval_samples)
                mask_opt = s_star_opt > 0
                h_sel_opt[mask_opt] = h[mask_opt, s_star_opt[mask_opt] - 1]
                sep_opt.append(torch.mean(calculate_sep(h_sel_opt, c1, c2)).item())

            results[(E_s_db, L)] = {'sep_ml': sep_ml, 'sep_opt': sep_opt, 'int_ml': int_ml}

    # Plot 1: SEP vs I_ave (E_s = 25, 30; L=4)
    
    plt.figure(figsize=(8, 6))
    plt.plot(I_ave_db_range, results[(30, 4)]['sep_opt'], label='Conventional (Es=30)', linestyle='-')
    plt.plot(I_ave_db_range, results[(30, 4)]['sep_ml'], label='Model Based (Es=30)', marker='d', linestyle='none')
    plt.plot(I_ave_db_range, results[(25, 4)]['sep_opt'], label='Conventional (Es=25)', linestyle='-')
    plt.plot(I_ave_db_range, results[(25, 4)]['sep_ml'], label='Model Based (Es=25)', marker='d', linestyle='none')
    plt.yscale('log')
    plt.xlabel('I_ave (dB)')
    plt.ylabel('Symbol Error Probability (SEP)')
    plt.title('Plot 1: SEP vs I_ave for E_s = 25 and 30 dBm')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig('plot1_sep_es.png')

    # Plot 2: Avg Interference vs I_ave (E_s = 25, 30; L=4)
    
    plt.figure(figsize=(8, 6))
    plt.plot(I_ave_db_range, 10**(I_ave_db_range/10), label='Interference Constraint', color='black', linestyle='--')
    plt.plot(I_ave_db_range, results[(30, 4)]['int_ml'], label='Model Based (Es=30)', marker='o')
    plt.plot(I_ave_db_range, results[(25, 4)]['int_ml'], label='Model Based (Es=25)', marker='s')
    plt.yscale('log')
    plt.xlabel('I_ave (dB)')
    plt.ylabel('Average Interference (Linear)')
    plt.title('Plot 2: Average Interference vs I_ave')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig('plot2_interference.png')

   
    # Plot 3: SEP vs I_ave (L = 2, 4; E_s=30)
   
    plt.figure(figsize=(8, 6))
    plt.plot(I_ave_db_range, results[(30, 4)]['sep_opt'], label='Conventional (L=4)', linestyle='-')
    plt.plot(I_ave_db_range, results[(30, 4)]['sep_ml'], label='Model Based (L=4)', marker='d', linestyle='none')
    plt.plot(I_ave_db_range, results[(30, 2)]['sep_opt'], label='Conventional (L=2)', linestyle='-')
    plt.plot(I_ave_db_range, results[(30, 2)]['sep_ml'], label='Model Based (L=2)', marker='d', linestyle='none')
    plt.yscale('log')
    plt.xlabel('I_ave (dB)')
    plt.ylabel('Symbol Error Probability (SEP)')
    plt.title('Plot 3: SEP vs I_ave for L = 2 and 4')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig('plot3_sep_L.png')
    
    print("Plots saved successfully as PNG files.")

if __name__ == "__main__":
    generate_plots() 
