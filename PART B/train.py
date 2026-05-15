import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from modules import ClassificationDNN, RegressionDNN, find_lambda_line_search, simplified_as_rule

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

def train_classification_model():
    print("--- Training Part A: Classification Model ---")
    L = 4
    c1, c2 = 0.5, 0.6
    mu_h, mu_g = 9.0, 7.0 # Normalized by sigma^2
    E_s_linear = 10**((30 - 30) / 10) # 30 dBm converted to Watts (1 Watt)
    
    I_ave_db_range = np.arange(-20, 21, 5) # Smaller steps for quick run.
    num_samples_per_I = 2000 
    
    X_data, y_data = [], []
    
    for I_ave_db in I_ave_db_range:
        I_ave_linear = 10**(I_ave_db/10)
        lam = find_lambda_line_search(I_ave_linear, E_s_linear, mu_h, mu_g, L, c1, c2)
        
        h = torch.empty(num_samples_per_I, L).exponential_(1.0 / mu_h)
        g = torch.empty(num_samples_per_I, L).exponential_(1.0 / mu_g)
        lam_tensor = torch.full((num_samples_per_I,), lam)
        
        s_star = simplified_as_rule(h, g, lam_tensor, c1, c2)
        I_ave_tensor = torch.full((num_samples_per_I, 1), I_ave_db) # Input I_ave in dB for better scaling
        
        features = torch.cat([h, g, I_ave_tensor], dim=1)
        X_data.append(features)
        y_data.append(s_star)

    X_data = torch.cat(X_data, dim=0)
    y_data = torch.cat(y_data, dim=0)
    
    # Train/Test Split (75/25)
    split_idx = int(0.75 * len(X_data))
    train_dataset = TensorDataset(X_data[:split_idx], y_data[:split_idx])
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    model = ClassificationDNN(L=L)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(50): # 50 epochs for demonstration (paper uses 1000)
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/50, Loss: {epoch_loss/len(train_loader):.4f}")
            
    torch.save(model.state_dict(), "classification_model.pth")
    print("Classification model saved.")

def train_regression_model():
    print("\n--- Training Part B: Regression Model ---")
    c1, c2 = 0.5, 0.6
    
    # Parameter grid for generalized dataset
    L_vals = [2, 4]
    M_vals = [4]
    mu_h_vals = [9.0]
    mu_g_vals = [7.0]
    Es_vals = [25, 30] # dBm
    I_ave_vals = np.arange(-20, 21, 2)
    
    X_data, y_data = [], []
    
    for L in L_vals:
        for M in M_vals:
            for mu_h in mu_h_vals:
                for mu_g in mu_g_vals:
                    for Es_db in Es_vals:
                        Es_linear = 10**((Es_db - 30) / 10)
                        for I_ave_db in I_ave_vals:
                            I_ave_linear = 10**(I_ave_db/10)
                            lam = find_lambda_line_search(I_ave_linear, Es_linear, mu_h, mu_g, L, c1, c2)
                            
                            features = [Es_db, I_ave_db, M, L, mu_h, mu_g]
                            X_data.append(features)
                            y_data.append([lam])
                            
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32)
    
    # Train/Test Split (80/20)
    split_idx = int(0.80 * len(X_tensor))
    train_dataset = TensorDataset(X_tensor[:split_idx], y_tensor[:split_idx])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = RegressionDNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    
    model.train()
    for epoch in range(500):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/100, MSE Loss: {epoch_loss/len(train_loader):.4f}")
            
    torch.save(model.state_dict(), "regression_model.pth")
    print("Regression model saved.")

if __name__ == "__main__":
    train_classification_model()
    train_regression_model()
