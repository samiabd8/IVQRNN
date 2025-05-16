import os
import uuid
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.quantile_regression import QuantReg
import time 
import warnings
warnings.filterwarnings("ignore", message="Maximum number of iterations")

N_OBS = 10000
N_TAUS = 9
OUTPUT_DIR = "results_ht"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HybridActivation(nn.Module):
    def __init__(self, m: int = 1):
        super().__init__()
        self.m = m
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.sigmoid(x)*x 

class QuantileSemiparametricNN(nn.Module):
    def __init__(self, input_dim: int, sample_size: int, dhat_index: int, q: float = 0.5, m: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.m = m
        self.q = q
        self.dhat_index = dhat_index
        
        d = input_dim - 1  
        alpha = 1.0
        r_n = int((sample_size / np.log(sample_size)) ** (1/(2*(1 + alpha/(d+1))))*10)
        self.hidden_dim = max(r_n, 1)
        self.B_n = np.sqrt(sample_size / np.log(sample_size))
        
        self.fc_nonparametric = nn.Linear(input_dim - 1, self.hidden_dim)
        self.activation = HybridActivation(m)
        self.fc_final = nn.Linear(self.hidden_dim, 1)
        self.dhat_coefficient = nn.Parameter(torch.tensor(0.0))
        self._initialize_weights()

    def _initialize_weights(self):
        W = np.random.randn(self.input_dim - 1, self.hidden_dim)
        norms = np.maximum(np.linalg.norm(W, axis=0), 1.0)
        W = W / (norms[None, :] ** self.m)
        scale = self.B_n / np.sum(np.abs(W))
        W = W * min(scale, 1.0)
        self.fc_nonparametric.weight.data = torch.tensor(W.T, dtype=torch.float32)
        self.fc_nonparametric.bias.data.zero_()
        
        v = np.random.randn(self.hidden_dim, 1)
        scale = self.B_n / np.sum(np.abs(v))
        v = v * min(scale, 1.0)
        self.fc_final.weight.data = torch.tensor(v.T, dtype=torch.float32)
        self.fc_final.bias.data.zero_()

    def forward(self, x):
        dhat = x[:, self.dhat_index]
        nonparametric_input = x[:, [i for i in range(x.size(1)) if i != self.dhat_index]]
        x_nonparametric = self.activation(self.fc_nonparametric(nonparametric_input))
        output_nonparametric = self.fc_final(x_nonparametric)
        linear_component = self.dhat_coefficient * dhat.unsqueeze(1)
        return output_nonparametric + linear_component

class QuantileLoss(nn.Module):
    def __init__(self, q):
        super().__init__()
        self.q = q
      
    def forward(self, preds, target):
        errors = target - preds
        return torch.mean(torch.max(self.q * errors, (self.q - 1) * errors))

def gen_ivqr_eg_mixed3(n=10**4, k=10, m=0.6, s=5, nonlinear=True, seed=None):
    if seed is not None:
        np.random.seed(seed)
    u = np.random.uniform(size=n)
    x_columns = []
    num_binomial = int(k * m)
    num_normal = k - num_binomial

    for _ in range(num_normal):
        x_columns.append(np.random.normal(size=n))
    for _ in range(num_binomial):
        x_columns.append(np.random.binomial(n=1, p=0.5, size=n))
    
    x = np.column_stack(x_columns)
    z = np.random.binomial(n=1, p=0.7, size=n)
    v = np.random.normal(size=n)
    d = z * (u > 0.5*v)   
    alpha_true = 1.0 * (u - 0.5) 
    beta_lin = np.linspace(0.2, 0.8, num_normal)
    beta_bin = np.random.uniform(-0.5, 0.5, num_binomial)
    y =  alpha_true*d  
    x_normal = x[:, :num_normal]
    if nonlinear:
        y += np.sum(beta_lin*x_normal, axis=1) + np.sum(np.sin(x_normal),axis=1)
    else:
        y += np.sum(beta_lin*x_normal, axis=1)   
    #y += np.sum(beta_bin * x[:, num_normal:], axis=1)
    y += norm.ppf(u, 0, 1) 
    
    return pd.DataFrame({'y': y, 'Z': z, 'D': d, 
                        **{f'x{i+1}': x[:,i] for i in range(k)}})

def prepare_ivqr_data(sim_data, val_size=0.1, test_size=0.2):
    y = sim_data['y'].values
    D = sim_data['D'].values
    X = sim_data.filter(like='x').values
    Z = sim_data['Z'].values
    
    X_int = np.column_stack((X, np.ones(len(y))))
    dhat = np.column_stack((Z, X)) @ np.linalg.pinv(np.column_stack((Z, X)).T @ np.column_stack((Z, X))) @ np.column_stack((Z, X)).T @ D
    
    y = StandardScaler(with_std=False).fit_transform(y.reshape(-1, 1)).flatten()
    dhat = StandardScaler(with_std=True).fit_transform(dhat.reshape(-1, 1)).flatten()
    X = StandardScaler().fit_transform(X)
    
    df = pd.DataFrame({'y': y, 'D': D, 'Z': Z, **{f'X{i}': X[:,i] for i in range(X.shape[1])}})
    train_val_df, test_df = train_test_split(df, test_size=test_size)
    train_df, val_df = train_test_split(train_val_df, test_size=val_size/(1 - test_size))
    
    return {
        'train': {
            'y': train_df['y'].values,
            'D': train_df['D'].values,
            'Z': train_df['Z'].values,
            'X': train_df.filter(like='X').values
        },
        'val': {
            'y': val_df['y'].values,
            'D': val_df['D'].values,
            'Z': val_df['Z'].values,
            'X': val_df.filter(like='X').values
        },
        'test': {
            'y': test_df['y'].values,
            'D': test_df['D'].values,
            'Z': test_df['Z'].values,
            'X': test_df.filter(like='X').values
        }
    }

def rq(X, y, tau):
    return QuantReg(y, X).fit(q=tau, max_iter=1000).params
    
def vciqrNN(alpha, nonpar_output, y, d, x, z, tau):
    y = np.asarray(y).flatten()
    d = np.asarray(d).flatten()
    x = np.asarray(x)
    z = np.asarray(z).reshape(-1, 1) if np.ndim(z) == 1 else np.asarray(z)
    nonpar_output = np.asarray(nonpar_output).flatten()
    
    n = len(y)
    assert len(d) == n, f"d dimension mismatch: {len(d)} vs {n}"
    assert x.shape[0] == n, f"x row mismatch: {x.shape[0]} vs {n}"
    assert z.shape[0] == n, f"z row mismatch: {z.shape[0]} vs {n}"

    e = y - (d * alpha + nonpar_output)
    h = 1.364 * (2 * np.sqrt(np.pi)) ** (-1/5) * np.std(e) * n ** (-1/5) #Silverman's rule
    kernel_vals = norm.pdf(e / h).reshape(-1, 1) 
    try:
        dx = np.column_stack((d.reshape(-1, 1), x))  
        zx = np.column_stack((z, x))               
    except ValueError as e:
        raise ValueError(f"Matrix construction failed: {str(e)}")
    J = (dx.T @ (zx * kernel_vals)) / (n * h) 
    try:
        J_inv = np.linalg.pinv(J) 
    except np.linalg.LinAlgError:
        raise RuntimeError("Matrix inversion failed")
    S = (zx.T @ zx) / n  
    vc = (tau * (1 - tau)) * J_inv @ S @ J_inv.T / n
    alpha_se = np.sqrt(np.abs(vc[0, 0])) 
    return alpha_se

def fit_for_tau_nn(tau, y_train, D_train, X_train, Z_train,
                   y_val, D_val, X_val, Z_val,
                   y_test, D_test, X_test, Z_test):
    start_time = time.time()

    X_train = np.column_stack((X_train, np.ones(len(y_train))))
    X_val = np.column_stack((X_val, np.ones(len(y_val))))
    X_test = np.column_stack((X_test, np.ones(len(y_test))))
    XZ_train = np.column_stack((Z_train, X_train))  
    coeffs = np.linalg.lstsq(XZ_train, D_train, rcond=None)[0]
    dhat_train = XZ_train @ coeffs
    z_train_combined = np.column_stack((dhat_train, X_train))
    XZ_val = np.column_stack((Z_val, X_val))
    dhat_val = XZ_val @ coeffs  
    z_val_combined = np.column_stack((dhat_val, X_val))
    XZ_test = np.column_stack((Z_test, X_test))
    dhat_test = XZ_test @ coeffs
    z_test_combined = np.column_stack((dhat_test, X_test))
    
    PSI_train = np.column_stack((z_train_combined[:, 0], X_train))
    two_sqr = rq(PSI_train, y_train, tau)
    e1 = y_train - PSI_train @ two_sqr
    mu1, sigma1 = np.mean(e1), np.var(e1)
    vc1 = (tau * (1 - tau) / (norm.pdf(norm.ppf(tau))**2) * np.linalg.inv(PSI_train.T @ PSI_train))
    std_2sqr = np.sqrt(np.diag(vc1))
    alpha_range = np.linspace(two_sqr[0] - 2*std_2sqr[0], two_sqr[0] + 2*std_2sqr[0], 40)
    
    best_g = float('inf')
    best_alpha = None
    best_model = None
    
    for alpha in alpha_range:
        model = QuantileSemiparametricNN(
            input_dim=z_train_combined.shape[1],
            sample_size=len(y_train),
            dhat_index=0,
            q=tau
        ).to(device)
        
        criterion = QuantileLoss(tau)
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-7)
        
        # Convert data to tensors
        z_train_tensor = torch.tensor(z_train_combined, dtype=torch.float32).to(device)
        y_train_adj = torch.tensor(y_train - alpha * D_train, dtype=torch.float32).to(device)
        z_val_tensor = torch.tensor(z_val_combined, dtype=torch.float32).to(device)
        y_val_adj = torch.tensor(y_val - alpha * D_val, dtype=torch.float32).to(device)
        
        best_val_loss = float('inf')
        patience = 20 #int(np.log(len(y_train)))
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            outputs = model(z_train_tensor).squeeze()
            loss = criterion(outputs, y_train_adj)
            loss.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(z_val_tensor).squeeze()
                val_loss = criterion(val_outputs, y_val_adj)
                
            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss.item()
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        current_g = abs(model.dhat_coefficient.item())
        if current_g < best_g:
            best_g = current_g
            best_alpha = alpha
            best_model = model
    
    best_model.eval()
    true_alpha = 1 * (tau - 0.5)
    bias = best_alpha - true_alpha
    
    with torch.no_grad():
        z_train_tensor = torch.tensor(z_train_combined, dtype=torch.float32).to(device)
        model_output = best_model(z_train_tensor).squeeze().cpu().numpy()
        dhat_coeff = best_model.dhat_coefficient.item()
        nonpar_output = model_output - dhat_coeff * z_train_combined[:, 0]
    
    y_train = y_train.flatten()
    D_train = D_train.flatten()
    Z_train = Z_train.flatten()
    X_train = np.column_stack((X_train, np.ones(len(y_train))))
    
    alpha_se = vciqrNN(
        alpha=best_alpha,
        nonpar_output=nonpar_output,
        y=y_train,
        d=D_train,
        x=X_train,
        z=Z_train,
        tau=tau
    )
    
    criterion = QuantileLoss(tau)
    y_train_adj_best = y_train - best_alpha * D_train
    z_train_tensor = torch.tensor(z_train_combined, dtype=torch.float32).to(device)
    y_train_adj_tensor = torch.tensor(y_train_adj_best, dtype=torch.float32).to(device)
    with torch.no_grad():
        train_outputs = best_model(z_train_tensor).squeeze()
        train_loss = criterion(train_outputs, y_train_adj_tensor).item()
    y_val_adj_best = y_val - best_alpha * D_val
    z_val_tensor = torch.tensor(z_val_combined, dtype=torch.float32).to(device)
    y_val_adj_tensor = torch.tensor(y_val_adj_best, dtype=torch.float32).to(device)
    with torch.no_grad():
        val_outputs = best_model(z_val_tensor).squeeze()
        val_loss = criterion(val_outputs, y_val_adj_tensor).item()
    y_test_adj_best = y_test - best_alpha * D_test
    z_test_tensor = torch.tensor(z_test_combined, dtype=torch.float32).to(device)
    y_test_adj_tensor = torch.tensor(y_test_adj_best, dtype=torch.float32).to(device)
    with torch.no_grad():
        test_outputs = best_model(z_test_tensor).squeeze()
        test_loss = criterion(test_outputs, y_test_adj_tensor).item()
    
    return {
        'tau': tau,
        'param': best_alpha,
        'bias': bias,
        'g_value': best_g,
        'alpha_se': alpha_se,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'test_loss': test_loss
    }

def run_simulation():
    data = gen_ivqr_eg_mixed3(n=N_OBS, k=25, m=0.0, s=25, nonlinear=True)
    prepared = prepare_ivqr_data(data)
    
    results = []
    for tau in np.linspace(0.1, 0.9, N_TAUS):
        start_time = time.time()
        
        result = fit_for_tau_nn(tau,
                              prepared['train']['y'], prepared['train']['D'], 
                              prepared['train']['X'], prepared['train']['Z'],
                              prepared['val']['y'], prepared['val']['D'], 
                              prepared['val']['X'], prepared['val']['Z'],
                              prepared['test']['y'], prepared['test']['D'], 
                              prepared['test']['X'], prepared['test']['Z'])
        
        computation_time = time.time() - start_time
        
        results.append({
            'tau': result['tau'],
            'alpha': result['param'],
            'alpha_se': result['alpha_se'],
            'bias': result['bias'],
            'g_value': result['g_value'],
            'computation_time': computation_time,
            'train_loss': result['train_loss'],
            'val_loss': result['val_loss'],
            'test_loss': result['test_loss']
        })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    unique_id = uuid.uuid4().hex
    output_path = f"{OUTPUT_DIR}/sim_{unique_id}.csv"
    
    results_df = run_simulation()
    results_df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")
