#Code for the linear IVQR model used to run simulations
import os
import uuid
import numpy as np
import pandas as pd
from scipy.stats import norm, t
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.quantile_regression import QuantReg
import warnings
import time
warnings.filterwarnings("ignore", message="Maximum number of iterations")

N_OBS = 10000
N_TAUS = 9
OUTPUT_DIR = "results_ht"

def quantile_loss(y_true, y_pred, tau):
    residuals = y_true - y_pred
    return np.mean(np.maximum(tau * residuals, (tau - 1) * residuals))

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

    y = alpha_true*d  
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

def inv_qr(y, D, X, Z, tau):
    X_ext = np.column_stack((X, np.ones(len(y))))
    dhat = np.column_stack((Z, X_ext)) @ np.linalg.pinv(np.column_stack((Z, X_ext)).T @ np.column_stack((Z, X_ext))) @ np.column_stack((Z, X_ext)).T @ D
    dhat = dhat.reshape(-1, 1)
    psi = np.column_stack((dhat, X_ext))
    beta_init = rq(psi, y, tau)
    alpha_grid = np.linspace(beta_init[0] - 2*np.std(beta_init), 
                            beta_init[0] + 2*np.std(beta_init), 40)
    
    best_g = float('inf')
    best_params = None
    
    for alpha in alpha_grid:
        beta = rq(psi, y - alpha*D, tau)
        g = np.linalg.norm(beta[:dhat.shape[1]])
        if g < best_g:
            best_g = g
            best_params = np.concatenate([[alpha], beta[dhat.shape[1]:]])
    
    return best_params, best_g

def vciqr(bhat, y, d, x, z, tau):
    n = y.shape[0]
    x = np.column_stack((x, np.ones(n)))
    k = np.column_stack((d, x)).shape[1]
    
    S = (1 / n) * np.column_stack((z, x)).T @ np.column_stack((z, x))
    e = y - np.column_stack((d, x)) @ bhat
    
    h = 1.364 * ((2 * np.sqrt(np.pi)) ** (-1/5)) * np.std(e) * n ** (-1/5)
    J = (1 / (n * h)) * (norm.pdf(e / h)[:, None] * np.column_stack((d, x))).T @ np.column_stack((z, x))
    
    vc = (1 / n) * (tau - tau**2) * np.linalg.pinv(J.T) @ S @ np.linalg.pinv(J)
    std_errors = np.sqrt(np.diag(vc))
    
    return np.column_stack((bhat, std_errors)), vc, np.linalg.inv(J)

def run_simulation():
    data = gen_ivqr_eg_mixed3(n=N_OBS, k=25, m=0, s=25, nonlinear=True)
    prepared = prepare_ivqr_data(data)
    results = []
    for tau in np.linspace(0.1, 0.9, N_TAUS):
        start_time = time.time()
        params, g = inv_qr(
            prepared['train']['y'],
            prepared['train']['D'],
            prepared['train']['X'],
            prepared['train']['Z'],
            tau
        )
        try:
            b_lin, _, _ = vciqr(params, prepared['train']['y'], prepared['train']['D'], 
                                prepared['train']['X'], prepared['train']['Z'], tau)
            alpha_se = b_lin[0, 1]
        except:
            alpha_se = np.nan
        def calculate_losses(data_split):
            X = data_split['X']
            D = data_split['D']
            y = data_split['y']
            X_ext = np.column_stack((X, np.ones(len(y))))
            y_pred = params[0] * D + X_ext @ params[1:]
            return quantile_loss(y, y_pred, tau)
        train_loss = calculate_losses(prepared['train'])
        val_loss = calculate_losses(prepared['val'])
        test_loss = calculate_losses(prepared['test'])        
        computation_time = time.time() - start_time
        
        results.append({
            'tau': tau,
            'param': params[0],
            'bias': params[0] - (tau - 0.5),
            'g': g,
            'alpha_se': alpha_se,
            'computation_time': computation_time,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_loss': test_loss
        })  
    return pd.DataFrame(results)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    unique_id = uuid.uuid4().hex
    output_path = f"{OUTPUT_DIR}/sim_{unique_id}.csv"
    
    results_df = run_simulation()
    results_df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")
