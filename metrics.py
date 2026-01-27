import torch
import numpy as np
import numpy as np
from scipy.stats import pearsonr
from sklearn import linear_model
from math import sqrt
from collections import OrderedDict
from functools import partial
from scipy.spatial import distance
import os
import time
from utils import scatter 


def all_means(features, targets, device='cpu'):
    mu_c_dict = {}
    mus = scatter(features, targets, dim=0, reduce='mean')
    K = mus.shape[0]
    
    for i in range(K):
        mu_c_dict[i] = mus[i].to(device)

    mu_G = torch.mean(features, dim=0).to(device)

    return mu_c_dict, mu_G


def all_vars(features, targets, mu_c_dict, device='cpu'):
    var_c_dict = {}
    K = len(mu_c_dict)
    
    for i in range(K):
        class_features = features[targets == i]
        n_k = class_features.shape[0]
        if n_k > 0:
            mu_c = mu_c_dict[i]
            centered_features = class_features - mu_c
            var_c = torch.mean(torch.linalg.norm(centered_features, ord=2, dim=1) ** 2)
            var_c_dict[i] = var_c.to(device)
        else:
            var_c_dict[i] = torch.tensor(0.0, device=device)

    return var_c_dict

    
def sigma_W(features, mu_c_dict, targets, device='cpu'):
    N = features.shape[0]
    d = features.shape[1]
    Sigma_W = torch.zeros((d, d), device=device)

    for j in range(0, N):
        mu_c = mu_c_dict[targets[j].item()]
        centered = (features[j] - mu_c).reshape(-1, 1)
        Sigma_W += centered @ centered.T

    Sigma_W /= N

    return Sigma_W


def sigma_B(mu_c_dict, mu_G, device='cpu'):
    d = mu_G.shape[0]
    Sigma_B = torch.zeros((d, d), device=device)
    K = len(mu_c_dict)
    for i in range(0, K):
        mu_c = mu_c_dict[i]
        centered = (mu_c - mu_G).reshape(-1, 1)  # Center the class mean around the global mean
        Sigma_B += centered @ centered.T

    Sigma_B /= K

    return Sigma_B


def global_centered_features(mu_c_dict, mu_G, device='cpu'):
    d = mu_G.shape[0]
    K = len(mu_c_dict)
    H_bar = torch.zeros((d, K), device=device)
    for i in range(0, K):
        H_bar[:, i] =  mu_c_dict[i] - mu_G
    return H_bar



########################################
# NEURAL COLLAPSE METRICS
########################################


def nc1_metric(sigma_W, sigma_B, K):

    sigma_B = torch.nan_to_num(sigma_B, nan=0.0, posinf=1e6, neginf=-1e6)

    try:
        pseudo_inv_B = torch.linalg.pinv(sigma_B)
    except np.linalg.LinAlgError as e:
        if "SVD did not converge" in str(e):
            regularized_Sigma_B = sigma_B + 1e-4 * torch.eye(sigma_B.shape[0], device=sigma_B.device)
            pseudo_inv_B = torch.linalg.pinv(regularized_Sigma_B)
        else:
            raise e
    M = sigma_W @ pseudo_inv_B
    return torch.trace(M) / K


def frob_from_etf(M, K):
    M = M / torch.linalg.norm(M, ord='fro')
    ETF = 1 / np.sqrt(K-1) * (torch.eye(K) - 1/K * torch.ones((K, K)))
    ETF = ETF.to(M.device)
    res_matrix = M - ETF
    return torch.linalg.norm(res_matrix, ord='fro')


def nc2_metric(weights, K):
    M = weights @ weights.T
    return frob_from_etf(M, K)


def nc3_metric(weights, mu_c_dict, mu_G):
    K = len(mu_c_dict)
    H_bar = global_centered_features(mu_c_dict, mu_G, device=weights.device)
    M = weights @ H_bar
    return frob_from_etf(M, K)


def nc4_metric(weights, mu_G, bias):
    
    if bias is None:
        return torch.linalg.norm(weights @ mu_G, ord=2)
    else:
        # If bias is provided, compute the norm with bias
        return torch.linalg.norm(weights @ mu_G + bias, ord=2) 


def nc2_cosine_metric(mu_c_dict, mu_G):
    
    nc2_cos1 = nc2_cos2 = None
    K = len(mu_c_dict)
    
    zeros = torch.zeros((K, K), device=mu_G.device)
    H_bar = global_centered_features(mu_c_dict, mu_G, device=mu_G.device).T
    row_norms = torch.linalg.norm(H_bar, ord=2, dim=1) # Kx1
    diff_matrix = torch.abs(row_norms.unsqueeze(0) - row_norms.unsqueeze(1))
    diff_matrix = torch.triu(diff_matrix, diagonal=0)
    nc2_cos1 = torch.abs(diff_matrix - zeros).sum() / (K * (K - 1) / 2)
    
    H_tilde = H_bar / torch.linalg.norm(H_bar, ord=2, dim=1, keepdim=True)
    H_cos = H_tilde @ H_tilde.T
    cos_mat = (torch.ones((K, K), device=mu_G.device) - torch.eye(K, device=mu_G.device)) * (-1.0 / (K - 1)) + torch.eye(K, device=mu_G.device)
    
    cos_mat = torch.triu(cos_mat, diagonal=0)
    H_cos = torch.triu(H_cos, diagonal=0)
    nc2_cos2 = torch.abs(H_cos - cos_mat).sum() / (K * (K - 1) / 2)

    return nc2_cos1, nc2_cos2
    
    

########################################
# TUNNEL EFFECT METRICS
########################################


def numerical_rank(embs):
    # embs = embs.to('cpu')
    eps = 1e-3
    embs = torch.nan_to_num(embs, nan=0.0, posinf=1e6, neginf=-1e6)
    L, _ = torch.linalg.eigh(embs @ embs.T + 1e-8 * torch.eye(embs.size(0), device=embs.device))
    threshold = L.real[0] * eps
    n_rank = torch.sum(L.real > threshold)
    return n_rank


def intra_class_var(mu_c_dict, features, targets):
    var = 0
    K = len(mu_c_dict)

    class_features = {i: features[targets == i] for i in range(K)}

    for i in range(0, K):
        n_k = class_features[i].shape[0]
        if n_k > 0:
            mu_c = mu_c_dict[i]
            centered_features = class_features[i] - mu_c
            var += torch.mean(torch.linalg.norm(centered_features, ord=2, dim=1) ** 2)

    return var / K


def inter_class_var(mu_c_dict):
    means = torch.stack([mu_c_dict[k] for k in mu_c_dict])  # (K, D)
    K = means.shape[0]

    norms = (means ** 2).sum(dim=1, keepdim=True)
    dist_sq = norms + norms.T - 2 * means @ means.T

    dist_sq.fill_diagonal_(0)

    return dist_sq.sum() / (K * (K - 1))


def hsic_0(M1, M2, device='cpu'):
    n = M1.shape[0]
    H = torch.eye(n, device=device) -  1/n * torch.ones((n, n), device=device)
    K = H @ M1 @ H
    L = H @ M2 @ H
    K_vec = K.flatten()
    L_vec = L.flatten()
    
    return K_vec @ L_vec / ((n - 1)**2)


def hsic_1(M1, M2, device='cpu'):
    n = M1.shape[0]
    K = M1 - torch.diag(M1.mean(axis=0))
    L = M2 - torch.diag(M2.mean(axis=0))

    ones = torch.ones((n), device=device)
    ones_tran = ones.reshape(1, -1)
    
    first_term = torch.trace(K @ L)
    second_term = ones_tran @ K @ ones * ones_tran @ L @ ones /((n-1) * (n-2))
    third_term = 2 * ones_tran @ K @ L @ ones / (n-2)

    return (first_term + second_term - third_term) / (n*(n-3))


def cka_similarity_0(embs1, embs2, device='cpu'):
    XX = embs1 @ embs1.T
    YY = embs2 @ embs2.T
    M1 = hsic_0(XX, YY, device=device)
    M2 = hsic_0(XX, XX, device=device)
    M3 = hsic_0(YY, YY, device=device)
    return M1 / torch.sqrt(M2 * M3)


def cka_similarity_1(embs1, embs2, device='cpu'):
    XX = embs1 @ embs1.T
    YY = embs2 @ embs2.T
    M1 = hsic_1(XX, YY, device=device)
    M2 = hsic_1(XX, XX, device=device)
    M3 = hsic_1(YY, YY, device=device)
    return M1 / torch.sqrt(M2 * M3)



########################################
# INTRINSIC DIMENSION METRIC
########################################

                     
def estimate(X, fraction=0.9, verbose=False):    
    '''
        Estimates the intrinsic dimension of a system of points from
        the matrix of their distances X
        
        Args:
        X : 2-D Matrix X (n,n) where n is the number of points
        fraction : fraction of the data considered for the dimensionality
        estimation (default : fraction = 0.9)

        Returns:            
        x : log(mu)    (*)
        y : -(1-F(mu)) (*)
        reg : linear regression y ~ x structure obtained with scipy.stats.linregress
        (reg.slope is the intrinsic dimension estimate)
        r : determination coefficient of y ~ x
        pval : p-value of y ~ x
            
        (*) See cited paper for description
        
        Usage:
            
        _,_,reg,r,pval = estimate(X,fraction=0.85)
            
        The technique is described in : 
            
        "Estimating the intrinsic dimension of datasets by a 
        minimal neighborhood information"       
        Authors : Elena Facco, Maria d'Errico, Alex Rodriguez & Alessandro Laio        
        Scientific Reports 7, Article number: 12140 (2017)
        doi:10.1038/s41598-017-11873-y
    
    '''             
     
    # sort distance matrix
    Y = np.sort(X,axis=1,kind='quicksort')

    # clean data
    k1 = Y[:,1]
    k2 = Y[:,2]

    zeros = np.where(k1 == 0)[0]
    if verbose:
        print('Found n. {} elements for which r1 = 0'.format(zeros.shape[0]))
        print(zeros)

    degeneracies = np.where(k1 == k2)[0]
    if verbose:
        print('Found n. {} elements for which r1 = r2'.format(degeneracies.shape[0]))
        print(degeneracies)

    good = np.setdiff1d(np.arange(Y.shape[0]), np.array(zeros) )
    good = np.setdiff1d(good,np.array(degeneracies))
    
    if verbose:
        print('Fraction good points: {}'.format(good.shape[0]/Y.shape[0]))
    
    k1 = k1[good]
    k2 = k2[good]    
    
    # n.of points to consider for the linear regression
    npoints = int(np.floor(good.shape[0]*fraction))

    # define mu and Femp
    N = good.shape[0]
    mu = np.sort(np.divide(k2, k1), axis=None,kind='quicksort')
    Femp = (np.arange(1,N+1,dtype=np.float64) )/N
    
    # take logs (leave out the last element because 1-Femp is zero there)
    x = np.log(mu[:-2])
    y = -np.log(1 - Femp[:-2])

    # regression
    regr = linear_model.LinearRegression(fit_intercept=False)
    try:
        regr.fit(x[0:npoints,np.newaxis],y[0:npoints,np.newaxis]) 
        return regr.coef_[0][0]
    except:
        print("Error in linear regression. Check the input data.")
        return -1
    # r,pval = pearsonr(x[0:npoints], y[0:npoints])  
                    

def intrinsic_dimension(embs, fraction=0.9, verbose=False, device='cpu'):
    dist_matrix = torch.cdist(embs, embs, p=2).cpu().numpy()
    return estimate(dist_matrix, fraction=fraction, verbose=verbose)


def ncc_metric_stable(all_means, all_vars, eps=1e-12):
    keys = list(all_means.keys())

    means = torch.stack([all_means[k] for k in keys])
    vars_ = torch.stack([all_vars[k] for k in keys])

    norms = (means ** 2).sum(dim=1, keepdim=True)
    dist_sq = norms + norms.T - 2 * means @ means.T

    # Numerical stability
    dist_sq = dist_sq.clamp_min(eps)

    # Exclude self-pairs
    dist_sq.fill_diagonal_(float('inf'))

    var_sum = vars_.unsqueeze(1) + vars_.unsqueeze(0)

    return (var_sum / dist_sq).sum() / 2



#########################################
# CALCULATE ALL METRICS PER LAYER
#########################################


def process_metrics_layer(i, layer_names, layers_features_dict, targets_np, device='cpu'):
    layer_name = layer_names[i]
    features = layers_features_dict[layer_name]
    
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features, dtype=torch.float32, device=device)
    else:
        features = features.to(device)
    if not isinstance(targets_np, torch.Tensor):
        targets_np = torch.tensor(targets_np, dtype=torch.int64, device=device)
    else:
        targets_np = targets_np.to(device)

    #print(f"Processing layer {layer_name} with shape {features.shape}, targets shape {targets_np.shape}")
    #t0 = time.time()
    
    mu_c_dict, mu_G = all_means(features, targets_np, device=device)
    
    class_vars = all_vars(features, targets_np, mu_c_dict, device=device)
    
    ncc = ncc_metric_stable(mu_c_dict, class_vars)

    intr_dim = intrinsic_dimension(features, fraction=0.9, verbose=False, device=device)
    
    n_rank = numerical_rank(features)

    intra_var = intra_class_var(mu_c_dict, features, targets_np)
   
    inter_var = inter_class_var(mu_c_dict)

    # CKA metrics
    if i < len(layer_names) - 1:
        next_features = layers_features_dict[layer_names[i+1]]
        
        if not isinstance(next_features, torch.Tensor):
            next_features = torch.tensor(next_features, dtype=torch.float32, device=device)
        else:
            next_features = next_features.to(device)
        cka_0 = cka_similarity_0(features, next_features, device=device)
        cka_1 = cka_similarity_1(features, next_features, device=device)

    else:
        cka_0 = cka_1 = None

    nc2_cos1, nc2_cos2 = nc2_cosine_metric(mu_c_dict, mu_G)

    #t1 = time.time()
    #print(f"Finished processing layer {layer_name} in {t1 - t0:.4f} seconds")
    
    return layer_name, {
            'nc2_cos1': nc2_cos1,
            'nc2_cos2': nc2_cos2,
            'ncc': ncc,
            'intrinsic_dim': intr_dim,
            'numerical_rank': n_rank,
            'intra_class_var': intra_var,
            'inter_class_var': inter_var,
            'proxy_nc': intra_var / (inter_var + 1e-8),
            'cka_0': cka_0,
            'cka_1': cka_1
    }


def nc_vals(features, targets_np, model_weights_bias, device='cpu'):
    
    nc1 = nc2 = nc3 = nc4 = None

    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features, dtype=torch.float32, device=device)
    else:
        features = features.to(device)
    if not isinstance(targets_np, torch.Tensor):
        targets_np = torch.tensor(targets_np, dtype=torch.int64, device=device)
    else:
        targets_np = targets_np.to(device)
        
    mu_c_dict, mu_G = all_means(features, targets_np, device=device)
    
    K = len(mu_c_dict)
    N = features.shape[0]
    d = features.shape[1]
    
    W, bias = model_weights_bias
        
    if not isinstance(W, torch.Tensor):
        W = torch.tensor(W, dtype=torch.float64, device=device)
    else:
        W = W.to(device)
    if bias is not None and not isinstance(bias, torch.Tensor):
        bias = torch.tensor(bias, dtype=torch.float64, device=device)
    else:
        bias = bias.to(device) if bias is not None else None
        
    sigma_W_val = sigma_W(features, mu_c_dict, targets_np, device=device)
    sigma_B_val = sigma_B(mu_c_dict, mu_G, device=device)

    nc1 = nc1_metric(sigma_W_val, sigma_B_val, K)
    nc2 = nc2_metric(W, K)
    nc3 = nc3_metric(W, mu_c_dict, mu_G)
    nc4 = nc4_metric(W, mu_G, bias)

    return nc1.item(), nc2.item(), nc3.item(), nc4.item()


def calculate_all_nc_metrics(model, targets, device='cpu', tunnel_mode=False, calculate_nc=False):
    
    layers_features = model.all_embs
    layer_names = list(layers_features.keys())
    metrics_per_layer = OrderedDict()

    # Pre-convert tensors to numpy arrays
    layers_features_dict = {}
    for name, tensor in layers_features.items():
        layers_features_dict[name] = tensor.clone().detach()

    if isinstance(targets, torch.Tensor):
        targets_np = targets.clone().detach()
    elif isinstance(targets, np.ndarray):
        targets_np = np.copy(targets)
    elif isinstance(targets, list):
        targets_np = np.array(targets)
    else:
        raise ValueError("Unsupported type for targets. Must be Tensor, ndarray, or list.")

    # Pre-fetch final layer weights
    W, bias = model.get_last_layer_weights()
    W = W.clone().detach()
    bias = bias.clone().detach() if bias is not None else None
    model_weights_bias = (W, bias)

    if not tunnel_mode:
        for i in range(len(layer_names)):
            layer_name, metrics = process_metrics_layer(
                i,
                layer_names,
                layers_features_dict,
                targets_np,
                device=device
            )
            if metrics is not None:
                metrics_per_layer[layer_name] = metrics
            else:
                raise ValueError(f"Failed to compute metrics for layer {layer_name}")
    calculate_nc = False
    if calculate_nc:
        nc1, nc2, nc3, nc4 = nc_vals(layers_features_dict[layer_names[len(layer_names) - 2]], targets_np, model_weights_bias, device)
    else:
        nc1 = nc2 = nc3 = nc4 = None
    ncs = [nc1, nc2, nc3, nc4]
    
    return metrics_per_layer, ncs


def save_metrics(run_path, metrics, epoch):

    if not os.path.exists(run_path):
        os.makedirs(run_path)

    metrics_file = os.path.join(run_path, "metrics.pth")

    if os.path.exists(metrics_file):
        all_exp_metric = torch.load(metrics_file, weights_only=False)
    else:
        all_exp_metric = dict()

    all_exp_metric[epoch] = metrics

    torch.save(all_exp_metric, metrics_file)
    