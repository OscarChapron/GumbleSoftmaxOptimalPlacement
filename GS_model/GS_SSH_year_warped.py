import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torch.distributions import Bernoulli
import xarray as xr
import xrft
import numpy as np
import matplotlib.pyplot as plt
import src.data
import src.utils
from scipy.stats import multivariate_normal
import pandas as pd
import functools as ft
from collections import namedtuple
from IPython.display import Markdown, display
from omegaconf import OmegaConf
import yaml
import inspect
import hydra
import os
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.colors as colors
from matplotlib.colors import LogNorm 
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from PIL import Image
import kornia.filters as kfilts
from torch.amp import autocast, GradScaler
from scipy.ndimage import gaussian_filter
from scipy.ndimage import binary_dilation
from skimage.morphology import disk, dilation
from sklearn.linear_model import LinearRegression

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"

def optimal_interpolation_chunk(
    data_with_nans, 
    length_scale=15.,
    sigma_f=1.0,
    sigma_n=0.1,
    chunk_size=1100
):
    """
    Perform optimal interpolation (kriging) on a 2D tensor with missing data.
    Uses chunking to reduce GPU memory usage when predicting missing points.
    """
    device = data_with_nans.device
    H, W = data_with_nans.shape

    observed_indices = ~torch.isnan(data_with_nans)
    missing_indices = torch.isnan(data_with_nans)
    observed_positions = torch.nonzero(observed_indices, as_tuple=False).float()
    missing_positions = torch.nonzero(missing_indices, as_tuple=False).float()

    observed_values = data_with_nans[observed_indices].float().to(device)

    # Coordinates [x, y]
    observed_coords = observed_positions[:, [1, 0]].to(device)  # (W, H) order
    missing_coords = missing_positions[:, [1, 0]].to(device)

    # Ensure hyperparameters are tensors on the correct device
    if not isinstance(length_scale, torch.Tensor):
        length_scale = torch.tensor(length_scale, dtype=torch.float32, device=device)
    if not isinstance(sigma_f, torch.Tensor):
        sigma_f = torch.tensor(sigma_f, dtype=torch.float32, device=device)
    if not isinstance(sigma_n, torch.Tensor):
        sigma_n = torch.tensor(sigma_n, dtype=torch.float32, device=device)

    def cov_func(x1, x2, length_scale, sigma_f):
        x1 = x1.unsqueeze(1)  # [N1, 1, 2]
        x2 = x2.unsqueeze(0)  # [1, N2, 2]
        sqdist = ((x1 - x2) ** 2).sum(dim=2)  # [N1, N2]
        return sigma_f**2 * torch.exp(-0.5 / length_scale**2 * sqdist)

    # Build the NxN covariance for observed points
    K_oo = cov_func(observed_coords, observed_coords, length_scale, sigma_f)
    K_oo += sigma_n**2 * torch.eye(K_oo.size(0), device=device)

    # Cholesky
    jitter = 1e-6 * torch.eye(K_oo.size(0), device=device)
    L = torch.linalg.cholesky(K_oo + jitter)
    alpha = torch.cholesky_solve(observed_values.unsqueeze(1), L)  # Nx1

    # Predict for missing in chunks
    num_missing = missing_coords.size(0)
    pred_list = []
    for start in range(0, num_missing, chunk_size):
        end = min(start+chunk_size, num_missing)
        K_s_chunk = cov_func(observed_coords, 
                             missing_coords[start:end], 
                             length_scale, sigma_f)  # Nx(chunk_size)
        pred_chunk = K_s_chunk.t().matmul(alpha).squeeze(1)
        pred_list.append(pred_chunk)
        # Optional: free GPU memory if needed
        del K_s_chunk, pred_chunk
        torch.cuda.empty_cache()

    mean_pred = torch.cat(pred_list, dim=0)

    # Fill back into the final output
    reconstructed_data = data_with_nans.clone()
    reconstructed_data[missing_indices] = mean_pred.to(reconstructed_data.dtype)

    return reconstructed_data


def optimal_interpolation(data_with_nans, length_scale=15., sigma_f=1.0, sigma_n=0.1):
    """
    Perform optimal interpolation (kriging) on a 2D tensor with missing data (NaNs).

    Parameters:
    - data_with_nans (torch.Tensor): 2D tensor with missing values represented as NaN.
    - length_scale (float or torch.Tensor): Length scale parameter for the RBF kernel.
    - sigma_f (float or torch.Tensor): Signal variance parameter for the RBF kernel.
    - sigma_n (float or torch.Tensor): Noise variance parameter.

    Returns:
    - reconstructed_data (torch.Tensor): 2D tensor with missing values reconstructed.
    """
    device = data_with_nans.device
    H, W = data_with_nans.shape

    observed_indices = ~torch.isnan(data_with_nans)
    observed_positions = torch.nonzero(observed_indices, as_tuple=False).float()  # Positions (i, j)
    observed_values = data_with_nans[observed_indices].to(device)

    missing_indices = torch.isnan(data_with_nans)
    missing_positions = torch.nonzero(missing_indices, as_tuple=False).float()

    observed_positions_x = observed_positions[:, 1] #/ (W - 1) * 10  
    observed_positions_y = observed_positions[:, 0] #/ (H - 1) * 10  
    observed_coords = torch.stack([observed_positions_x, observed_positions_y], dim=1)

    missing_positions_x = missing_positions[:, 1] #/ (W - 1) * 10
    missing_positions_y = missing_positions[:, 0] #/ (H - 1) * 10
    missing_coords = torch.stack([missing_positions_x, missing_positions_y], dim=1)

    # Define the covariance function (RBF kernel)
    def cov_func(x1, x2, length_scale, sigma_f):
        """
        Compute the covariance matrix using the RBF kernel.
        """
        x1 = x1.unsqueeze(1)  # Shape: [N1, 1, 2]
        x2 = x2.unsqueeze(0)  # Shape: [1, N2, 2]
        sqdist = ((x1 - x2) ** 2).sum(dim=2)  # Shape: [N1, N2]
        return sigma_f ** 2 * torch.exp(-0.5 / length_scale ** 2 * sqdist)

    # Ensure hyperparameters are tensors on the correct device
    if not isinstance(length_scale, torch.Tensor):
        length_scale = torch.tensor(length_scale, device=device)
    if not isinstance(sigma_f, torch.Tensor):
        sigma_f = torch.tensor(sigma_f, device=device)
    if not isinstance(sigma_n, torch.Tensor):
        sigma_n = torch.tensor(sigma_n, device=device)

    # Compute covariance matrices
    K = cov_func(observed_coords, observed_coords, length_scale, sigma_f)
    K += sigma_n ** 2 * torch.eye(K.size(0), device=device)  # Add noise variance
    K_s = cov_func(observed_coords, missing_coords, length_scale, sigma_f)

    # Cholesky decomposition
    L = torch.linalg.cholesky(K + 1e-6 * torch.eye(K.size(0), device=device))  # Add jitter for numerical stability

    # Solve for alpha
    alpha = torch.cholesky_solve(observed_values.unsqueeze(1), L)

    # Predict mean at missing points
    mean_pred = K_s.t().matmul(alpha).squeeze()
    mean_pred = mean_pred.to(torch.float32)

    # Reconstruct the data tensor
    reconstructed_data = data_with_nans.clone()
    reconstructed_data[missing_indices] = mean_pred

    return reconstructed_data

def generate_correlated_fields(N, L, T_corr, sigma, num_fields=10, device=device,seed=41):
    """
    Generate a series of 2D fields with both spatial and temporal correlations.

    Parameters:
        N (int): Grid size (assumed to be square, NxN).
        L (float): Spatial correlation length.
        T_corr (float): Temporal correlation length.
        sigma (float): Standard deviation of the field.
        num_fields (int): Number of fields to generate (default is 10).
        device (str): Device to run the computations on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Generated fields of shape (num_fields, N, N).
    """
    torch.manual_seed(seed)
    # Define the time points for the fields
    time_points = torch.linspace(0, num_fields - 1, num_fields, device=device)
    # Compute the temporal covariance matrix
    C_temporal = torch.exp(-abs(time_points[:, None] - time_points[None, :]) / T_corr)
    # Perform Cholesky decomposition to get the temporal correlation factors
    L_chol = torch.linalg.cholesky(C_temporal)
    # Generate independent Gaussian white noise for each time point
    white_noises = torch.randn((num_fields, N, N), device=device)
    # Combine white noise using the Cholesky factors to induce temporal correlation
    temporal_correlated_noises = torch.matmul(L_chol, white_noises.view(num_fields, -1)).view(num_fields, N, N)
    # Generate 2D grid of wavenumbers for spatial correlation
    kx = torch.fft.fftfreq(N, device=device) * N
    ky = torch.fft.fftfreq(N, device=device) * N
    k = torch.sqrt(kx[:, None]**2 + ky[None, :]**2)
    cutoff_mask = (k < 20).float()  # High-frequency cutoff
    # apply - if we apply the same approach to vorticity, and then obtain 
    # stream function, 
    # Spatial covariance (Power spectrum) for Gaussian covariance
    P_k = torch.exp(-0.5 * (k * L)**3)
    P_k[0, 0] = 0.0
    P_k = P_k / torch.sum(P_k)
    # Generate fields using Fourier transform
    fields = []
    for i in range(num_fields):
        noise_ft = torch.fft.fft2(temporal_correlated_noises[i])
        field_ft = noise_ft * sigma**2 * torch.sqrt(P_k) * cutoff_mask
        field = torch.fft.ifft2(field_ft).real
        fields.append(field)
    return torch.stack(fields)

# We now perturb the fields at the same way,
def warp_field(field, dx, dy):
    """
    Warp a 2D field based on displacement fields dx and dy.
    field (torch.Tensor): Input field of shape (batch_size, height, width)
    dx (torch.Tensor): X-displacement field of shape (batch_size, height, width)
    dy (torch.Tensor): Y-displacement field of shape (batch_size, height, width)
    """
    batch_size, _, height, width = field.shape
    
    # Create base grid
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    base_grid = torch.stack((x, y), dim=-1).float()
    dx=dx.to(field.dtype)  # Adjust size as needed
    dy=dy.to(field.dtype) 
    # Add batch dimension and move to the same device as input field
    base_grid = base_grid.unsqueeze(0).repeat(batch_size,1,1,1).to(field.device)

    # Apply displacements
    sample_grid = base_grid + torch.stack((dx, dy), dim=-1)
    sample_grid[..., 0] = sample_grid[..., 0] % (width)
    sample_grid[..., 1] = sample_grid[..., 1] % (height)
    

    # Normalize grid to [-1, 1] range
    sample_grid[..., 0] = 2 * sample_grid[..., 0] / (width) - 1
    sample_grid[..., 1] = 2 * sample_grid[..., 1] / (height) - 1

    # Perform sampling
    warped_field = F.grid_sample(field, sample_grid, mode='bilinear', padding_mode='reflection', align_corners=False)
    return warped_field,torch.max(sample_grid), torch.min(sample_grid)

# Function to create a 2D Gaussian kernel
def gaussian_kernel(kernel_size=5, sigma=1.0, dtype=torch.float32, device='cuda'):
    # Create a 2D grid of coordinates (x, y)
    x = torch.arange(kernel_size, dtype=dtype, device=device) - (kernel_size - 1) / 2
    y = torch.arange(kernel_size, dtype=dtype, device=device) - (kernel_size - 1) / 2
    xx, yy = torch.meshgrid(x, y)

    # Compute the Gaussian function on the grid
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    # Normalize the kernel to ensure the sum of the weights is 1
    kernel = kernel / kernel.sum()
    return kernel

# Function to apply Gaussian smoothing to a batch of images
def apply_gaussian_smoothing(input_tensor, kernel_size=15, sigma=1.0):
    dtype = input_tensor.dtype
    device = input_tensor.device
    kernel = gaussian_kernel(kernel_size, sigma, dtype=dtype, device=device)
    
    # Reshape the kernel to be [1, 1, kernel_size, kernel_size] (for 2D convolution)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)

    # Expand the kernel to apply it across all channels and the batch
    kernel = kernel.expand(1, 1, kernel_size, kernel_size)

    padding=kernel_size//2
    # Pad the input tensor with circular padding
    padded_input = F.pad(input_tensor, (padding, padding, padding, padding), mode='reflect')

    # Apply the convolution (use groups=batch_size to apply the kernel individually per batch element)
    smoothed_tensor = F.conv2d(padded_input, kernel, groups=1)

    # Remove the extra channel dimension
    return smoothed_tensor 

class GSModel_warped(nn.Module):
    def __init__(self, time, lat, lon, rate, device):
        super(GSModel_warped, self).__init__()
        self.logits = nn.Parameter(torch.zeros((2, time, lat, lon)))
        #self.logits = torch.zeros((2, time, lat, lon))
        self.logits.data[1, :, :, :] = np.log(rate / (1 - rate))
        self.time_window, self.lat, self.lon  = time, lat, lon
        self.device = device
        self.length_scale = nn.Parameter(torch.tensor(25.0))

    def gumbel_softmax(self, logits, tau=1.0, hard=False):
        gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim=0)
        if hard:
            index = y_soft.max(dim=0, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim=0, index=index, value=1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
        return ret

    def compute_logits_loss(self, batch, budget_obs = 0.01, tau=1.0, weight=1.0, hard=True):
        logits = self.logits.to(self.device)
        batch_size = batch.tgt.size(0)
        total_loss = 0
        total_normalized_selected_points = 0
        total_loss_mse = 0
        #avg_logits = torch.zeros_like(logits)
        for i in range(batch_size):
            tgt_inp = batch.tgt[i].clone()
            #logits = self.logits.to(self.device)
            # Generate mask using Gumbel-Softmax
            gs_output = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=0)[0, :, :, :]
            mask_input = gs_output.view(self.time_window, self.lat, self.lon)
            mask_input = mask_input.unsqueeze(0)  # Add batch dimension if necessary
            normalized_selected_points = mask_input.mean()
            # Create mask for observed data
            mask = mask_input
            # Apply mask to target input
            selected_tgt_inp = tgt_inp * mask
            selected_tgt_inp[mask == 0] = float('nan')
            # Perform optimal interpolation
            output = optimal_interpolation(selected_tgt_inp[0,0], length_scale=self.length_scale)
            output = output.unsqueeze(0)
            output = output.unsqueeze(0)
            # Ensure the data types match
            output = output.to(tgt_inp.dtype)
            # Define the padding size
            padding_size = 10  # Adjust this value as needed
            center_mask = torch.zeros_like(output)
            center_mask[:,:, padding_size:-padding_size, padding_size:-padding_size] = 1
            output_center = output #* center_mask
            tgt_center = batch.tgt[i] #* center_mask
            loss_mse = F.mse_loss(output_center, tgt_center)
            if normalized_selected_points >= budget_obs:
                weight = weight
            else:
                weight = 0
            loss = loss_mse + weight*normalized_selected_points

            total_loss += loss
            total_normalized_selected_points += normalized_selected_points
            total_loss_mse += loss_mse
        avg_loss = total_loss / batch_size
        avg_normalized_selected_points = total_normalized_selected_points / batch_size
        avg_loss_mse = total_loss_mse / batch_size
        
        return output, avg_loss, mask_input, avg_normalized_selected_points, avg_loss_mse

    def forward(self, batch, budget_obs, tau, weight, n_draw):
        total_loss = 0
        total_points = 0
        total_mse = 0
        for _ in range(n_draw):
            output, loss_draw, mask, normalized_selected_points, loss_mse = self.compute_logits_loss(batch, budget_obs=budget_obs, tau=tau, weight=weight, hard=True)
            total_loss += loss_draw
            total_points += normalized_selected_points
            total_mse += loss_mse
            del output, mask
        mean_loss = total_loss / n_draw
        mean_loss = mean_loss 
        mean_points = total_points / n_draw
        mean_mse = total_mse / n_draw
        return mean_loss, mean_points, mean_mse, self.length_scale
    
n_draw = 2
n_iter = 1000
rate = 0.999
budget_obs = 1 - 0.999
accumulation_steps = 30

warmup_iterations = 50
delta_weight = 0.2
max_weight = 120.0
start_temp = 15.0
min_temp = 4.0
alpha = 0.9
lr = 1e-1

# Select the first 3 time steps to check the code
tgt_ds = xr.open_dataset('/Odyssey/public/natl60/ssh/NATL60-CJM165-ssh-2012-2013-1_20.nc')
#tgt_ds = tgt_ds.isel(time=slice(0, 3))

# Initialize lists to store results for all time steps
all_mean_loss_list = []
all_mean_points_list = []
all_mean_mse_list = []
all_length_scale_list = []
all_time_steps = []
all_mean_rmse = []

# Initialize lists to store logits and observation probabilities for all time steps
all_logits_list = []
all_obs_prob_list = []

for time_step in tgt_ds.time[::5]:
    print(time_step)
    inp_da = tgt_ds.ssh.sel(time=time_step)
    inp_da_GS = inp_da.sel(lat=slice(28, 45.), 
                           lon=slice(-66, -49.))
    border_size = 30

    lat_shape = len(inp_da_GS.lat)
    lon_shape = len(inp_da_GS.lon)

    crop_lat_start = border_size
    crop_lat_end = lat_shape - border_size
    crop_lon_start = border_size
    crop_lon_end = lon_shape - border_size

    inp_da_GS_crop = inp_da_GS.isel(lat=slice(crop_lat_start, crop_lat_end),
                                    lon=slice(crop_lon_start, crop_lon_end))

    inp_da_GS_crop = inp_da_GS_crop.fillna(0.0)
    

    # Prepare data
    mean_tgt = inp_da_GS_crop.mean().item()
    std_tgt = inp_da_GS_crop.std().item()

    TrainingItem = namedtuple('TrainingItem', ['input', 'tgt'])

    N = lat_shape  # Grid size
    L = 5.0  # Spatial correlation length
    T_corr = 1.0  # Temporal correlation length
    sigma = 35  # Standard deviation of the field
    
    batch_size = 10
    channels = 1
    
    # Generate a Gaussian random field with a high-frequency cutoff
    displacement_x = generate_correlated_fields(N, L, T_corr, sigma, num_fields=batch_size, device=device, seed=41)
    displacement_y = generate_correlated_fields(N, L, T_corr, sigma, num_fields=batch_size, device=device, seed=42)
    
    # Prepare data
    mean_tgt = inp_da_GS.mean().item()
    std_tgt = inp_da_GS.std().item()

    tens_inp = torch.from_numpy(inp_da_GS.values).float().to(device)
    tens_inp = (tens_inp - mean_tgt) / std_tgt
    tens_inp.unsqueeze_(0)

    tens_inp = tens_inp.repeat(batch_size, 1, 1, 1)
    # Create a sample field (e.g., a gradient)
    field = tens_inp

    # Perform warping
    warped_field,max_grid,min_grid = warp_field(field, displacement_x, displacement_y)

    rmses = []
    for i in range(batch_size):
        w_f = warped_field[i,0,border_size:-border_size,border_size:-border_size].cpu().numpy() 
        f =field[i,0,border_size:-border_size,border_size:-border_size].cpu().numpy()
        rmse = np.sqrt(np.mean((f - w_f) ** 2))
        rmses.append(rmse)
    mean_rmse = np.mean(rmses)
    
    inp_da_GS_warped = warped_field[:, 0, border_size:-border_size, border_size:-border_size]
    lat = inp_da_GS_warped.shape[1]
    lon = inp_da_GS_warped.shape[2]
    time = 1

    inputs = []
    targets = []

    for j in range(batch_size):
        inp = inp_da_GS_warped[j]
        # Prepare data
        mean_tgt = inp.mean().item()
        std_tgt = inp.std().item()

        tens_inp_da_warped = inp
        tens_inp_da_warped = (tens_inp_da_warped - mean_tgt) / std_tgt
        tens_inp_da_warped = tens_inp_da_warped.unsqueeze(0)
        tens_inp_da_warped = tens_inp_da_warped.unsqueeze(0)
        tens_inp_da_warped.requires_grad_(True)

        inputs.append(tens_inp_da_warped)
        targets.append(tens_inp_da_warped)

    # Stack inputs and targets to create a batch
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    batch = TrainingItem(input=inputs, tgt=targets)

    temperature = start_temp
    mean_loss_list = []
    mean_points_list = []
    mean_mse_list = []
    length_scale_list =[]

    decay_rate = (min_temp / start_temp) ** (1 / (n_iter - 1))
    # Initialize model
    model = GSModel_warped(time, lat, lon,  rate, device).to(device)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()

    # Training loop
    for k in range(n_iter):
        optimizer.zero_grad()
        total_points = 0
        total_mse = 0
        #weight = (1/1)*(k/n_iter)
        # Adjust weight based on the current iteration
        if k < warmup_iterations:
            weight = 0
        else:
            # Calculate weight incrementally after warmup
            weight = min((k - warmup_iterations + 1) * delta_weight, max_weight)
        for i in range(accumulation_steps):
            with autocast('cuda'):  # Mixed precision training
                mean_loss, normalized_selected_points, loss_mse, length_scale = model(batch, budget_obs=budget_obs, tau=temperature, n_draw=n_draw, weight = weight)
                mean_loss = mean_loss / accumulation_steps  # Normalize loss by accumulation steps
                #mean_loss = mean_loss / accumulation_steps  # Normalize loss by accumulation steps
                total_points += normalized_selected_points / accumulation_steps  # Accumulate normalized points
                total_mse += loss_mse / accumulation_steps  # Accumulate MSE loss

            scaler.scale(mean_loss).backward()

        scaler.step(optimizer)
        scaler.update()

        mean_loss_list.append(mean_loss.item())
        mean_points_list.append(total_points.item())
        mean_mse_list.append(total_mse.item())
        temperature = start_temp * (min_temp / start_temp) ** (1 / (n_iter - 1)) 
        length_scale_list.append(length_scale.detach().cpu().numpy())
        
        if (k + 1) % 100 == 0:
            print(f'Iteration {k + 1}/{n_iter}: Loss = {mean_loss.item():.4f}, Points = {total_points.item():.4f}, MSE = {total_mse.item():.4f}, Length scale = {length_scale.item():.4f}')
    # Store results for this time step
    all_mean_loss_list.append(mean_loss_list)
    all_mean_points_list.append(mean_points_list)
    all_mean_mse_list.append(mean_mse_list)
    all_length_scale_list.append(length_scale_list)
    all_mean_rmse.append(mean_rmse)
    all_time_steps.append(time_step.item())

    # Save logits and observation probabilities for this time step
    logits_da = xr.DataArray(model.logits[1, 0, :, :].detach().cpu().numpy(), dims=['lat', 'lon'], coords={'lat': inp_da_GS_crop.lat, 'lon': inp_da_GS_crop.lon}, name='logits')
    obs_prob_da = xr.DataArray(1 - 1 / (1 + np.exp(-model.logits[1, 0, :, :].detach().cpu().numpy())), dims=['lat', 'lon'], coords={'lat': inp_da_GS_crop.lat, 'lon': inp_da_GS_crop.lon}, name='obs_prob')

    all_logits_list.append(logits_da)
    all_obs_prob_list.append(obs_prob_da)

# Convert lists to xarray DataArrays
mean_loss_da = xr.DataArray(all_mean_loss_list, dims=['time', 'iteration'], coords={'time': all_time_steps, 'iteration': np.arange(n_iter)})
mean_points_da = xr.DataArray(all_mean_points_list, dims=['time', 'iteration'], coords={'time': all_time_steps, 'iteration': np.arange(n_iter)})
mean_mse_da = xr.DataArray(all_mean_mse_list, dims=['time', 'iteration'], coords={'time': all_time_steps, 'iteration': np.arange(n_iter)})
length_scale_da = xr.DataArray(all_length_scale_list, dims=['time', 'iteration'], coords={'time': all_time_steps, 'iteration': np.arange(n_iter)})
mean_rmse_da = xr.DataArray(all_mean_rmse, dims=['time'], coords={'time': all_time_steps}, name='mean_rmse')

# Combine DataArrays into a Dataset
results_ds = xr.Dataset({
    'mean_loss': mean_loss_da,
    'mean_points': mean_points_da,
    'mean_mse': mean_mse_da,
    'length_scale': length_scale_da,
    'mean_rmse_warped': mean_rmse_da
    })

# Save to a NetCDF file
results_filename = f'/Odyssey/private/ochapron/ConcreteAE/GS_model/outputs_v3/results_natl60_sigma_{sigma}.nc'
results_ds.to_netcdf(results_filename)

logits_da = xr.concat(all_logits_list, dim='time')
obs_prob_da = xr.concat(all_obs_prob_list, dim='time')

logits_filename = f'/Odyssey/private/ochapron/ConcreteAE/GS_model/outputs_v3/logits_natl60_sigma_{sigma}.nc'
obs_prob_filename = f'/Odyssey/private/ochapron/ConcreteAE/GS_model/outputs_v3/obs_prob_natl60_sigma_{sigma}.nc'

logits_da.to_netcdf(logits_filename)
obs_prob_da.to_netcdf(obs_prob_filename)
