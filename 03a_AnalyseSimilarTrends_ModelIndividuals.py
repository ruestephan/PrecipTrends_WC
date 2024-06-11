# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:49:43 2024

@author: Ruth Stephan
"""


#packages:
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import statsmodels.api as sm
import matplotlib as mpl
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import seaborn as sns



#paths:
os.getcwd()
data_path = '/mnt/share/scratch/rs1155/data/CMIP6/output_data/'
plot_path = '/mnt/share/scratch/rs1155/plots/model_individuals/'


# Set a font that supports Unicode characters
sns.set_style('ticks')
plt.rcParams['font.family'] = 'DejaVu Sans'

'''
Functions
'''

def calc_vardiff(data, var):
    """
    calculates the difference between period 2075-2100 and 2000-2025
    """
    
    #data = sdelta_ds
    #var = 'pr'
    
    var20002025 = data[var].sel(time=slice('2000', '2025')).mean(dim='time')
    var20752100 = data[var].sel(time=slice('2075', '2100')).mean(dim='time')
       
    vardiff21002025 = var20752100-var20002025
    
    #vardiff21002025.abs() >= var20002025*0.1
    
    return vardiff21002025

def calc_vardiff_season(data, var):
    """
    calculates the difference between period 2075-2100 and 2000-2025
    """
    svar20002025 = data[var].sel(time=slice('2000', '2025')).groupby('time.season').mean(dim='time')
    svar20752100 = data[var].sel(time=slice('2075', '2100')).groupby('time.season').mean(dim='time')
   
    svardiff21002025 = svar20752100-svar20002025
    
    
    return svardiff21002025


def trend_single_cell(var, time):
    
    # Remove NaN values from both arrays
    valid_mask = ~np.isnan(var)
    var_valid = var[valid_mask]
    time_valid = time[valid_mask]
    
    if var_valid.size == 0:
     # If there are no valid values, return an array of NaNs with the same shape as var
     return np.full_like(var, np.nan)
    
    # Compute the lowess 
    lowess_result1 = sm.nonparametric.lowess(var_valid, time_valid, frac=0.25)
    y_lowess = lowess_result1[:, 1]
    
    # Reconstruct the array with NaNs in the original places
    result = np.full_like(var, np.nan)
    result[valid_mask] = y_lowess
    
    return result

def trend_all_cells(data, var, start_time, end_time):
    
    #data= detrended_ds 
    #var='pr' 
    #start_time = '2000'
    #end_time = '2100'
    
    data = data.sel(time=slice(start_time, end_time))
    numeric_time = np.arange(data.time.size)

    input_vars = [data[var], numeric_time]
    
    detrend_results = xr.apply_ufunc(
        trend_single_cell,
        *input_vars,
        input_core_dims=[['time'], ['time']],
        output_dtypes=[np.float64],
        output_core_dims=[['time']],
        vectorize=True,
        keep_attrs=True
        )
    
    return detrend_results

def trend_seasons_all_cells(data, var, start_time, end_time, season):
    
    #data= detrended_ds 
    #var='pr' 
    #start_time = '2000'
    #end_time = '2100'
    #season = 'DJF'
    
    data = data.sel(time=slice(start_time, end_time))
    seasonal_data = data.where(data['time'].dt.season == season, drop=True)
    numeric_time = np.arange(seasonal_data.time.size)

    input_vars = [seasonal_data[var], numeric_time]
    
    detrend_results = xr.apply_ufunc(
        trend_single_cell,
        *input_vars,
        input_core_dims=[['time'], ['time']],
        output_dtypes=[np.float64],
        output_core_dims=[['time']],
        vectorize=True,
        keep_attrs=True
        )
    
    return detrend_results

# Get to the files that should be uses:
files_in_directory = os.listdir(data_path)
combined_files = [file for file in files_in_directory if 'combined' in file]
model_names = [filename.split('_combined.nc')[0] for filename in combined_files]    

all_models_delta = []
all_models_sdelta = []
    
model_name = model_names[0]
  
combined = xr.open_dataset(data_path + model_name + '_combined.nc')
combined = combined.sel(time=slice('2000', '2100'))

trended_ds = combined.copy()
trended_ds = trended_ds.fillna(np.nan)

DJF_trended_ds = trended_ds.where(trended_ds['time'].dt.season == 'DJF', drop=True)
MAM_trended_ds = trended_ds.where(trended_ds['time'].dt.season == 'MAM', drop=True)
JJA_trended_ds = trended_ds.where(trended_ds['time'].dt.season == 'JJA', drop=True)
SON_trended_ds = trended_ds.where(trended_ds['time'].dt.season == 'SON', drop=True)


trended_ds['tas'] = trend_all_cells(combined, 'tas', start_time='2000', end_time='2100')
trended_ds['pr'] = trend_all_cells(combined, 'pr', start_time='2000', end_time='2100')
trended_ds['evspsbl'] = trend_all_cells(combined, 'evspsbl', start_time='2000', end_time='2100')
trended_ds['mrso'] = trend_all_cells(combined, 'mrso', start_time='2000', end_time='2100')
trended_ds['mrro'] = trend_all_cells(combined, 'mrro', start_time='2000', end_time='2100')
trended_ds['PETR'] = trend_all_cells(combined, 'PETR', start_time='2000', end_time='2100')


# Plot the original and smoothed data for a specific cell
combined['mrso'].sel(lat=48, lon=2).sel(time=slice('2050', '2100')).plot(label='Original Data')
trended_ds['mrso'].sel(lat=48, lon=2).sel(time=slice('2050', '2100')).plot(label='Detrended Data')
plt.legend()
plt.show()

combined['tas'].sel(lat=48, lon=2).sel(time=slice('2050', '2100')).plot(label='Original Data')
#deseasonalized_ds['tas'].sel(lat=48, lon=2).sel(time=slice('2050', '2100')).plot(label='Deseasonalized Data')
trended_ds['tas'].sel(lat=48, lon=2).sel(time=slice('2050', '2100')).plot(label='Detrended Data')
DJF_trended_ds['tas'].sel(lat=48, lon=2).sel(time=slice('2010', '2020')).plot(label='Detrended Data')
DJF_trended_ds.time

plt.legend()
plt.show()

DJF_trended_ds['tas'] = trend_seasons_all_cells(combined_ds, 'tas', start_time='2000', end_time='2100', season='DJF')
DJF_trended_ds['pr'] = trend_seasons_all_cells(combined_ds, 'pr', start_time='2000', end_time='2100', season='DJF')
DJF_trended_ds['evspsbl'] = trend_seasons_all_cells(combined_ds, 'evspsbl', start_time='2000', end_time='2100', season='DJF')
DJF_trended_ds['mrso'] = trend_seasons_all_cells(combined_ds, 'mrso', start_time='2000', end_time='2100', season='DJF')
DJF_trended_ds['mrro'] = trend_seasons_all_cells(combined_ds, 'mrro', start_time='2000', end_time='2100', season='DJF')
DJF_trended_ds['PETR'] = trend_seasons_all_cells(combined_ds, 'PETR', start_time='2000', end_time='2100', season='DJF')

MAM_trended_ds['tas'] = trend_seasons_all_cells(combined_ds, 'tas', start_time='2000', end_time='2100', season='MAM')
MAM_trended_ds['pr'] = trend_seasons_all_cells(combined_ds, 'pr', start_time='2000', end_time='2100', season='MAM')
MAM_trended_ds['evspsbl'] = trend_seasons_all_cells(combined_ds, 'evspsbl', start_time='2000', end_time='2100', season='MAM')
MAM_trended_ds['mrso'] = trend_seasons_all_cells(combined_ds, 'mrso', start_time='2000', end_time='2100', season='MAM')
MAM_trended_ds['mrro'] = trend_seasons_all_cells(combined_ds, 'mrro', start_time='2000', end_time='2100', season='MAM')
MAM_trended_ds['PETR'] = trend_seasons_all_cells(combined_ds, 'PETR', start_time='2000', end_time='2100', season='MAM')

JJA_trended_ds['tas'] = trend_seasons_all_cells(combined_ds, 'tas', start_time='2000', end_time='2100', season='JJA')
JJA_trended_ds['pr'] = trend_seasons_all_cells(combined_ds, 'pr', start_time='2000', end_time='2100', season='JJA')
JJA_trended_ds['evspsbl'] = trend_seasons_all_cells(combined_ds, 'evspsbl', start_time='2000', end_time='2100', season='JJA')
JJA_trended_ds['mrso'] = trend_seasons_all_cells(combined_ds, 'mrso', start_time='2000', end_time='2100', season='JJA')
JJA_trended_ds['mrro'] = trend_seasons_all_cells(combined_ds, 'mrro', start_time='2000', end_time='2100', season='JJA')
JJA_trended_ds['PETR'] = trend_seasons_all_cells(combined_ds, 'PETR', start_time='2000', end_time='2100', season='JJA')

SON_trended_ds['tas'] = trend_seasons_all_cells(combined_ds, 'tas', start_time='2000', end_time='2100', season='SON')
SON_trended_ds['pr'] = trend_seasons_all_cells(combined_ds, 'pr', start_time='2000', end_time='2100', season='SON')
SON_trended_ds['evspsbl'] = trend_seasons_all_cells(combined_ds, 'evspsbl', start_time='2000', end_time='2100', season='SON')
SON_trended_ds['mrso'] = trend_seasons_all_cells(combined_ds, 'mrso', start_time='2000', end_time='2100', season='SON')
SON_trended_ds['mrro'] = trend_seasons_all_cells(combined_ds, 'mrro', start_time='2000', end_time='2100', season='SON')
SON_trended_ds['PETR'] = trend_seasons_all_cells(combined_ds, 'PETR', start_time='2000', end_time='2100', season='SON')

def plot_seasonal_longterm_trend(var, lat, lon):
    #var = 'tas'
    #lat = 48
    #lon = 2
    
    DJF_dt = DJF_trended_ds[var].sel(lat=lat, lon=lon)
    MAM_dt = MAM_trended_ds[var].sel(lat=lat, lon=lon)
    JJA_dt = JJA_trended_ds[var].sel(lat=lat, lon=lon)
    SON_dt = SON_trended_ds[var].sel(lat=lat, lon=lon)
    
    # Create numeric time arrays
    numeric_time_DJF = np.arange(DJF_dt.time.size)
    numeric_time_MAM = np.arange(MAM_dt.time.size)
    numeric_time_JJA = np.arange(JJA_dt.time.size)
    numeric_time_SON = np.arange(SON_dt.time.size)
    
    # Create the plot
    sns.set(style="ticks")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    plt.figure(figsize=(12, 8))
    
    # Plot each dataset
    plt.plot(numeric_time_DJF, DJF_dt, label='DJF')
    plt.plot(numeric_time_MAM, MAM_dt, label='MAM')
    plt.plot(numeric_time_JJA, JJA_dt, label='JJA')
    plt.plot(numeric_time_SON, SON_dt, label='SON')
    
    # Add labels and title
    plt.xlabel('Time')
    plt.ylabel(f'{var}')
    plt.title(f'{var} trend at lat={lat}, lon={lon} for Different Seasons')
    plt.legend()
    plt.savefig(f'Plots/Global/{var}_seasonal_trend.png', dpi=300)
    plt.show()

def plot_longterm_trend(lat, lon):
    #lat = 48
    #lon = 2
    
    pr_dt = trended_ds['pr'].sel(lat=lat, lon=lon)
    evspsbl_dt = trended_ds['evspsbl'].sel(lat=lat, lon=lon)
    mrro_dt = trended_ds['mrro'].sel(lat=lat, lon=lon)
    PETR_dt = trended_ds['PETR'].sel(lat=lat, lon=lon)
    
    sns.set(style="ticks")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    plt.figure(figsize=(12, 8))
    pr_dt.sel(time=slice('2000', '2100')).plot(label='Precipitation')
    evspsbl_dt.sel(time=slice('2000', '2100')).plot(label='Evapotranspiration')
    mrro_dt.sel(time=slice('2000', '2100')).plot(label='Runoff')
    PETR_dt.sel(time=slice('2000', '2100')).plot(label='Residual')
    plt.xlabel('Time')
    plt.title(f'Longterm trends at lat={lat}, lon={lon} for Different Seasons')
    plt.legend()
    plt.savefig('Plots/Global/allvars_longterm_trend.png', dpi=300)
    plt.show()

plot_longterm_trend(lat=48, lon=2)

plot_seasonal_longterm_trend(var='tas', lat=48, lon=2)
plot_seasonal_longterm_trend(var='pr', lat=48, lon=2)
plot_seasonal_longterm_trend(var='mrro', lat=48, lon=2)
plot_seasonal_longterm_trend(var='PETR', lat=48, lon=2)

#save the data:
trended_ds.to_netcdf(os.path.join(output_path, 'trended_ds.nc'))
DJF_trended_ds.to_netcdf(os.path.join(output_path, 'DJF_trended_ds.nc'))
MAM_trended_ds.to_netcdf(os.path.join(output_path, 'MAM_trended_ds.nc'))
JJA_trended_ds.to_netcdf(os.path.join(output_path, 'JJA_trended_ds.nc'))
SON_trended_ds.to_netcdf(os.path.join(output_path, 'SON_trended_ds.nc'))


# load the data:
trended_ds = xr.open_dataset(os.path.join(data_path + 'trended_ds.nc'))
DJF_trended_ds = xr.open_dataset(os.path.join(data_path + 'DJF_trended_ds.nc'))
MAM_trended_ds = xr.open_dataset(os.path.join(data_path + 'MAM_trended_ds.nc'))
JJA_trended_ds = xr.open_dataset(os.path.join(data_path + 'JJA_trended_ds.nc'))
SON_trended_ds = xr.open_dataset(os.path.join(data_path + 'SON_trended_ds.nc'))



### Calculate Correlations for those longterm-trends: 
def corr_func_single_cell(var1: np.ndarray, var2: np.ndarray) -> np.float64:
    # Remove NaN values from both arrays
    valid_mask = ~np.isnan(var1) & ~np.isnan(var2)
    var1_valid = var1[valid_mask]
    var2_valid = var2[valid_mask]
    
    #if len(var1_valid) == 0 or len(var2_valid) == 0:
    #    return np.nan
    
    # Compute the correlation coefficient
    correlation_coefficient = np.corrcoef(var1_valid, var2_valid)[0, 1]
    return correlation_coefficient


def corr_all_cells(data, var1, var2
                   #, start_time, end_time
                   ):
    
    #data= detrended_ds 
    #var1='d_pr' 
    #var2='d_evspsbl'
    
    #data = data.sel(time=slice(start_time, end_time))

    input_vars = [data[var1], data[var2]]
    
    corr_results = xr.apply_ufunc(
        corr_func_single_cell,
        *input_vars,
        input_core_dims=[['time'], ['time']],
        output_dtypes=[np.float64],
        vectorize=True,
        )
    
    cmap = mpl.colormaps.get_cmap('RdBu')
    cmap.set_bad(color='lightgray')

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 3),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    corr_results.plot.imshow(ax=axes, cmap=cmap, extend='both',
                                  center=0, vmin=-1, vmax=1,
                                  cbar_kwargs={'label': 'Pearson Correlation', 'shrink' : 0.9})
    axes.coastlines()
    axes.set_title(f'Correlation between {var1} and {var2}')
    plt.tight_layout()
    plt.savefig(f'Plots/Global/Corr_{var1}_{var2}.png', dpi=300, transparent=True)
    plt.show()
    
    return corr_results


def find_high_index(arr1, arr2, arr3):
    # Mask NaN values in all arrays
    mask_arr1 = np.isnan(arr1)
    mask_arr2 = np.isnan(arr2)
    mask_arr3 = np.isnan(arr3)
    
    # Create masked arrays to handle NaN values
    arr1_masked = np.ma.masked_array(arr1, mask_arr1)
    arr2_masked = np.ma.masked_array(arr2, mask_arr2)
    arr3_masked = np.ma.masked_array(arr3, mask_arr3)
    
    # Find the maximum value among non-NaN value
    max_arr = np.maximum(abs(arr1_masked), np.maximum(abs(arr2_masked), abs(arr3_masked)))
    
    # Determine which array has the maximum value
    index_arr = np.where(max_arr >= 0.5, 
                         np.where(max_arr == abs(arr1_masked), 1, 
                                  np.where(max_arr == abs(arr2_masked), 2, 3)), 
                         np.nan).astype(float)
    
    # Set NaN values back to NaN
    index_arr[mask_arr1] = np.nan
    index_arr[mask_arr2] = np.nan
    index_arr[mask_arr3] = np.nan
    
    return index_arr

def high_corr_allcells(data, var1, var2, var3, save_string):
    input_vars = [abs(data[var1]), abs(data[var2]), abs(data[var3])]
    
    highest_corr = xr.apply_ufunc(
        find_high_index,
        *input_vars,
        input_core_dims=[['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon']],
        output_core_dims=[['lat', 'lon']],
        dask='parallelized',
        output_dtypes=[float]
    )
    
    # Prepare barplot inset:
    colors = ['forestgreen', 'skyblue', 'darkgoldenrod']    

    PET_cells = np.sum(highest_corr== 1)
    PR_cells = np.sum(highest_corr== 2)
    PSM_cells = np.sum(highest_corr== 3)

    total_cells = PET_cells + PSM_cells +PR_cells

    perc_PET = (PET_cells / total_cells) * 100
    perc_SM = (PSM_cells / total_cells) * 100
    perc_PR = (PR_cells / total_cells) * 100

    categories = ['Cor(P,ET)', 'Cor(P,R)', 'Cor(P,SM)']
    values = [perc_PET, perc_PR, perc_SM]
        
    cmap = ListedColormap(colors)
    cmap.set_bad('lightgray')


    ## Plot:
    sns.set(style="ticks")
    plt.rcParams['font.family'] = 'DejaVu Sans'
        
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 3), subplot_kw={'projection': ccrs.PlateCarree()})        

    highest_corr.plot.imshow(ax=axes, cmap=cmap, extend='both', cbar_kwargs={'label': 'Highest Correlation'})
    axes.coastlines()

    # Create an inset for the bar plot
    left, bottom, width, height = 0.1, 0.15, 0.10, 0.19
    ax_inset = fig.add_axes([left, bottom, width, height], facecolor='white')
    
    # Plot the bar plot
    ax_inset.bar(categories, values, color=colors, edgecolor='black')
    ax_inset.set_xlabel('')
    ax_inset.set_title('No. of cells [%]')
    
    # Hide xticks and xtick labels
    ax_inset.set_yticks([0, 20, 40, 60])
    ax_inset.set_xticks([])
    ax_inset.set_xticklabels([])
    
    plt.tight_layout()
    plt.savefig(f'Plots/Global/High_Corr_{save_string}.png', dpi=300)
    plt.show()
    
    return highest_corr

corr_PET = corr_all_cells(data=trended_ds, var1='pr', var2='evspsbl')
corr_PR = corr_all_cells(data=trended_ds, var1='pr', var2='mrro')
corr_PPETR = corr_all_cells(data=trended_ds, var1='pr', var2='PETR')

# Create a new xarray.Dataset
corr_ds = xr.Dataset({
    'corr_PET': corr_PET,
    'corr_PR': corr_PR,
    'corr_PPETR': corr_PPETR 
})

highest_corr = high_corr_allcells(corr_ds, 'corr_PET', 'corr_PR', 'corr_PPETR', save_string='yearly')
np.unique(highest_corr)
highest_corr.sel(lat=48, lon=2)
corr_PET.sel(lat=48, lon=2)
corr_PR.sel(lat=48, lon=2)
corr_PPETR.sel(lat=48, lon=2)

fig, axs = plt.subplots(3, 1, figsize=(8, 12))  # 3 rows, 1 column

Cor_PET = np.corrcoef(trended_ds['pr'].sel(lat=48, lon=2), trended_ds['evspsbl'].sel(lat=48, lon=2))[0, 1]
axs[0].scatter(trended_ds['pr'].sel(lat=48, lon=2), trended_ds['evspsbl'].sel(lat=48, lon=2), color='lightblue')
axs[0].set_title(f'Correlation: {Cor_PET:.2f}')
axs[0].set_xlabel('pr')
axs[0].set_ylabel('evspsbl')
axs[0].grid(True)
axs[0].legend()

Cor_PR = np.corrcoef(trended_ds['pr'].sel(lat=48, lon=2), trended_ds['mrro'].sel(lat=48, lon=2))[0, 1]
axs[1].scatter(trended_ds['pr'].sel(lat=48, lon=2), trended_ds['mrro'].sel(lat=48, lon=2), color='purple')
axs[1].set_title(f'Correlation: {Cor_PR:.2f}')
axs[1].set_xlabel('pr')
axs[1].set_ylabel('mrro')
axs[1].grid(True)
axs[1].legend()

Cor_PPETR = np.corrcoef(trended_ds['pr'].sel(lat=48, lon=2), trended_ds['PETR'].sel(lat=48, lon=2))[0, 1]
axs[2].scatter(trended_ds['pr'].sel(lat=48, lon=2), trended_ds['PETR'].sel(lat=48, lon=2), color='brown')
axs[2].set_title(f'Correlation: {Cor_PPETR:.2f}')
axs[2].set_xlabel('pr')
axs[2].set_ylabel('PETR')
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()
plt.show()


### analyse in addition if the Precipitation will increase or decrease:
delta_ds = combined_ds.copy()   


## Calculate Difference between 2100 and 2025
delta_ds = delta_ds.assign(d_pr = calc_vardiff(delta_ds, 'pr'))
#if change for precip is lower than 3% the cell is not relevant for the analysis:
pr_20002025 = delta_ds['pr'].sel(time=slice('2000', '2025')).mean(dim='time')
pr_20752100 = delta_ds['pr'].sel(time=slice('2075', '2100')).mean(dim='time')
change_greater = np.abs(delta_ds['d_pr'])/pr_20002025 >= 0.03
delta_ds['d_pr'] = delta_ds['d_pr'].where(change_greater)

    
m_neg = delta_ds['d_pr'] < 0

np.unique(highest_corr)
data_modified = highest_corr.where(~m_neg, highest_corr+3)
np.unique(data_modified)
highest_corr.sel(lat=48, lon=2)
data_modified.sel(lat=48, lon=2)
delta_ds['d_pr'].sel(lat=48, lon=2)


#Barplot global:
PET_cells_pos = np.sum(data_modified == 1)
PR_cells_pos = np.sum(data_modified == 2)
PPETR_cells_pos = np.sum(data_modified ==3)

PET_cells_neg = np.sum(data_modified == 4)
PR_cells_neg = np.sum(data_modified == 5)
PPETR_cells_neg = np.sum(data_modified ==6)

total_cells = PET_cells_pos + PPETR_cells_pos + PR_cells_pos + PET_cells_neg + PPETR_cells_neg + PR_cells_neg

perc_PET_pos = (PET_cells_pos / total_cells) * 100
perc_PPETR_pos = (PPETR_cells_pos / total_cells) * 100
perc_PR_pos = (PR_cells_pos / total_cells) * 100

perc_PET_neg = (PET_cells_neg / total_cells) * 100
perc_PPETR_neg = (PPETR_cells_neg / total_cells) * 100
perc_PR_neg = (PR_cells_neg / total_cells) * 100


categories = ['$\Delta$ET-$\Delta$P pos', '$\Delta$R-$\Delta$P pos', '$\Delta$(P-ET-R)-$\Delta$P pos', 
              '$\Delta$ET-$\Delta$P neg', '$\Delta$R-$\Delta$P neg', '$\$\Delta$(P-ET-R)-$\Delta$P neg']
values = [perc_PET_pos, perc_PR_pos, perc_PPETR_pos, 
          perc_PET_neg, perc_PR_neg, perc_PPETR_neg]


colors = ['forestgreen', 'steelblue', 'darkgoldenrod', 'lightgreen', 'skyblue','wheat'] 

## Plot:
cmap = ListedColormap(colors)
cmap.set_bad('lightgray')

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 3), subplot_kw={'projection': ccrs.PlateCarree()})        

# Plot the main map
data_modified.plot.imshow(ax=axes, cmap=cmap,  extend='both', 
                                      cbar_kwargs={'label': 'Highest Change'}, 
                                      vmin=1, vmax=6)
axes.coastlines()
axes.set_title('Similar Change, Annual Mean')



# Create an inset for the bar plot
left, bottom, width, height = 0.13, 0.1, 0.14, 0.19
ax_inset = fig.add_axes([left, bottom, width, height], facecolor='white')

# Plot the bar plot
ax_inset.bar(categories, values, color=colors, edgecolor='black')
ax_inset.set_xlabel('')
ax_inset.set_title('No. of cells [%]')

# Hide xticks and xtick labels
ax_inset.set_yticks([0, 10, 20, 30, 40])
ax_inset.set_xticks([])
ax_inset.set_xticklabels([])

plt.tight_layout()
plt.savefig('Plots/Global/High_Correlation_Yearly_Pmask.png', dpi=300)
plt.show()





###################### seasons:

DJF_corr_PET = corr_all_cells(data=DJF_trended_ds, var1='pr', var2='evspsbl')
DJF_corr_PR = corr_all_cells(data=DJF_trended_ds, var1='pr', var2='mrro')
DJF_corr_PPETR = corr_all_cells(data=DJF_trended_ds, var1='pr', var2='PETR')

DJF_corr_ds = xr.Dataset({
    'DJF_corr_PET': DJF_corr_PET,
    'DJF_corr_PR': DJF_corr_PR,
    'DJF_corr_PPETR': DJF_corr_PPETR 
})

DJF_highest_corr = high_corr_allcells(DJF_corr_ds, 'DJF_corr_PET', 'DJF_corr_PR', 'DJF_corr_PPETR', save_string='yearly')


MAM_corr_PET = corr_all_cells(data=MAM_trended_ds, var1='pr', var2='evspsbl')
MAM_corr_PR = corr_all_cells(data=MAM_trended_ds, var1='pr', var2='mrro')
MAM_corr_PPETR = corr_all_cells(data=MAM_trended_ds, var1='pr', var2='PETR')

MAM_corr_ds = xr.Dataset({
    'MAM_corr_PET': MAM_corr_PET,
    'MAM_corr_PR': MAM_corr_PR,
    'MAM_corr_PPETR': MAM_corr_PPETR 
})

MAM_highest_corr = high_corr_allcells(MAM_corr_ds, 'MAM_corr_PET', 'MAM_corr_PR', 'MAM_corr_PPETR', save_string='yearly')

JJA_corr_PET = corr_all_cells(data=JJA_trended_ds, var1='pr', var2='evspsbl')
JJA_corr_PR = corr_all_cells(data=JJA_trended_ds, var1='pr', var2='mrro')
JJA_corr_PPETR = corr_all_cells(data=JJA_trended_ds, var1='pr', var2='PETR')

JJA_corr_ds = xr.Dataset({
    'JJA_corr_PET': JJA_corr_PET,
    'JJA_corr_PR': JJA_corr_PR,
    'JJA_corr_PPETR': JJA_corr_PPETR 
})

JJA_highest_corr = high_corr_allcells(JJA_corr_ds, 'JJA_corr_PET', 'JJA_corr_PR', 'JJA_corr_PPETR', save_string='yearly')

SON_corr_PET = corr_all_cells(data=SON_trended_ds, var1='pr', var2='evspsbl')
SON_corr_PR = corr_all_cells(data=SON_trended_ds, var1='pr', var2='mrro')
SON_corr_PPETR = corr_all_cells(data=SON_trended_ds, var1='pr', var2='PETR')

SON_corr_ds = xr.Dataset({
    'SON_corr_PET': SON_corr_PET,
    'SON_corr_PR': SON_corr_PR,
    'SON_corr_PPETR': SON_corr_PPETR 
})

SON_highest_corr = high_corr_allcells(SON_corr_ds, 'SON_corr_PET', 'SON_corr_PR', 'SON_corr_PPETR', save_string='yearly')

high_cor_seasons = xr.Dataset({
    'DJF' : DJF_highest_corr,
    'MAM' : MAM_highest_corr,
    'JJA' : JJA_highest_corr,
    'SON' : SON_highest_corr})

seasons = ['DJF', 'MAM', 'JJA', 'SON']

bar_data = []
for i in seasons:
    PET_cells = np.sum(high_cor_seasons[i] == 1)
    PR_cells = np.sum(high_cor_seasons[i] == 2)
    PPETR_cells = np.sum(high_cor_seasons[i] == 3)
    
    total_cells = PET_cells + PPETR_cells +PR_cells
    
    perc_PET = (PET_cells / total_cells) * 100
    perc_PPETR = (PPETR_cells / total_cells) * 100
    perc_PR = (PR_cells / total_cells) * 100
    
    values = [perc_PET.item(), perc_PR.item(), perc_PPETR.item()]
    
    bar_data.append(values)

x_values = np.arange(len(bar_data[0]))


colors = ['forestgreen', 'skyblue', 'darkgoldenrod']        
cmap = ListedColormap(colors)
cmap.set_bad('lightgray')
    
fig, axes = plt.subplots(nrows=len(seasons), ncols=1, figsize=(7, 12), subplot_kw={'projection': ccrs.PlateCarree()})

for i, s  in zip([0,1,2,3], ['DJF', 'MAM', 'JJA', 'SON']):
        
        ax = axes[i]
       
        high_cor_seasons[s].plot.imshow(ax=ax, cmap=cmap, extend='both', 
                                              cbar_kwargs={'label': 'Highest Correlation'}, 
                                              vmin=1, vmax=3)
        ax.coastlines()
        ax.set_title(f'{s}')
        
        # Create an inset for the bar plot
        left, bottom, width, height = 0.10, 0.12, 0.12, 0.22
        inset_ax = ax.inset_axes([left, bottom, width, height], facecolor='white')
        
        # Plot the bar plot
        inset_ax.bar(x_values, bar_data[i], color=colors, edgecolor='black')
        
        inset_ax.set_xlabel('')
        inset_ax.set_title('No. of cells [%]')
        
        # Hide xticks and xtick labels
        inset_ax.set_yticks([0, 25, 50, 75])
        inset_ax.set_xticks([])
        inset_ax.set_xticklabels([])


plt.tight_layout()
plt.savefig('Plots/Global/High_Correlation_Seasons.png', dpi=300)
plt.show()


### analyse in addition if the Precipitation will increase or decrease:
sdelta_ds = combined_ds.copy() 


## Calculate Difference between 2100 and 2025
sdelta_ds = sdelta_ds.assign(d_pr = calc_vardiff_season(sdelta_ds, 'pr'))
#if change for precip is lower than 5% the cell is not relevant for the analysis:
pr_20002025 = delta_ds['pr'].sel(time=slice('2000', '2025')).groupby('time.season').mean(dim='time')
pr_20752100 = delta_ds['pr'].sel(time=slice('2075', '2100')).groupby('time.season').mean(dim='time')
s_change_greater = np.abs(sdelta_ds['d_pr'])/pr_20002025 >= 0.03
sdelta_ds['d_pr'] = sdelta_ds['d_pr'].where(s_change_greater)

    
s_neg = sdelta_ds['d_pr'] < 0

data_modified_ls = high_cor_seasons
for i in seasons:
    print(np.unique(high_cor_seasons[i]))
    data_modified_ls[i] = high_cor_seasons[i].where(~s_neg.sel(season=i), high_cor_seasons[i]+3)
    print(np.unique(data_modified_ls[i]))

#Get barplot data:
bar_data = []
for i in seasons:
    PET_cells_pos = np.sum(high_cor_seasons[i] == 1)
    PR_cells_pos = np.sum(high_cor_seasons[i] == 2)
    PPETR_cells_pos = np.sum(high_cor_seasons[i] == 3)

    PET_cells_neg = np.sum(high_cor_seasons[i] == 4)
    PR_cells_neg = np.sum(high_cor_seasons[i] == 5)
    PPETR_cells_neg = np.sum(high_cor_seasons[i] == 6)
    
    total_cells = PET_cells_pos + PPETR_cells_pos + PR_cells_pos + PET_cells_neg + PPETR_cells_neg + PR_cells_neg
    perc_PET_pos = (PET_cells_pos / total_cells) * 100
    perc_PPETR_pos = (PPETR_cells_pos / total_cells) * 100
    perc_PR_pos = (PR_cells_pos / total_cells) * 100

    perc_PET_neg = (PET_cells_neg / total_cells) * 100
    perc_PPETR_neg = (PPETR_cells_neg / total_cells) * 100
    perc_PR_neg = (PR_cells_neg / total_cells) * 100

    
    values = [perc_PET_pos.item(), perc_PR_pos.item(), perc_PPETR_pos.item(), 
              perc_PET_neg.item(), perc_PR_neg.item(), perc_PPETR_neg.item()]
    
    bar_data.append(values)

x_values = np.arange(len(bar_data[0]))

colors = ['forestgreen', 'steelblue', 'darkgoldenrod', 'lightgreen', 'skyblue','wheat'] 
cmap = ListedColormap(colors)
cmap.set_bad('lightgray')


fig, axes = plt.subplots(nrows=len(seasons), ncols=1, figsize=(7, 12), subplot_kw={'projection': ccrs.PlateCarree()})

for i, s  in zip([0,1,2,3],
                      ['DJF', 'MAM', 'JJA', 'SON']):
        
    
        ax = axes[i]
       
        high_cor_seasons[s].plot.imshow(ax=ax, cmap=cmap, extend='both', 
                                              cbar_kwargs={'label': 'Highest Similarity'},
                                              vmin=1, vmax=6)
        ax.coastlines()
        ax.set_title(f'{s}')
        
        # Create an inset for the bar plot
        left, bottom, width, height = 0.10, 0.12, 0.12, 0.24
        inset_ax = ax.inset_axes([left, bottom, width, height], facecolor='white')
        
        # Plot the bar plot
        inset_ax.bar(x_values, bar_data[i], color=colors, edgecolor='black')
        inset_ax.set_xlabel('')
        inset_ax.set_title('No. of cells [%]')
        
        # Hide xticks and xtick labels
        inset_ax.set_yticks([0, 10, 20, 30, 40])
        inset_ax.set_xticks([])
        inset_ax.set_xticklabels([])


plt.tight_layout()
plt.savefig('Plots/Global/High_Correlation_Seasons_Pmask.png', dpi=300)
plt.show()











def corr_func(data, var1, var2, time_start, time_end, save_string):
    import cartopy.crs as ccrs
    import xarray as xr
    
    #data = trended_ds
    #var1 = 'pr'
    #var2 = 'evspsbl'
    #time_start = '2000'
    #time_end = '2100'
    #save_string = 'PET_yearly'
    
    corr_results = xr.Dataset(
        {'corr': (['lat', 'lon'], np.random.rand(len(corr_ds.lat), len(corr_ds.lon)))},
        coords={'lat': corr_ds.lat, 'lon': corr_ds.lon}
    )
    
    for i, lat_val in enumerate(tqdm(corr_ds.lat)):
        for j, lon_val in enumerate(corr_ds.lon):
            # Select data for the current cell
            cell_data = data.sel(time=slice(time_start, time_end)).isel(lat=i, lon=j).to_dataframe()
            
            if ~np.all(np.isnan(cell_data[var1])):
                corr = cell_data[var1].corr(cell_data[var2])
            else:
                corr = np.nan
            # Store the lowess result in the array
            corr_results['corr'].loc[lat_val, lon_val] = corr
            
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 4),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    corr_results['corr'].plot.imshow(ax=axes, cmap='RdBu', extend='both',
                                  center=0, vmin=-1, vmax=1,
                                  cbar_kwargs={'label': 'Pearson Correlation'})
    axes.coastlines()
    axes.set_title(f'Correlation between {var1} and {var2}')
    plt.tight_layout()
    #plt.savefig(f'Plots/Global/{save_string}.png', dpi=300, transparent=True)
    plt.show()
            
    return corr_results

Cor_PET_loop = corr_func(data = trended_ds, var1='pr', var2 = 'evspsbl', time_start='2000', time_end='2100', save_string='Cor_PET_yearly')
Cor_PR_loop = corr_func(data = trended_ds, var1='pr', var2 = 'mrro', time_start='2000', time_end='2100', save_string='Cor_PR_yearly')
Cor_PPETR_loop = corr_func(data = trended_ds, var1='pr', var2 = 'PETR', time_start='2000', time_end='2100', save_string='Cor_PPETR_yearly')

# Extract the 'corr' DataArray from each Dataset
Cor_PR_da = Cor_PR_loop['corr']
Cor_PPETR_da = Cor_PPETR_loop['corr']
Cor_PET_da = Cor_PET_loop['corr']  # Assuming Cor_PET_loop is similarly defined

# Create a new Dataset combining the extracted DataArrays
corr_ds_loop = xr.Dataset({
    'Cor_PET_loop': Cor_PET_da,
    'Cor_PR_loop': Cor_PR_da,
    'Cor_PPETR_loop': Cor_PPETR_da
})



highest_corr_loop = high_corr_allcells(corr_ds_loop, 'Cor_PET_loop', 'Cor_PR_loop', 'Cor_PPETR_loop', save_string='yearly_loop')
np.unique(highest_corr)
highest_corr_loop.sel(lat=48, lon=2)
Cor_PET_loop.sel(lat=48, lon=2)
Cor_PR_loop.sel(lat=48, lon=2)
Cor_PPETR_loop.sel(lat=48, lon=2)

Cor_PET_loop.sel(lat=48, lon=2)
