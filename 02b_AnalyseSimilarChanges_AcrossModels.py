# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 08:01:10 2024

@author: Ruth Stephan
"""

#packages:
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import seaborn as sns
import cartopy.feature as cfeature
import pickle



#paths:
os.getcwd()
data_path = '/mnt/share/scratch/rs1155/data/CMIP6/output_data/'
plot_path = '/mnt/share/scratch/rs1155/plots/across_models/'

# Set several styles for plotting:
sns.set_style('ticks')
plt.rcParams['font.family'] = 'DejaVu Sans'
ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '110m', edgecolor='face', facecolor='white')
pal_PRGn = mpl.colormaps.get_cmap('PRGn') 
pal_PRGn.set_bad(color='lightgray')
pal_coolwarm = mpl.colormaps.get_cmap('coolwarm') 
pal_coolwarm.set_bad(color='lightgray')
pal_coolwarm_r = mpl.colormaps.get_cmap('coolwarm_r') 
pal_coolwarm_r.set_bad(color='lightgray')
pal_RdBu = mpl.colormaps.get_cmap('RdBu') 
pal_RdBu.set_bad(color='lightgray')
pal_Reds = mpl.colormaps.get_cmap('Reds') 
pal_Reds.set_bad(color='lightgray')
pal_Greens = mpl.colormaps.get_cmap('Greens') 
pal_Greens.set_bad(color='lightgray')
pal_Purples = mpl.colormaps.get_cmap('Purples') 
pal_Purples.set_bad(color='lightgray')
pal_Greens_r = mpl.colormaps.get_cmap('Greens_r') 
pal_Greens_r.set_bad(color='lightgray')
pal_Reds_r = mpl.colormaps.get_cmap('Reds_r')  
pal_Reds_r.set_bad(color='lightgray')
colors_highest_sim2 = ['forestgreen', 'skyblue', 'darkgoldenrod']
cmap_highest_sim2 = ListedColormap(colors_highest_sim2)
cmap_highest_sim2.set_bad('lightgray')
colors_highest_sim = ['forestgreen', 'skyblue', 'darkgoldenrod', 'black']
cmap_highest_sim = ListedColormap(colors_highest_sim)
cmap_highest_sim.set_bad('lightgray')
colors_highest_sim_Pmask = ['forestgreen', 'steelblue', 'darkgoldenrod', 'black', 'lightgreen', 'skyblue','wheat', 'darkgray'] 
cmap_highest_sim_Pmask = ListedColormap(colors_highest_sim_Pmask)
cmap_highest_sim_Pmask.set_bad('lightgray')
colors_highest_sim_Pmask2 = ['forestgreen', 'steelblue', 'darkgoldenrod', 'lightgreen', 'skyblue','wheat'] 
cmap_highest_sim_Pmask2 = ListedColormap(colors_highest_sim_Pmask2)
cmap_highest_sim_Pmask2.set_bad('lightgray')

# Load the list from the file
with open(data_path + 'all_models_delta.pkl', 'rb') as f:
    all_models_delta = pickle.load(f)



# calculate cell area:
# Calculation of an xarray that stores the area info (km2) in each cell:
def calculate_cell_area(lat, lon):
    """
    Calculate the area of each cell in a latitude-longitude grid.

    Parameters:
    - lat: Array of latitudes.
    - lon: Array of longitudes.

    Returns:
    - area: Array of areas in square kilometers.
    """
    R = 6371.0  # Radius of the Earth in kilometers
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    #lat_diff = np.abs(np.diff(lat_rad))
    lon_diff = np.abs(np.diff(lon_rad))
    
    # Calculate the area for each cell
    areas = np.zeros((len(lat), len(lon)))
    
    for i in range(len(lat) - 1):
        for j in range(len(lon) - 1):
            areas[i, j] = (R ** 2) * np.abs(np.sin(lat_rad[i]) - np.sin(lat_rad[i + 1])) * lon_diff[j]
    
    # Handle the last row and column
    areas[-1, :] = areas[-2, :]
    areas[:, -1] = areas[:, -2]
    
    return areas

    
latitudes = all_models_delta[0]['lat'].values
longitudes = all_models_delta[0]['lon'].values
cell_areas = calculate_cell_area(latitudes, longitudes)
cell_areas_da = xr.DataArray(cell_areas, dims=['lat', 'lon'], coords={'lat': latitudes, 'lon': longitudes})


# Define the function to count NaNs in a single DataArray
def count_nans(data_variable):
    return np.isnan(data_variable).astype(int)

def count_pos(data_variable):
    return data_variable >= 0

def count_neg(data_variable):
    return data_variable <= 0


def across_models(model_ls, var, function):

    # Initialize an array to store the NaN counts
    init_array = np.zeros((len(model_ls[0]['lat']), len(model_ls[0]['lon'])))
    
    # Loop over all models and accumulate the NaN counts
    for model in model_ls:
        model_var = model[var]
        
        # Apply the count_nans function over the dataset using apply_ufunc
        result = xr.apply_ufunc(
            function,
            model_var,
            input_core_dims=[['lat', 'lon']],
            output_core_dims=[['lat', 'lon']],
            vectorize=True,
            dask='parallelized',  # Optional: enable parallel execution with dask
            output_dtypes=[np.int64]
        )
        
        # Update the nan_agreement_count with the new counts
        init_array += result.values
        
        # Convert nan_agreement_count back to an xarray DataArray 
        result_da = xr.DataArray(init_array, dims=['lat', 'lon'], coords={'lat': model_ls[0]['lat'], 'lon': model_ls[0]['lon']})

    return result_da


######################################################
##   Evaluate results across models: Annual Mean    ##
######################################################


lat = all_models_delta[0].coords['lat']
lon = all_models_delta[0].coords['lon']


###################################
#  1. Check agreement on Masks:  ##
###################################

nan_agreement_count = across_models(all_models_delta, var='sim_change', function = count_nans)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 3), subplot_kw={'projection': ccrs.PlateCarree()})        
nan_agreement_count.plot.imshow(ax=axes, cmap=pal_Reds,  extend='both', 
                                      cbar_kwargs={'label': 'No. of models'}, 
                                      vmin=0, vmax=8
                                      )
axes.add_feature(ocean)
axes.coastlines()
axes.set_title('Agreement: Masking')


plt.tight_layout()
plt.savefig(plot_path + 'NAN_Agreement.png', dpi=300)
plt.show()


Psurplus_agreement_count = across_models(all_models_delta, var='d_pr', function = count_pos)
Pdeficit_agreement_count = across_models(all_models_delta, var='d_pr', function = count_neg)


# Plot Agreement on P surplus or deficit:
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 3), subplot_kw={'projection': ccrs.PlateCarree()})

Psurplus_agreement_count.plot.imshow(ax=axes[0], cmap=pal_Greens, extend='both', 
                                      cbar_kwargs={'label': 'No. of models'}, 
                                      vmin=0, vmax=8)
axes[0].add_feature(ocean)
axes[0].coastlines()
axes[0].set_title('Agreement: Precipitation surplus')

Pdeficit_agreement_count.plot.imshow(ax=axes[1], cmap=pal_Purples, extend='both', 
                                      cbar_kwargs={'label': 'No. of models'}, 
                                      vmin=0, vmax=8)
axes[1].add_feature(ocean)
axes[1].coastlines()
axes[1].set_title('Agreement: Precipitation deficit')

plt.tight_layout()
plt.savefig(plot_path + 'Pmask_Agreement.png', dpi=300)
plt.show()


##############################################
# Check agreement on results without Pmask: ##
##############################################

# Initialize value_counts
value_counts = {v: xr.DataArray(np.zeros((len(lat), len(lon)), dtype=int), coords=[lat, lon], dims=["lat", "lon"]) for v in range(9)}
value_counts[np.nan] = xr.DataArray(np.zeros((len(lat), len(lon)), dtype=int), coords=[lat, lon], dims=["lat", "lon"])
most_frequent_value = xr.DataArray(np.zeros((len(lat), len(lon))), coords=[lat, lon], dims=["lat", "lon"])


# Function to count occurrences of values across datasets
def count_values(data_variable, value_counts):
    for value in range(len(value_counts)-1):
        value_counts[value] += (data_variable == value).astype(int)
    value_counts[np.nan] += np.isnan(data_variable).astype(int)
    return value_counts

# Loop over all models and update value_counts
for dataset in all_models_delta:
    sim_change = dataset['sim_change']
    value_counts = count_values(sim_change, value_counts)

# Function to determine the most frequent value
def most_frequent_func(*value_counts):
    all_values = list(range(len(value_counts)-1)) + [np.nan]
    counts = np.stack(value_counts, axis=-1)
    most_frequent = np.full(counts.shape[:-1], np.nan)

    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            most_frequent[i, j] = all_values[np.nanargmax(counts[i, j])]

    return most_frequent

# Combine value_counts into a list for xr.apply_ufunc
value_counts_list = [value_counts[v] for v in range(9)] + [value_counts[np.nan]]

# Apply the function using xr.apply_ufunc
most_frequent_value = xr.apply_ufunc(
    most_frequent_func,
    *value_counts_list,
    input_core_dims=[['lat', 'lon']] * 10,
    output_core_dims=[['lat', 'lon']],
    vectorize=True,
    output_dtypes=[np.float64]
)

# Plot most frequent values:
PET_area = float((cell_areas_da * (most_frequent_value == 1)).sum())
PR_area = float((cell_areas_da * (most_frequent_value == 2)).sum())
PPETR_area = float((cell_areas_da * (most_frequent_value == 3)).sum())
unclear_area = float((cell_areas_da * (most_frequent_value == 4)).sum())

# Total area
total_area = PET_area + PR_area + PPETR_area + unclear_area

perc_PET = (PET_area / total_area) * 100
perc_PR = (PR_area / total_area) * 100
perc_PPETR = (PPETR_area / total_area) * 100
perc_unclear = (unclear_area / total_area) * 100

categories = ['$\Delta$ET/$\Delta$P', '$\Delta$R/$\Delta$P', '$\Delta$RES/$\Delta$P', 'unclear']
values = [perc_PET, perc_PR, perc_PPETR, perc_unclear]


## Plot Most frequent value:
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 3), subplot_kw={'projection': ccrs.PlateCarree()})        

# Plot the main map
most_frequent_value.plot.imshow(ax=axes, cmap=cmap_highest_sim,  extend='both', 
                                      cbar_kwargs={'label': 'Highest Change'}, 
                                      vmin=1, vmax=4)
axes.add_feature(ocean)
axes.coastlines()
axes.set_title('Highest Similarity - Yearly')

# Inset for the bar plot
left, bottom, width, height = 0.13, 0.16, 0.10, 0.19
ax_inset = fig.add_axes([left, bottom, width, height], facecolor='white')
ax_inset.bar(categories, values, color=colors_highest_sim2, edgecolor='black')
ax_inset.set_xlabel('')
ax_inset.set_title('Area [%]')
ax_inset.set_yticks([0, 20, 40, 60])
ax_inset.set_xticks([])
ax_inset.set_xticklabels([])

plt.tight_layout()
plt.savefig(plot_path + 'HighSimilarity_Ratio_Yearly.png', dpi=300)
plt.show()


# Check agreement on most frequent value:
def count_agreement(var, array_to_check):
    agreement = ((np.isnan(var) & np.isnan(array_to_check)) | 
                 (~np.isnan(var) & (var == array_to_check)))
    return agreement.astype(int)

# count_agreement across models
def count_agreements_across_models(model_ls, var, array_to_check):
    init_array = np.zeros((len(model_ls[0]['lat']), len(model_ls[0]['lon'])))
    
    for model in model_ls:
        model_var = model[var]
        
        # Apply the count_agreement function over the dataset using apply_ufunc
        result = xr.apply_ufunc(
            count_agreement,
            model_var,
            array_to_check,
            input_core_dims=[['lat', 'lon'], ['lat', 'lon']],
            output_core_dims=[['lat', 'lon']],
            vectorize=True,
            dask='parallelized',  # Optional: enable parallel execution with dask
            output_dtypes=[np.int64]
        )
        
        # Update the init_array with the new counts
        init_array += result.values
        
    # Normalize by the number of models to get the percentage
    total_models = len(model_ls)
    percentage_array = (init_array / total_models) * 100
    
    # Convert init_array back to an xarray DataArray 
    result_da = xr.DataArray(percentage_array, dims=['lat', 'lon'], coords={'lat': model_ls[0]['lat'], 'lon': model_ls[0]['lon']})
    result_da = result_da.where(~np.isnan(array_to_check), other=np.nan)
    
    return result_da


agreement_count = count_agreements_across_models(model_ls=all_models_delta, var='sim_change', array_to_check=most_frequent_value)
agreement_count_values = agreement_count.values.flatten()


# Plot Agreement
agreement_palette = ['indianred', 'lightcoral', 'white', 'lightskyblue', 'skyblue', 'steelblue', '#313695']

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 3), subplot_kw={'projection': ccrs.PlateCarree()})        
agreement_count.plot.imshow(ax=axes, cmap=pal_RdBu, extend='both', 
                            cbar_kwargs={'label': 'Model Agreement [%]'}, 
                            vmin=0, vmax=100)
axes.add_feature(ocean)
axes.coastlines()
axes.set_title('Agreement: Highest Similarity - Yearly')

# Inset for the bar plot
left, bottom, width, height = 0.11, 0.22, 0.14, 0.19
ax_inset = fig.add_axes([left, bottom, width, height], facecolor='white')
#ax_inset.hist(agreement_count_values, bins=7, color='gray', edgecolor='black')
# Histogram with custom colors
n, bins, patches = ax_inset.hist(agreement_count_values, bins=7, edgecolor='black')
for i, patch in enumerate(patches):
    patch.set_facecolor(agreement_palette[i % len(agreement_palette)])

ax_inset.set_xlabel('')
ax_inset.set_ylabel('')
ax_inset.set_title('Agreement')
ax_inset.set_xticks([0, 25, 50, 75, 100])
ax_inset.set_yticks([])
ax_inset.set_yticklabels([])


plt.tight_layout()
plt.savefig(plot_path + 'Agreement_HighSimilarity_Ratio_Yearly.png', dpi=300)
plt.show()


##############################################
#   Check agreement on results with Pmask:  ##
##############################################
value_counts = {v: xr.DataArray(np.zeros((len(lat), len(lon)), dtype=int), coords=[lat, lon], dims=["lat", "lon"]) for v in range(9)}
value_counts[np.nan] = xr.DataArray(np.zeros((len(lat), len(lon)), dtype=int), coords=[lat, lon], dims=["lat", "lon"])
most_frequent_value = xr.DataArray(np.zeros((len(lat), len(lon))), coords=[lat, lon], dims=["lat", "lon"])

# Function to count occurrences of values across datasets
def count_values(data_variable, value_counts):
    for value in range(len(value_counts)-1):
        value_counts[value] += (data_variable == value).astype(int)
    value_counts[np.nan] += np.isnan(data_variable).astype(int)
    return value_counts

# Loop over all models and update value_counts
for dataset in all_models_delta:
    sim_change_Pmask = dataset['sim_change_Pmask']
    value_counts = count_values(sim_change_Pmask, value_counts)

# Function to determine the most frequent value
def most_frequent_func(*value_counts):
    all_values = list(range(len(value_counts)-1)) + [np.nan]
    counts = np.stack(value_counts, axis=-1)
    most_frequent = np.full(counts.shape[:-1], np.nan)

    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            most_frequent[i, j] = all_values[np.nanargmax(counts[i, j])]

    return most_frequent

# Combine value_counts into a list for xr.apply_ufunc
value_counts_list = [value_counts[v] for v in range(len(value_counts)-1)] + [value_counts[np.nan]]

# Apply the function using xr.apply_ufunc
most_frequent_value = xr.apply_ufunc(
    most_frequent_func,
    *value_counts_list,
    input_core_dims=[['lat', 'lon']] * 10,
    output_core_dims=[['lat', 'lon']],
    vectorize=True,
    output_dtypes=[np.float64]
)

# Plot Most Frequent Value:
PET_area_pos = float((cell_areas_da * (most_frequent_value == 1)).sum())
PR_area_pos = float((cell_areas_da * (most_frequent_value == 2)).sum())
PPETR_area_pos = float((cell_areas_da * (most_frequent_value == 3)).sum())
unclear_area_pos = float((cell_areas_da * (most_frequent_value == 4)).sum())

PET_area_neg = float((cell_areas_da * (most_frequent_value == 5)).sum())
PR_area_neg = float((cell_areas_da * (most_frequent_value == 6)).sum())
PPETR_area_neg = float((cell_areas_da * (most_frequent_value == 7)).sum())
unclear_area_neg = float((cell_areas_da * (most_frequent_value == 8)).sum())

# Total area
total_area = PET_area_pos + PR_area_pos + PPETR_area_pos + unclear_area_pos + PET_area_neg + PR_area_neg + PPETR_area_neg + unclear_area_neg

perc_PET_pos = (PET_area_pos / total_area) * 100
perc_PR_pos = (PR_area_pos / total_area) * 100
perc_PPETR_pos = (PPETR_area_pos / total_area) * 100
perc_unclear_pos = (unclear_area_pos / total_area) * 100

perc_PET_neg = (PET_area_neg / total_area) * 100
perc_PR_neg = (PR_area_neg / total_area) * 100
perc_PPETR_neg = (PPETR_area_neg / total_area) * 100
perc_unclear_neg = (unclear_area_neg / total_area) * 100

categories = range(8)
values = [perc_PET_pos, perc_PR_pos, perc_PPETR_pos, perc_unclear_pos,
          perc_PET_neg, perc_PR_neg, perc_PPETR_neg, perc_unclear_neg]



fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 3), subplot_kw={'projection': ccrs.PlateCarree()})        

# Plot the main map
most_frequent_value.plot.imshow(ax=axes, cmap=cmap_highest_sim_Pmask,  extend='both', 
                                      cbar_kwargs={'label': 'Highest Change'}, 
                                      vmin=1, vmax=8)
axes.add_feature(ocean)
axes.coastlines()
axes.set_title('Highest Similarity - Yearly')

# Inset for the bar plot
left, bottom, width, height = 0.11, 0.16, 0.14, 0.19
ax_inset = fig.add_axes([left, bottom, width, height], facecolor='white')
ax_inset.bar(categories, values, color=colors_highest_sim_Pmask, edgecolor='black')
ax_inset.set_xlabel('')
ax_inset.set_title('Area [%]')
ax_inset.set_yticks([0, 10, 20, 30, 40])
ax_inset.set_xticks([])
ax_inset.set_xticklabels([])

plt.tight_layout()
plt.savefig(plot_path + 'HighSimilarity_Ratio_Pmask_Yearly.png', dpi=300)
plt.show()
    
# Check agreement on most frequent value:
agreement_count = count_agreements_across_models(model_ls=all_models_delta, var='sim_change_Pmask', array_to_check=most_frequent_value)
agreement_count_values = agreement_count.values.flatten()
mean_per_lat = agreement_count.mean(dim='lon', skipna=True)

# Plot the mean values per latitude
plt.figure(figsize=(2, 3))
plt.plot(mean_per_lat, mean_per_lat['lat'],  linestyle='-')
plt.xlabel('Mean Agreement')
plt.ylabel('Latitude')
plt.ylim(-90, 90)
plt.yticks([-90, -60, -30, 0, 30, 60, 90])
plt.xlim(0, 100)
plt.axvline(x=50, color='r', linestyle='--', linewidth=1) 
plt.show()


# Plot Agreement
agreement_palette = ['indianred', 'lightcoral', 'white', 'lightskyblue', 'skyblue', 'steelblue', '#313695']

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 3), subplot_kw={'projection': ccrs.PlateCarree()})        
agreement_count.plot.imshow(ax=axes, cmap=pal_RdBu, extend='both', 
                            cbar_kwargs={'label': 'Model Agreement [%]'}, 
                            vmin=0, vmax=100)
axes.add_feature(ocean)
axes.coastlines()
axes.set_title('Agreement: Highest Similarity - Yearly')

# Inset for the bar plot
left, bottom, width, height = 0.11, 0.22, 0.14, 0.19
ax_inset = fig.add_axes([left, bottom, width, height], facecolor='white')
#ax_inset.hist(agreement_count_values, bins=7, color='gray', edgecolor='black')
# Histogram with custom colors
n, bins, patches = ax_inset.hist(agreement_count_values, bins=7, edgecolor='black')
for i, patch in enumerate(patches):
    patch.set_facecolor(agreement_palette[i % len(agreement_palette)])
ax_inset.set_xlabel('')
ax_inset.set_ylabel('')
ax_inset.set_title('Agreement')
ax_inset.set_xticks([0, 25, 50, 75, 100])
ax_inset.set_yticks([])
ax_inset.set_yticklabels([])


plt.tight_layout()
plt.savefig(plot_path + 'Agreement_HighSimilarity_Ratio_Pmask_Yearly.png', dpi=300)
plt.show()

######################################################
##     Evaluate results across models: Seasons      ##
######################################################

# Load the list from the file
with open(data_path + 'all_models_sdelta.pkl', 'rb') as f:
    all_models_sdelta = pickle.load(f)

lat = all_models_sdelta[0].coords['lat']
lon = all_models_sdelta[0].coords['lon']

###################################
#  1. Check agreement on Masks:  ##
###################################

Pmask_pos_agreement_count_per_season = []
Pmask_neg_agreement_count_per_season = []

seasons = ['DJF', 'MAM', 'JJA', 'SON']

# Loop through each season
for season in seasons:
    #season = 'DJF'
    
    sel_season = [model.sel(season=season) for model in all_models_sdelta]
    
    Psurplus_agreement_count = across_models(sel_season, var='d_pr', function = count_pos)
    Pdeficit_agreement_count = across_models(sel_season, var='d_pr', function = count_neg)
     
    Pmask_pos_agreement_count_per_season.append(Psurplus_agreement_count)
    Pmask_neg_agreement_count_per_season.append(Pdeficit_agreement_count)
 
# Plot Agreement on P surplus or deficit:
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(14, 12), subplot_kw={'projection': ccrs.PlateCarree()})

for i, season in zip([0,1,2,3], ['DJF', 'MAM', 'JJA', 'SON']):
    
    # Positive agreement subplot
    Pmask_pos_agreement_count_per_season[i].plot.imshow(ax=axes[i, 0], cmap=pal_Greens, extend='both', 
                                                            cbar_kwargs={'label': 'No. of models'}, 
                                                            vmin=0, vmax=8)
    axes[i, 0].add_feature(ocean)
    axes[i, 0].coastlines()
    axes[i, 0].set_title(f'Agreement: Precipitation surplus ({season})')
    
    # Negative agreement subplot
    Pmask_neg_agreement_count_per_season[i].plot.imshow(ax=axes[i, 1], cmap=pal_Purples, extend='both', 
                                                            cbar_kwargs={'label': 'No. of models'}, 
                                                            vmin=0, vmax=8)
    axes[i, 1].add_feature(ocean)
    axes[i, 1].coastlines()
    axes[i, 1].set_title(f'Agreement: Precipitation deficit ({season})')

plt.tight_layout()
plt.savefig(plot_path + 'Pmask_Agreement_Seasons.png', dpi=300)
plt.show()


##############################################
# Check agreement on results without Pmask: ##
##############################################

most_frequent_value_per_season = []
agreement_count_per_season = []
agreement_count_values_per_season = []

# Loop through each season
for season in seasons:
    #season = 'MAM'
    
    sel_season = [model.sel(season=season) for model in all_models_sdelta]
    
    value_counts = {v: xr.DataArray(np.zeros((len(lat), len(lon)), dtype=int), coords=[lat, lon], dims=["lat", "lon"]) for v in range(9)}
    value_counts[np.nan] = xr.DataArray(np.zeros((len(lat), len(lon)), dtype=int), coords=[lat, lon], dims=["lat", "lon"])
    most_frequent_value = xr.DataArray(np.zeros((len(lat), len(lon))), coords=[lat, lon], dims=["lat", "lon"])

    # Loop over all models and update value_counts
    for dataset in sel_season:
        sim_change = dataset['sim_change']
        value_counts = count_values(sim_change, value_counts)

    # Combine value_counts into a list for xr.apply_ufunc
    value_counts_list = [value_counts[v] for v in range(len(value_counts)-1)] + [value_counts[np.nan]]

    # Apply the function using xr.apply_ufunc
    most_frequent_value = xr.apply_ufunc(
        most_frequent_func,
        *value_counts_list,
        input_core_dims=[['lat', 'lon']] * 10,
        output_core_dims=[['lat', 'lon']],
        vectorize=True,
        output_dtypes=[np.float64]
    )    

    # Append the count for the current season to the list
    most_frequent_value_per_season.append(most_frequent_value)
    
    # Create an array to count the number of agreements
    agreement_count = count_agreements_across_models(model_ls=sel_season, var='sim_change', array_to_check=most_frequent_value)
    agreement_count_values = agreement_count.values.flatten()

    agreement_count_per_season.append(agreement_count)
    agreement_count_values_per_season.append(agreement_count_values)
    
# Plot most frequent values:    
bar_data = []
for i in [0,1,2,3]:
    PET_area = float((cell_areas_da * (most_frequent_value_per_season[i] == 1)).sum())
    PR_area = float((cell_areas_da * (most_frequent_value_per_season[i]  == 2)).sum())
    PPETR_area = float((cell_areas_da * (most_frequent_value_per_season[i]  == 3)).sum())
    unclear_area = float((cell_areas_da * (most_frequent_value_per_season[i]  == 4)).sum())
    
    total_area = PET_area + PR_area + PPETR_area + unclear_area
    
    perc_PET = (PET_area / total_area) * 100
    perc_PR = (PR_area / total_area) * 100
    perc_PPETR = (PPETR_area / total_area) * 100
    perc_unclear = (unclear_area / total_area) * 100
    
    values = [perc_PET, perc_PR, perc_PPETR, perc_unclear]
    
    bar_data.append(values)

x_values = np.arange(len(bar_data[0]))

   
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(7, 12), subplot_kw={'projection': ccrs.PlateCarree()})

for i, s  in zip([0,1,2,3], ['DJF', 'MAM', 'JJA', 'SON']):
        
        ax = axes[i]
        most_frequent_value_per_season[i].plot.imshow(ax=ax, cmap=cmap_highest_sim, extend='both', 
                                              cbar_kwargs={'label': 'Highest Similarity'}, 
                                              vmin=1, vmax=4)
        ax.add_feature(ocean)
        ax.coastlines()
        ax.set_title(f'{s}')
        
        # Create an inset for the bar plot
        left, bottom, width, height = 0.10, 0.12, 0.12, 0.22
        inset_ax = ax.inset_axes([left, bottom, width, height], facecolor='white')
        inset_ax.bar(x_values, bar_data[i], color=colors_highest_sim, edgecolor='black')
        inset_ax.set_xlabel('')
        inset_ax.set_title('Area [%]')
        inset_ax.set_yticks([0, 25, 50, 75])
        inset_ax.set_xticks([])
        inset_ax.set_xticklabels([])

plt.tight_layout()
plt.savefig(plot_path + 'HighSimilarity_Ratio_Seasons.png', dpi=300)
plt.show()
                        
## Plot Agreement:
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(7, 12), subplot_kw={'projection': ccrs.PlateCarree()})

for i, s  in zip([0,1,2,3], ['DJF', 'MAM', 'JJA', 'SON']):
        
        ax = axes[i]

        agreement_count_per_season[i].plot.imshow(ax=ax, cmap=pal_RdBu,  extend='both', 
                                              cbar_kwargs={'label': 'Model Agreement [%]'}, 
                                              vmin=0, vmax=100)
        ax.add_feature(ocean)
        ax.coastlines()
        ax.set_title(f'Agreement: Highest Similarity - {s}')
        
        # Create an inset for the bar plot
        left, bottom, width, height = 0.10, 0.12, 0.12, 0.22
        inset_ax = ax.inset_axes([left, bottom, width, height], facecolor='white')
        
plt.tight_layout()
plt.savefig(plot_path + 'Agreement_HighSimilarity_Ratio_Seasons.png', dpi=300)
plt.show()

##############################################
#   Check agreement on results with Pmask:  ##
##############################################

most_frequent_value_per_season = []
agreement_count_per_season = []

# Loop through each season
for season in seasons:
    #season = 'MAM'
    sel_season = [model.sel(season=season) for model in all_models_sdelta]
    
    value_counts = {v: xr.DataArray(np.zeros((len(lat), len(lon)), dtype=int), coords=[lat, lon], dims=["lat", "lon"]) for v in range(9)}
    value_counts[np.nan] = xr.DataArray(np.zeros((len(lat), len(lon)), dtype=int), coords=[lat, lon], dims=["lat", "lon"])
    most_frequent_value = xr.DataArray(np.zeros((len(lat), len(lon))), coords=[lat, lon], dims=["lat", "lon"])

    # Loop over all models and update value_counts
    for dataset in sel_season:
        sim_change = dataset['sim_change']
        value_counts = count_values(sim_change, value_counts)

    # Combine value_counts into a list for xr.apply_ufunc
    value_counts_list = [value_counts[v] for v in range(len(value_counts)-1)] + [value_counts[np.nan]]

    # Apply the function using xr.apply_ufunc
    most_frequent_value = xr.apply_ufunc(
        most_frequent_func,
        *value_counts_list,
        input_core_dims=[['lat', 'lon']] * 10,
        output_core_dims=[['lat', 'lon']],
        vectorize=True,
        output_dtypes=[np.float64]
    )    

    # Append the count for the current season to the list
    most_frequent_value_per_season.append(most_frequent_value)
    
    # Create an array to count the number of agreements
    agreement_count = count_agreements_across_models(model_ls=sel_season, var='sim_change', array_to_check=most_frequent_value)
    
    agreement_count_per_season.append(agreement_count)
    
bar_data = []
for i in [0,1,2,3]:
    PET_area_pos = float((cell_areas_da * (most_frequent_value_per_season[i] == 1)).sum())
    PR_area_pos = float((cell_areas_da * (most_frequent_value_per_season[i] == 2)).sum())
    PPETR_area_pos = float((cell_areas_da * (most_frequent_value_per_season[i] == 3)).sum())
    unclear_area_pos = float((cell_areas_da * (most_frequent_value_per_season[i] == 4)).sum())
    
    PET_area_neg = float((cell_areas_da * (most_frequent_value_per_season[i] == 5)).sum())
    PR_area_neg = float((cell_areas_da * (most_frequent_value_per_season[i] == 6)).sum())
    PPETR_area_neg = float((cell_areas_da * (most_frequent_value_per_season[i] == 7)).sum())
    unclear_area_neg = float((cell_areas_da * (most_frequent_value_per_season[i] == 8)).sum())
    
    # Total area
    total_area = PET_area_pos + PR_area_pos + PPETR_area_pos + unclear_area_pos + PET_area_neg + PR_area_neg + PPETR_area_neg + unclear_area_neg
    
    perc_PET_pos = (PET_area_pos / total_area) * 100
    perc_PR_pos = (PR_area_pos / total_area) * 100
    perc_PPETR_pos = (PPETR_area_pos / total_area) * 100
    perc_unclear_pos = (unclear_area_pos / total_area) * 100
    
    perc_PET_neg = (PET_area_neg / total_area) * 100
    perc_PR_neg = (PR_area_neg / total_area) * 100
    perc_PPETR_neg = (PPETR_area_neg / total_area) * 100
    perc_unclear_neg = (unclear_area_neg / total_area) * 100
    
    values = [perc_PET_pos, perc_PR_pos, perc_PPETR_pos, perc_unclear_pos,
              perc_PET_neg, perc_PR_neg, perc_PPETR_neg, perc_unclear_neg]
    
    bar_data.append(values)

x_values = np.arange(len(bar_data[0]))

   
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(7, 12), subplot_kw={'projection': ccrs.PlateCarree()})

for i, s  in zip([0,1,2,3], ['DJF', 'MAM', 'JJA', 'SON']):
        
        ax = axes[i]
        most_frequent_value_per_season[i].plot.imshow(ax=ax, cmap=cmap_highest_sim_Pmask, extend='both', 
                                              cbar_kwargs={'label': 'Highest Similarity'}, 
                                              vmin=1, vmax=8)
        ax.add_feature(ocean)
        ax.coastlines()
        ax.set_title(f'{s}')
        
        # Create an inset for the bar plot
        left, bottom, width, height = 0.10, 0.12, 0.12, 0.22
        inset_ax = ax.inset_axes([left, bottom, width, height], facecolor='white')
        inset_ax.bar(x_values, bar_data[i], color=colors_highest_sim_Pmask, edgecolor='black')
        inset_ax.set_xlabel('')
        inset_ax.set_title('Area [%]')
        inset_ax.set_yticks([0, 25, 50, 75])
        inset_ax.set_xticks([])
        inset_ax.set_xticklabels([])

plt.tight_layout()
plt.savefig(plot_path + 'HighSimilarity_Ratio_Pmask_Seasons.png', dpi=300)
plt.show()



## Plot Agreement:
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(7, 12), subplot_kw={'projection': ccrs.PlateCarree()})

for i, s  in zip([0,1,2,3], ['DJF', 'MAM', 'JJA', 'SON']):
        
    ax = axes[i]

    agreement_count_per_season[i].plot.imshow(ax=ax, cmap=pal_RdBu,  extend='both', 
                                          cbar_kwargs={'label': 'Model Agreement [%]'}, 
                                          vmin=0, vmax=100)
    ax.add_feature(ocean)
    ax.coastlines()
    ax.set_title(f'Agreement: Highest Similarity - {s}')

plt.tight_layout()
plt.savefig(plot_path + 'Agreement_HighSimilarity_Ratio_Pmask_Season.png', dpi=300)
plt.show()
