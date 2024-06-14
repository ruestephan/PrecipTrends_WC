# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:38:46 2024

@author: Ruth Stephan

"""

#packages:
import os
import numpy as np
import xarray as xr
import matplotlib as mpl
import pandas as pd

os.getcwd()

########################
## 1. Mask out oceans ##
########################

# open terrestrial mask:
path_masks = '/mnt/share/scratch/rs1155/data/CMIP6/'
terr_mask = xr.open_dataset(path_masks + 'terrestrial_mask_2x2degrees.nc')


def mask(var, scenario, realization = 'r1i1p1f1'):

################################
### collect files and check ####
################################

    #var = 'mrro'
    #scenario = 'ssp585'
    
    data_path = '/mnt/share/data/CMIP6/2d00_monthly/' + var + '/' + scenario + '/'
    output_path = '/mnt/share/scratch/rs1155/data/CMIP6/model_individuals/'
    
    
    # Get to the files that should be uses:
    files_in_directory = os.listdir(data_path)
    
    # Check if all files start with 'mrro'
    var_check = all(file.startswith(var) for file in files_in_directory)
    print(f"All files start with {var}: {var_check}")
    
    # If var_check is False, stop the function
    if not var_check:
        return 
    
    # Filter ralization and time period:
    wholeperiod_files = [file for file in files_in_directory if '185001-209912' in file]
    r1i1p1f1_files = [file for file in wholeperiod_files if realization in file]
    
    
    
    # Check if all entries are unique
    model_list = ['_'.join(filename.split('_')[2:3]) for filename in r1i1p1f1_files]
    are_unique = len(model_list) == len(set(model_list))
    print(f"All models (n = {len(model_list)}) are unique: {are_unique}")
    
    # If var_check is False, stop the function
    if not are_unique:
        return 
    
    #Check which models are already masked:
    # Get to the files that should be uses:
    #files_in_output_directory = os.listdir(output_path)
    #existing_files = [file for file in files_in_output_directory if var in file]
    
    #masked_models = np.unique((['_'.join(filename.split('_')[2:3]) for filename in existing_files]))
    #r1i1p1f1_files =  [file for file in files_in_output_directory if var in file]


    
#######################
### model  masking ####
#######################
    
    
    # 1. get all files of the variable in the folder:
    var_files_to_mask = [file for file in r1i1p1f1_files]
    
    for file in var_files_to_mask:
        
        
        model_name = '_'.join(file.split('_')[2:3])
        
       # if model_name in masked_models:
       #     print(f'model {model_name} already masked.')
       # else:
        # 2. open those files
        vardt = xr.open_dataset(os.path.join(data_path, file))
        # 3. time coordinates to year and month:
        vardt['time'] = vardt['time'].dt.strftime('%Y-%m')
        # 4. mask the data:
        masked_data = vardt.where(terr_mask['mrro'])
        #masked_data = masked_data.where(masked_data['lat'] >= -60, np.nan)
        # 5. safe 
        masked_data.to_netcdf(output_path + f'{var}_{scenario}_{model_name}_2x2degrees_masked.nc')
        print(f'Masked and saved variable {var} in scenario {scenario} for model {model_name}.')
            
    
mask(var='tas', scenario = 'ssp585')    
mask(var='pr', scenario = 'ssp585')    
mask(var='mrro', scenario = 'ssp585')    
mask(var='mrso', scenario = 'ssp585')       
mask(var='evspsbl', scenario = 'ssp585') 



#############################################
## 2. Combine variables in one netcdf file ##
#############################################


#paths:
data_path = '/mnt/share/scratch/rs1155/data/CMIP6/model_individuals/'
output_path = '/mnt/share/scratch/rs1155/data/CMIP6/output_data/'

# Get to the files that should be uses:
files_in_directory = os.listdir(data_path)
model_names = (['_'.join(filename.split('_')[2:3]) for filename in files_in_directory])

from collections import Counter

# Convert the list of model names to a set to get unique model names
unique_model_names = set(model_names)
unique_model_count = len(unique_model_names)
model_name_counts = Counter(model_names)

# Output the result
print(f"Unique model names: {unique_model_names}")
print(f"Number of unique model names: {unique_model_count}")

# Output the counts
for model_name, count in model_name_counts.items():
    print(f"Model name: {model_name}, Count: {count}")


# Filter out model names with a count not equal to 5 (tas, pr, evspsbl, mrro, mrso are available):
model_name = {mname for mname in unique_model_names if model_name_counts[mname] == 5}

# Filter out specific models
models_to_remove = ['CESM2-WACCM', 'CESM2'] # time coordinate wrong
unique_model_names_array = np.array(list(model_name))
model_names = unique_model_names_array[~np.isin(unique_model_names_array, models_to_remove)]




def combine_variables(model_name, plot=True):
    #model_name = 'MRI-ESM2-0'
    
    mask = 'masked'
    scenario = 'ssp585'
    
    # 1. get the data 
    tas_ds = xr.open_dataset(data_path + 'tas_' + scenario + '_' + model_name + '_2x2degrees_' + mask +'.nc')
    pr_ds = xr.open_dataset(data_path + 'pr_' + scenario + '_' + model_name + '_2x2degrees_' + mask +'.nc')
    evspsbl_ds = xr.open_dataset(data_path + 'evspsbl_' + scenario + '_' + model_name + '_2x2degrees_' + mask +'.nc')
    mrro_ds = xr.open_dataset(data_path + 'mrro_' + scenario + '_' + model_name + '_2x2degrees_' + mask +'.nc')
    mrso_ds = xr.open_dataset(data_path + 'mrso_' + scenario + '_' + model_name + '_2x2degrees_' + mask +'.nc')
    
    combined = xr.merge([tas_ds, pr_ds, evspsbl_ds, mrro_ds, mrso_ds])
    #combined = xr.merge([tas_ds, pr_ds,  mrro_ds, mrso_ds])
    
    # 2. reckognize time as datetime coordinate to faciliate access to years and months: 
    combined['time'] = pd.to_datetime(combined['time'])
    combined = combined.set_index(time='time')
    
    # 3. Clean coordinates and attributes:
    if 'height' in combined.variables:
        combined = combined.drop_vars('height')
        
    if combined.attrs:
        combined.attrs = {}

    # 4. combine the variables:
    combined = combined.assign(PETR = combined['pr']-combined['evspsbl']-combined['mrro'])
    combined = combined.assign(PET =  np.abs(combined['pr']) + np.abs(combined['evspsbl']))
    
    # 5. plot:
    if plot == True:
        palette = mpl.colormaps.get_cmap('viridis')
        palette.set_bad(color='lightgray')
        combined['mrro'].sel(time = slice('2000', '2100')).mean(dim='time').plot.imshow(cmap=palette)
        
    # 9. save the data:
    combined.to_netcdf(output_path + model_name + '_combined.nc')
    
    print(f'Combined, masked and saved model {model_name} for scenario {scenario}.')


for mname in model_names:
    combine_variables(model_name = mname)


combine_variables(model_name = 'CESM2-WACCM')






################################
## 3. Calculate ensemble mean ##
################################
mmean_output_path = '/mnt/share/scratch/rs1155/data/CMIP6/model_means/'

def mean_and_mask(var, scenario, realization = 'r1i1p1f1', plot=True):

################################
### collect files and check ####
################################

    #var = 'tas'
    #scenario = 'ssp585'
    
    data_path = '/mnt/share/data/CMIP6/2d00_monthly/' + var + '/' + scenario + '/'
    output_path = '/mnt/share/scratch/rs1155/data/CMIP6/model_means/'
    
    
    # Get to the files that should be uses:
    files_in_directory = os.listdir(data_path)
    
    # Check if all files start with 'mrro'
    var_check = all(file.startswith(var) for file in files_in_directory)
    print(f"All files start with {var}: {var_check}")
    
    # If var_check is False, stop the function
    if not var_check:
        return 
    
    # Filter ralization and time period:
    wholeperiod_files = [file for file in files_in_directory if '185001-209912' in file]
    r1i1p1f1_files = [file for file in wholeperiod_files if realization in file]
    
    # Check if all entries are unique
    model_list = ['_'.join(filename.split('_')[2:3]) for filename in r1i1p1f1_files]
    are_unique = len(model_list) == len(set(model_list))
    print(f"All models (n = {len(model_list)}) are unique: {are_unique}")
    
    # If var_check is False, stop the function
    if not are_unique:
        return 

   
    # Filter to files, where I know that the models provide all variables --> see filtering above
    filtered_files = [filename for filename in r1i1p1f1_files if '_'.join(filename.split('_')[2:3]) in model_names]

    ###############################
    ### model mean and masking ####
    ###############################
    
    
    # 1. get all files of the variable in the folder:
    var_files_to_mean = [file for file in filtered_files]
    
    vardt_ls = []
    vararray_ls = []
    for file in var_files_to_mean:
        # 2. open those files
        vardt = xr.open_dataset(os.path.join(data_path, file))
        vardt_ls.append(vardt)
        vararray_ls.append(vardt[var].values)
    # 3. calculate the means
    mean_array = np.mean(vararray_ls, axis=0)
    # 4. store mean vals again in a netcdf file:
    mean_vardt = vardt_ls[0].copy()  # Assuming vardt_ls[0] is your template dataset
    mean_vardt[var].values = mean_array
    
    # 5. time coordinates to year and month:
    mean_vardt['time'] = mean_vardt['time'].dt.strftime('%Y-%m')
    
    # 6. mask the data:
    masked_data = mean_vardt.where(terr_mask['mrro'])
      
    # 7. safe 
    mean_vardt.to_netcdf(output_path + f'{var}_{scenario}_mean_2x2degrees.nc')
    masked_data.to_netcdf(output_path + f'{var}_{scenario}_mean_2x2degrees_masked.nc')
    
    # 8. plot:
    if plot == True:
        palette = mpl.colormaps.get_cmap('viridis')
        palette.set_bad(color='lightgray')
    
        masked_data[var].sel(time = slice('2000', '2100')).mean(dim='time').plot.imshow(cmap=palette)
        
    print(f'Calculated mean, masked and saved variable {var} in scenario {scenario}.')
    
    
mean_and_mask(var='tas', scenario = 'ssp585')    
mean_and_mask(var='mrro', scenario = 'ssp585')    
mean_and_mask(var='mrso', scenario = 'ssp585')    
mean_and_mask(var='pr', scenario = 'ssp585')    
mean_and_mask(var='evspsbl', scenario = 'ssp585')    


# Combine the variables:
# 1. get the data 
scenario = 'ssp585'
tas_ds = xr.open_dataset(mmean_output_path + 'tas_' + scenario + '_mean_2x2degrees_masked.nc')
pr_ds = xr.open_dataset(mmean_output_path + 'pr_' + scenario + '_mean_2x2degrees_masked.nc')
evspsbl_ds = xr.open_dataset(mmean_output_path + 'evspsbl_' + scenario  + '_mean_2x2degrees_masked.nc')
mrro_ds = xr.open_dataset(mmean_output_path + 'mrro_' + scenario + '_mean_2x2degrees_masked.nc')
mrso_ds = xr.open_dataset(mmean_output_path + 'mrso_' + scenario  + '_mean_2x2degrees_masked.nc')

combined = xr.merge([tas_ds, pr_ds, evspsbl_ds, mrro_ds, mrso_ds])

# 2. reckognize time as datetime coordinate to faciliate access to years and months: 
combined['time'] = pd.to_datetime(combined['time'])
combined = combined.set_index(time='time')

# 3. drop the coordinate representing height
combined = combined.drop_vars('height')

# 4. combine the variables:
combined = combined.assign(PETR = combined['pr']-combined['evspsbl']-combined['mrro'])
combined = combined.assign(PET =  np.abs(combined['pr']) + np.abs(combined['evspsbl']))
    
# 5. save the data:
combined.to_netcdf(mmean_output_path + 'mean_combined.nc')

# Calculate Precip Masks:
'''
Function to calculate difference between period 2075-2100 and 2000-2025
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

modelmean = xr.open_dataset(mmean_output_path + 'mean_combined.nc')

yearly_minimum = 200 
minimum_change = 0.05
    

# mask out areas with precip < 200mm/year (deserts), see Wang et al.:
yearly_precip = modelmean['pr'].sel(time=slice('2000', '2100')).mean(dim='time')*60*60*24*30*12
yearly_precip_threshold = yearly_precip >= yearly_minimum

# and mask out areas that change less than 5 % per year:
d_pr = calc_vardiff(modelmean, 'pr')
pr_20002025 = modelmean['pr'].sel(time=slice('2000', '2025')).mean(dim='time')
yearly_change_greater = np.abs(d_pr)/pr_20002025 >= minimum_change


# and mask out areas that change less than 5 % per year:
d_pr = calc_vardiff_season(modelmean, 'pr')
pr_20002025 = modelmean['pr'].sel(time=slice('2000', '2025')).groupby('time.season').mean(dim='time')
seasons_change_greater = np.abs(d_pr)/pr_20002025 >= minimum_change

# save the masks:
yearly_precip_threshold.to_netcdf(path_masks + 'YearlyPrecip_Mask.nc')
yearly_change_greater.to_netcdf(path_masks + 'YearlyPrecipChange_Mask.nc')
seasons_change_greater.to_netcdf(path_masks + 'SeasonsPrecip_Mask.nc')


'''

# Create a mask that is True where all values are not NaN
terr_mask = mean_vardt.mrro.notnull().all(dim='time')
terr_mask.to_netcdf('/mnt/share/scratch/rs1155/data/CMIP6/terrestrial_mask_2x2degrees.nc')

# Plot intermediate data:
palette = mpl.colormaps.get_cmap('viridis')
palette.set_bad(color='lightgray')

mean_vardt['pr'].sel(time = ('1850-01')).plot.imshow()
masked_data['pr'].sel(time = slice('2000', '2100')).mean(dim='time').plot.imshow(cmap=palette)

'''
