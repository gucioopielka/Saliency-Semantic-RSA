import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pickle
import re
import seaborn as sns
import re
os.chdir('/home/c12049018/Documents/Experiment_NEW/Results/All_layers')

conditions = ['Baseline',
                'Control',
                'Salient',
                'Semantic',
                'Semantic_Salient']

def load(file):   
    file = open(file, 'rb')
    data = pickle.load(file)
    file.close()
    return data

def layers(df, cor_type):
    num_layers = df['layer_type'].nunique()
    
    # Using a seaborn color palette
    colors = sns.color_palette("dark", n_colors=num_layers)
    color_dict = {layer: color for layer, color in zip(df['layer_type'].unique(), colors)}

    # Create a new figure
    plt.figure(figsize=(10, 6))

    # Loop through each layer type and plot the bars
    for layer_type, color in color_dict.items():
        subset = df[df['layer_type'] == layer_type]
        plt.bar(subset.index, subset[cor_type], label=layer_type, color=color, alpha=0.7)
    
    # Optionally, if you'd like a legend
    plt.legend(title='Layer Type')

    # Draw vertical lines between blocks
    previous_block = df['block'].iloc[0]
    block_ticks = []
    block_labels = []
    i_previous = 0  # Initializing i_previous to avoid UnboundLocalError
    for i, current_block in enumerate(df['block']):
        if current_block != previous_block:
            plt.axvline(x=i-0.5, color='k', linestyle='--', alpha=0.6)
            block_ticks.append((i-1 + i_previous) / 2)
            block_labels.append(str(previous_block))
            i_previous = i
        previous_block = current_block

    # Adding labels and title
    plt.xlabel('Layer')
    plt.ylabel('Rho')
    plt.title(cor_type)
    ymax = 0.5 if 'sem' in cor_type else 0.3 # df[cor_type].max()
    plt.ylim((df[cor_type].min(), ymax))

    ax = plt.gca()  # Get current Axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Adding custom x-ticks
    block_ticks.append((i + i_previous) / 2)  # Add last block's middle
    block_labels.append(str(previous_block))  # Add last block number
    plt.xticks(block_ticks, block_labels)

    #plt.xticks(ticks=range(len(df['layer_type'])), labels=df['layer_type'], rotation=45)
    #plt.grid(axis='y')
    plt.tight_layout()

    plt.show()

def check_condition(layer, cond):
    # Check if 'Semantic_Salient' is in the input string
    if cond == 'Semantic' or cond == 'Salient':
        if cond in layer and 'Semantic_Salient' not in layer:
            return True
        else:
            return False   
    else:
        return cond in layer 


#%% RN50
rn50 = load('rn50_delta.pkl')

# Cleaning layer names
raw_layers = [i for i in rn50['saliency_RSA'].keys() if 'clip' not in i]
n_layers = int(len(raw_layers)/5)
raw_layers = raw_layers[:n_layers]

layer_types = [layer.split('_')[-1].rstrip('0123456789.') for layer in raw_layers]
layer_types = [name.split('.')[-1] if '.' in name else name for name in layer_types]

# Extracting block numbers
layer_blocks = []
for name in raw_layers:
    match = re.search(r'layer(\d+)\.', name)
    if match:
        layer_blocks.append(match.group(1))
    else:
        layer_blocks.append(None)
layer_blocks[:4] = ['pre_res_block']*4
layer_blocks[-1] = 'post_res_block' 

# Extracting bottleneck number
bottleneck_number = []
for name in raw_layers:
    match = re.search(r'layer(\d+)\.(.+?)(?:\.|$)', name)
    if match:
        bottleneck_number.append(match.group(2))
    else:
        bottleneck_number.append(None)
bottleneck_number[:4] = ['pre_res_block']*4
bottleneck_number[-1] = 'post_res_block' 


rn50_df = pd.DataFrame({
            'model' : 'rn50',
            'layer_type' : layer_types,
            'block' : layer_blocks, 
            'bottleneck_number' : bottleneck_number,
            'sal_Baseline' : np.zeros(n_layers),
            'sem_Baseline' : np.zeros(n_layers),
            'sal_Control' : np.zeros(n_layers),
            'sem_Control' : np.zeros(n_layers),
            'sal_Salient' : np.zeros(n_layers),
            'sem_Salient' : np.zeros(n_layers),              
            'sal_Semantic' : np.zeros(n_layers),
            'sem_Semantic' : np.zeros(n_layers), 
            'sal_Semantic_Salient' : np.zeros(n_layers),
            'sem_Semantic_Salient' : np.zeros(n_layers), 
            })

# Key pairs to iterate over both conditions
key_pairs = [('sal', 'saliency_RSA'), ('sem', 'semantic_RSA')]

# Iterate over both pairs
for col_prefix, rsa_key in key_pairs:
    # Extract relevant layers and values based on the condition, excluding 'clip'
    relevant_vals = {layer: val for layer, val in rn50[rsa_key].items() if 'clip' not in layer}
    
    for cond in conditions:
        # Filter the relevant layers/values based on another condition
        filtered_vals = {layer: val for layer, val in relevant_vals.items() if check_condition(layer, cond)}
        
        # Assign the values to the dataframe
        for idx, (layer, val) in enumerate(filtered_vals.items()):
            rn50_df.loc[idx, f'{col_prefix}_{cond}'] = val 

# Loop over each column type (sal and sem)
for col_type in ['sal', 'sem']:
    baseline_col = f"{col_type}_Baseline"
    # Iterate over the other columns with the same type
    for col in rn50_df.columns:
        if col.startswith(col_type) and col != baseline_col:
            rn50_df[col] -= rn50_df[baseline_col]


#%% CLIP RN50
# Cleaning layer names
raw_layers = [i for i in rn50['saliency_RSA'].keys() if 'clip' in i]
n_layers = int(len(raw_layers)/5)
raw_layers = raw_layers[:n_layers]

layer_types = [layer.split('_')[-1].rstrip('0123456789-.') for layer in raw_layers]
layer_types = [name.split('.')[-1] if '.' in name else name for name in layer_types]

# Extracting block numbers
layer_blocks = []
for name in raw_layers:
    match = re.search(r'layer(\d+)\.', name)
    if match:
        layer_blocks.append(match.group(1))
    else:
        layer_blocks.append(None)
layer_blocks[:10] = ['pre_res_block']*10

# Extracting bottleneck number
bottleneck_number = []
for name in raw_layers:
    match = re.search(r'layer(\d+)\.(.+?)(?:\.|$)', name)
    if match:
        bottleneck_number.append(match.group(2))
    else:
        bottleneck_number.append(None)
bottleneck_number[:10] = ['pre_res_block']*10

clip_rn50_df = pd.DataFrame({
            'model' : 'clip_rn50',
            'layer_type' : layer_types,
            'block' : layer_blocks, 
            'bottleneck_number' : bottleneck_number,
            'sal_Baseline' : np.zeros(n_layers),
            'sem_Baseline' : np.zeros(n_layers),
            'sal_Control' : np.zeros(n_layers),
            'sem_Control' : np.zeros(n_layers),
            'sal_Salient' : np.zeros(n_layers),
            'sem_Salient' : np.zeros(n_layers),              
            'sal_Semantic' : np.zeros(n_layers),
            'sem_Semantic' : np.zeros(n_layers), 
            'sal_Semantic_Salient' : np.zeros(n_layers),
            'sem_Semantic_Salient' : np.zeros(n_layers), 
            })

# Key pairs to iterate over both conditions
key_pairs = [('sal', 'saliency_RSA'), ('sem', 'semantic_RSA')]

# Iterate over both pairs
for col_prefix, rsa_key in key_pairs:
    # Extract relevant layers and values based on the condition, excluding 'clip'
    relevant_vals = {layer: val for layer, val in rn50[rsa_key].items() if 'clip' in layer}
    
    for cond in conditions:
        # Filter the relevant layers/values based on another condition
        filtered_vals = {layer: val for layer, val in relevant_vals.items() if check_condition(layer, cond)}
        
        # Assign the values to the dataframe
        for idx, (layer, val) in enumerate(filtered_vals.items()):
            clip_rn50_df.loc[idx, f'{col_prefix}_{cond}'] = val 

# Loop over each column type (sal and sem)
for col_type in ['sal', 'sem']:
    baseline_col = f"{col_type}_Baseline"
    # Iterate over the other columns with the same type
    for col in clip_rn50_df.columns:
        if col.startswith(col_type) and col != baseline_col:
            clip_rn50_df[col] -= clip_rn50_df[baseline_col]


#%% VIT
vit = load('vit_delta.pkl')               
saliency = [val for layer, val in vit['saliency_RSA'].items() if 'clip' not in layer]
semantic = [val for layer, val in vit['semantic_RSA'].items() if 'clip' not in layer]

# Layer Names
raw_layers = [i for i in vit['saliency_RSA'].keys() if 'clip' not in i]
raw_layers[:2] = ['Baseline_pre_att.conv', 'Baseline_pre_att.dropout']
n_layers = int(len(raw_layers)/5)
raw_layers = raw_layers[:n_layers]

layer_types = [layer.rstrip('_.0123456789') for layer in raw_layers]
layer_types = [name.split('.')[-1] if '.' in name else name for name in layer_types]

# Layer numbers
layer_nums = []
for name in raw_layers:
    match = re.search(r'layer_(\d+)\.', name)
    if match:
        layer_nums.append(int(match.group(1))+1)
    else:
        layer_nums.append(None)
layer_nums[:2], layer_nums[-2:] = ['pre_att']*2,['post_att']*2

vit_df = pd.DataFrame({
            'model' : 'vit',
            'layer_type' : layer_types,
            'block' : layer_nums, 
            'bottleneck_number' : np.full(n_layers, np.nan),
            'sal_Baseline' : np.zeros(n_layers),
            'sem_Baseline' : np.zeros(n_layers),
            'sal_Control' : np.zeros(n_layers),
            'sem_Control' : np.zeros(n_layers),
            'sal_Salient' : np.zeros(n_layers),
            'sem_Salient' : np.zeros(n_layers),              
            'sal_Semantic' : np.zeros(n_layers),
            'sem_Semantic' : np.zeros(n_layers), 
            'sal_Semantic_Salient' : np.zeros(n_layers),
            'sem_Semantic_Salient' : np.zeros(n_layers), 
            })

# Iterate over both pairs
for col_prefix, rsa_key in key_pairs:
    # Extract relevant layers and values based on the condition, excluding 'clip'
    relevant_vals = {layer: val for layer, val in vit[rsa_key].items() if 'clip' not in layer}
    
    for cond in conditions:
        # Filter the relevant layers/values based on another condition
        filtered_vals = {layer: val for layer, val in relevant_vals.items() if check_condition(layer, cond)}
        # Assign the values to the dataframe
        for idx, (layer, val) in enumerate(filtered_vals.items()):
            vit_df.loc[idx, f'{col_prefix}_{cond}'] = val 

# Loop over each column type (sal and sem)
for col_type in ['sal', 'sem']:
    baseline_col = f"{col_type}_Baseline"
    # Iterate over the other columns with the same type
    for col in vit_df.columns:
        if col.startswith(col_type) and col != baseline_col:
            vit_df[col] -= vit_df[baseline_col]


#%% CLIP VIT
vit = load('vit_delta.pkl')               
saliency = [val for layer, val in vit['saliency_RSA'].items() if 'clip' in layer]
semantic = [val for layer, val in vit['semantic_RSA'].items() if 'clip' in layer]

# Layer Names
raw_layers = [i for i in vit['saliency_RSA'].keys() if 'clip' in i]
raw_layers[:2] = ['pre_att.conv', 'pre_att.dropout']
n_layers = int(len(raw_layers)/5)
raw_layers = raw_layers[:n_layers]

layer_types = [layer.rstrip('_.0123456789') for layer in raw_layers]
layer_types = [name.split('.')[-1] if '.' in name else name for name in layer_types]

# Layer numbers
layer_nums = []
for name in raw_layers:
    match = re.search(r'resblocks.(\d+)\.', name)
    if match:
        layer_nums.append(int(match.group(1))+1)
    else:
        layer_nums.append(None)
layer_nums[:2], layer_nums[-1] = ['pre_att']*2, 'post_att'

clip_vit_df = pd.DataFrame({
            'model' : 'clip_vit',
            'layer_type' : layer_types,
            'block' : layer_nums, 
            'bottleneck_number' : np.full(n_layers, np.nan),
            'sal_Baseline' : np.zeros(n_layers),
            'sem_Baseline' : np.zeros(n_layers),
            'sal_Control' : np.zeros(n_layers),
            'sem_Control' : np.zeros(n_layers),
            'sal_Salient' : np.zeros(n_layers),
            'sem_Salient' : np.zeros(n_layers),              
            'sal_Semantic' : np.zeros(n_layers),
            'sem_Semantic' : np.zeros(n_layers), 
            'sal_Semantic_Salient' : np.zeros(n_layers),
            'sem_Semantic_Salient' : np.zeros(n_layers), 
            })

clip_vit_df = clip_vit_df.replace(['c_fc', 'c_proj'], 'mlp')    

# Iterate over both pairs
for col_prefix, rsa_key in key_pairs:
    # Extract relevant layers and values based on the condition, excluding 'clip'
    relevant_vals = {layer: val for layer, val in vit[rsa_key].items() if 'clip' in layer}
    
    for cond in conditions:
        # Filter the relevant layers/values based on another condition
        filtered_vals = {layer: val for layer, val in relevant_vals.items() if check_condition(layer, cond)}
        
        # Assign the values to the dataframe
        for idx, (layer, val) in enumerate(filtered_vals.items()):
            clip_vit_df.loc[idx, f'{col_prefix}_{cond}'] = val 

# Loop over each column type (sal and sem)
for col_type in ['sal', 'sem']:
    baseline_col = f"{col_type}_Baseline"
    # Iterate over the other columns with the same type
    for col in clip_vit_df.columns:
        if col.startswith(col_type) and col != baseline_col:
            clip_vit_df[col] -= clip_vit_df[baseline_col]



# %%
pd.concat([rn50_df, clip_rn50_df, vit_df, clip_vit_df]).to_csv('/home/c12049018/Documents/Experiment_NEW/Results/All_layers/sal_sem_RSA_experiment.csv', index = False)
#%% Depth check 
rn101 = load('rn101.pkl')

# Cleaning layer names
raw_layers = [i for i in rn101['saliency_RSA'].keys() if 'clip' not in i]
n_layers = int(len(raw_layers)/5)
raw_layers = raw_layers[:n_layers]

layer_types = [layer.split('_')[-1].rstrip('0123456789.') for layer in raw_layers]
layer_types = [name.split('.')[-1] if '.' in name else name for name in layer_types]

# Extracting block numbers
layer_blocks = []
for name in raw_layers:
    match = re.search(r'layer(\d+)\.', name)
    if match:
        layer_blocks.append(match.group(1))
    else:
        layer_blocks.append(None)
layer_blocks[:4] = ['pre_res_block']*4
layer_blocks[-1] = 'post_res_block' 

# Extracting bottleneck number
bottleneck_number = []
for name in raw_layers:
    match = re.search(r'layer(\d+)\.(.+?)(?:\.|$)', name)
    if match:
        bottleneck_number.append(match.group(2))
    else:
        bottleneck_number.append(None)
bottleneck_number[:4] = ['pre_res_block']*4
bottleneck_number[-1] = 'post_res_block' 


rn101_df = pd.DataFrame({
            'model' : 'rn101',
            'layer_type' : layer_types,
            'block' : layer_blocks, 
            'bottleneck_number' : bottleneck_number,
            'sal_Baseline' : np.zeros(n_layers),
            'sem_Baseline' : np.zeros(n_layers),
            'sal_Control' : np.zeros(n_layers),
            'sem_Control' : np.zeros(n_layers),
            'sal_Salient' : np.zeros(n_layers),
            'sem_Salient' : np.zeros(n_layers),              
            'sal_Semantic' : np.zeros(n_layers),
            'sem_Semantic' : np.zeros(n_layers), 
            'sal_Semantic_Salient' : np.zeros(n_layers),
            'sem_Semantic_Salient' : np.zeros(n_layers), 
            })

# Key pairs to iterate over both conditions
key_pairs = [('sal', 'saliency_RSA'), ('sem', 'semantic_RSA')]

# Iterate over both pairs
for col_prefix, rsa_key in key_pairs:
    # Extract relevant layers and values based on the condition, excluding 'clip'
    relevant_vals = {layer: val for layer, val in rn101[rsa_key].items() if 'clip' not in layer}
    
    for cond in conditions:
        # Filter the relevant layers/values based on another condition
        filtered_vals = {layer: val for layer, val in relevant_vals.items() if check_condition(layer, cond)}
        
        # Assign the values to the dataframe
        for idx, (layer, val) in enumerate(filtered_vals.items()):
            rn101_df.loc[idx, f'{col_prefix}_{cond}'] = val 

# Loop over each column type (sal and sem)
for col_type in ['sal', 'sem']:
    baseline_col = f"{col_type}_Baseline"
    # Iterate over the other columns with the same type
    for col in rn101_df.columns:
        if col.startswith(col_type) and col != baseline_col:
            rn101_df[col] -= rn101_df[baseline_col]


clip_vit_l = load('vit-l.pkl')               
saliency = [val for layer, val in clip_vit_l['saliency_RSA'].items() if 'clip' in layer]
semantic = [val for layer, val in clip_vit_l['semantic_RSA'].items() if 'clip' in layer]

# Layer Names
raw_layers = [i for i in clip_vit_l['saliency_RSA'].keys() if 'clip' in i]
raw_layers[:2] = ['pre_att.conv', 'pre_att.dropout']
n_layers = int(len(raw_layers)/5)
raw_layers = raw_layers[:n_layers]

layer_types = [layer.rstrip('_.0123456789') for layer in raw_layers]
layer_types = [name.split('.')[-1] if '.' in name else name for name in layer_types]

# Layer numbers
layer_nums = []
for name in raw_layers:
    match = re.search(r'resblocks.(\d+)\.', name)
    if match:
        layer_nums.append(int(match.group(1))+1)
    else:
        layer_nums.append(None)
layer_nums[:2], layer_nums[-1] = ['pre_att']*2, 'post_att'

clip_vit_df = pd.DataFrame({
            'model' : 'clip_vit_l',
            'layer_type' : layer_types,
            'block' : layer_nums, 
            'bottleneck_number' : np.full(n_layers, np.nan),
            'sal_Baseline' : np.zeros(n_layers),
            'sem_Baseline' : np.zeros(n_layers),
            'sal_Control' : np.zeros(n_layers),
            'sem_Control' : np.zeros(n_layers),
            'sal_Salient' : np.zeros(n_layers),
            'sem_Salient' : np.zeros(n_layers),              
            'sal_Semantic' : np.zeros(n_layers),
            'sem_Semantic' : np.zeros(n_layers), 
            'sal_Semantic_Salient' : np.zeros(n_layers),
            'sem_Semantic_Salient' : np.zeros(n_layers), 
            })

clip_vit_df = clip_vit_df.replace(['c_fc', 'c_proj'], 'mlp')    

key_pairs = [('sal', 'saliency_RSA'), ('sem', 'semantic_RSA')]

# Iterate over both pairs
for col_prefix, rsa_key in key_pairs:
    # Extract relevant layers and values based on the condition, excluding 'clip'
    relevant_vals = {layer: val for layer, val in clip_vit_l[rsa_key].items() if 'clip' in layer}
    
    for cond in conditions:
        # Filter the relevant layers/values based on another condition
        filtered_vals = {layer: val for layer, val in relevant_vals.items() if check_condition(layer, cond)}
        
        # Assign the values to the dataframe
        for idx, (layer, val) in enumerate(filtered_vals.items()):
            clip_vit_df.loc[idx, f'{col_prefix}_{cond}'] = val 

# Loop over each column type (sal and sem)
for col_type in ['sal', 'sem']:
    baseline_col = f"{col_type}_Baseline"
    # Iterate over the other columns with the same type
    for col in clip_vit_df.columns:
        if col.startswith(col_type) and col != baseline_col:
            clip_vit_df[col] -= clip_vit_df[baseline_col]


#%%
pd.concat([rn101_df, clip_vit_df]).to_csv('/home/c12049018/Documents/Experiment_NEW/Results/All_layers/sal_sem_RSA_experiment_depth.csv', index = False)
#%% epochs
#epoch = 1
for i in range(1,6):
    print('epoch', i)

    batch = i
    rn50 = load(f'/home/c12049018/Documents/training/ResNet/RSA/epoch_{str(batch)}.pkl')
    saliency = [val for layer, val in rn50['saliency_RSA'].items() if 'clip' not in layer and 'Baseline' in layer]
    semantic = [val for layer, val in rn50['semantic_RSA'].items() if 'clip' not in layer and 'Baseline' in layer]

    raw_layers = [i for i in rn50['saliency_RSA'].keys() if 'clip' not in i and 'Baseline' in i]
    layer_types = [layer.split('_')[-1].rstrip('0123456789') for layer in raw_layers]
    layer_types = [name.split('.')[-1] if '.' in name else name for name in layer_types]

    # Extracting block numbers
    layer_blocks = []
    for name in raw_layers:
        match = re.search(r'layer(\d+)\.', name)
        if match:
            layer_blocks.append(match.group(1))
        else:
            layer_blocks.append(None)

    rn50_df = pd.DataFrame({'model' : 'rn50',
                    'layer_type' : layer_types,
                    'block' : layer_blocks, 
                'saliency' : saliency,
                'semantic' : semantic})

    filtered_df = rn50_df[~rn50_df['layer_type'].isin(['', 'fc', 'bn'])]
    filtered_df = filtered_df[filtered_df['block'].notnull()].reset_index()

    filtered_df = filtered_df[filtered_df['layer_type'] == 'relu']

    layers(filtered_df, 'saliency')
    layers(filtered_df, 'semantic')

# %%
swsl = load('swsl.pkl')

# Cleaning layer names
raw_layers = [i for i in swsl['saliency_RSA'].keys() if 'clip' not in i]
n_layers = int(len(raw_layers)/5)
raw_layers = raw_layers[:n_layers]

layer_types = [layer.split('_')[-1].rstrip('0123456789.') for layer in raw_layers]
layer_types = [name.split('.')[-1] if '.' in name else name for name in layer_types]

# Extracting block numbers
layer_blocks = []
for name in raw_layers:
    match = re.search(r'layer(\d+)\.', name)
    if match:
        layer_blocks.append(match.group(1))
    else:
        layer_blocks.append(None)
layer_blocks[:4] = ['pre_res_block']*4
layer_blocks[-1] = 'post_res_block' 

# Extracting bottleneck number
bottleneck_number = []
for name in raw_layers:
    match = re.search(r'layer(\d+)\.(.+?)(?:\.|$)', name)
    if match:
        bottleneck_number.append(match.group(2))
    else:
        bottleneck_number.append(None)
bottleneck_number[:4] = ['pre_res_block']*4
bottleneck_number[-1] = 'post_res_block' 


swsl_df = pd.DataFrame({
            'model' : 'swsl',
            'layer_type' : layer_types,
            'block' : layer_blocks, 
            'bottleneck_number' : bottleneck_number,
            'sal_Baseline' : np.zeros(n_layers),
            'sem_Baseline' : np.zeros(n_layers),
            'sal_Control' : np.zeros(n_layers),
            'sem_Control' : np.zeros(n_layers),
            'sal_Salient' : np.zeros(n_layers),
            'sem_Salient' : np.zeros(n_layers),              
            'sal_Semantic' : np.zeros(n_layers),
            'sem_Semantic' : np.zeros(n_layers), 
            'sal_Semantic_Salient' : np.zeros(n_layers),
            'sem_Semantic_Salient' : np.zeros(n_layers), 
            })

# Key pairs to iterate over both conditions
key_pairs = [('sal', 'saliency_RSA'), ('sem', 'semantic_RSA')]

conditions = ['Baseline', 'Control', 'Salient', 'Semantic', 'Semantic_Salient']

# Iterate over both pairs
for col_prefix, rsa_key in key_pairs:
    # Extract relevant layers and values based on the condition, excluding 'clip'
    relevant_vals = {layer: val for layer, val in swsl[rsa_key].items() if 'clip' not in layer}
    
    for cond in conditions:
        # Filter the relevant layers/values based on another condition
        filtered_vals = {layer: val for layer, val in relevant_vals.items() if check_condition(layer, cond)}
        
        # Assign the values to the dataframe
        for idx, (layer, val) in enumerate(filtered_vals.items()):
            swsl_df.loc[idx, f'{col_prefix}_{cond}'] = val 

# Loop over each column type (sal and sem)
for col_type in ['sal', 'sem']:
    baseline_col = f"{col_type}_Baseline"
    # Iterate over the other columns with the same type
    for col in swsl_df.columns:
        if col.startswith(col_type) and col != baseline_col:
            swsl_df[col] -= swsl_df[baseline_col]

swsl_df.to_csv('/home/c12049018/Documents/Experiment_NEW/Results/All_layers/swsl_results.csv', index = False)

# %%
