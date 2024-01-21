import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pickle
import re
import seaborn as sns
import re
os.chdir('/home/c12049018/Documents/Experiment_NEW/Results/crop_test')

conditions = ['Baseline',
                'Control',
                'Salient',
                'Semantic',
                'Semantic_Salient']

files = os.listdir(os.getcwd())

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
resnet = [i for i in files if 'resnet' in i]

def process_resnet(file):

    rn50 = load(file)

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
                'model' : file[:-4],
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
    
    return rn50_df

resnet_df = pd.DataFrame()
for file in resnet:
    df = process_resnet(file)
    resnet_df = pd.concat([resnet_df, df])

#%% CLIP RN50

def preprocess_clip_resnet(file):
    rn50 = load(file)

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
                'model' : 'clip_'+file[:-4],
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

    return clip_rn50_df

Clip_resnet_df = pd.DataFrame()
for file in resnet:
    df = preprocess_clip_resnet(file)
    Clip_resnet_df = pd.concat([Clip_resnet_df, df])

#%% VIT

vit = [i for i in files if 'vit' in i]

def preprocess_vit(file):

    vit = load(file)               
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
    
    key_pairs = [('sal', 'saliency_RSA'), ('sem', 'semantic_RSA')]

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

vit_df = pd.DataFrame()
for file in vit:
    df = preprocess_vit(file)
    vit_df = pd.concat([vit_df, df])

#%% CLIP VIT
vit = load('vit.pkl')               
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
sal_sem_df = pd.concat([rn50_df, clip_rn50_df, vit_df, clip_vit_df]).reset_index()
sal_sem_df = pd.concat([resnet_df, Clip_resnet_df]).reset_index()
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pickle
from statsmodels.stats.multitest import fdrcorrection
import seaborn as sns
from scipy.stats import pearsonr as corr
from sklearn.manifold import MDS
from scipy.stats import ks_2samp
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon
from scipy.stats import ttest_rel

def load(file):   
    file = open(file, 'rb')
    data = pickle.load(file)
    file.close()
    return data

def dump(file_name, file):
    file_name = open(file_name, 'wb')
    # dump information to that file
    pickle.dump(file, file_name)
    # close the file
    file_name.close()

def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]

root = '/home/c12049018/Documents/NSD/more_layers/results'

# Helper functions

model_colors = {
    'vit': get_color_gradient("#15696a", "#26abac", 4),
    'clip_vit': get_color_gradient("#5d2f67", "#a558b6", 4),
    'rn50': get_color_gradient("#976110", "#f8a326", 4),
    'clip_rn50': get_color_gradient("#623631", "#FB575D", 4),
    'rn101': get_color_gradient("#976110", "#f8a326", 4),
    'clip_vit_l': get_color_gradient("#5d2f67", "#a558b6", 4),
}

conditions = ['Control', 'Salient', 'Semantic', 'Semantic_Salient']

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
    plt.title('')

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


# Plotting baselines


#%% Histograms
df = sal_sem_df

colors = {
    'vit': '#209596',
    'rn101': '#d88d1f',
    'clip_vit': '#8d4a9c',
    'clip_rn50': '#c84c4e',
    'rn50': '#d88d1f',
}

models = [['rn50', 'clip_rn50'], ['vit', 'clip_vit']]

legend = {
    'vit': 'ViT',
    'clip_vit': 'CLIP',
    'clip_rn50': 'CLIP',
    'rn50': 'RN',
    }

# Set the style and size
sns.set_style("ticks")
plt.figure(figsize=(6, 5))

for i, model in enumerate(models):
    print(model)
    
    # Plot the density plots for the first two models
    plt.subplot(1, 2, i+1)
    sns.kdeplot(df[df['model'] == model[0]]['sal_Baseline'], color=colors[model[0]], label=model[0], shade=True, alpha=0.6)
    sns.kdeplot(df[df['model'] == model[1]]['sal_Baseline'], color=colors[model[1]], label=model[1], shade=True, alpha=0.6)
    
    plt.xlabel('RSA', fontsize=16)
    if i == 0:
        plt.ylabel('Density', fontsize=16) 
        plt.title('Saliency Maps', fontsize=18)
    else: 
        plt.ylabel('', fontsize=16)

plt.show()

# Set the style and size
sns.set_style("ticks")
plt.figure(figsize=(6, 5))

for i, model in enumerate(models):
    print(model)
    
    # Plot the density plots for the first two models
    plt.subplot(1, 2, i+1)
    sns.kdeplot(df[df['model'] == model[0]]['sem_Baseline'], color=colors[model[0]], label=model[0], shade=True, alpha=0.6)
    sns.kdeplot(df[df['model'] == model[1]]['sem_Baseline'], color=colors[model[1]], label=model[1], shade=True, alpha=0.6)
    
    plt.xlabel('RSA', fontsize=16)
    if i == 0:
        plt.ylabel('Density', fontsize=16) 
        plt.title(f'Caption Embeddings', fontsize=18)
    else: 
        plt.ylabel('', fontsize=16)

plt.show()


# Kolmogorov
for i in ['sal', 'sem']:
    for e in models:
        data1 = df[df['model'] == e[0]][f'{i}_Baseline']
        data2 = df[df['model'] == e[1]][f'{i}_Baseline']

        stat, p = ks_2samp(data1, data2)

        print(f'{i} : {e}', 'Statistics=%.3f, p=%.3f' % (stat, p))

# Mann-Whitney
for i in ['sal', 'sem']:
    for e in models:
        data1 = df[df['model'] == e[0]][f'{i}_Baseline']
        data2 = df[df['model'] == e[1]][f'{i}_Baseline']

        stat, p = mannwhitneyu(data1, data2)

        print(f'{i} : {e}', 'Statistics=%.3f, p=%.3f' % (stat, p))

#%% Layer filters

# ResNet
df = sal_sem_df

# Generate a group identifier for each sequence of repeating bottleneck values
#df = df[df['model'].isin(['resnet', 'clip_rn50'])]
df['group'] = (df['bottleneck_number'] != df['bottleneck_number'].shift()).cumsum()

# For 'relu', get the last row for each group
last_relus = df[df.layer_type == 'relu'].groupby('group').tail(1)

# For 'conv', get all rows
all_convs = df[df.layer_type == 'conv']

# Combine the indices of last_relus and all_convs
selected_indices = last_relus.index.union(all_convs.index)

clip_rn50_pre_res_block_indices = [125, 128]
selected_indices = selected_indices.difference(clip_rn50_pre_res_block_indices)

# Create the boolean column
df['is_selected'] = df.index.isin(selected_indices)
rn101_filter = list(df[(df['model'] == 'resnet101') | (df['model'] == 'clip_resnet101')]['is_selected'])
rn50_filter = list(df[(df['model'] == 'resnet50') | (df['model'] == 'clip_resnet50')]['is_selected'])

# VIT
df = sal_sem_df
df = df[df['model'].isin(['vit', 'clip_vit'])]
# For 'relu', get the last row for each group
last_mlp = df.groupby(['model', 'block']).tail(4)
# For 'conv', get all rows
all_attn = df[(df.layer_type == 'attn') | (df.layer_type == 'self_attention')]
# Combine the indices of last_relus and all_convs
selected_indices = last_mlp.index.union(all_attn.index)
# Create the boolean column
df['is_selected'] = list(df.index.isin(selected_indices))
vit_filter = list(df['is_selected'][df['model'] == 'vit'])
vit_filter[-1] = False
clip_vit_filter = list(df['is_selected'][df['model'] == 'clip_vit'])
#%% Baseline plot all layers
df = sal_sem_df#[sal_sem_df['layer_type'].isin(['relu', 'conv', 'mlp', 'self_attention', 'attn', 'ln'])]

sns.set_style("ticks")
plt.figure(figsize=(20, 4))

n = 20

colors = {
    'vit': '#209596',
    'clip_vit': '#8d4a9c',
    'clip_rn50': '#c84c4e',
    'rn50': '#f8a326',
    }

model_title = {
    'vit': 'ImageNet ViT',
    'clip_vit': 'CLIP ViT',
    'clip_rn50': 'CLIP ResNet',
    'rn50': 'ImageNet ResNet',
    }

filters = {
    'vit': vit_filter,
    'clip_vit': clip_vit_filter,
    'rn50': rn50_filter,
    'clip_rn50': clip_rn50_filter,
    }

models = df['model'].unique()
for i, model in enumerate(models):
    #ax = plt.subplot(2, 2, i+1)
    subset = df[df['model'] == model].reset_index()
    sns.barplot(x=subset.index, y=subset['sal_Baseline'], color=colors[model])
    plt.title(model, fontsize=20)
    plt.show()


    plt.ylim(df['sal_Baseline'].min(), df['sal_Baseline'].max())
    plt.xlabel('Layer', fontsize=16)
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.title(model, fontsize=20)
    plt.ylabel('RSA', fontsize=20)    # Adjust x-axis ticks
    locs, labels = plt.xticks()  
    plt.xticks(locs[::n], subset.index[::n])

    #ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, color='gray')

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
    
    plt.tight_layout()

plt.show()

sns.set_style("ticks")
plt.figure(figsize=(12, 10))

models = df['model'].unique()
for i, model in enumerate(models):
    ax = plt.subplot(2, 2, i+1)
    subset = df[df['model'] == model][filters[model]].reset_index()
    sns.barplot(x=subset.index, y=subset['sem_Baseline'], color=colors[model])
    plt.ylim(df['sem_Baseline'].min(), df['sem_Baseline'].max())
    plt.xlabel('Layer', fontsize=16)
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.ylabel('RSA', fontsize=20)
    plt.title(model_title[model], fontsize=20)
    # Adjust x-axis ticks
    locs, labels = plt.xticks()  
    plt.xticks(locs[::n], subset.index[::n])

    ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, color='gray')

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
    
    plt.tight_layout()
plt.show()

#%% Correlation between saliency and semantics
corr(df['sal_Baseline'], df['sem_Baseline'])

#%% Delta Plots
#sns.set(style="whitegrid")

resnet_filter = rn50_filter + clip_rn50_filter
vistrnsfrm_filter = vit_filter + clip_vit_filter

conditions = ['Control', 'Salient', 'Semantic', 'Semantic_Salient']

colors = {
    'vit': '#209596',
    'clip_vit': '#8d4a9c',
    'clip_rn50': '#FF1018',
    'rn50': '#b6c648',
    }


def make_line_plot_moreLayers(df, models, cor_type, condition, model_colors, filter_ = None, ymin=None, ymax=None, ax=None):
    
    df = df[(df['model'].isin(models))].reset_index()
    if filter_ is not None:
        df = df[filter_].reset_index()
    model_names = df['model'].unique()
    column = f'{cor_type}_{condition}'
    #df = df.groupby(['model','block'])[column].mean().abs().reset_index()
    df[column] = df[column].abs()
    df = df[['model', column]]
    layer_order = list(df.index)
    #df = sal_sem_df[sal_sem_df['model'].isin(['rn50', 'clip_rn50'])]
                #& sal_sem_df['layer_type'].isin(['conv', 'relu'])]
    #df.groupby(['model','block'])['sal_Salient'].mean().abs().to_frame()
    #df['Layer'] = pd.Categorical(df['Layer'], categories=layer_order, ordered=True)

    # Ensuring there are as many colors as layers
    num_layers = len(layer_order)
    
    for model_name in model_names:
        model_df = df[df['model'] == model_name]
        model_layers = list(model_df.index)
        model_rsa = model_df[column].abs()

        # Using model_colors to color the lines. Adjust as needed.
        line_color = model_colors[model_name] # Default to 'black' if not found.

        ax.plot(model_layers, model_rsa, marker='', color=line_color, label=model_name)

    # Setting x-ticks labels as layer names
    #ax.set_xticks(np.arange(num_layers))
    #ax.set_xticklabels(layer_order)
    
    ax.set_ylabel(u'|∆ RSA|', fontsize=20)
    ax.set_title(f'{cor_type} : {condition}', fontsize=20)
    ax.set_ylim(ymin, ymax)
    
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    
    plt.tight_layout()

def plot_shared_layers(df, models, cor_type, condition, model_colors, ymin=None, ymax=None, ax=None):
    
    if ax is None:
        ax = plt.gca()

    sns.set(style="ticks")
    plt.figure(figsize=(12, 10))

    # Create a new x-tick for each unique layer (row)
    x_ticks = range(int(df.shape[0]/2))

    for model in models:
        # Filter data based on model
        model_data = df[df['model'] == model]
        rsa_values = model_data[f'{cor_type}_{condition}'].abs()
        
        sns.lineplot(x=x_ticks, y=rsa_values, marker='', color=model_colors[model], 
                     linewidth=1.5, label=model, 
                     ax=ax, legend=False)

    ax.set_xlabel('Layer', fontsize=20)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_ylabel(u'|∆ RSA|', fontsize=20)
    if condition == 'Semantic_Salient':
        ax.set_title('Semantic & Salient', fontsize=26)   
    else:
        ax.set_title(condition, fontsize=26)   

    ax.set_ylim(ymin, ymax)

    if 'rn' in models:
        y_ticks = np.arange(0, ymax, 0.02)  # Generate y-ticks
        ax.set_yticks(y_ticks)  # Set y-ticks
    elif ('vit' in models) and 'sem' == cor_type:
        y_ticks = np.arange(0, ymax, 0.02)  # Generate y-ticks
        ax.set_yticks(y_ticks)  # Set y-ticks 
        if condition == 'Control' or condition == 'Salient':
            ax.set_ylim(ymin, 0.12)
    
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
    
    plt.tight_layout()

def plot_delta(df_, models, cor_type, colors, layers_type = None, filter_ = None):

    df_ = df_[df_['model'].isin((models))]
    if layers_type is not None:
        df_ = df_[df_['layer_type'].isin(layers_type)]
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))

    if filter_ is not None:
        df_ = df_[filter_].reset_index()
        ymax = df_[[f'{cor_type}_'+i for i in conditions]].abs().max().max()
        for i, ax in enumerate(axes.ravel()):            
            plot_shared_layers(df_, models, cor_type, conditions[i], colors, ax=ax, ymax=ymax)
    else:
        ymax = df_[[f'{cor_type}_'+i for i in conditions]].abs().max().max()
        for i, ax in enumerate(axes.ravel()):   
            make_line_plot_moreLayers(df_, models, cor_type, conditions[i], colors, filter_ = filter_, ax=ax, ymax=ymax)


plot_delta(sal_sem_df, ['vit', 'clip_vit'], 'sal', colors)
plot_delta(sal_sem_df, ['rn50', 'clip_rn50'], 'sal', colors)


plot_delta(sal_sem_df, ['rn50', 'clip_rn50'], 'sal', colors, filter_=resnet_filter) 
plot_delta(sal_sem_df, ['rn50', 'clip_rn50'], 'sem', colors, filter_=resnet_filter) 

plot_delta(sal_sem_df, ['vit', 'clip_vit'], 'sal', colors, filter_=vistrnsfrm_filter)
plot_delta(sal_sem_df, ['vit', 'clip_vit'], 'sem', colors, filter_=vistrnsfrm_filter)


colors = {
    'clip_resnet50': '#FF1018',
    'resnet50': '#b6c648',
    'clip_resnet101': '#FF1018',
    'resnet101': '#b6c648',
    }
plot_delta(sal_sem_df, ['resnet101', 'clip_resnet101'], 'sal', colors)
