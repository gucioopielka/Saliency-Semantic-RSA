import numpy as np
import pandas as pd
import pickle
import re

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


def check_condition(layer, cond):
    # Check if 'Semantic_Salient' is in the input string
    if cond == 'Semantic' or cond == 'Salient':
        if cond in layer and 'Semantic_Salient' not in layer:
            return True
        else:
            return False   
    else:
        return cond in layer 
    

def populate_model_df(raw_data, df):

    models = df['model'].unique()
    model_indices = {model: np.where(df['model'] == model)[0] for model in models}

    # Populate columns with RSA values
    for model in models:

        for col_prefix, rsa_key in [('sal', 'saliency_RSA'), ('sem', 'semantic_RSA')]:
            relevant_vals = {layer: val for layer, val in raw_data[rsa_key].items() if layer.startswith(model)}
            
            for cond in conditions:
                filtered_vals = {layer: val for layer, val in relevant_vals.items() if check_condition(layer, cond)}
                
                for idx, (layer, val) in enumerate(filtered_vals.items()):
                    df.loc[model_indices[model][idx], f'{col_prefix}_{cond}'] = val 

    # Calculate Delta RSA (Distractor RSA - Baseline RSA)
    for col_type in ['sal', 'sem']:
        baseline_col = f"{col_type}_Baseline"

        for col in df.columns:
            if col.startswith(col_type) and col != baseline_col:
                df[col] -= df[baseline_col]

    return df


def calculate_n_model_layers(raw_data, model):
    return int(len([layer for layer in raw_data['saliency_RSA'].keys() if layer.startswith(model+'_Baseline')]))


def clean_layer_names_IN_ResNet(raw_layers):

    # Only baseline layers (others are repeated)
    n_layers = int(len(raw_layers)/5)
    raw_layers = raw_layers[:n_layers]

    # Extracting layer types
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

    # Extracting bottleneck number
    bottleneck_number = []
    for name in raw_layers:
        match = re.search(r'layer(\d+)\.(.+?)(?:\.|$)', name)
        if match:
            bottleneck_number.append(match.group(2))
        else:
            bottleneck_number.append(None)

    return layer_types, layer_blocks, bottleneck_number


def clean_layer_names_CLIP_ResNet(raw_layers):

    # Only baseline layers (others are repeated)
    n_layers = int(len(raw_layers)/5)
    raw_layers = raw_layers[:n_layers]

    # Extracting layer types
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
    #layer_blocks[:10] = ['pre_res_block']*10

    # Extracting bottleneck number
    bottleneck_number = []
    for name in raw_layers:
        match = re.search(r'layer(\d+)\.(.+?)(?:\.|$)', name)
        if match:
            bottleneck_number.append(match.group(2))
        else:
            bottleneck_number.append(None)
    #bottleneck_number[:10] = ['pre_res_block']*10

    return layer_types, layer_blocks, bottleneck_number


def clean_layer_names_CLIP_ViT(raw_layers):

    # Only baseline layers (others are repeated)
    n_layers = int(len(raw_layers)/5)
    raw_layers = raw_layers[:n_layers]

    # Extracting layer types
    layer_types = [layer.rstrip('_.0123456789') for layer in raw_layers]
    layer_types = [name.split('.')[-1] if '.' in name else name for name in layer_types]

    # Extracting block numbers
    layer_nums = []
    for name in raw_layers:
        match = re.search(r'resblocks.(\d+)\.', name)
        if match:
            layer_nums.append(int(match.group(1))+1)
        else:
            layer_nums.append(None)

    #layer_nums[:2], layer_nums[-1] = ['pre_att']*2, 'post_att'
    return layer_types, layer_nums


def clean_layer_names_IN_ViT(raw_layers):

    # Only baseline layers (others are repeated)
    #raw_layers[:2] = ['Baseline_pre_att.conv', 'Baseline_pre_att.dropout']
    n_layers = int(len(raw_layers)/5)
    raw_layers = raw_layers[:n_layers]

    # Extracting layer types
    layer_types = [layer.rstrip('_.0123456789') for layer in raw_layers]
    layer_types = [name.split('.')[-1] if '.' in name else name for name in layer_types]

    # Extracting block numbers
    layer_nums = []
    for name in raw_layers:
        match = re.search(r'layer_(\d+)\.', name)
        if match:
            layer_nums.append(int(match.group(1))+1)
        else:
            layer_nums.append(None)

    #layer_nums[:2], layer_nums[-2:] = ['pre_att']*2,['post_att']*2
    return layer_types, layer_nums



if __name__ == '__main__':

    ### ResNets ###
    resnet = load('data/resnet_raw.pkl')
    models = ['rn50', 'rn101', 'clip_rn50', 'clip_rn101']

    # Cleaning layer names
    layer_types, layer_blocks, bottleneck_number = [], [], []
    for model in models:

        if 'clip' in model:
            layer_types_model, layer_blocks_model, bottleneck_number_model = clean_layer_names_CLIP_ResNet(
                [layer for layer in resnet['saliency_RSA'].keys() if layer.startswith(model)])
        else:
            layer_types_model, layer_blocks_model, bottleneck_number_model = clean_layer_names_IN_ResNet(
                [layer for layer in resnet['saliency_RSA'].keys() if layer.startswith(model)])
            
        layer_types.extend(layer_types_model)
        layer_blocks.extend(layer_blocks_model)
        bottleneck_number.extend(bottleneck_number_model)

    # Creating the dataframe
    model_names = np.array([])
    for model in models:
        model_names = np.concatenate((
            model_names, 
            np.repeat(model, calculate_n_model_layers(raw_data=resnet, model=model))), 
            axis=None)

    ResNet_df = pd.DataFrame({
                'model' : model_names,
                'layer_type' : layer_types,
                'block' : layer_blocks, 
                'bottleneck_number' : bottleneck_number,
                'sal_Baseline' : np.zeros(len(model_names)),
                'sem_Baseline' : np.zeros(len(model_names)),
                'sal_Control' : np.zeros(len(model_names)),
                'sem_Control' : np.zeros(len(model_names)),
                'sal_Salient' : np.zeros(len(model_names)),
                'sem_Salient' : np.zeros(len(model_names)),              
                'sal_Semantic' : np.zeros(len(model_names)),
                'sem_Semantic' : np.zeros(len(model_names)), 
                'sal_Semantic_Salient' : np.zeros(len(model_names)),
                'sem_Semantic_Salient' : np.zeros(len(model_names)), 
                })

    ResNet_df = populate_model_df(raw_data=resnet, df=ResNet_df)


    ### ViTs ###
    vit = load('data/vit_raw.pkl') 
    models = ['vit_b16', 'vit_b32', 'vit_l14', 'clip_vit_b16', 'clip_vit_b32', 'clip_vit_l14']              

    # Cleaning layer names
    layer_types, layer_blocks = [], []
    for model in models:

        if 'clip' in model:
            layer_types_model, layer_blocks_model = clean_layer_names_CLIP_ViT(
                [layer for layer in vit['saliency_RSA'].keys() if layer.startswith(model)])
        else:
            layer_types_model, layer_blocks_model = clean_layer_names_IN_ViT(
                [layer for layer in vit['saliency_RSA'].keys() if layer.startswith(model)])
            
        layer_types.extend(layer_types_model)
        layer_blocks.extend(layer_blocks_model)

    # Creating the dataframe
    model_names = np.array([])
    for model in models:
        model_names = np.concatenate((
            model_names, 
            np.repeat(model, calculate_n_model_layers(raw_data=vit, model=model))), 
            axis=None)

    ViT_df = pd.DataFrame({
                'model' : model_names,
                'layer_type' : layer_types,
                'block' : layer_blocks, 
                'bottleneck_number' : np.full(len(model_names), np.nan),
                'sal_Baseline' : np.zeros(len(model_names)),
                'sem_Baseline' : np.zeros(len(model_names)),
                'sal_Control' : np.zeros(len(model_names)),
                'sem_Control' : np.zeros(len(model_names)),
                'sal_Salient' : np.zeros(len(model_names)),
                'sem_Salient' : np.zeros(len(model_names)),              
                'sal_Semantic' : np.zeros(len(model_names)),
                'sem_Semantic' : np.zeros(len(model_names)), 
                'sal_Semantic_Salient' : np.zeros(len(model_names)),
                'sem_Semantic_Salient' : np.zeros(len(model_names)), 
                })

    ViT_df = populate_model_df(raw_data=vit, df=ViT_df)
    ViT_df = ViT_df.replace(['c_fc', 'c_proj'], 'mlp')    

    pd.concat([ResNet_df, ViT_df]).to_csv('data/model_RSA.csv', index = False)



# #epoch = 1
# for i in range(1,6):
#     print('epoch', i)

#     batch = i
#     rn50 = load(f'/home/c12049018/Documents/training/ResNet/RSA/epoch_{str(batch)}.pkl')
#     saliency = [val for layer, val in rn50['saliency_RSA'].items() if 'clip' not in layer and 'Baseline' in layer]
#     semantic = [val for layer, val in rn50['semantic_RSA'].items() if 'clip' not in layer and 'Baseline' in layer]

#     raw_layers = [i for i in rn50['saliency_RSA'].keys() if 'clip' not in i and 'Baseline' in i]
#     layer_types = [layer.split('_')[-1].rstrip('0123456789') for layer in raw_layers]
#     layer_types = [name.split('.')[-1] if '.' in name else name for name in layer_types]

#     # Extracting block numbers
#     layer_blocks = []
#     for name in raw_layers:
#         match = re.search(r'layer(\d+)\.', name)
#         if match:
#             layer_blocks.append(match.group(1))
#         else:
#             layer_blocks.append(None)

#     rn50_df = pd.DataFrame({'model' : 'rn50',
#                     'layer_type' : layer_types,
#                     'block' : layer_blocks, 
#                 'saliency' : saliency,
#                 'semantic' : semantic})

#     filtered_df = rn50_df[~rn50_df['layer_type'].isin(['', 'fc', 'bn'])]
#     filtered_df = filtered_df[filtered_df['block'].notnull()].reset_index()

#     filtered_df = filtered_df[filtered_df['layer_type'] == 'relu']

#     layers(filtered_df, 'saliency')
#     layers(filtered_df, 'semantic')


