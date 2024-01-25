import os
import pandas as pd
import pickle
import numpy as np
from NSD import NSD_get_shared_sub_data
from create_distractors import applyCropToImg
from RSA import get_RDM, compute_RSA, RSA
from tqdm import tqdm
import ast
from torch.utils.data import Dataset
from PIL import Image
import torchvision
import clip
from torch import cuda
device = "cuda" if cuda.is_available() else "cpu"


n_cores = "12"
os.environ["OMP_NUM_THREADS"] = n_cores
os.environ["OPENBLAS_NUM_THREADS"] = n_cores
os.environ["MKL_NUM_THREADS"] = n_cores
os.environ["VECLIB_MAXIMUM_THREADS"] = n_cores
os.environ["NUMEXPR_NUM_THREADS"] = n_cores


# Load NSD info
nsd_info_shared = pd.read_csv('/home/c12049018/nsd_shared_imgs.csv', dtype={'cocoId': str, 'nsdId': str})


class NSD_SharedDataset(Dataset):
    def __init__(self, preprocess, device):
        self.imgs_paths = [f"/home/Public/Datasets/COCO/train2017/{file+'.jpg'}" for file in nsd_info_shared['cocoId']]
        self.cropboxes = nsd_info_shared['cropBox']
        self.preprocess = preprocess
        self.device = device

    def __len__(self):
        return len(self.imgs_paths)
    
    def __getitem__(self, idx):
        # Load the image
        img_path = self.imgs_paths[idx]
        cropBox = self.cropboxes[idx]
        img = Image.open(img_path)
        img = applyCropToImg(np.array(img), ast.literal_eval(cropBox))

        # Preprocess the image and send it to the chosen device ('cpu' or 'cuda')
        if self.preprocess:
            img = self.preprocess(Image.fromarray(img)).to(self.device)
        
        return img 


### Resnet Models
rn50 = torchvision.models.resnet50(weights='IMAGENET1K_V1').to(device).eval()
clip_rn50 = clip.load("RN50", device=device, jit=False)[0]
rn101 = torchvision.models.resnet101(weights='IMAGENET1K_V1').to(device).eval()
clip_rn101 = clip.load("RN101", device=device, jit=False)[0]

# Load preprocessing pipeline from CLIP
preprocess_resnet = clip.load("RN50", device=device, jit=False)[1]

# Initiate RSA class
resnets = RSA(
    device=device,
    general_preprocess=preprocess_resnet,
    rn50 = rn50,
    clip_rn50 = clip_rn50.visual,
    rn101 = rn101,
    clip_rn101 = clip_rn101.visual
    )


### Vit Models
vit_b16 = torchvision.models.vit_b_16(weights='IMAGENET1K_V1').to(device).eval()
clip_vit_b16 = clip.load("ViT-B/16", device=device, jit=False)[0]
vit_b32 = torchvision.models.vit_b_32(weights='IMAGENET1K_V1').to(device).eval()
clip_vit_b32 = clip.load("ViT-B/32", device=device, jit=False)[0]
vit_l14 = torchvision.models.vit_l_16(weights='IMAGENET1K_V1').to(device).eval()
clip_vit_l14 = clip.load("ViT-L/14", device=device, jit=False)[0]

# Load preprocessing pipeline from CLIP
preprocess_vit = clip.load("ViT-B/16", device=device, jit=False)[1]

# CLIP ViT only accepts half tensors
specific_preprocess = {
    'vit_b16': None,
    'clip_vit_b16': lambda x: x.half(),
    'vit_b32': None,
    'clip_vit_b32': lambda x: x.half(),
    'vit_l14': None,
    'clip_vit_l14': lambda x: x.half()
}

# Initiate RSA class
vits = RSA(
    device=device,
    general_preprocess=preprocess_vit,
    specific_preprocess=specific_preprocess,
    vit_b16 = vit_b16,
    clip_vit_b16 = clip_vit_b16.visual,
    vit_b32 = vit_b32,
    clip_vit_b32 = clip_vit_b32.visual,
    vit_l14 = vit_l14,
    clip_vit_l14 = clip_vit_l14.visual
    )

### Load NSD dataset
model_names = ['rn50', 'clip_rn50', 'rn101', 'clip_rn101', 'vit_b16', 'clip_vit_b16', 'vit_b32', 'clip_vit_b32', 'vit_l14', 'clip_vit_l14']
dataset_resnets = NSD_SharedDataset(preprocess=preprocess_resnet, device=device)
dataset_vits = NSD_SharedDataset(preprocess=preprocess_vit, device=device)

### Compute RDMs for each model and layer

if 'model_rdms_raw.pkl' not in os.listdir('data'):
    model_rdms = {}
    models_seen = []
else:
    model_rdms = pickle.load(open('data/model_rdms_raw.pkl', 'rb'))
    models_seen = list(model_rdms.keys())

for model_idx, model in enumerate(model_names):
    if model in models_seen:
        continue

    print(f'Computing activations for {model}...')
    model_run = eval(model) if 'clip' not in model else eval(model).visual
    if 'rn' in model:
        model_activations = resnets.model_activations(
                        dataset=dataset_resnets, 
                        model=model_run, 
                        layers=resnets.layer_specs[model], 
                        specific_preprocess=None)
    else:
        model_activations = vits.model_activations(
                        dataset=dataset_vits, 
                        model=model_run, 
                        layers=vits.layer_specs[model], 
                        specific_preprocess=specific_preprocess[model])
    
    print(f'Converting to RDMs...')
    model_rdms[model] = np.zeros((len(model_activations), len(dataset_resnets), len(dataset_resnets)))
    for layer_idx, layer in tqdm(enumerate(model_activations)):
        rdm = get_RDM(model_activations[layer_idx])
        model_rdms[model][layer_idx, :, :] = rdm
    
    del model_activations

pickle.dump(model_rdms, open('data/model_rdms_raw.pkl', 'wb'))
        

### Compute FMRI-MODEL RSA
fmri_model_RSA = {model: np.zeros((7, model_rdms[model].shape[0], 32)) for model in model_names}

for sub_idx, sub in tqdm(enumerate([1, 2, 3, 4, 5, 6, 8])):
    print(f'Subject : {sub}')

    sub_data = NSD_get_shared_sub_data(
                data_dir='/home/Public/Datasets/algonauts', 
                subj_num=sub, 
                img_filter=nsd_info_shared['nsdId'].to_list())
    fmri_roi, _  = sub_data.get_fmri_data_roi()
    
    for roi_idx, roi in enumerate(fmri_roi):

        if (roi[0].shape[1] == 0) or (roi[1].shape[1] == 0): # If no data in one of the hemis for this ROI
            for model in model_names:
                for layer_idx in range(model_rdms[model].shape[0]):
                    fmri_model_RSA[model][sub_idx, layer_idx, roi_idx] = np.nan

        else:
            rdm_l_hemi, rdm_r_hemi = get_RDM(roi[0]), get_RDM(roi[1])
            roi_rdm = np.mean(np.stack((rdm_l_hemi, rdm_r_hemi), axis=0), axis=0) # Avg across hemis

        for model_idx, model in enumerate(model_names):
            feature_rdms = model_rdms[model]

            for layer_idx in range(feature_rdms.shape[0]):
                fmri_model_RSA[model][sub_idx, layer_idx, roi_idx] = compute_RSA(roi_rdm, feature_rdms[layer_idx])


pickle.dump(fmri_model_RSA, open('data/fmri_model_RSA_raw.pkl', 'wb'))