import os
import pickle
from torch import cuda
import torchvision
import clip
from RSA import RSA

n_cores = "12"
os.environ["OMP_NUM_THREADS"] = n_cores
os.environ["OPENBLAS_NUM_THREADS"] = n_cores
os.environ["MKL_NUM_THREADS"] = n_cores
os.environ["VECLIB_MAXIMUM_THREADS"] = n_cores
os.environ["NUMEXPR_NUM_THREADS"] = n_cores

def load(file):   
    file = open(file, 'rb')
    data = pickle.load(file)
    file.close()
    return data

def dump(file_name, file):
    file_name = open(file_name, 'wb')
    pickle.dump(file, file_name)
    file_name.close()

device = "cuda" if cuda.is_available() else "cpu"


### Resnet Models
rn50 = torchvision.models.resnet50(weights='IMAGENET1K_V1').to(device).eval()
clip_rn50 = clip.load("RN50", device=device, jit=False)[0]
rn101 = torchvision.models.resnet101(weights='IMAGENET1K_V1').to(device).eval()
clip_rn101 = clip.load("RN101", device=device, jit=False)[0]

# Load preprocessing pipeline from CLIP
preprocess_resnet = clip.load("RN50", device=device, jit=False)[1]

# Initiate RSA class
rsa = RSA(
    device=device,
    general_preprocess=preprocess_resnet,
    rn50 = rn50,
    clip_rn50 = clip_rn50.visual,
    rn101 = rn101,
    clip_rn101 = clip_rn101.visual
    )

rsa_results = rsa.perform_RSA('delta')
dump('data/resnet_raw.pkl', rsa_results)

del rn50, clip_rn50, rn101, clip_rn101, preprocess_resnet


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
rsa = RSA(
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

rsa_results = rsa.perform_RSA('delta')
dump('data/vit_raw.pkl', rsa_results)



