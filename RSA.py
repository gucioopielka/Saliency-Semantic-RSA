import os
import numpy as np
import warnings
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import pickle
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import torchvision.transforms as T
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr


def get_RDM(arr: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        arr = np.stack(arr)
        return pairwise_distances(np.reshape(arr, (arr.shape[0], -1)), metric='cosine')


def compute_RSA(rdm1, 
                rdm2):
    """Compute the RSA between two RDMs."""
    return spearmanr(squareform(rdm1), squareform(rdm2))[0]


class RSA:
    def __init__(self, 
                 device: str = 'cpu', 
                 layer_specs: dict = None,
                 general_preprocess: dict = None, 
                 specific_preprocess: dict = None,
                 img_directory: str = os.getcwd(),
                 test: int = False,
                 **models: nn.Module) -> None:
        """
        Initialize RSA class.
        
        Parameters:
            device (str): Device to run the models on.
            layer_specs (dict): A dictionary specifying the layers of interest for each model.
            general_preprocess (callable): A general preprocessing function applicable to all models.
            specific_preprocess (callable): Model-specific preprocessing functions.
            **models (nn.Module): One or more PyTorch models for analysis.
        """
        
        self.device = device
        self.test = test
        self.root = img_directory 

        # Distractor experimental conditions names
        self.conditions = [
                        'Baseline',
                        'Control',
                        'Salient',
                        'Semantic',
                        'Semantic_Salient'
                        ]
        
        # Models for analysis
        self.models = {name: model for name, model in models.items()} 
        self.model_names = list(models.keys())

        # General preprocessing function
        self.general_preprocess = general_preprocess if general_preprocess else T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop((224, 224)),
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),  # Convert to RGB
            T.ToTensor(),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

        # Model-specific preprocessing function
        if specific_preprocess:
            self.specific_preprocess = specific_preprocess 
        else:
            self.specific_preprocess = {model: lambda x: x for model in self.model_names}
        
        # Layer specifications for each model
        if layer_specs is None:
            self.layer_specs = {model: self.get_layers(self.models[model]) for model in self.model_names}
        else:  
            self.layer_specs = layer_specs  

        # Load Baseline Saliency/Semantic RDMs
        self.saliency_rdm = np.load(os.path.join(self.root, 'image_RDMs/saliency_RDM.npy'))
        self.semantic_rdm = np.load(os.path.join(self.root, 'image_RDMs/semantic_RDM.npy'))

        if self.test:
            self.saliency_rdm = self.saliency_rdm[:self.test, :self.test]
            self.semantic_rdm = self.semantic_rdm[:self.test, :self.test]


    class ImageDataset(Dataset):
        def __init__(self, imgs_paths, preprocess, device):
            self.imgs_paths = np.array(imgs_paths)
            self.preprocess = preprocess
            self.device = device

        def __len__(self):
            return len(self.imgs_paths)
        
        def load(self, file):
            file = open(file, 'rb')
            data = pickle.load(file)
            file.close()
            return data   

        def __getitem__(self, idx):
            # Load the image
            img_path = self.imgs_paths[idx]
            img = self.load(img_path)['Img']

            # Preprocess the image and send it to the chosen device ('cpu' or 'cuda')
            if self.preprocess:
                img = self.preprocess(Image.fromarray(img)).to(self.device)
            
            return img 
    

    def model_activations(self, dataset, model, layers, specific_preprocess):
        layer_names = list(layers.keys())
        layers = [layers[layer] for layer in layers.keys()]

        activation = {}
        def getActivation(name):
            # the hook signature
            def hook(model, input, output):
                # Check if the output is a tuple
                if isinstance(output, tuple):
                    # If yes, extract the output tensor from the tuple
                    output_tensor = output[0]
                else:
                    # If the output is not a tuple, use it as it is
                    output_tensor = output
                activation[name] = output_tensor.detach().cpu().numpy()
            return hook
        
        # register forward hooks
        hooks = [layer.register_forward_hook(getActivation(layer_names[i])) 
                 for i, layer in enumerate(layers)]
        
        activations = [[] for _ in layers]
        n_images = len(dataset)

        for idx in tqdm(range(n_images), miniters=int(n_images / 10)):

            # a dict to store the activations
            activation = {}

            # input
            if specific_preprocess is None:
                image_input = dataset[idx].unsqueeze(0)  
            else:
                image_input = specific_preprocess(dataset[idx].unsqueeze(0))

            # forward pass -- getting the outputs
            out = model(image_input)

            for i, layer in enumerate(layers):
                try:
                    # Squeezing while extracting and converting
                    activations[i].append(np.squeeze(np.array(activation[layer_names[i]])))
                except KeyError:
                    continue

        # detach the hooks
        for hook in hooks:
            hook.remove()

        # Updating layer names
        model_name = [name for name, mod in self.models.items() if mod == model][0]
        self.layer_specs[model_name] = {layer_name: layers[i] for i, layer_name in enumerate(layer_names) if layer_name in activation}

        return [act for act in activations if len(act) == n_images]


    def dataset_condition(self, cond):
        directory = os.path.join(self.root, 'distractors', 'Baseline')
        files = sorted(os.listdir(directory))
        if self.test:
            files = files[:self.test]
        imgs_paths = [Path(os.path.join(self.root, 'distractors', cond, img)) 
                      for img in files]
        return self.ImageDataset(imgs_paths, self.general_preprocess, self.device)


    def get_layers(self, model, prefix='', specified_layers=None):
        layers = {}
        for name, child in model.named_children():
            child_prefix = f'{prefix}.{name}' if prefix else name
            
            # Check if the child is a singular layer by examining its type
            is_singular_layer = isinstance(child, (nn.Conv2d, nn.Linear, nn.LayerNorm, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.AvgPool2d, nn.Dropout))
            
            # Check if the child is a self-attention block
            is_self_attention_block = isinstance(child, nn.MultiheadAttention) 
            
            if is_singular_layer or is_self_attention_block:
                if specified_layers is None or any(substring in child_prefix for substring in specified_layers):
                    # Add singular layers or self-attention blocks to the layers dict
                    layers[child_prefix] = child
                    if specified_layers is not None:
                        continue
            # Recursively get sub-layers
            layers.update(self.get_layers(child, child_prefix, specified_layers))
        return layers
        

    def perform_RSA(self, 
                    base_delta: str = 'delta') -> dict:
        """
        Perform RSA on the models and layers specified in the class initialization.

        Parameters:
            base_delta (str): Whether to perform RSA on the baseline or delta conditions.
        """

        print('=== Performing RSA ===')

        if base_delta == 'delta':
            conditions = self.conditions
        elif base_delta == 'base':
            conditions = [self.conditions[0]] # Only use Baseline

        saliency_RSA = {}
        semantic_RSA = {}
        
        for model_idx, model in enumerate(self.models):
            print(f'___Model: {self.model_names[model_idx]}___')

            for cond_idx, cond in enumerate(conditions):
                print(f'Condition: {cond}')

                # Extract Features to Distractors
                print('Getting Features...')
                dataset = self.dataset_condition(cond)
                act = self.model_activations(dataset, 
                                                self.models[model], 
                                                self.layer_specs[model], 
                                                self.specific_preprocess[model])

                print('Processing Layer Activations...')               
                for layer_idx in tqdm(range(len(self.layer_specs[model]))):
                    
                    # Convert Features into RDM
                    feature_rdm = get_RDM(act[layer_idx])
                    
                    # Create dictionary key
                    model_name = self.model_names[model_idx]
                    layer_name = list(self.layer_specs[model].keys())[layer_idx]
                    key = f"{model_name}_{cond}_{layer_name}"

                    # Perform RSA
                    saliency_RSA[key] = compute_RSA(self.saliency_rdm, feature_rdm)
                    semantic_RSA[key] = compute_RSA(self.semantic_rdm, feature_rdm)
                    
                del act
        
        # Save results
        rsa_results = {
            "saliency_RSA": saliency_RSA,
            "semantic_RSA": semantic_RSA,
        }

        return rsa_results
