import os
import numpy as np
import pickle
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import cv2

baseline_dir = os.getcwd() + '/Distractors/Baseline'
directory = sorted(os.listdir(baseline_dir))

# Saliency Maps
sal_maps = np.zeros((len(directory), 224, 224))

for idx, file in tqdm(enumerate(directory)):
    with open(os.path.join(baseline_dir, file), 'rb') as pickle_file:
        img = pickle.load(pickle_file)
    sal_maps[idx] = cv2.resize(img['Saliency_Map'], 
                               dsize=(224, 224), 
                               interpolation=cv2.INTER_CUBIC)

sal_maps = np.reshape(sal_maps,(sal_maps.shape[0], -1))
rdm = pairwise_distances(sal_maps, metric='cosine')
print(rdm.shape)

with open(os.getcwd() + '/image_RDMs/saliency_RDM.npy', 'wb') as f:
    np.save(f, rdm)

# Caption Embeddings
captions = np.zeros((len(directory), 512))

for idx, file in tqdm(enumerate(directory)):
    with open(os.path.join(baseline_dir, file), 'rb') as pickle_file:
        img = pickle.load(pickle_file)
    captions[idx] = img['Avg Text Embedding']

rdm = pairwise_distances(captions, metric='cosine')
print(rdm.shape)

with open(os.getcwd() + '/image_RDMs/semantic_RDM.npy', 'wb') as f:
    np.save(f, rdm)
