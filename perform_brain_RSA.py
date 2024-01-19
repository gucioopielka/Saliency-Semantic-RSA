import os
import pandas as pd
import numpy as np
from NSD import NSD_get_shared_sub_data
import pickle
import cv2
from create_distractors import open_img, saliency
from RSA import get_RDM, compute_RSA
from tqdm import tqdm

n_cores = "12"
os.environ["OMP_NUM_THREADS"] = n_cores
os.environ["OPENBLAS_NUM_THREADS"] = n_cores
os.environ["MKL_NUM_THREADS"] = n_cores
os.environ["VECLIB_MAXIMUM_THREADS"] = n_cores
os.environ["NUMEXPR_NUM_THREADS"] = n_cores


# Load NSD info
nsd_info_shared = pd.read_csv('/home/c12049018/nsd_shared_imgs.csv', dtype={'cocoId': str, 'nsdId': str})
# Load COCO caption embeddings
coco_dict = pickle.load(open('data/coco_dict_train2017', 'rb'))
coco_dict = [img for img in coco_dict if str(img['id']).zfill(12) in nsd_info_shared['cocoId'].to_list()] # Filter out images not in NSD
coco_dict = {img['file_name'][:-4]: img['Avg Text Embedding'] for img in coco_dict}


# Get saliency maps for each image
sal_maps = []
for row in tqdm(range(len(nsd_info_shared))):
    img = open_img(file = nsd_info_shared.loc[row, 'cocoId']+'.jpg')
    sal_map = saliency(img)
    sal_map_resized = cv2.resize(
                    sal_map, 
                    dsize=(224, 224), 
                    interpolation=cv2.INTER_CUBIC)
    sal_maps.append(sal_map_resized)

sal_rdm = get_RDM(sal_maps)


# Get caption embeddings for each image
caption_embeddings = []
for row in range(len(nsd_info_shared)):
    coco_id = nsd_info_shared.loc[row, 'cocoId']
    caption_embeddings.append(coco_dict[coco_id])

sem_rdm = get_RDM(caption_embeddings)


### Compute FMRI-SALIENCY/SEMANTIC RSA
fmri_sal_RSA = np.zeros((7, 32))
fmri_sem_RSA = np.zeros((7, 32))

for sub_idx, sub in enumerate([1, 2, 3, 4, 5, 6, 8]):
    print(f'Subject : {sub}')

    sub_data = NSD_get_shared_sub_data(
                data_dir='/home/Public/Datasets/algonauts', 
                subj_num=sub, 
                img_filter=nsd_info_shared['nsdId'].to_list())
    fmri_roi, _  = sub_data.get_fmri_data_roi()
    
    for roi_idx, roi in enumerate(fmri_roi):

        if (roi[0].shape[1] == 0) or (roi[1].shape[1] == 0): # If no data in one of the hemis for this ROI
            fmri_sal_RSA[sub_idx, roi_idx] = np.nan
            fmri_sem_RSA[sub_idx, roi_idx] = np.nan

        else:
            rdm_l_hemi, rdm_r_hemi = get_RDM(roi[0]), get_RDM(roi[1])
            roi_rdm = np.mean(np.stack((rdm_l_hemi, rdm_r_hemi), axis=0), axis=0) # Avg across hemis
            fmri_sal_RSA[sub_idx, roi_idx] = compute_RSA(roi_rdm, sal_rdm)
            fmri_sem_RSA[sub_idx, roi_idx] = compute_RSA(roi_rdm, sem_rdm)


# Save RSA results
np.save('data/fmri_sal_RSA', fmri_sal_RSA)
np.save('data/fmri_sem_RSA', fmri_sem_RSA)