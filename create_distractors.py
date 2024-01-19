import pickle
from scipy.spatial import distance
import sys
sys.path.append('/home/c12049018/pySaliencyMap')
import pySaliencyMapDefs
from pySaliencyMap import pySaliencyMap
from pySaliencyMap import pySaliencyMap
from PIL import Image
import numpy as np
import cv2
import random
import os
import pandas as pd
import pandas as pd
import ast


# Pickle load function
def load(file):   
    file = open(file, 'rb')
    data = pickle.load(file)
    file.close()
    return data

# Pickle dump function
def dump(file_name, file):
    file_name = open(file_name, 'wb')
    pickle.dump(file, file_name)
    file_name.close()

# Paste function
def paste(img, img_distractor):
    distractor_ratio = 0.1

    img_edit = img.copy()

    # Calculate the area of img_edit
    area_org = img_edit.shape[0] * img_edit.shape[1]
    # Calculate the target area for img_distractor
    target_area = int(area_org * distractor_ratio)
    # Calculate the aspect ratio for img_distractor
    aspect_ratio_dist = img_distractor.shape[1] / img_distractor.shape[0]
    # Calculate the new dimensions for img_distractor
    new_height_dist = int((target_area / aspect_ratio_dist)**0.5)
    new_width_dist = int(new_height_dist * aspect_ratio_dist)

    # Resize img_distractor
    img_distractor = cv2.resize(img_distractor, (new_width_dist, new_height_dist))

    # Place the distractor randomly
    edge = random.choice(['left', 'right', 'top', 'bottom'])
    if edge == 'left':
        x_offset = 0
    elif edge == 'right':
        x_offset = img.shape[1] - img_distractor.shape[1]
    elif edge == 'top':
        y_offset = 0
    else: # edge == 'bottom'
        y_offset = img.shape[0] - img_distractor.shape[0]

    if edge in ['left', 'right']:
        y_offset = random.randint(0, img.shape[0] - img_distractor.shape[0])
    else: # edge in ['top', 'bottom']
        x_offset = random.randint(0, img.shape[1] - img_distractor.shape[1])

    x_end = x_offset + img_distractor.shape[1]
    y_end = y_offset + img_distractor.shape[0]

    img_edit[y_offset:y_end, x_offset:x_end] = img_distractor
    
    # save coordinates
    area_ratio = img_distractor.shape[0] * img_distractor.shape[1] / (img_edit.shape[0] * img_edit.shape[1])
    coordinates = {'x_offset': x_offset, 'y_offset': y_offset, 'x_end': x_end, 'y_end': y_end, 'area_ratio': area_ratio}

    return img_edit, coordinates

# Crop function
def applyCropToImg(img, box):
    '''
    applyCropToImg(img, cropBox)
    img ~ any h x w x n image
    cropBox ~ (top, bottom, left, right) in fractions of image size
    '''
    if box[0]+box[1] >= 1:
        raise ValueError('top and bottom crop must sum to less than 1')
    if box[2]+box[3] >= 1:
        raise ValueError('left and right crop must sum to less than 1')
    shape = img.shape
    topCrop = np.round(shape[0]*box[0]).astype(int)
    bottomCrop = np.round(shape[0]*box[1]).astype(int)
    leftCrop = np.round(shape[1]*box[2]).astype(int)
    rightCrop = np.round(shape[1]*box[3]).astype(int)
    croppedImage = img[topCrop:(shape[0]-bottomCrop),leftCrop:(shape[1]-rightCrop)]
    return croppedImage

def getCropBox(img):
    id = int(coco[img]['file_name'][:-4].lstrip('0'))
    idx = nsd_info['cocoId'] == id
    cropBox = nsd_info.loc[idx, 'cropBox']
    return ast.literal_eval(cropBox.values[0])

# Saliency map function
def saliency(img):
    # initialize
    imgsize = img.shape
    img_width  = imgsize[1]
    img_height = imgsize[0]  
    sm = pySaliencyMap(img_width, img_height)
    # computation
    saliency_map = sm.SMGetSM(img)

    return saliency_map

# Save img 
def save_img(img1, img_edit, sem_sim, sal_sim, coordinates, sm):
    img_dict = {}
    img_dict['Idx'] = img1
    img_dict['id1'] = coco[img1]['id']
    img_dict['id2'] = coco[img2]['id']      
    img_dict['Img'] = img_edit
    img_dict['Paste Coordinates'] = coordinates
    img_dict['Sem_Sim'] = sem_sim
    img_dict['Sal_Sim'] = sal_sim
    img_dict['Saliency_Map'] = sm
    return img_dict

# Open img
def open_img(file):
    return np.array(Image.open(f"/home/Public/Datasets/COCO/train2017/{file}"))


if __name__ == "__main__":
    
    n_cores = "16"
    os.environ["OMP_NUM_THREADS"] = n_cores
    os.environ["OPENBLAS_NUM_THREADS"] = n_cores
    os.environ["MKL_NUM_THREADS"] = n_cores
    os.environ["VECLIB_MAXIMUM_THREADS"] = n_cores
    os.environ["NUMEXPR_NUM_THREADS"] = n_cores

    start, end = int(sys.argv[1]), int(sys.argv[2])

    nsd_info = pd.read_csv('/home/c12049018/nsd_stim_info_merged.csv')
    nsd_info = nsd_info[nsd_info['cocoSplit'] == 'train2017']
    nsd_info = nsd_info[['cocoId', 'cropBox']].reset_index()

    dataset = 'train2017'

    coco = load(f"/home/c12049018/Documents/Coco-5000/COCO-2017_TRAIN_DICT/coco_dict_{dataset}")
    #cocoids_nsd = nsd_info['cocoId'].to_list()


    root = "/home/c12049018/Documents/create_images/Distractors"
    ids_found = [i[:-7] for i in os.listdir(f'{root}/Baseline')]

    # Thresholds
    semantic_similarity = 0.32
    semantic_dissimilarity = 0.92

    non_salient_distractor = 0.019
    salient_distractor = 0.24



    for img1 in range(start, end):
        
        id = coco[img1]['file_name'][:-4]
        if id in ids_found:
            continue

        found_imgs = len(os.listdir(f'{root}/Baseline'))
        print(f'{found_imgs} images found so far.')

        salient_to_find = True
        control_to_find = True
        semantic_salient_to_find = True
        semantic_to_find = True

        # Saliency map org
        try:
            img_org = open_img(coco[img1]['file_name'])
            img_org = applyCropToImg(img_org , getCropBox(img1)) # Crop according to NSD
            img_org_sm = saliency(img_org)
        except (ZeroDivisionError, ValueError, IndexError) as error:
            continue


        for img2 in range(len(coco)):  

            # Exclude the diagonal
            if img1 == img2:
                continue
            
            sem_sim = distance.cosine(coco[img1]['Avg Text Embedding'], coco[img2]['Avg Text Embedding'])
            
            # Semantic similarity
            if sem_sim < semantic_similarity and any([salient_to_find, control_to_find]): 

                # Edit
                try:
                    img_distractor = open_img(coco[img2]['file_name'])
                    img_edit, coordinates = paste(img_org, img_distractor)
                except ValueError:
                    continue
                
                # Saliency map Edit
                try:
                    img_edit_sm = saliency(img_edit)
                    sal_sim = distance.cosine(img_org_sm.flatten(), img_edit_sm.flatten())
                except (ZeroDivisionError, ValueError) as error:
                    continue
                
                # Non-salient distractor
                if sal_sim < non_salient_distractor and control_to_find:
                    control_img = save_img(img1, img_edit, sem_sim, sal_sim, coordinates, img_edit_sm)
                    control_to_find = False
                
                # Salient distractor
                if sal_sim > salient_distractor and salient_to_find:
                    salient_img = save_img(img1, img_edit, sem_sim, sal_sim, coordinates, img_edit_sm)
                    salient_to_find = False
            
            # Semantic dissimilarity
            if sem_sim > semantic_dissimilarity and any([semantic_salient_to_find, semantic_to_find]): 
                
                # Edit
                try:
                    img_distractor = open_img(coco[img2]['file_name'])
                    img_edit, coordinates = paste(img_org, img_distractor)            
                except ValueError:
                    continue

                #Saliency Map
                try:
                    img_edit_sm = saliency(img_edit)
                    sal_sim = distance.cosine(img_org_sm.flatten(), img_edit_sm.flatten())
                except ZeroDivisionError:
                    continue
                
                # Non-salient distractor
                if sal_sim < non_salient_distractor and semantic_to_find:
                    semantic_img = save_img(img1, img_edit, sem_sim, sal_sim, coordinates, img_edit_sm)
                    semantic_to_find = False

                # Salient distractor
                if sal_sim > salient_distractor and semantic_salient_to_find:
                    semantic_salient_img = save_img(img1, img_edit, sem_sim, sal_sim, coordinates, img_edit_sm)
                    semantic_salient_to_find = False
            
            if not any([semantic_salient_to_find, semantic_to_find, salient_to_find, control_to_find]):
                
                filename = os.path.splitext(coco[img1]['file_name'])[0]

                dump(f"{root}/Control/{filename}.pickle", control_img)
                dump(f"{root}/Salient/{filename}.pickle", salient_img)
                dump(f"{root}/Semantic/{filename}.pickle", semantic_img)
                dump(f"{root}/Semantic_Salient/{filename}.pickle", semantic_salient_img)

                baseline = {}
                baseline['Img'] = img_org
                baseline['Saliency_Map'] = img_org_sm
                baseline['Avg Text Embedding'] = coco[img1]['Avg Text Embedding']

                dump(f"{root}/Baseline/{filename}.pickle", baseline)

                print(f"Found! After {img2} iterations.")
                break

            
