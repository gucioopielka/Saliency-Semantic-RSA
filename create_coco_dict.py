import pickle
import sys
sys.path.append('/home/c12049018/pySaliencyMap')
import pySaliencyMapDefs
from pySaliencyMap import pySaliencyMap
from pySaliencyMap import pySaliencyMap
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_hub as hub
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
from sklearn.metrics import pairwise_distances

### DATASET ####
dataset = str(sys.argv[1])
print("Dataset: ", dataset)

# Pickle load function
def load(file):   
    root = '/home/c12049018/Documents/Experiment/'
    file = open(root+file, 'rb')
    data = pickle.load(file)
    file.close()
    return data

# Pickle dump function
def dump(file_name, file):
    file_name = open(file_name, 'wb')
    # dump information to that file
    pickle.dump(file, file_name)
    # close the file
    file_name.close()


model_url = 'https://tfhub.dev/google/universal-sentence-encoder/4'
model = hub.load(model_url)

def embed(input):
  return model(input)

# Init
dataDir = '/home/c12049018/Documents/Coco-5000/root/fiftyone/coco-2017'
dataType = dataset

segment_annFile = f'/home/c12049018/Documents/Coco-5000/root/fiftyone/coco-2017/raw/instances_{dataset}.json'
coco_segment = COCO(segment_annFile)

caps_annFile = f'/home/c12049018/Documents/Coco-5000/root/fiftyone/coco-2017/raw/captions_{dataset}.json'
coco_caps = COCO(caps_annFile)

# Coco
coco = []
n_coco = len(coco_segment.loadImgs(coco_segment.getImgIds()))
for idx in range(n_coco):
    if idx % 5000 == 0:
       print("Log:", idx)

    coco_dict = {}
    coco_dict['Idx'] = idx

    # Load image
    img = coco_segment.loadImgs(coco_segment.getImgIds()[idx])[0]
    
    annIds_segment = coco_segment.getAnnIds(imgIds=img['id'])
    anns_segment = coco_segment.loadAnns(annIds_segment)

    annIds_caps = coco_caps.getAnnIds(imgIds=img['id'])
    anns_caps = coco_caps.loadAnns(annIds_caps)

    # Get categories
    cats = []
    for i in range(len(anns_segment)):
        entity_id = anns_segment[i]["category_id"]
        entity = coco_segment.loadCats(entity_id)[0]["name"]
        cats.append(entity)

    coco_dict['Category'] = cats

    # Get caption
    captions = [caption['caption'] for caption in anns_caps]
    coco_dict['Caption'] = captions

    # Get filename (local)
    coco_dict['file_name'] = img['file_name']

    # COCO ID
    coco_dict['id'] = img['id']

    coco.append(coco_dict)

# Avg embeddings
ids_to_remove = []
for i in range(n_coco):
  try:
    m = np.mean(embed(coco[i]['Caption']).numpy(), axis=0)
    coco[i]['Avg Text Embedding'] = m

  except:
    print(i, "Error")
    ids_to_remove.append(i)
    continue

coco[0]['Avg Text Embedding'].shape

dump(f"/home/c12049018/Documents/Coco-5000/COCO-2017_TRAIN_DICT/coco_dict_{dataset}", coco)