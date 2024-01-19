import os
import numpy as np


class NSD_get_shared_sub_data:
    def __init__(self, data_dir, subj_num, img_filter: list):
        self.subj_num = format(subj_num, '02')
        self.data_dir = os.path.join(data_dir, 'subj'+self.subj_num)
        self.fmri_dir = os.path.join(self.data_dir, 'training_split', 'training_fmri')
        self.img_dir = os.path.join(self.data_dir, 'training_split', 'training_images')
        self.img_filter = img_filter
        self.shared_imgs = self.imgs_shared()


    def get_fmri_data(self):
        lh_fmri = np.load(os.path.join(self.fmri_dir, 'lh_training_fmri.npy'))[self.shared_imgs]
        rh_fmri = np.load(os.path.join(self.fmri_dir, 'rh_training_fmri.npy'))[self.shared_imgs]
        return lh_fmri, rh_fmri


    def imgs_shared(self) -> list:
        '''
        Returns a boolean list of the shared images between all subjects the NSD.
        '''
        img_list = sorted(os.listdir(self.img_dir))
        shared_nsd_list = [str(img).zfill(5) for img in self.img_filter] # Leading zeros
        return [img[15:-4] in shared_nsd_list for img in img_list]
    

    def get_roi_name_maps(self):
            roi_mapping_files = ['mapping_prf-visualrois.npy', 'mapping_floc-bodies.npy',
            'mapping_floc-faces.npy', 'mapping_floc-places.npy',
            'mapping_floc-words.npy', 'mapping_streams.npy']

            roi_name_maps = []
            for r in roi_mapping_files:
                roi_name_maps.append(np.load(os.path.join(self.data_dir, 'roi_masks', r),
                    allow_pickle=True).item())

            return roi_name_maps


    def get_roi_brain_maps(self):
        lh_challenge_roi_files = ['lh.prf-visualrois_challenge_space.npy',
            'lh.floc-bodies_challenge_space.npy', 'lh.floc-faces_challenge_space.npy',
            'lh.floc-places_challenge_space.npy', 'lh.floc-words_challenge_space.npy',
            'lh.streams_challenge_space.npy']
        rh_challenge_roi_files = ['rh.prf-visualrois_challenge_space.npy',
            'rh.floc-bodies_challenge_space.npy', 'rh.floc-faces_challenge_space.npy',
            'rh.floc-places_challenge_space.npy', 'rh.floc-words_challenge_space.npy',
            'rh.streams_challenge_space.npy']
        lh_challenge_rois = []
        rh_challenge_rois = []

        for r in range(len(lh_challenge_roi_files)):
            lh_challenge_rois.append(np.load(os.path.join(self.data_dir, 'roi_masks',
                lh_challenge_roi_files[r])))
            rh_challenge_rois.append(np.load(os.path.join(self.data_dir, 'roi_masks',
                rh_challenge_roi_files[r])))
            
        return lh_challenge_rois, rh_challenge_rois
    

    def get_fmri_data_roi(self):
        roi_name_maps = self.get_roi_name_maps()
        lh_challenge_rois, rh_challenge_rois = self.get_roi_brain_maps()
        lh_fmri, rh_fmri = self.get_fmri_data()

        roi_names = []
        roi = []
        for r1 in range(len(lh_challenge_rois)):
            for r2 in roi_name_maps[r1].items():
                if r2[0] != 0: # zeros indicate to vertices falling outside the ROI of interest
                    roi_names.append(r2[1])
                    lh_roi_idx = np.where(lh_challenge_rois[r1] == r2[0])[0]
                    rh_roi_idx = np.where(rh_challenge_rois[r1] == r2[0])[0]
                    roi.append((lh_fmri[:, lh_roi_idx], rh_fmri[:, rh_roi_idx]))
        roi_names.append('All vertices')
        roi.append((lh_fmri, rh_fmri))

        return roi, roi_names