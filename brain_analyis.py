import numpy as np   
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sal_RSA = np.load('data/fmri_sal_RSA.npy')
sem_RSA = np.load('data/fmri_sem_RSA.npy')
roi_names = np.load('data/roi_names.npy')
roi_noise_ceilings = np.load('data/roi_noise_ceilings.npy')

sal_RSA_m, sem_RSA_m = np.mean(sal_RSA, axis=0), np.mean(sem_RSA, axis=0)
sal_RSA_sem, sem_RSA_sem = np.std(sal_RSA, axis=0)/np.sqrt(len(sem_RSA)), np.std(sem_RSA, axis=0)/np.sqrt(len(sem_RSA))

# filter out nans
roi_names, roi_noise_ceilings = roi_names[~np.isnan(sal_RSA_m)], roi_noise_ceilings[~np.isnan(sal_RSA_m)]
sal_RSA_m, sem_RSA_m = sal_RSA_m[~np.isnan(sal_RSA_m)], sem_RSA_m[~np.isnan(sem_RSA_m)]
sal_RSA_sem, sem_RSA_sem = sal_RSA_sem[~np.isnan(sal_RSA_sem)], sem_RSA_sem[~np.isnan(sem_RSA_sem)]

# increase font size
plt.rcParams.update({'font.size': 16})
bar_width = 0.35
n_groups = len(sal_RSA_m)
plt.figure(figsize=(12, 5))
plt.bar([x + 0.2 for x in range(n_groups)],sem_RSA_m,width =bar_width,label='Caption Embeddings', color='steelblue')
plt.bar([x - 0.2 for x in range(n_groups)],sal_RSA_m,width =bar_width,label='Saliency Maps', color='indianred')
plt.errorbar([x - 0.2 for x in range(n_groups)], sal_RSA_m, yerr=sal_RSA_sem, fmt='none', c='grey', capsize=1)
plt.errorbar([x + 0.2 for x in range(n_groups)], sem_RSA_m, yerr=sem_RSA_sem, fmt='none', c='grey', capsize=1)
plt.xticks(range(n_groups), roi_names, rotation=70)
plt.ylabel('RSA', fontsize=16)
plt.legend(frameon=False)
plt.show()


