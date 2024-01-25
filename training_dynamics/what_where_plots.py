import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os
os.chdir('/home/c12049018/Documents/training')
import pickle
import pandas as pd

files = os.listdir('/home/c12049018/Documents/training/ResNet/RSA')
n_epochs = len([i for i in files if len(i) == 11 or len(i) == 12])

df = pd.read_csv('/home/c12049018/Documents/Experiment_NEW/Results/All_layers/sal_sem_RSA_experiment.csv')
df = df[df['model'] == 'rn50']

def load(file):   
    file = open(file, 'rb')
    data = pickle.load(file)
    file.close()
    return data

def get_RSA(epoch):
    if epoch == 0:
        rn50 = load(f'/home/c12049018/Documents/training/ResNet/RSA/epoch_0_0.pkl')
    else:
        rn50 = load(f'/home/c12049018/Documents/training/ResNet/RSA/epoch_{str(epoch)}.pkl')
    saliency = [val for layer, val in rn50['saliency_RSA'].items()]
    semantic = [val for layer, val in rn50['semantic_RSA'].items()]
    return saliency, semantic

def get_RSA_first_epoch(batch):
    rn50 = load(f'/home/c12049018/Documents/training/ResNet/RSA/epoch_0_{str(batch)}.pkl')
    saliency = [val for layer, val in rn50['saliency_RSA'].items()]
    semantic = [val for layer, val in rn50['semantic_RSA'].items()]
    return saliency, semantic

sal = np.zeros((n_epochs, 125))
sem = np.zeros((n_epochs, 125))

for idx, epoch in enumerate(range(n_epochs)):
    sal[idx] = get_RSA(epoch)[0]
    sem[idx] = get_RSA(epoch)[1]

#%% plot
filter_ = df['layer_type'] == 'relu' 

layers = np.arange(1, sum(filter_)+1, dtype=int)  # 125 layers
epochs = np.array(range(n_epochs))   # 50 epochs
X, Y = np.meshgrid(layers, epochs)
Z = sal[:,filter_]

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(X.ravel(), Y.ravel(), Z.ravel(),
                cmap='inferno', linewidths=0.2)
surf = ax.plot_trisurf(X.ravel(), Y.ravel(), Z.ravel(),
                       cmap='inferno', linewidths=0.2)

# # Add a vertical plane where X is equal to 5
# y = np.linspace(0, 5, 50)  # spanning across your epochs range
# z = np.linspace(Z.min(), Z.max(), 50)  # spanning across your RSA values range
# Y_plane, Z_plane = np.meshgrid(y, z)
# X_plane = np.full_like(Y_plane, 4)  # constant value of 5
# ax.plot_surface(X_plane, Y_plane, Z_plane, color='black', alpha=0.4)

ax.set_xticks(list(range(1, len(layers)+1, 2)))

cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
cbar.set_label('RSA Values')

ax.set_xlabel('RELU layer')
ax.set_ylabel('Epoch')
ax.set_zlabel('RSA', rotation=90)
ax.view_init(60, 190) 
plt.show()

#%%
n_epochs = 11

def get_RSA_first_epoch(batch):
    rn50 = load(f'/home/c12049018/Documents/training/ResNet/RSA/epoch_0_{str(batch)}.pkl')
    saliency = [val for layer, val in rn50['saliency_RSA'].items()]
    semantic = [val for layer, val in rn50['semantic_RSA'].items()]
    return saliency, semantic

sal = np.zeros((n_epochs, 125))
sem = np.zeros((n_epochs, 125))

for idx, batch in enumerate(range(0,5500,500)):
    sal[idx] = get_RSA_first_epoch(batch)[0]
    sem[idx] = get_RSA_first_epoch(batch)[1]

filter_ = df['layer_type'] == 'relu' 

layers = np.arange(1, sum(filter_)+1)  # 125 layers
epochs = np.array(range(n_epochs))   # 50 epochs
X, Y = np.meshgrid(layers, epochs)
Z = sal[:,filter_]

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(X.ravel(), Y.ravel(), Z.ravel(),
                cmap='inferno', linewidths=0.2)

# # Add a vertical plane where X is equal to 5
# y = np.linspace(0, 5, 50)  # spanning across your epochs range
# z = np.linspace(Z.min(), Z.max(), 50)  # spanning across your RSA values range
# Y_plane, Z_plane = np.meshgrid(y, z)
# X_plane = np.full_like(Y_plane, 4)  # constant value of 5
# ax.plot_surface(X_plane, Y_plane, Z_plane, color='black', alpha=0.4)

ax.set_xlabel('RELU layer')
ax.set_ylabel('Batch')
ax.set_zlabel('RSA')
ax.view_init(40, 170) 
plt.show()


# %%
loss = []
for i in range(17):
    if i == 0:
        loss.append(load('/home/c12049018/Documents/training/ResNet/Checkpoint/epoch_0_0.pkl')['loss'])
    else:
        loss.append(load(f'/home/c12049018/Documents/training/ResNet/Checkpoint/train_results_epoch_{(str(i))}.pkl')['loss'])


# %%
load('/home/c12049018/Documents/training/ResNet/Checkpoint/train_results_epoch_10.pkl')

# %%
sal = np.array(get_RSA_first_epoch(2500)[1])[rn50_filter]

sns.set(style="ticks")
plt.figure(figsize=(24, 12))
ticks =list(range(len(sal)))
n = 20
models = ['Training step 2500']
for i, model in enumerate(models):
    ax = plt.subplot(2, 2, i+1)
    sns.barplot(x=ticks, y=sal, color='#f8a326')
    plt.xlabel('Layer', fontsize=16)
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.title(models[0], fontsize=20)
    plt.ylabel('RSA', fontsize=20)    # Adjust x-axis ticks
    locs, labels = plt.xticks()  
    plt.xticks(locs[::n], ticks[::n])

    ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, color='gray')

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
    
    plt.tight_layout()

plt.show()
# %%
rn50 =load('/home/c12049018/Documents/Experiment_NEW/Results/All_layers/rn50_train.pkl')
sal = np.array([val for layer, val in rn50['semantic_RSA'].items()])[rn50_filter]

sns.set(style="ticks")
plt.figure(figsize=(24, 12))
ticks =list(range(len(sal)))
n = 20
models = ['Training mode']
for i, model in enumerate(models):
    ax = plt.subplot(2, 2, i+1)
    sns.barplot(x=ticks, y=sal, color='#f8a326')
    plt.xlabel('Layer', fontsize=16)
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.title(models[0], fontsize=20)
    plt.ylabel('RSA', fontsize=20)    # Adjust x-axis ticks
    locs, labels = plt.xticks()  
    plt.xticks(locs[::n], ticks[::n])

    ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, color='gray')

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
    
    plt.tight_layout()

plt.show()
# %%
