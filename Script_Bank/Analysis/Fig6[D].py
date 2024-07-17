######################################
######## Optimize Exploration ########
######################################

#%%# Catalyzer

sup_comp = False # Super Computer?
inference_prod_activate = True # Activate Inference Procedure?
data_path = 'Optimize_Exploration_1' # 'Shallow_Grid_1_N_Link'
acts = [0] # [0, 1, ...]
act = max(acts)
observers = [1, 2] # {0, 1, ...} # L ~ {1, 2}
curb = 'Mid' # {'Weak', 'Mid', 'Strong'}
restrict = {
    'Weak': {'G_EA': (750, 1500), 'N_EA': (750, 1500)},
    'Mid': {'G_EA': (0, 1000), 'N_EA': (0, 1000)},
    'Strong': {'G_EA': (0, 750), 'N_EA': (0, 750)}
}

import os
import sys
path = os.path.dirname(os.path.realpath(__file__)) + '/../Resources'
sys.path.append(path)
path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path)
import re
import numpy as np
import torch
import time
import pickle
if not sup_comp:
    import matplotlib.pyplot as plt
    fig_resolution = 500 # {250, 375, 500, 750, 1000}
    plt.rcParams['figure.dpi'] = fig_resolution
    plt.rcParams['savefig.dpi'] = fig_resolution
    fig_size_base = 10 # {2.5, 3.75, 5, 7.5, 10}

#%%# Data Preparation

from Fig0_Services import make_sieve_rag, discover_appraisal_opals

data_paths = ['Shallow_Grid_1_N_Link', 'Shallow_Grid_1_N_Link', 'Optimize_Exploration_1', 'Optimize_Exploration_1'] # data_path
prep_data_paths = ['Shallow_Grid', 'Shallow_Grid', 'Optimize_Exploration', 'Optimize_Exploration']
nooks = [0, 0, 0, 0] # nook
acts = [7, 9, 9, 9] # acts = [7, 7, 7, 7] # act
observers = [1, 2, 1, 2] # observe
curbs = ['Mid', 'Mid', 'Mid', 'Mid'] # curb
para_key_sets = [ # para_key_set
    ['N_N', 'G_G', 'FC_N', 'G_EA', 'G_N', 'N_G', 'FC_G', 'N_EA', 'MRNA', 'PRO', 'td_FC', 'td_FM', 'tau_C', 'tau_M', 'tau_ef_EA', 'tau_eb_EA', 'tau_pf_NP', 'tau_pb_NP', 'chi_auto']
]*4
para_value_sets = [ # para_value_set
    [(0, 1000), (0, 1000), (0, 1000), (0, 1000), (0, 1000), (0, 1000), (0, 1000), (0, 1000), (0, 250), (0, 1000), (300, 28800), (300, 28800), (30, 5*900), (30, 2*2100), (10*30, 43200), (30, 43200), (10*30, 43200), (30, 43200), (0, 1)]
]*4
para_set_modes = [0, 0, 0, 0] # para_set_mode

keys = ['data_path', 'prep_data_path', 'nook', 'act', 'observe', 'curb']
values = [data_paths, prep_data_paths, nooks, acts, observers, curbs]
memos = [{key: value[_] for key, value in zip(keys, values)} for _ in range(len(data_paths))]

species = ['N', 'G', 'NP'] # ['N', 'G', 'N_MRNA', 'G_MRNA', 'NP', 'FC', 'FC_MRNA', 'FM', 'FD', 'EI', 'EA']
time_mini = 0
time_maxi = 48
time_unit = 'Hours'
time_delta = 0.25
time_slit = 1
score_mini = 0
score_maxi = 1
score_slit = 0.05

cellulates = [(10, 10)] # (cell_layers, layer_cells)

# Make Sieve Rag

sieve_lit = [0, 25, 50, 75, 80, 85, 90, 95, 96, 97, 98, 99, 100] # [0, 25, 50, 75, 80, 85, 90, 95, 96, 97, 98, 99, 100]
lots = [0, 1, 2, 3]
sieve_indices = range(0, len(sieve_lit)) # range(min(sieve_lit), max(sieve_lit)+1)
sieve_rag_shape = (len(lots), 4, len(sieve_indices))
sieve_rag = np.full(shape = sieve_rag_shape, fill_value = np.nan)
pats = ['AI_L1', 'AI_L2', 'SA_L1', 'SA_L2']
rules = [0, 1]
aim_NT = [1, 2/5]
aim_G = [1-_ for _ in aim_NT]
cellulates = [(10, 10)]
cells = [cellulate[0]*cellulate[1] for cellulate in cellulates]
species = ['N', 'G', 'NP']
L = 1

tau_mini = time_maxi-8
tau_maxi = time_maxi
cellulate = cellulates[0]
keywords = {
    'sieve_lit': sieve_lit,
    'memos': memos,
    'species': species,
    'time_mini': time_mini,
    'time_maxi': time_maxi,
    'time_unit': time_unit,
    'time_delta': time_delta,
    'rules': rules,
    'aim_NT': aim_NT,
    'aim_G': aim_G,
    'L': L,
    'sieve_rag': sieve_rag
}

sieve_rag, theta_sets = make_sieve_rag(lots, cellulate, tau_mini, tau_maxi, keywords)

appraisal_opal_discoveries = discover_appraisal_opals(lots, sieve_rag)

exam = np.invert(np.all(np.equal(acts, acts[0])))

#%%# Simulation [Preparation]

from Utilities_Optimize_Exploration import make_paras

para_set_raw = {
    'N_N': (100, (0, 1000), 1, None), 'G_G': (200, (0, 1000), 1, None), 'FC_N': (500, (0, 1000), 1, None), 'G_EA': (750, restrict[curb]['G_EA'], 1, None),
    'G_N': (50, (0, 1000), 1, None), 'N_G': (400, (0, 1000), 1, None), 'FC_G': (500, (0, 1000), 1, None), 'N_EA': (750, restrict[curb]['N_EA'], 1, None),
    'MRNA': (50, (0, 250), 1, None), 'PRO': (200, (0, 1000), 1, None),
    'td_FC': (7200, (300, 28800), 1, None), 'td_FM': (7200, (300, 28800), 1, None),
    'tau_C': (450, (30, 5*900), 1, None), 'tau_M': (450, (30, 2*2100), 1, None),
    'tau_ef_EA': (17100, (10*30, 43200), 1, None), 'tau_eb_EA': (300, (30, 43200), 1, None), 'tau_pf_NP': (1710, (10*30, 43200), 1, None), 'tau_pb_NP': (171, (30, 43200), 1, None),
    'chi_auto': (0.5, (0, 1), 0, None)
}
para_set_mode = 0 # {0, 1} # {'No Remap', 'Remap: [A, B] ---->>>> [0, 1]'}
para_set, para_set_true = make_paras(para_set_raw = para_set_raw, para_set_mode = para_set_mode, verbose = True)

#%%# SBI Data

from Fig0_Services import retrieve_posteriors_mapes, synthesizer_post, geometric_median

_data_paths = ['Shallow_Grid_1_N_Link', 'Shallow_Grid_1_N_Link'] # data_path
_nooks = [0, 0] # nook
_acts = [acts[0], acts[1]] # act
_observers = [1, 2] # observe
_curbs = ['Mid', 'Mid'] # curb
posteriors, mapes = retrieve_posteriors_mapes(_data_paths, _acts, _observers, _curbs, verbose = True) # mape

#%%# Extract SBI Data!

q_lit = list()
gem_q_lit = list()
for index in range(len(_observers)):
    observe = _observers[index]
    posterior = posteriors[index]
    mape = mapes[index]
    print(f'Observe {observe} Synthesize Post')
    posterior_samples, theta = synthesizer_post(posterior = posterior, observation = None, posterior_sample_shape = tuple([100000]))
    q = posterior_samples.numpy()
    q_lit.append(q)
    # print(f'Observe {observe} Geom Med')
    # gem_q = geometric_median(q, list(para_set_raw.keys()), para_set_true)
    # gem_q_lit.append(gem_q)

#%%# Comparison (ANN, SA) [Extract SA Data!]

from Utilities_Optimize_Exploration import synopsis_data_load

path = os.path.dirname(os.path.realpath(__file__)) + f'/../../Data_Bank/Optimize_Exploration/{data_path}/' # f'/media/mars-fias/MARS/MARS_Data_Bank/World/Optimize_Exploration/{data_path}/'
paras = ['N_N', 'G_G', 'FC_N', 'G_EA', 'G_N', 'N_G', 'FC_G', 'N_EA', 'MRNA', 'PRO', 'td_FC', 'td_FM', 'tau_C', 'tau_M', 'tau_ef_EA', 'tau_eb_EA', 'tau_pf_NP', 'tau_pb_NP', 'chi_auto']

a_lit = list()
gem_a_lit = list()
for index in range(len(_observers)):
    observe = _observers[index]
    lot = index+2
    sieve = np.round(sieve_lit[appraisal_opal_discoveries[lot][0]]/100, 2)
    act_range = (0, acts[lot]+1)
    for act in range(act_range[0], act_range[1]):
        tag = f'Act_{act}_Observe_{observe}_{curb.capitalize()}'
        task_pins_info = None
        synopses, tasks_info = synopsis_data_load(task_pins_info, path, tag)
        for j in range(0, len(synopses)):
            _synopsis = synopses[j][0, :, :]
            if j == 0:
                synopsis = _synopsis
            else:
                synopsis = np.concatenate((synopsis, _synopsis))
        _a = synopsis
        if act == np.min(act_range):
            a_temp = _a
        else:
            a_temp = np.concatenate((a_temp, _a))
    a = a_temp[a_temp[:, -1] >= sieve, :]
    a_lit.append(a)
    gem_a = geometric_median(a[:, 0:-1], paras, para_set_true)
    print(f'Observe {observe} {a_temp.shape} {a.shape}\n\t{gem_a}')
    print(f'Lot {lot} Pat {pats[lot]} Sieve {sieve}\n\t{theta_sets[(lot, appraisal_opal_discoveries[lot][0])]}')
    gem_a_lit.append(gem_a)

#%%# Comparison (ANN, SA)

def make_heat_mask(heat_mate):
    heat_mask = np.invert(heat_mate.astype(bool))
    rows, cols = heat_mate.shape
    for row in range(rows):
        for col in range(cols):
            if col < row:
                heat_mask[row, col] = True
            elif col == row:
                heat_mask[row, col] = False
    return heat_mask

#%%# Compare Parameter Sets! [Test]

from scipy.spatial import distance

_mape_norm_1 = [0, 0, 0, 0]
_mape_norm_2 = [0, 0, 1, 1]
_gem_a_norm_1 = [1, 1, 0, 0]
_gem_a_norm_2 = [1, 1, 1, 1]

m = 1 # {0, 1, 2}
metrics = [None, 'cityblock', 'euclidean']
metric = metrics[m]

X = np.array([_mape_norm_1, _mape_norm_2, _gem_a_norm_1, _gem_a_norm_2])
_Y = distance.squareform(distance.pdist(X, metric))
Y = _Y/np.power(X.shape[1], 1/m)
Z = np.round(100*Y, 2)
print(Z)

#%%# Compare Parameter Sets! [Separate]

from scipy.spatial import distance

_mape_norm_1 = np.array([mapes[0][i]/(para_set_true[paras[i]][1][1]-para_set_true[paras[i]][1][0]) for i in range(0, len(paras))])
_mape_norm_2 = np.array([mapes[1][i]/(para_set_true[paras[i]][1][1]-para_set_true[paras[i]][1][0]) for i in range(0, len(paras))])
_gem_a_norm_1 = np.array([gem_a_lit[0][i]/(para_set_true[paras[i]][1][1]-para_set_true[paras[i]][1][0]) for i in range(0, len(paras))])
_gem_a_norm_2 = np.array([gem_a[i]/(para_set_true[paras[i]][1][1]-para_set_true[paras[i]][1][0]) for i in range(0, len(paras))])

m = 1 # {0, 1, 2}
metrics = [None, 'cityblock', 'euclidean']
_metrics = [None, 'L1', 'L2']
metric = metrics[m]
_metric = _metrics[m]

X = np.array([_mape_norm_1, _mape_norm_2, _gem_a_norm_1, _gem_a_norm_2])
_Y = distance.squareform(distance.pdist(X, metric))
Y = _Y/np.power(X.shape[1], 1/m)
Z = np.round(100*Y, 2)
print(Z)

#%%# Disc Plot!

from seaborn import heatmap, light_palette

fig_rows = 1
fig_cols = 1
fig_size_base = 10 # {2.5, 3.75, 5, 7.5, 10}
fig_size = (fig_cols*fig_size_base, fig_rows*fig_size_base)

fig = plt.figure(figsize = fig_size, layout = "constrained")
axe_mat = fig.subplots(nrows = fig_rows, ncols = fig_cols, squeeze = False)
axe = axe_mat

font_size_alp = 23 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_bet = 19 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_chi = 17 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_ate = 23 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53} # Annotate!

prey = 100
n_colors = int(prey/2.5) # {1, ..., 100}
colors = ['darkcyan', 'royalblue', 'steelblue', 'midnightblue']
cmap = light_palette(color = colors[3], n_colors = n_colors, reverse = 0, as_cmap = False)

tick_step = (prey-0)/n_colors
ticks = np.arange(start = 0, stop = prey+tick_step, step = tick_step).tolist()
_tick_labels = [tick if ticks.index(tick) % 4 == 0 else None for tick in ticks]
tick_labels = [int(tick_label) if tick_label is not None and np.round(tick_label) == tick_label else tick_label for tick_label in _tick_labels]
_pats = pats # _pats = [pat+'_'+f'R{acts[pats.index(pat)]}' for pat in pats] if exam else pats
heat_tick_step = 1
heat_ticks = np.arange(start = 0.5, stop = len(lots), step = heat_tick_step).tolist()

heat_mask = make_heat_mask(Z)
heat_map = heatmap(data = Z, mask = heat_mask, cmap = cmap, vmin = 0, vmax = prey, linewidths = 0, linecolor = 'white', annot = True, fmt = '.4g', annot_kws = {'fontsize': font_size_bet}, cbar = True, square = True, ax = axe[0, 0], cbar_kws = {'ticks': None, 'location': 'bottom', 'fraction': 0.05, 'pad': 0.025})
heat_map.set_xticks(ticks = heat_ticks, labels = _pats, fontsize = font_size_bet)
heat_map.set_yticks(ticks = heat_ticks, labels = _pats, fontsize = font_size_bet)
axe[0, 0].vlines(x = [0, 1, 2, 3, 4], ymin = 0, ymax = 4, colors = 'w', linewidths = 2, alpha = 1)
axe[0, 0].hlines(y = [0, 1, 2, 3, 4], xmin = 0, xmax = 4, colors = 'w', linewidths = 2, alpha = 1)
axe[0, 0].figure.axes[-1].set_xticks(ticks = ticks, labels = tick_labels, fontsize = font_size_bet)
axe[0, 0].set_title(label = f'ITWT · Parameter Value Set (Comparison)\n{X.shape[1]}-Dimensional Vector Space · {_metric} Distance [%]', x = 0.5, y = 1, pad = 0, fontsize = font_size_alp)
axe[0, 0].set_title(label = '[D]', loc = 'left', pad = 11, fontsize = font_size_alp, fontweight = 'bold', position = (-0.0625, 1))
plt.show()

#%%# Save Fig!

fig_path = os.path.dirname(os.path.realpath(__file__))
fig_nick = 'Fig6[D]'
fig_forts = ['tiff', 'svg', 'eps', 'pdf', 'png'] # {TIFF, SVG, EPS, PDF, PNG}
fig_keros = [{'compression': 'tiff_lzw'}, None, None, None, None]
fig_flags = [fig_kero is not None for fig_kero in fig_keros]

for fig_fort in fig_forts:
    fig_fort_index = fig_forts.index(fig_fort)
    fig_kero = fig_keros[fig_fort_index]
    fig_flag = fig_flags[fig_fort_index]
    fig_clue = f'{fig_path}/{fig_nick}.{fig_fort}'
    print(fig_clue)
    if fig_flag:
        fig.savefig(fname = fig_clue, dpi = fig_resolution, format = fig_fort, pil_kwargs = fig_kero)
    else:
        fig.savefig(fname = fig_clue, dpi = fig_resolution, format = fig_fort)
