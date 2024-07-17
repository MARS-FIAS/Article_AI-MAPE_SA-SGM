######################################
######## Optimize Exploration ########
######################################

#%%# Catalyzer

sup_comp = False # Super Computer?
inference_prod_activate = True # Activate Inference Procedure?
data_path = 'Optimize_Exploration_1_Kern' # 'Shallow_Grid_1_Rule_Kern'
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

data_paths = ['Shallow_Grid_1_Rule_Kern', 'Shallow_Grid_1_Rule_Kern', 'Optimize_Exploration_1_Kern', 'Optimize_Exploration_1_Kern'] # data_path
prep_data_paths = ['Shallow_Grid', 'Shallow_Grid', 'Optimize_Exploration', 'Optimize_Exploration']
nooks = [0]*4 # nook
acts = [3]*4 # act
observers = [1, 2, 1, 2] # observe
curbs = ['Mid']*4 # curb
para_key_sets = [ # para_key_set
    ['N_N', 'G_G', 'G_N', 'N_G']
]*4
para_value_sets = [ # para_value_set
    [(0, 1000), (0, 1000), (0, 1000), (0, 1000)]
]*4
para_set_modes = [0]*4 # para_set_mode

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
pats = ['AI_1', 'AI_2', 'SA_1', 'SA_2']
rules = [0]
aim_NT = [2/5]
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

#%%# SBI Data

from Fig0_Services import retrieve_posteriors_mapes, retrieve_para_set_truths

picks = [0, 1]

_data_paths = [data_paths[pick] for pick in picks] # data_path
_nooks = [nooks[pick] for pick in picks] # nook
_acts = [acts[pick] for pick in picks] # act
_observers = [observers[pick] for pick in picks] # observe
_curbs = [curbs[pick] for pick in picks] # curb
posteriors, mapes = retrieve_posteriors_mapes(_data_paths, _acts, _observers, _curbs, verbose = True) # mape
_para_key_sets = [para_key_sets[pick] for pick in picks]
_para_value_sets = [para_value_sets[pick] for pick in picks]
_para_set_modes = [para_set_modes[pick] for pick in picks]
para_set_truths = retrieve_para_set_truths(mapes, _para_key_sets, _para_value_sets, _para_set_modes, verbose = True) # para_set_true

#%%# Conditional Posterior Synthesizer

from Fig0_Services import synthesizer_post_coda, cure_axe_mat

#%%# Synthetic Creationism

pick = 0 # [0, 1] # picks

posterior = posteriors[pick]
para_key_set = para_key_sets[pick]
para_value_set = para_value_sets[pick]
para_set_mode = para_set_modes[pick]
para_set_true = para_set_truths[pick]

para_key_label_set = [ # para_key_label_set
    r'$\mathit{Nanog}$_NANOG', r'$\mathit{Gata6}$_GATA6', r'$\mathit{Gata6}$_NANOG', r'$\mathit{Nanog}$_GATA6'
]

para_key_elect_set = [ # para_key_elect_set
    ['N_N', 'G_G', 'G_N', 'N_G']
]

para_key_caste_set = list()
for para_key_elect in para_key_elect_set:
    para_key_caste = [para_key_set.index(para_key) for para_key in para_key_elect]
    para_key_caste_set.append(para_key_caste)

_theta_set_opals = list()
for lot in lots:
    appraisal_opal_discovery = appraisal_opal_discoveries[lot]
    theta_set_oppal = theta_sets[(lot, appraisal_opal_discovery[0])]
    _theta_set_opals.append(theta_set_oppal)
theta_set_opals = torch.tensor(_theta_set_opals)

#%%# Synthetic Plot! (Only Lot = {0, 1}!)

from seaborn import heatmap, diverging_palette

if pick not in picks:
    mess = "Oops! This script section is only compatible with the data sets 0 and 1!"
    raise RuntimeError(mess)

theta_set_picks = {'post': pick, 'coda': 0} # {'post': {{0, 1}, picks}, 'coda': {{0, 1, 2, 3}U{4}, lots}}

fig_rows = 2
fig_cols = 2
# fig_size_base = 10 # {2.5, 3.75, 5, 7.5, 10}
fig_size = (fig_cols*fig_size_base, fig_rows*fig_size_base)
fig_size_panel = (fig_size_base, fig_size_base)
verbose = True

font_size_alp = 23 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_bet = 19 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_chi = 17 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
mark_size_alp = 13 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}

font_size_tick = 17
font_size_label = 19
plt.rcParams['xtick.labelsize'] = font_size_tick
plt.rcParams['axes.labelsize'] = font_size_label
plt.rcParams['axes.labelweight'] = 'normal'

theta_set_cocos = ['tab:red', 'tab:purple', 'tab:olive', 'tab:green', 'magenta']
cmap = diverging_palette(h_neg = 330, h_pos = 110, s = 100, l = 40, sep = int(0.625*pow(2, 8)), n = pow(2, 8), center = 'light', as_cmap = True) # {'PiYG', 'Spectral', 'RdBu', 'coolwarm', 'bwr'}

sieve_opals = [sieve_lit[appraisal_opal_discoveries[lot][0]] for lot in lots]
_pats = [f'{pats[0]} R{acts[0]}', f'{pats[1]} R{acts[1]}', f'{pats[2]} R{acts[2]} {sieve_opals[2]}%', f'{pats[3]} R{acts[3]} {sieve_opals[3]}%']
theta_set_pats = _pats+['Dummy']
cure = {'tick', 'label', 'legend'}

para_key_label = para_key_label_set # [_ for _ in para_key_label_set if para_key_label_set.index(_) in para_key_caste]

#%%# Mate Test

posterior = posteriors[pick]

fig_rows = 1
fig_cols = 1
# fig_size_base = 10 # {2.5, 3.75, 5, 7.5, 10}
fig_size = (fig_cols*fig_size_base, fig_rows*fig_size_base)
fig = plt.figure(figsize = fig_size, layout = "constrained")
axe_mat = fig.subplots(nrows = fig_rows, ncols = fig_cols, squeeze = False)
axe = axe_mat

codas = [0, 1, 2, 3]
heat_cores = list()
heat_core_codas = list()
for coda in codas:
    theta_set_picks = {'post': pick, 'coda': coda}
    for para_key_elect_index in range(len(para_key_elect_set)):
        para_key_caste = para_key_caste_set[para_key_elect_index]
        chart, chart_coda, posterior_samples, core_coda = synthesizer_post_coda(
            posterior = posterior, observation = None, posterior_sample_shape = tuple([250000]), parameter_set_true = para_set_true, mape_calculate = False, fig_size = fig_size_panel, verbose = verbose,
            theta_set_opals = theta_set_opals, theta_set_picks = theta_set_picks, theta_set_cocos = theta_set_cocos,
            para_value_set = para_value_set, para_key_caste = para_key_caste, para_key_elect = para_key_elect, para_key_label = para_key_label,
            fig = None, axes = None, fig_coda = None, axes_coda = None,
            mark_size = mark_size_alp
        )
        core = np.corrcoef(posterior_samples[:, para_key_caste].T)
        heat_cores.append(core)
        heat_core_codas.append(core_coda)

#%%# Color Map Test

color_map_test = plt.get_cmap(name = 'PiYG')
_color_map_zero_RGBA = color_map_test(0)
color_map_zero_RGBA = [np.round(color_map_test.N*color, 0).astype(int) for color in _color_map_zero_RGBA]
_color_map_last_RGBA = color_map_test(color_map_test.N-1)
color_map_last_RGBA = [np.round(color_map_test.N*color, 0).astype(int) for color in _color_map_last_RGBA]
color_map_zero_HSL = {'h': 325, 's': 95, 'l': 25}
color_map_last_HSL = {'h': 110, 's': 60, 'l': 25}

def make_heat_mask(heat_mate):
    heat_mask = np.invert(heat_mate.astype(bool))
    rows, cols = heat_mate.shape
    for row in range(rows):
        for col in range(cols):
            if col < row:
                heat_mask[row, col] = True
            else:
                continue
    return heat_mask

def cure_heat_mask(heat_mask):
    rows, cols = heat_mask.shape
    cure = [(index+1, index) for index in range(0, cols, 2)] # [(index+1, index) for index in range(rows)]
    for row in range(rows):
        for col in range(cols):
            if (row, col) in cure:
                heat_mask[row, col] = False
            else:
                continue
    return heat_mask

#%%# It only works with 4 cases for now and perhaps perfect square numbers!

fig_rows = 1
fig_cols = 1
# fig_size_base = 10 # {2.5, 3.75, 5, 7.5, 10}
fig_size = (fig_cols*fig_size_base, fig_rows*fig_size_base)
cmap = diverging_palette(h_neg = 330, h_pos = 110, s = 100, l = 40, sep = int(0.5*pow(2, 8)), n = pow(2, 8), center = 'light', as_cmap = True) # {'PiYG', 'Spectral', 'RdBu', 'coolwarm', 'bwr'}

paras = ['A', 'B', 'C', 'D']
cases = [0, 1, 2, 3]
dire = int(len(cases)/2)
rows, cols = len(paras), len(paras)
heat_dime = (len(paras)*dire, len(paras)*dire)
heat_delts = [(0, 0), (1, 1), (0, 1), (1, 0)]
heat_mates = list() # 10/12.5
heat_masks = list() # Small Model · Large Model

tick_step = (1-0)/pow(2, 3)
ticks = np.arange(start = -1, stop = 1+tick_step, step = tick_step).tolist()
_tick_labels = [tick if ticks.index(tick) % 2 == 0 else None for tick in ticks]
tick_labels = [int(tick_label) if tick_label is not None and np.round(tick_label) == tick_label else tick_label for tick_label in _tick_labels]

fig = plt.figure(figsize = fig_size, layout = "constrained")
axe_mat = fig.subplots(nrows = fig_rows, ncols = fig_cols, squeeze = False)
axe = axe_mat

case = 0
heat_core = heat_cores[case]
index = heat_core.shape[0] # index = heat_core.shape[0]*dire
heat_pate = np.full(shape = heat_core.shape, fill_value = len(paras), dtype = np.float32)
heat_pate_mask = np.full(shape = heat_core.shape, fill_value = True, dtype = bool)
heat_pate_anna = np.full(shape = heat_core.shape, fill_value = '', dtype = 'object')
heat_pate_here = (index-1, 0)
annotate_heat_pate = pats[pick]
heat_pate[heat_pate_here[0], heat_pate_here[1]] = pick
heat_pate_mask[heat_pate_here[0], heat_pate_here[1]] = False
heat_pate_anna[heat_pate_here[0], heat_pate_here[1]] = annotate_heat_pate
heatmap(data = heat_pate, cmap = theta_set_cocos[:-1]+['white'], annot = heat_pate_anna, fmt = '', annot_kws = {'color': theta_set_cocos[pick], 'fontsize': font_size_bet, 'fontweight': 'bold'}, cbar = False, square = True, xticklabels = [], yticklabels = [], ax = axe[0, 0], alpha = 0.25)
heat_mask = make_heat_mask(heat_core)
heatmap(data = heat_core, mask = heat_mask, cmap = cmap, vmin = -1, vmax = 1, linewidths = 0, linecolor = 'white', annot = True, annot_kws = {'fontsize': font_size_bet}, cbar = True, square = True, xticklabels = [], yticklabels = [], ax = axe[0, 0], cbar_kws = {'ticks': None, 'location': 'bottom', 'fraction': 0.05, 'pad': 0.0125})
heat_tick_step = 1
heat_ticks = np.arange(start = 0.5, stop = len(paras), step = heat_tick_step).tolist()
axe[0, 0].tick_params(bottom = False, top = True, left = False, right = True, labelbottom = False, labeltop = True, labelleft = False, labelright = True)
axe[0, 0].set_xticks(ticks = heat_ticks, labels = para_key_label, fontsize = font_size_bet)
axe[0, 0].set_yticks(ticks = heat_ticks, labels = para_key_label, fontsize = font_size_bet, rotation = 270, rotation_mode = 'anchor', horizontalalignment = 'center', verticalalignment = 'bottom')
axe[0, 0].vlines(x = [0, 1, 2, 3, 4], ymin = 0, ymax = 4, colors = 'w', linewidths = 2, alpha = 1)
axe[0, 0].hlines(y = [0, 1, 2, 3, 4], xmin = 0, xmax = 4, colors = 'w', linewidths = 2, alpha = 1)
axe[0, 0].figure.axes[-1].set_xticks(ticks = ticks, labels = tick_labels, fontsize = font_size_bet)
axe[0, 0].set_title(label = 'RTM · Unconditional Correlation Coefficient', y = -0.05, pad = 0, fontsize = font_size_alp) # "Pearson's Product-Moment Correlation Coefficient"
axe[0, 0].text(s = '[A]', x = -0.1875, y = -0.125, ha = 'center', va = 'center', fontsize = font_size_alp, fontweight = 'bold')
plt.show()

#%%# Save Fig!

fig_path = os.path.dirname(os.path.realpath(__file__))
fig_nick = 'Fig12[A]'
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
