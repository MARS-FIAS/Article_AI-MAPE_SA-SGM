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

rows = 1
cols = 1
fig_size = (cols*fig_size_base, rows*fig_size_base)
fig_size_panel = (fig_size_base, fig_size_base)
fig = plt.figure(figsize = fig_size, layout = "constrained")
fig_mat = fig.subfigures(nrows = rows, ncols = cols, squeeze = False)
plt.set_cmap('Greys_r')
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
cmap = 'PiYG' # diverging_palette(h_neg = 310, h_pos = 170, s = 100, l = 50, sep = 10, n = 100, center = 'light', as_cmap = True) # {'PiYG', 'Spectral', 'RdBu', 'coolwarm', 'bwr'}

tits_alp = ['[A]', '[A]']
tits_bet = ['[B]', '[B]']

mape = [int(np.round(numb, 0)) if numb > 1 else np.round(numb, 2) for numb in posterior.map().tolist()]
if theta_set_picks['coda'] != len(lots):
    geon = [int(np.round(numb, 0)) if numb > 1 else np.round(numb, 2) for numb in theta_set_opals[theta_set_picks['coda']].tolist()]
else:
    _geon = [np.mean(rang) for rang in para_value_set]
    geon = [int(np.round(numb, 0)) if numb > 1 else np.round(numb, 2) for numb in _geon]

sieve_opals = [sieve_lit[appraisal_opal_discoveries[lot][0]] for lot in lots]
_pats = pats # _pats = [f'{pats[0]} R{acts[0]}', f'{pats[1]} R{acts[1]}', f'{pats[2]} R{acts[2]} {sieve_opals[2]}%', f'{pats[3]} R{acts[3]} {sieve_opals[3]}%']
theta_set_pats = _pats+['Dummy']
cure = {'tick', 'label', 'legend'}

case = 0 # {'post': 0, 'coda': 1}

for para_key_elect_index in range(len(para_key_elect_set)):
    print(f"{'~'*2*8}Elect Index {para_key_elect_index}{'~'*2*8}")
    para_key_caste = para_key_caste_set[para_key_elect_index]
    para_key_elect = para_key_elect_set[para_key_elect_index]
    para_key_label = para_key_label_set
    row_plan = para_key_elect_index // cols
    col_plan = para_key_elect_index % cols
    fig_vet = fig_mat[row_plan, col_plan] if case == 0 else None
    axe_mat = fig_vet.subplots(nrows = len(para_key_caste), ncols = len(para_key_caste), squeeze = False, subplot_kw = {'box_aspect': 1}) if case == 0 else None
    fig_vet_coda = fig_mat[row_plan, col_plan] if case == 1 else None
    axe_mat_coda = fig_vet_coda.subplots(nrows = len(para_key_caste), ncols = len(para_key_caste), squeeze = False, subplot_kw = {'box_aspect': 1}) if case == 1 else None
    print(f'Plan\n\t({row_plan}, {col_plan}) ~ ({row_plan}, {col_plan+1})')
    chart, chart_coda, posterior_samples, core_coda = synthesizer_post_coda(
        posterior = posterior, observation = None, posterior_sample_shape = tuple([250000]), parameter_set_true = para_set_true, mape_calculate = False, fig_size = fig_size_panel, verbose = verbose,
        theta_set_opals = theta_set_opals, theta_set_picks = theta_set_picks, theta_set_cocos = theta_set_cocos,
        para_value_set = para_value_set, para_key_caste = para_key_caste, para_key_elect = para_key_elect, para_key_label = para_key_label,
        fig = fig_vet, axes = axe_mat, fig_coda = fig_vet_coda, axes_coda = axe_mat_coda,
        mark_size = mark_size_alp
    )
    tit_alp = tits_alp[para_key_elect_index]
    tit_bet = tits_bet[para_key_elect_index]
    if case == 0:
        fig_vet.suptitle(t = tit_alp, x = 0.025, y = 1.025, ha = 'center', va = 'center', fontsize = font_size_alp, fontweight = 'bold')
        fig_vet.text(s = 'RTM · Unconditional Posterior Marginals {1D, 2D}', x = 0.5, y = 1.025, ha = 'center', va = 'center', fontsize = font_size_alp, fontweight = 'normal')
        axe_mat = cure_axe_mat(
            axe_mat, para_key_caste, para_value_set, mape,
            font_size_tick = font_size_tick, font_size_label = font_size_tick, font_size_loge = font_size_label, cure = cure, verbose = verbose,
            theta_set_opals = theta_set_opals, theta_set_picks = theta_set_picks, theta_set_cocos = theta_set_cocos, theta_set_pats = theta_set_pats, coda = False
        )
    elif case == 1:
        fig_vet_coda.suptitle(t = tit_bet, x = 0.025, y = 1.025, ha = 'center', va = 'center', fontsize = font_size_alp, fontweight = 'bold')
        fig_vet_coda.text(s = 'RTM · Conditional Posterior Marginals {1D, 2D}', x = 0.5, y = 1.025, ha = 'center', va = 'center', fontsize = font_size_alp, fontweight = 'normal')
        axe_mat_coda = cure_axe_mat(
            axe_mat_coda, para_key_caste, para_value_set, geon,
            font_size_tick = font_size_tick, font_size_label = font_size_tick, font_size_loge = font_size_label, cure = cure, verbose = verbose,
            theta_set_opals = theta_set_opals, theta_set_picks = theta_set_picks, theta_set_cocos = theta_set_cocos, theta_set_pats = theta_set_pats, coda = True
        )
    else:
        raise NotImplementedError('Oops!')
plt.show()

#%%# Save Fig!

fig_path = os.path.dirname(os.path.realpath(__file__))
fig_nick = 'Fig4[A]'
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
        fig.savefig(fname = fig_clue, dpi = fig_resolution, format = fig_fort, bbox_inches = 'tight', pil_kwargs = fig_kero)
    else:
        fig.savefig(fname = fig_clue, dpi = fig_resolution, format = fig_fort, bbox_inches = 'tight')
