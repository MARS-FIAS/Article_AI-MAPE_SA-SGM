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

#%%# Posterior Samples

posterior_sample_shape = tuple([250000])
observation = posterior.default_x
posterior_samples = posterior.sample(sample_shape = posterior_sample_shape, x = observation)

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

#%%# {1D, 2D} Susceptibility AND Distance Fun

def disc_fun(bins, pose_opal, hist_bing, hist_coda_bing, verbose = False):
    from scipy.spatial import distance
    pose_opal_norm = pose_opal/bins[0]
    hist_bing_corn = np.argwhere(hist_bing)/bins[0]
    hist_bing_disc_temp = np.min(distance.cdist(hist_bing_corn, pose_opal_norm, 'cityblock'))/2
    hist_bing_disc = np.round(100*hist_bing_disc_temp, 2)
    if verbose: print(f'Distance {pose_opal.shape[1]}D Post\n\t{hist_bing_disc}%')
    hist_coda_bing_corn = np.argwhere(hist_coda_bing.T)/bins[0]
    hist_coda_bing_disc_temp = np.min(distance.cdist(hist_coda_bing_corn, pose_opal_norm, 'cityblock'))/2
    hist_coda_bing_disc = np.round(100*hist_coda_bing_disc_temp, 2)
    if verbose: print(f'Distance {pose_opal.shape[1]}D Coda\n\t{hist_coda_bing_disc}%')
    disc = (hist_bing_disc_temp, hist_coda_bing_disc_temp)
    return disc

def make_suss_disc(dimes_1D, dimes_2D, keywords = dict(), verbose = False):
    
    posterior, posterior_samples = keywords.get('posterior'), keywords.get('posterior_samples')
    para_key_label_set, para_value_set = keywords.get('para_key_label_set'), keywords.get('para_value_set')
    edge, bins, resh, limes, condition = keywords.get('edge'), keywords.get('bins'), keywords.get('resh'), keywords.get('limes'), keywords.get('condition')
    
    from sbi.analysis import conditional_density
    
    shape_1D = (2, len(dimes_1D))
    suss_1D = np.full(shape = shape_1D, fill_value = np.nan)
    disc_1D = np.full(shape = shape_1D, fill_value = np.nan)
    
    for dime in dimes_1D:
        print(f"{'·'*8}{para_key_label_set[dime]}{'·'*8}")
        dime_index = dimes_1D.index(dime)
        pose_opal = np.array(bins[0]*(condition[0, dime]/(para_value_set[dime][1]-para_value_set[dime][0]))).reshape((1, -1))
        # 1D Hist!
        hist_temp = np.histogram(a = posterior_samples[:, dime].numpy(), bins = bins[0], range = limes[dime], density = True)
        hist = hist_temp[0]
        hist_maxi = np.max(hist)
        if np.isclose(0, edge):
            hist_bing = np.invert(np.isclose(0, hist))
        else:
            hist_bing = (hist >= hist_maxi*edge)
        hist_pile = np.count_nonzero(hist_bing)
        hist_lent_temp = hist_pile/bins[0]
        hist_lent = np.round(100*hist_lent_temp, 2)
        if verbose: print(f'Length Ratio Post\n\t{hist_lent}%')
        # hist_plat = np.where(hist_bing, np.ones(hist_bing.shape), np.zeros(hist_bing.shape))
        # 1D Hist Coda!
        hist_coda_temp = conditional_density.eval_conditional_density(density = posterior, condition = condition, limits = torch.tensor(limes), dim1 = dime, dim2 = dime, resolution = resh)
        hist_coda = hist_coda_temp.numpy()
        hist_coda_maxi = np.max(hist_coda)
        if np.isclose(0, edge):
            hist_coda_bing = np.invert(np.isclose(0, hist_coda))
        else:
            hist_coda_bing = (hist_coda >= hist_coda_maxi*edge)
        hist_coda_pile = np.count_nonzero(hist_coda_bing)
        hist_coda_lent_temp = hist_coda_pile/bins[0]
        hist_coda_lent = np.round(100*hist_coda_lent_temp, 2)
        if verbose: print(f'Length Ratio Coda\n\t{hist_coda_lent}%')
        # hist_coda_plat = np.where(hist_coda_bing, np.ones(hist_coda_bing.shape), np.zeros(hist_coda_bing.shape))
        # Distance!
        disc = disc_fun(bins, pose_opal, hist_bing, hist_coda_bing, verbose)
        # Safe Data!
        suss_1D[0, dime_index] = hist_lent_temp
        suss_1D[1, dime_index] = hist_coda_lent_temp
        disc_1D[0, dime_index] = disc[0]
        disc_1D[1, dime_index] = disc[1]
    
    shape_2D = (2, len(dimes_2D))
    suss_2D = np.full(shape = shape_2D, fill_value = np.nan)
    disc_2D = np.full(shape = shape_2D, fill_value = np.nan)
    
    for dimes in dimes_2D:
        print(f"{'·'*8}{para_key_label_set[dimes[0]]}{'·'*8}{para_key_label_set[dimes[1]]}{'·'*8}")
        dimes_index = dimes_2D.index(dimes)
        pose_opal = np.array([bins[0]*(condition[0, dime]/(para_value_set[dime][1]-para_value_set[dime][0]))for dime in dimes]).reshape((1, -1))
        # 2D Hist!
        hist_temp = np.histogram2d(x = posterior_samples[:, dimes[0]].numpy(), y = posterior_samples[:, dimes[1]].numpy(), bins = bins, range = [limes[dimes[0]], limes[dimes[1]]], density = True)
        hist = hist_temp[0]
        hist_maxi = np.max(hist)
        if np.isclose(0, edge):
            hist_bing = np.invert(np.isclose(0, hist))
        else:
            hist_bing = (hist >= hist_maxi*edge)
        hist_pile = np.count_nonzero(hist_bing)
        hist_area_temp = hist_pile/(bins[0]*bins[1])
        hist_area = np.round(100*hist_area_temp, 2)
        if verbose: print(f'Area Ratio Post\n\t{hist_area}%')
        # hist_plat = np.where(hist_bing, np.ones(hist_bing.shape), np.zeros(hist_bing.shape))
        # 2D Hist Coda!
        hist_coda_temp = conditional_density.eval_conditional_density(density = posterior, condition = condition, limits = torch.tensor(limes), dim1 = dimes[0], dim2 = dimes[1], resolution = resh)
        hist_coda = hist_coda_temp.numpy()
        hist_coda_maxi = np.max(hist_coda)
        if np.isclose(0, edge):
            hist_coda_bing = np.invert(np.isclose(0, hist_coda))
        else:
            hist_coda_bing = (hist_coda >= hist_coda_maxi*edge)
        hist_coda_pile = np.count_nonzero(hist_coda_bing)
        hist_coda_area_temp = hist_coda_pile/(bins[0]*bins[1])
        hist_coda_area = np.round(100*hist_coda_area_temp, 2)
        if verbose: print(f'Area Ratio Coda\n\t{hist_coda_area}%')
        # hist_coda_plat = np.where(hist_coda_bing, np.ones(hist_coda_bing.shape), np.zeros(hist_coda_bing.shape))
        # Distance!
        disc = disc_fun(bins, pose_opal, hist_bing, hist_coda_bing, verbose)
        # Safe Data!
        suss_2D[0, dimes_index] = hist_area_temp
        suss_2D[1, dimes_index] = hist_coda_area_temp
        disc_2D[0, dimes_index] = disc[0]
        disc_2D[1, dimes_index] = disc[1]
    
    ret = (suss_1D, disc_1D, suss_2D, disc_2D)
    return ret

#%%# Make Suss Disc

from itertools import combinations

suss_disc_lit = list()
edge = 0.25
bins = [250]*2
resh = bins[0]
limes = [[para_value_set[index][0], para_value_set[index][1]] for index in range(posterior_samples.shape[1])]
_dimes_1D = list(range(len(para_key_set)))
_dimes_2D = list(combinations(_dimes_1D, 2))
dimes_1D = [para_key_set.index(para) for para in para_key_elect_set[0]] # Permutation
dimes_2D = list(combinations(dimes_1D, 2))
for coda in lots:
    condition = theta_set_opals[[coda], :]
    keywords = {
        'posterior': posterior, 'posterior_samples': posterior_samples,
        'para_key_label_set': para_key_label_set, 'para_value_set': para_value_set,
        'edge': edge, 'bins': bins, 'resh': resh, 'limes': limes, 'condition': condition
    }
    suss_disc = make_suss_disc(dimes_1D, dimes_2D, keywords, verbose = False)
    suss_disc_lit.append(suss_disc)

#%%# It only works with 4 cases for now and perhaps perfect square numbers! [Suss]

from seaborn import light_palette

kind = 0 # {0, 1} # {'loge', 'pert'}
chop = 1 # {0, 1} # {'post', 'coda'}
_prey = pow(2, 4)
prey_norm_1D = np.log10(bins[0])
prey_norm_2D = np.log10(bins[0]*bins[1])
prey = 100 # 1 if kind == 0 else 100 # prey = np.floor(np.log10(bins[0]*bins[1])) if kind == 0 else 100
n_colors = _prey # _prey*prey if kind == 0 else _prey

fig_rows = 1
fig_cols = 1
# fig_size_base = 10 # {2.5, 3.75, 5, 7.5, 10}
fig_size = (fig_cols*fig_size_base, fig_rows*fig_size_base)
cmap = light_palette(color = ['darkred', 'darkolivegreen'][kind], n_colors = n_colors, reverse = kind, as_cmap = False) # diverging_palette(h_neg = 330, h_pos = 110, s = 100, l = 40, sep = int(0.625*pow(2, 8)), n = pow(2, 8), center = 'light', as_cmap = True) # {'PiYG', 'Spectral', 'RdBu', 'coolwarm', 'bwr'}

fig = plt.figure(figsize = fig_size, layout = "constrained")
axe_mat = fig.subplots(nrows = fig_rows, ncols = fig_cols, squeeze = False)
axe = axe_mat

tick_step = (prey-0)/n_colors
ticks = np.arange(start = 0, stop = prey+tick_step, step = tick_step).tolist()
_tick_labels = [tick if ticks.index(tick) % 2 == 0 else None for tick in ticks]
tick_labels = [int(tick_label) if tick_label is not None and np.round(tick_label) == tick_label else tick_label for tick_label in _tick_labels]

paras = ['A', 'B', 'C', 'D']
cases = [0, 1, 2, 3]
dire = int(len(cases)/2)
rows, cols = len(paras), len(paras)
heat_dime = (len(paras)*dire, len(paras)*dire)
heat_delts = [(0, 0), (1, 1), (0, 1), (1, 0)]
heat_mates = list() # 10/12.5
heat_masks = list() # Small Model · Large Model

if chop == 1:
    heat_tick_step = 2
    heat_ticks = np.arange(start = 1, stop = len(paras)*dire, step = heat_tick_step).tolist()
    index = rows*dire # index = cols*dire
    heat_pate = np.full(shape = heat_dime, fill_value = len(paras), dtype = np.float32)
    heat_pate_mask = np.full(shape = heat_dime, fill_value = True, dtype = bool)
    heat_pate_anna = np.full(shape = heat_dime, fill_value = '', dtype = 'object')
    heat_pate_where = [(index-2, 0), (index-1, 1), (index-2, 1), (index-1, 0)]
    annotate_heat_pate = [pats[0], pats[1], pats[2], pats[3]]
    for heat_pate_here in heat_pate_where:
        pose = heat_pate_where.index(heat_pate_here)
        heat_pate[heat_pate_here[0], heat_pate_here[1]] = pose
        heat_pate_mask[heat_pate_here[0], heat_pate_here[1]] = False
        heat_pate_anna[heat_pate_here[0], heat_pate_here[1]] = annotate_heat_pate[pose]
    heatmap(data = heat_pate, cmap = theta_set_cocos[:-1]+['white'], annot = heat_pate_anna, fmt = '', annot_kws = {'fontsize': font_size_bet}, cbar = False, square = True, ax = axe[0, 0])
    for case in cases:
        heat_delt = heat_delts[case]
        heat_mate = np.full(shape = heat_dime, fill_value = np.nan, dtype = np.float32)
        suss_1D, _, suss_2D, _ = suss_disc_lit[case]
        for row in range(rows):
            for col in range(cols):
                index_1D = row # index_1D = col
                _index_2D = (row, col)
                flag_1D = row == col
                flag_2D = _index_2D in dimes_2D
                row_case = row*dire + heat_delt[0]
                col_case = col*dire + heat_delt[1]
                if flag_1D:
                    entry = -np.log10(suss_1D[chop][index_1D])/prey_norm_1D if kind == 0 else suss_1D[chop][index_1D]
                    heat_mate[row_case, col_case] = np.round(100*entry, 4)
                elif flag_2D:
                    index_2D = dimes_2D.index(_index_2D)
                    entry = -np.log10(suss_2D[chop][index_2D])/prey_norm_2D if kind == 0 else suss_2D[chop][index_2D]
                    heat_mate[row_case, col_case] = np.round(100*entry, 4)
                else:
                    continue
        flag = True if case == len(cases)-1 else False
        heat_mask = cure_heat_mask(make_heat_mask(heat_mate))
        print(np.abs(np.sign(heat_mate)))
        heat_map = heatmap(data = heat_mate, mask = heat_mask, cmap = cmap, vmin = 0, vmax = prey, linewidths = 0, linecolor = 'white', annot = True, annot_kws = {'fontsize': font_size_bet}, cbar = flag, square = True, ax = axe[0, 0], cbar_kws = {'ticks': None, 'location': 'bottom', 'fraction': 0.05, 'pad': 0.0125})
    axe[0, 0].vlines(x = [0, 2, 4, 6, 8], ymin = 0, ymax = 8, colors = 'w', linewidths = 2, alpha = 1)
    axe[0, 0].hlines(y = [0, 2, 4, 6, 8], xmin = 0, xmax = 8, colors = 'w', linewidths = 2, alpha = 1)
    axe[0, 0].figure.axes[-1].set_xticks(ticks = ticks, labels = tick_labels, fontsize = font_size_bet)
    # plt.show()
elif chop == 0:
    heat_tick_step = 1
    heat_ticks = np.arange(start = 0.5, stop = len(paras), step = heat_tick_step).tolist()
    case = pick
    index = len(paras)
    heat_mate = np.full(shape = (int(heat_dime[0]/dire), int(heat_dime[1]/dire)), fill_value = np.nan, dtype = np.float32)
    suss_1D, _, suss_2D, _ = suss_disc_lit[case]
    heat_pate = np.full(shape = heat_mate.shape, fill_value = len(paras), dtype = np.float32)
    heat_pate_mask = np.full(shape = heat_mate.shape, fill_value = True, dtype = bool)
    heat_pate_anna = np.full(shape = heat_mate.shape, fill_value = '', dtype = 'object')
    heat_pate_here = (index-1, 0)
    annotate_heat_pate = pats[pick]
    heat_pate[heat_pate_here[0], heat_pate_here[1]] = pick
    heat_pate_mask[heat_pate_here[0], heat_pate_here[1]] = False
    heat_pate_anna[heat_pate_here[0], heat_pate_here[1]] = annotate_heat_pate
    heatmap(data = heat_pate, cmap = theta_set_cocos[:-1]+['white'], annot = heat_pate_anna, fmt = '', annot_kws = {'color': theta_set_cocos[pick], 'fontsize': font_size_bet, 'fontweight': 'bold'}, cbar = False, square = True, ax = axe[0, 0], alpha = 0.25)
    for row in range(rows):
        for col in range(cols):
            index_1D = row # index_1D = col
            _index_2D = (row, col)
            flag_1D = row == col
            flag_2D = _index_2D in dimes_2D
            if flag_1D:
                entry = -np.log10(suss_1D[chop][index_1D])/prey_norm_1D if kind == 0 else suss_1D[chop][index_1D]
                heat_mate[row, col] = np.round(100*entry, 4)
            elif flag_2D:
                index_2D = dimes_2D.index(_index_2D)
                entry = -np.log10(suss_2D[chop][index_2D])/prey_norm_2D if kind == 0 else suss_2D[chop][index_2D]
                heat_mate[row, col] = np.round(100*entry, 4)
            else:
                continue
    flag = True
    heat_mask = make_heat_mask(heat_mate)
    heat_map = heatmap(data = heat_mate, mask = heat_mask, cmap = cmap, vmin = 0, vmax = prey, linewidths = 0, linecolor = 'white', annot = True, annot_kws = {'fontsize': font_size_bet}, cbar = flag, square = True, ax = axe[0, 0], cbar_kws = {'ticks': None, 'location': 'bottom', 'fraction': 0.05, 'pad': 0.0125})
    plt.vlines(x = [0, 1, 2, 3, 4], ymin = 0, ymax = 4, colors = 'w', linewidths = 2, alpha = 1)
    plt.hlines(y = [0, 1, 2, 3, 4], xmin = 0, xmax = 4, colors = 'w', linewidths = 2, alpha = 1)
    axe[0, 0].figure.axes[-1].set_xticks(ticks = ticks, labels = tick_labels, fontsize = font_size_bet)
    # plt.show()
else:
    raise NotImplementedError('Oops!')
axe[0, 0].tick_params(bottom = False, top = True, left = False, right = True, labelbottom = False, labeltop = True, labelleft = False, labelright = True)
heat_map.set_xticks(ticks = heat_ticks, labels = para_key_label, fontsize = font_size_bet)
heat_map.set_yticks(ticks = heat_ticks, labels = para_key_label, fontsize = font_size_bet, rotation = 270, rotation_mode = 'anchor', horizontalalignment = 'center', verticalalignment = 'bottom')
tit = 'RTM'+' · '+['Unconditional', 'Conditional'][chop]+' '+['Sensitivity Coefficient [%]', 'Coverage Coefficient [%]'][kind]
axe[0, 0].set_title(label = tit, y = -0.05, pad = 0, fontsize = font_size_alp)
axe[0, 0].text(s = '[D]', x = [-0.1875, -0.375][chop], y = [-0.125, -0.25][chop], ha = 'center', va = 'center', fontsize = font_size_alp, fontweight = 'bold')
plt.show()

#%%# Save Fig!

fig_path = os.path.dirname(os.path.realpath(__file__))
fig_nick = 'Fig12[D]'
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
