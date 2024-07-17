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
lots = range(len(data_paths))
sieve_indices = range(0, len(sieve_lit)) # range(min(sieve_lit), max(sieve_lit)+1)
sieve_rag_shape = (len(lots), 4, len(sieve_indices))
sieve_rag = np.full(shape = sieve_rag_shape, fill_value = np.nan)
pats = ['AI_1', 'AI_2', 'SA_1', 'SA_2'] # pats = ['AI (·1)', 'AI (·2)', 'SA (·1)', 'SA (·2)'] # pats = ['AI [·1]', 'AI [·2]', 'SA [·1]', 'SA [·2]']
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

#%%# Simulation [Preparation]

from Utilities_Optimize_Exploration import make_paras

para_set_raw = {
    'N_N': (100, (0, 1000), 1, None), 'G_G': (200, (0, 1000), 1, None),
    'G_N': (50, (0, 1000), 1, None), 'N_G': (400, (0, 1000), 1, None)
}
para_set_mode = 0 # {0, 1} # {'No Remap', 'Remap: [A, B] ---->>>> [0, 1]'}
para_set, para_set_true = make_paras(para_set_raw = para_set_raw, para_set_mode = para_set_mode, verbose = True)

#%%# SBI Data

from Fig0_Services import retrieve_posteriors_mapes, synthesizer_post, geometric_median

_data_paths = ['Shallow_Grid_1_Rule_Kern', 'Shallow_Grid_1_Rule_Kern'] # data_path
_nooks = [0, 0] # nook
_acts = [3, 3] # act
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

path = f'/media/mars-fias/MARS/MARS_Data_Bank/World/Optimize_Exploration/{data_path}/'
paras = list(para_set_true.keys())
act_range = (0, 4)

a_lit = list()
gem_a_lit = list()
for index in range(len(_observers)):
    observe = _observers[index]
    lot = index+2
    sieve = np.round(sieve_lit[appraisal_opal_discoveries[lot][0]]/100, 2)
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

#%%# Synthetic Creationism

pick = 0 # [0, 1] # picks

posterior = posteriors[pick]
para_key_set = para_key_sets[pick]
para_value_set = para_value_sets[pick]
para_set_mode = para_set_modes[pick]

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

#%%# {Make Periphery, Make Surface, Periphery Disc Fun, Surface Disc Fun}

from skimage.morphology import convex_hull_image
from scipy.stats import gaussian_kde

def make_periphery(data, dime, keywords = dict(), soft = False, verbose = False):
    para_key_label_set, para_value_set = keywords.get('para_key_label_set'), keywords.get('para_value_set')
    bins, limes = keywords.get('bins'), keywords.get('limes')
    coco, pat = keywords.get('coco'), keywords.get('pat')
    print(f"{'·'*8}{para_key_label_set[dime]}{'·'*8}")
    variety = 'refer' if data.shape[1] == len(para_value_set) else 'alter'
    w = data[:, dime]
    weights = None if variety == 'refer' else data[:, -1]
    hist_temp = np.histogram(a = w, bins = bins, range = limes[dime], weights = weights, density = False)
    hist = hist_temp[0]
    hist_pose = hist_temp[1]
    hist_maxi = np.max(hist)
    hist_1D = hist/np.sum(hist)
    silhouette = (hist >= edge*hist_maxi).astype(float)
    if soft:
        _KDE = gaussian_kde(w)
        KDE = _KDE(hist_pose)
        KDE_maxi = np.max(KDE)
        silhouette_soft = KDE/KDE_maxi
    else:
        silhouette_soft = None
    if verbose:
        plt.plot(hist_pose[1:len(hist_pose)], hist/hist_maxi, alpha = 0.5, color = 'tab:gray' if soft else coco, drawstyle = 'steps-pre', linewidth = 1)
        plt.plot(hist_pose[1:len(hist_pose)], silhouette, alpha = 1, color = 'tab:gray' if soft else coco, drawstyle = 'steps-pre', linewidth = 1)
        if soft: plt.plot(hist_pose, silhouette_soft, alpha = 0.75, color = coco, drawstyle = 'default', linewidth = 2)
        # plt.xlim(para_value_set[dime][0], para_value_set[dime][1])
        plt.ylim(0-0.05, 1+0.05)
        plt.title(f'{pat}\n{para_key_label_set[dime]}')
        plt.show()
    ret = {'hist_1D': hist_1D, 'silhouette': silhouette, 'silhouette_soft': silhouette_soft}
    return ret

def make_surface(data, dimes, keywords = dict(), soft = False, verbose = False):
    para_key_label_set, para_value_set = keywords.get('para_key_label_set'), keywords.get('para_value_set')
    bins, limes, edge = keywords.get('bins'), keywords.get('limes'), keywords.get('edge')
    coco, pat = keywords.get('coco'), keywords.get('pat')
    print(f"{'·'*8}{para_key_label_set[dimes[0]]}{'·'*8}{para_key_label_set[dimes[1]]}{'·'*8}")
    variety = 'refer' if data.shape[1] == len(para_value_set) else 'alter'
    x = data[:, dimes[0]]
    y = data[:, dimes[1]]
    weights = None if variety == 'refer' else data[:, -1]
    hist_temp = np.histogram2d(x = x, y = y, bins = bins, range = [limes[dimes[0]], limes[dimes[1]]], density = False, weights = weights)
    hist = hist_temp[0]
    hist_pose_temp = np.meshgrid(hist_temp[1], hist_temp[2])
    hist_pose = np.vstack([hist_pose_temp[0].ravel(), hist_pose_temp[1].ravel()])
    hist_maxi = np.max(hist)
    hist_2D = hist/np.sum(hist)
    contour_temp = (hist >= edge*hist_maxi)
    contour = convex_hull_image(contour_temp).astype(float)
    if soft:
        z = np.vstack([x, y])
        _KDE = gaussian_kde(z)
        KDE_temp = _KDE(hist_pose)
        KDE = np.transpose(np.reshape(KDE_temp, (hist_temp[1].size, hist_temp[2].size)))
        KDE_maxi = np.max(KDE)
        contour_soft = (KDE >= edge*KDE_maxi).astype(float)
    else:
        contour_soft = None
    if verbose:
        plt.imshow(X = hist_2D, cmap = 'Greys_r', origin = 'lower')
        if soft: plt.contour(KDE, levels = [level*KDE_maxi for level in [edge*mule for mule in range(0, int(1/edge))]], colors = 'white', alpha = 0.75, linewidths = 1, linestyles = 'solid')
        plt.contour(contour, colors = 'magenta', alpha = 0.75, linewidths = 1, linestyles = 'solid')
        if soft: plt.contour(contour_soft, colors = coco, alpha = 0.75, linewidths = 2, linestyles = 'solid')
        plt.title(f"{pat}\n{para_key_label_set[dimes[0]]}{'·'*8}{para_key_label_set[dimes[1]]}")
        plt.show()
    ret = {'hist_2D': hist_2D, 'contour': contour, 'contour_soft': contour_soft}
    return ret

from scipy.spatial.distance import jensenshannon

def periphery_disc_fun(peripheries, pat_reference, pats, dimes_1D, cocos, metric = 'JS', kind = 'raw', show = False, verbose = False):
    # {'raw', 'rough', 'soft'} # {'hist_1D', 'silhouette', 'silhouette_soft'}
    check = pat_reference in pats
    mess = 'Oops! The reference is not an element of the source set!'
    assert check, mess
    kind_1D = {'raw': 'hist_1D', 'rough': 'silhouette', 'soft': 'silhouette_soft'}
    _kind = kind_1D[kind]
    print(f"{'~'*8}Distance 1D [Start] {kind.capitalize()}{'~'*8}")
    periphery_disc_mat = np.full(shape = (len(dimes_1D), len(pats)), fill_value = np.nan)
    if show:
        fig_rows = len(dimes_1D)
        fig_cols = len(dimes_1D)
        fig_size_base = 5
        fig_size = (fig_cols*fig_size_base, fig_rows*fig_size_base)
        fig = plt.figure(figsize = fig_size, layout = "constrained")
        axe_mat = fig.subplots(nrows = fig_rows, ncols = fig_cols, squeeze = False)
        _ = [axe_mat[fig_row, fig_col].remove() for fig_row in range(fig_rows) for fig_col in range(fig_cols) if fig_row != fig_col]
    for dime in dimes_1D:
        if verbose: print(f"{'·'*8}{dime}{'·'*8}")
        index = dimes_1D.index(dime)
        _p = peripheries[pat_reference][dime][_kind]
        p = _p/np.sum(_p)
        for pat in pats:
            _q = peripheries[pat][dime][_kind]
            q = _q/np.sum(_q)
            if metric == 'JS':
                disc = jensenshannon(p, q, 2)
            elif metric == 'L1':
                disc = np.sum(np.abs(p-q))/2
            else:
                raise NotImplementedError('Oops!')
            if verbose: print(f"{pat_reference}{' '*4}{pat}{' '*4}{metric}{' '*4}{np.round(100*disc, 2)}%")
            periphery_disc_mat[dimes_1D.index(dime), pats.index(pat)] = disc
            if show:
                axe_mat[index, index].plot(q, alpha = 0.25, color = cocos[pats.index(pat)], linestyle = '--', linewidth = 2)
                axe_mat[index, index].fill_between(x = np.arange(len(q)), y1 = q, alpha = 0.125, color = cocos[pats.index(pat)])
                axe_mat[index, index].spines[['right', 'top']].set_visible(False)
                axe_mat[index, index].set_xticks(ticks = [])
                axe_mat[index, index].set_yticks(ticks = [])
    if show: plt.show()
    print(f"{'~'*8}Distance 1D [Final] {kind.capitalize()}{'~'*8}")
    return periphery_disc_mat

def surface_disc_fun(surfaces, pat_reference, pats, dimes_2D, cocos, metric = 'JS', kind = 'raw', show = False, verbose = False):
    # {'raw', 'rough', 'soft'} # {'hist_2D', 'contour', 'contour_soft'}
    check = pat_reference in pats
    mess = 'Oops! The reference is not an element of the source set!'
    assert check, mess
    kind_2D = {'raw': 'hist_2D', 'rough': 'contour', 'soft': 'contour_soft'}
    _kind = kind_2D[kind]
    print(f"{'~'*8}Distance 2D [Start] {kind.capitalize()}{'~'*8}")
    surface_disc_mat = np.full(shape = (len(dimes_2D), len(pats)), fill_value = np.nan)
    if show:
        scope = np.sort(np.unique(list(zip(*dimes_2D))))
        fig_rows = len(scope)
        fig_cols = len(scope)
        fig_size_base = 5
        fig_size = (fig_cols*fig_size_base, fig_rows*fig_size_base)
        fig = plt.figure(figsize = fig_size, layout = "constrained")
        axe_mat = fig.subplots(nrows = fig_rows, ncols = fig_cols, squeeze = False)
        _ = [axe_mat[fig_row, fig_col].remove() for fig_row in range(fig_rows) for fig_col in range(fig_cols) if fig_row > fig_col]
        _ = [axe_mat[fig_row, fig_col].spines[['left', 'right', 'top', 'bottom']].set_visible(False) for fig_row in range(fig_rows) for fig_col in range(fig_cols) if fig_row == fig_col]
        _ = [axe_mat[fig_row, fig_col].set_xticks(ticks = []) for fig_row in range(fig_rows) for fig_col in range(fig_cols) if fig_row == fig_col]
        _ = [axe_mat[fig_row, fig_col].set_yticks(ticks = []) for fig_row in range(fig_rows) for fig_col in range(fig_cols) if fig_row == fig_col]
    for dimes in dimes_2D:
        if verbose: print(f"{'·'*8}{dimes}{'·'*8}")
        index = dimes
        _p = surfaces[pat_reference][dimes][_kind]
        p = (_p/np.sum(_p)).flatten()
        for pat in pats:
            _q = surfaces[pat][dimes][_kind]
            q = (_q/np.sum(_q)).flatten()
            if metric == 'JS':
                disc = jensenshannon(p, q, 2)
            elif metric == 'L1':
                disc = np.sum(np.abs(p-q))/2
            else:
                raise NotImplementedError('Oops!')
            if verbose: print(f"{pat_reference}{' '*4}{pat}{' '*4}{metric}{' '*4}{np.round(100*disc, 2)}%")
            surface_disc_mat[dimes_2D.index(dimes), pats.index(pat)] = disc
            if show:
                axe_mat[index[0], index[1]].contour(_q, levels = [edge*np.max(_q), np.max(_q)], colors = cocos[pats.index(pat)], alpha = 0.5, linewidths = 2, linestyles = 'solid')
                axe_mat[index[0], index[1]].spines[['right', 'top']].set_visible(False)
                axe_mat[index[0], index[1]].set_xticks(ticks = [])
                axe_mat[index[0], index[1]].set_yticks(ticks = [])
    if show: plt.show()
    print(f"{'~'*8}Distance 2D [Final] {kind.capitalize()}{'~'*8}")
    return surface_disc_mat

#%%# Process Data

from itertools import combinations

_dimes_1D = list(range(len(para_key_set)))
_dimes_2D = list(combinations(_dimes_1D, 2))
dimes_1D = [para_key_set.index(para) for para in para_key_elect_set[0]] # Permutation
dimes_2D = list(combinations(dimes_1D, 2))

data_lit = q_lit + a_lit # pats = ['AI_1', 'AI_2', 'SA_1', 'SA_2']
# para_key_label_set
# para_value_set
bins = 250
limes = [[para_value_set[index][0], para_value_set[index][1]] for index in range(posterior_samples.shape[1])]
edge = 0.25
# pats
cocos = ['tab:red', 'tab:purple', 'tab:olive', 'tab:green']
soft = True
verbose = True

safe = False # Careful!
trail = os.path.dirname(os.path.realpath(__file__)) + '/../../Data_Bank/Fig_Support/'
_loci = ['Data_Periphery.pkl', 'Data_Surface.pkl']
loci = ['Fig5[A]'+'_'+_ for _ in _loci]
periphery_locus = trail+loci[0]
surface_locus = trail+loci[1]

#%%# Process Data 1D [Make]

if safe:
    peripheries = {pat: {dime: None for dime in dimes_1D} for pat in pats}
    for lot in lots:
        data = data_lit[lot]
        keywords = {
            'para_key_label_set': para_key_label_set, 'para_value_set': para_value_set,
            'bins': bins, 'limes': limes, 'edge': edge,
            'pat': pats[lot], 'coco': cocos[lot]
        }
        for dime in _dimes_1D:
            periphery = make_periphery(data, dime, keywords, soft, verbose)
            peripheries[pats[lot]][dime] = periphery
    with open(periphery_locus, 'wb') as portfolio:
        pickle.dump(peripheries, portfolio)
else:
    with open(periphery_locus, 'rb') as portfolio:
        peripheries = pickle.load(portfolio)

#%%# Process Data 2D [Make]

if safe:
    surfaces = {pat: {dimes: None for dimes in dimes_2D} for pat in pats}
    for lot in lots:
        data = data_lit[lot]
        keywords = {
            'para_key_label_set': para_key_label_set, 'para_value_set': para_value_set,
            'bins': bins, 'limes': limes, 'edge': edge,
            'pat': pats[lot], 'coco': cocos[lot]
        }
        for dimes in _dimes_2D:
            surface = make_surface(data, dimes, keywords, soft, verbose)
            surfaces[pats[lot]][dimes] = surface
    with open(surface_locus, 'wb') as portfolio:
        pickle.dump(surfaces, portfolio)
else:
    with open(surface_locus, 'rb') as portfolio:
        surfaces = pickle.load(portfolio)

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

#%%# Process Data {1D, 2D} [Disc Fun]

kind_cate = 'raw'
kind_show = 'soft'
kind_1D = {'raw': 'hist_1D', 'rough': 'silhouette', 'soft': 'silhouette_soft'}
kind_2D = {'raw': 'hist_2D', 'rough': 'contour', 'soft': 'contour_soft'}
# {peripheries, surfaces}
# pat_reference
# pats
# {dimes_1D, dimes_2D}
# cocos
metric = 'JS' # {'JS', 'L1'}
kind = kind_cate
show = False
verbose = True
_Z_1D = np.full(shape = (len(pats), len(pats)), fill_value = np.nan)
_Z_2D = np.full(shape = (len(pats), len(pats)), fill_value = np.nan)
for pat_reference in pats:
    index = pats.index(pat_reference)
    periphery_disc_mat = periphery_disc_fun(peripheries, pat_reference, pats, dimes_1D, cocos, metric, kind, show, verbose)
    _Z_1D[index, :] = np.mean(periphery_disc_mat, 0)
    surface_disc_mat = surface_disc_fun(surfaces, pat_reference, pats, dimes_2D, cocos, metric, kind, show, verbose)
    _Z_2D[index, :] = np.mean(surface_disc_mat, 0)
Z_1D = np.round(100*_Z_1D, 2)
Z_2D = np.round(100*_Z_2D, 2)
_Z = [Z_1D, Z_2D]

#%%# Draw Data {1D, 2D}

from seaborn import heatmap, light_palette

font_size_alp = 23 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_bet = 19 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_chi = 17 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_ate = 23 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}

prey = 100
n_colors = int(prey/2.5) # {1, ..., 100}
colors = ['darkcyan', 'royalblue', 'steelblue', 'midnightblue']
cmap = light_palette(color = colors[3], n_colors = n_colors, reverse = 0, as_cmap = False)

tick_step = (prey-0)/n_colors
ticks = np.arange(start = 0, stop = prey+tick_step, step = tick_step).tolist()
_tick_labels = [tick if ticks.index(tick) % 4 == 0 else None for tick in ticks]
tick_labels = [int(tick_label) if tick_label is not None and np.round(tick_label) == tick_label else tick_label for tick_label in _tick_labels]

heat_tick_step = 1
heat_ticks = np.arange(start = 0.5, stop = len(paras), step = heat_tick_step).tolist()

fig_rows = 1
fig_cols = 1
fig_size_base = 10 # {2.5, 3.75, 5, 7.5, 10}
fig_size = (fig_cols*fig_size_base, fig_rows*fig_size_base)

for index in range(len(_Z)):
    Z = _Z[index]
    fig = plt.figure(figsize = fig_size, layout = "constrained")
    axe_mat = fig.subplots(nrows = fig_rows, ncols = fig_cols, squeeze = False)
    heat_mask = make_heat_mask(Z)
    heat_map = heatmap(data = Z, mask = heat_mask, cmap = cmap, vmin = 0, vmax = prey, linewidths = 0, linecolor = 'white', annot = True, fmt = '.4g', annot_kws = {'fontsize': font_size_bet}, cbar = True, square = True, ax = axe_mat[0, 0], cbar_kws = {'ticks': None, 'location': 'bottom', 'fraction': 0.05, 'pad': 0.025})
    heat_map.set_xticks(ticks = heat_ticks, labels = pats, fontsize = font_size_bet)
    heat_map.set_yticks(ticks = heat_ticks, labels = pats, fontsize = font_size_bet)
    axe_mat[0, 0].vlines(x = [0, 1, 2, 3, 4], ymin = 0, ymax = 4, colors = 'w', linewidths = 2, alpha = 1)
    axe_mat[0, 0].hlines(y = [0, 1, 2, 3, 4], xmin = 0, xmax = 4, colors = 'w', linewidths = 2, alpha = 1)
    axe_mat[0, 0].figure.axes[-1].set_xticks(ticks = ticks, labels = tick_labels, fontsize = font_size_bet)
    tit = f'RTM · Parameter Posterior (Comparison)\n{index+1}D Marginals · Average {metric} Distance [%]'
    axe_mat[0, 0].set_title(label = tit, x = 0.5, y = 1, pad = 0, fontsize = font_size_alp)
    axe_mat[0, 0].set_title(label = ['[B]', '[C]'][index], loc = 'left', pad = 11, fontsize = font_size_alp, fontweight = 'bold', position = (-0.0625, 1))
    # fig.suptitle(t = tit, fontsize = font_size_alp-2)
    plt.show()

#%%# Save Fig!

fig_path = os.path.dirname(os.path.realpath(__file__))
fig_nick = 'Fig5[B][C]'
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
