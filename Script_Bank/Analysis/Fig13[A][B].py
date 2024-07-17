#####################################
######## Paper Draft Cartoon ########
#####################################

#%%# Catalyzer

import os
import sys
path = os.path.dirname(os.path.realpath(__file__)) + '/../Resources'
sys.path.append(path)
path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path)
import numpy as np
import torch
import time
import pickle
import matplotlib
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
acts = [7, 7, 7, 7] # acts = [7, 7, 7, 7] # act
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

#%%# Create Figure! [Rule 0]

from Fig0_Services import retrieve_data_cellulate, objective_fun_counter
from ObjectiveFun import ObjectiveFunPortion # from ObjectiveFunL1L2 import ObjectiveFunPortionRule

rows = 1
cols = 1
fig_size = (cols*fig_size_base, rows*fig_size_base)
fig = plt.figure(figsize = fig_size, layout = "constrained")
axe_mat = fig.subplots(nrows = rows, ncols = cols, squeeze = False)

font_size_alp = 23 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_bet = 19 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_chi = 17 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_ate = 23 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53} # Annotate!

cocos = ['tab:red', 'tab:purple', 'tab:olive', 'tab:green']
cocks = ['tab:gray', 'tab:gray'] # ['tab:blue', 'tab:orange']
caps = ['full', 'none']
clues = ['NT', 'G'] # ['NT', 'G', 'U']
rule = 0

axe = axe_mat
_axe = axe[0, 0].inset_axes([0.5, 0.125, 0.5, 0.15])

for lot in lots:
    sieve = sieve_lit[appraisal_opal_discoveries[lot][0]]
    memo = memos[lot]
    axe = axe_mat
    rule_index = rules.index(rule)
    row = 0
    col = 0
    for cellulate in cellulates:
        cellulate_index = cellulates.index(cellulate)
        reiteration = f'Reiterate_{cellulate[0]}_{cellulate[1]}'
        _trajectory_set = retrieve_data_cellulate(memo, cellulate, reiteration, lot, sieve, verbose = True)[1]
        trajectory_set = _trajectory_set[:, int(_trajectory_set.shape[1]/2):_trajectory_set.shape[1]] if rule == 1 else _trajectory_set[:, 0:int(_trajectory_set.shape[1]/2)] # Temporary fix!
        objective_fun = ObjectiveFunPortion(trajectory_set, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi = 10, verbose = False, show = False, aim_NT = aim_NT[rule_index], aim_G = aim_G[rule_index])
        counter = objective_fun_counter(objective_fun, tau_mini = time_mini, tau_maxi = time_maxi)
        x = objective_fun.tau
        y = counter
        coco = cocos[lot]
        cap = caps[cellulate_index // len(cocos)]
        for clue in clues:
            clue_index = clues.index(clue)
            if clue_index == 0:
                _label = '' # _label = '_'+f'R{acts[lot]}' if exam else ''
                label = pats[lot]+_label
            else:
                label = None
            line_style = '-' if clue == 'NT' else '--'
            if clue != 'U':
                y_mean = np.mean(y[clue], 0).flatten()
                y_sand = np.std(y[clue], 0).flatten()
            else:
                y_mean = np.mean(100-(y['NT']+y['G']), 0).flatten()
                y_sand = np.std(100-(y['NT']+y['G']), 0).flatten()
            axe[row, col].plot(x, y_mean, color = coco, fillstyle = cap, linestyle = line_style, marker = 'None', linewidth = 3 if lot in [2, 3] else 3, label = label, alpha = 0.75 if lot in [2, 3] else 1)
            _x = np.array([_ for _ in x if _ % 4 == 0]).astype('int')
            _y = np.linspace(0, 100, 21).astype('int')
            axe[row, col].set_xticks(ticks = _x, labels = [_ if _ % 8 == 0 else None for _ in _x], fontsize = font_size_bet, rotation = 0)
            axe[row, col].set_yticks(ticks = _y, labels = [f'{_}' if _ % 10 == 0 else None for _ in _y], fontsize = font_size_bet, rotation = 0)
            axe[row, col].set_xlabel('Time [Hour]', fontsize = font_size_bet)
            axe[row, col].set_ylabel('Cell Count', fontsize = font_size_bet)
            axe[row, col].set_xlim(time_mini, time_maxi) # x_limes
            axe[row, col].set_ylim(0-2.5, 100+2.5) # y_limes
            axe[row, col].grid(linestyle = ':', alpha = 0.25)
            _axe.plot(x, y_sand, color = coco, fillstyle = cap, linestyle = line_style, marker = 'None', linewidth = 3 if lot in [2, 3] else 3, label = label, alpha = 0.75 if lot in [2, 3] else 1)
            _axe_x_ticks = _x
            _axe_y_ticks = np.linspace(0, 3, 4).astype('int')
            _axe.set_xticks(ticks = _axe_x_ticks, labels = [], fontsize = font_size_bet)
            _axe.set_yticks(ticks = _axe_y_ticks, labels = _axe_y_ticks, fontsize = font_size_bet)
            _axe.set_xlim(time_mini+24, time_maxi)
            _axe.set_ylim(min(_axe_y_ticks), max(_axe_y_ticks))
            # _axe.set_xlabel('Time [Hour]', fontsize = font_size_bet)
            # _axe.set_ylabel('Cell Count', fontsize = font_size_bet)
            _axe.set_title(label = 'ITWT (σ)', fontsize = font_size_alp)
        axe[row, col].set_title(label = f'ITWT (μ) · Target (PRE = {int(100*aim_G[rule_index])}, EPI = {int(100*aim_NT[rule_index])})', fontsize = font_size_alp)
        axe[row, col].set_title(label = '[A]', loc = 'left', pad = 11, fontsize = font_size_alp, fontweight = 'bold', position = (0, 1))
    labels = ['EPI Target', 'PRE Target']
    axe[row, col].axhline(100*aim_NT[rule_index], time_mini/(time_maxi-time_mini), time_maxi/(time_maxi-time_mini), color = cocks[0], linestyle = '-', linewidth = 2, alpha = 0.25, label = None)
    axe[row, col].axhline(100*aim_G[rule_index], time_mini/(time_maxi-time_mini), time_maxi/(time_maxi-time_mini), color = cocks[1], linestyle = '--', linewidth = 2, alpha = 0.25, label = None)
    where_mu = (2*8.5, 52.5)
    where_mu_sigma = (2*8.5, 47.5)
    axe[row, col].axhline(52.5, 4*3/(time_maxi-time_mini), 8*2/(time_maxi-time_mini), color = 'tab:gray', linestyle = '-', linewidth = 2, alpha = 0.25, label = None)
    axe[row, col].axhline(47.5, 4*3/(time_maxi-time_mini), 8*2/(time_maxi-time_mini), color = 'tab:gray', linestyle = '--', linewidth = 2, alpha = 0.25, label = None)
    axe[row, col].annotate(text = 'EPI', xy = where_mu, xytext = where_mu, color = 'tab:gray', fontsize = font_size_ate, fontweight = 'book', horizontalalignment = 'left', verticalalignment = 'center')
    axe[row, col].annotate(text = 'PRE', xy = where_mu_sigma, xytext = where_mu_sigma, color = 'tab:gray', fontsize = font_size_ate, fontweight = 'book', horizontalalignment = 'left', verticalalignment = 'center')
    axe[row, col].legend(fontsize = font_size_chi, frameon = True, framealpha = 0.25, loc = (0.5725, 0.725), ncols = 2)
plt.show()

#%%# Create Figure! [Rule 1]

from Fig0_Services import retrieve_data_cellulate, objective_fun_counter
from ObjectiveFun import ObjectiveFunPortion # from ObjectiveFunL1L2 import ObjectiveFunPortionRule

rows = 1
cols = 1
fig_size = (cols*fig_size_base, rows*fig_size_base)
fig = plt.figure(figsize = fig_size, layout = "constrained")
axe_mat = fig.subplots(nrows = rows, ncols = cols, squeeze = False)

font_size_alp = 23 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_bet = 19 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_chi = 17 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_ate = 23 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53} # Annotate!

cocos = ['tab:red', 'tab:purple', 'tab:olive', 'tab:green']
cocks = ['tab:gray', 'tab:gray'] # ['tab:blue', 'tab:orange']
caps = ['full', 'none']
clues = ['NT', 'G'] # ['NT', 'G', 'U']
rule = 1

axe = axe_mat
_axe = axe[0, 0].inset_axes([0.5, 0.025, 0.5, 0.1])

for lot in lots:
    sieve = sieve_lit[appraisal_opal_discoveries[lot][0]]
    memo = memos[lot]
    axe = axe_mat
    rule_index = rules.index(rule)
    row = 0
    col = 0
    for cellulate in cellulates:
        cellulate_index = cellulates.index(cellulate)
        reiteration = f'Reiterate_{cellulate[0]}_{cellulate[1]}'
        _trajectory_set = retrieve_data_cellulate(memo, cellulate, reiteration, lot, sieve, verbose = True)[1]
        trajectory_set = _trajectory_set[:, int(_trajectory_set.shape[1]/2):_trajectory_set.shape[1]] if rule == 1 else _trajectory_set[:, 0:int(_trajectory_set.shape[1]/2)] # Temporary fix!
        objective_fun = ObjectiveFunPortion(trajectory_set, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi = 10, verbose = False, show = False, aim_NT = aim_NT[rule_index], aim_G = aim_G[rule_index])
        counter = objective_fun_counter(objective_fun, tau_mini = time_mini, tau_maxi = time_maxi)
        x = objective_fun.tau
        y = counter
        coco = cocos[lot]
        cap = caps[cellulate_index // len(cocos)]
        for clue in clues:
            clue_index = clues.index(clue)
            if clue_index == 0:
                _label = '' # _label = '_'+f'R{acts[lot]}' if exam else ''
                label = pats[lot]+_label
            else:
                label = None
            line_style = '-' if clue == 'NT' else '--'
            if clue != 'U':
                y_mean = np.mean(y[clue], 0).flatten()
                y_sand = np.std(y[clue], 0).flatten()
            else:
                y_mean = np.mean(100-(y['NT']+y['G']), 0).flatten()
                y_sand = np.std(100-(y['NT']+y['G']), 0).flatten()
            axe[row, col].plot(x, y_mean, color = coco, fillstyle = cap, linestyle = line_style, marker = 'None', linewidth = 3 if lot in [2, 3] else 3, label = label, alpha = 0.75 if lot in [2, 3] else 1)
            _x = np.array([_ for _ in x if _ % 4 == 0]).astype('int')
            _y = np.linspace(0, 100, 21).astype('int')
            axe[row, col].set_xticks(ticks = _x, labels = [_ if _ % 8 == 0 else None for _ in _x], fontsize = font_size_bet, rotation = 0)
            axe[row, col].set_yticks(ticks = _y, labels = [f'{_}' if _ % 10 == 0 else None for _ in _y], fontsize = font_size_bet, rotation = 0)
            axe[row, col].set_xlabel('Time [Hour]', fontsize = font_size_bet)
            axe[row, col].set_ylabel('Cell Count', fontsize = font_size_bet)
            axe[row, col].set_xlim(time_mini, time_maxi) # x_limes
            axe[row, col].set_ylim(0-2.5, 100+2.5) # y_limes
            axe[row, col].grid(linestyle = ':', alpha = 0.25)
            _axe.plot(x, y_sand, color = coco, fillstyle = cap, linestyle = line_style, marker = 'None', linewidth = 3 if lot in [2, 3] else 3, label = label, alpha = 0.75 if lot in [2, 3] else 1)
            _axe_x_ticks = _x
            _axe_y_ticks = np.linspace(3, 5, 3).astype('int')
            _axe.set_xticks(ticks = _axe_x_ticks, labels = [], fontsize = font_size_bet)
            _axe.set_yticks(ticks = _axe_y_ticks, labels = _axe_y_ticks, fontsize = font_size_bet)
            _axe.set_xlim(time_mini+24, time_maxi)
            _axe.set_ylim(min(_axe_y_ticks), max(_axe_y_ticks))
            # _axe.set_xlabel('Time [Hour]', fontsize = font_size_bet)
            # _axe.set_ylabel('Cell Count', fontsize = font_size_bet)
            _axe.set_title(label = 'ITWT (σ)', fontsize = font_size_alp)
        axe[row, col].set_title(label = f'ITWT (μ) · Target (PRE = {int(100*aim_G[rule_index])}, EPI = {int(100*aim_NT[rule_index])})', fontsize = font_size_alp)
        axe[row, col].set_title(label = '[B]', loc = 'left', pad = 11, fontsize = font_size_alp, fontweight = 'bold', position = (0, 1))
    labels = ['EPI Target', 'PRE Target']
    axe[row, col].axhline(100*aim_NT[rule_index], time_mini/(time_maxi-time_mini), time_maxi/(time_maxi-time_mini), color = cocks[0], linestyle = '-', linewidth = 2, alpha = 0.25, label = None)
    axe[row, col].axhline(100*aim_G[rule_index], time_mini/(time_maxi-time_mini), time_maxi/(time_maxi-time_mini), color = cocks[1], linestyle = '--', linewidth = 2, alpha = 0.25, label = None)
    where_mu = (8.5, 52.5)
    where_mu_sigma = (8.5, 47.5)
    axe[row, col].axhline(52.5, 4*1/(time_maxi-time_mini), 8*1/(time_maxi-time_mini), color = 'tab:gray', linestyle = '--', linewidth = 2, alpha = 0.25, label = None)
    axe[row, col].axhline(47.5, 4*1/(time_maxi-time_mini), 8*1/(time_maxi-time_mini), color = 'tab:gray', linestyle = '-', linewidth = 2, alpha = 0.25, label = None)
    axe[row, col].annotate(text = 'PRE', xy = where_mu, xytext = where_mu, color = 'tab:gray', fontsize = font_size_ate, fontweight = 'book', horizontalalignment = 'left', verticalalignment = 'center')
    axe[row, col].annotate(text = 'EPI', xy = where_mu_sigma, xytext = where_mu_sigma, color = 'tab:gray', fontsize = font_size_ate, fontweight = 'book', horizontalalignment = 'left', verticalalignment = 'center')
    axe[row, col].legend(fontsize = font_size_chi, frameon = True, framealpha = 0.25, loc = 'upper right', ncols = 2)
plt.show()

#%%# Save Fig!

fig_path = os.path.dirname(os.path.realpath(__file__))
fig_nick = 'Fig13[A][B]'
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
