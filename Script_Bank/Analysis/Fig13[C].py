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
pats = ['AI_L1', 'AI_L2', 'SA_L1', 'SA_L2']

#%%# Retrieve Data [Cellulate /\ Memo]

from Fig0_Services import retrieve_data_cellulate

# Override ObjectiveFun Class! [Sieve]

from ObjectiveFunL1L2 import ObjectiveFunPortionRule # from ObjectiveFun import ObjectiveFunPortion

class ObjectiveFunPortionRuleTemp(ObjectiveFunPortionRule):
    
    def appraise(self, **keywords):
        check = self.data_objective is None
        mess = f'Please, we must synthesize (or execute/apply) the inference procedure!\ndata_objective = {self.data_objective}'
        assert not check, mess
        perks = keywords.get('perks', [25, 50, 75])
        percentiles = np.percentile(a = self.data_objective, q = perks, axis = 0, interpolation = 'nearest').reshape((len(perks), -1)).T
        alp = percentiles[:, 1]
        bet = percentiles[:, 2]-percentiles[:, 0]
        chi = alp*(1-bet)
        appraisal = np.round(np.mean(chi), 4)
        ret = (percentiles, appraisal)
        return ret

#%%# Create Figure! [Data | Full Sieve]

sieve_lit = [0, 25, 50, 75, 80, 85, 90, 95, 96, 97, 98, 99, 100] # [0, 25, 50, 75, 80, 85, 90, 95, 96, 97, 98, 99, 100]
lots = [0, 1, 2, 3]
sieve_indices = range(0, len(sieve_lit)) # range(min(sieve_lit), max(sieve_lit)+1)
sieve_rag_shape = (len(lots), 4, len(sieve_indices))
sieve_rag = np.full(shape = sieve_rag_shape, fill_value = np.nan)
rules = [0, 1]
aim_NT = [1, 2/5]
aim_G = [1-_ for _ in aim_NT]
cellulates = [(10, 10)]
cells = [cellulate[0]*cellulate[1] for cellulate in cellulates]
species = ['N', 'G', 'NP']
L = 1

for sieve in sieve_lit:

    sieve_index = sieve_lit.index(sieve) # sieve_index = sieve # Convenience!
    
    for lot in lots:
        
        memo = memos[lot]
        
        for cellulate in cellulates:
            cellulate_index = cellulates.index(cellulate)
            cellulate_cells = cells[cellulate_index]
            reiteration = f'Reiterate_{cellulate[0]}_{cellulate[1]}'
            trajectory_set = retrieve_data_cellulate(memo, cellulate, reiteration, lot, sieve, verbose = True)[1]
            objective_fun = ObjectiveFunPortionRuleTemp(trajectory_set, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi = 10, verbose = False, show = False, rules = rules, aim_NT = aim_NT, aim_G = aim_G, L = L)
            tau_mini = 32
            tau_maxi = 48
            objective_fun.apply(tau_mini = tau_mini, tau_maxi = tau_maxi)
            percentiles, appraisal = objective_fun.appraise()
            alp = np.mean(percentiles[:, 1])
            _bet = np.mean(percentiles[:, [0, 2]], 0)
            bet = np.abs(alp-_bet)
        
        sieve_rag[lot, 0, sieve_index] = alp
        sieve_rag[lot, [1, 2], sieve_index] = bet
        sieve_rag[lot, 3, sieve_index] = appraisal

#%%# Create Figure! [Plot | Full Sieve]

rows = 2
cols = 1
fig_size = (cols*fig_size_base, (rows-1)*fig_size_base)
fig = plt.figure(figsize = fig_size, layout = "constrained")
axe_mat = fig.subplots(nrows = rows, ncols = cols, squeeze = False, width_ratios = [1], height_ratios = [1, 1])
axe = axe_mat # Convenience!

font_size_alp = 23 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_bet = 19 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_chi = 17 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_ate = 23 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53} # Annotate!

cocos = ['tab:red', 'tab:purple', 'tab:olive', 'tab:green']
lites = [3, 3, 3, 3]
lisps = ['-', '-', '-', '-']
marks = ['', '', '.', '.']
mabes = ['', '', 'X', 'X']
ergos = [0, 0, -0.125, 0.125]
capes = [0, 0, 5, 5]
fores = ['', '', '.', '.']
almas = [0.75, 0.75, 1, 1]

_y_limes = [(0.5, 1, 0.05), (0.5, 1, 0.05)]
y_limes = _y_limes[L-1]

exam = np.invert(np.all(np.equal(acts, acts[0])))

for lot in lots:
    
    coco = cocos[lot]
    lite = lites[lot]
    lisp = lisps[lot]
    mark = marks[lot]
    mabe = mabes[lot]
    ergo = ergos[lot]
    cape = capes[lot]
    fore = fores[lot]
    alma = almas[lot]
    
    x_ticks = sieve_indices
    x_tick_labels = [int(sieve/100) if np.round(sieve/100) == sieve/100 else sieve/100 for sieve in sieve_lit] # x_tick_labels = [f'{sieve}%' for sieve in sieve_lit]
    y_ticks = np.round(np.linspace(0.5, 1, 11), 2)
    _y_tick_labels = [int(y_tick) if np.round(y_tick) == y_tick else y_tick for y_tick in y_ticks]
    y_tick_labels = [y_tick_label if _y_tick_labels.index(y_tick_label) % 2 == 0 else None for y_tick_label in _y_tick_labels]
    
    appraisal = sieve_rag[lot, 3, :]
    appraisal_opal_sieve = np.argmax(appraisal)
    appraisal_opal = np.max(appraisal)
    
    _pat = '' # _pat = '_'+f'R{acts[lot]}' if exam else ''
    pat = pats[lot]+_pat
    
    row = 0
    col = 0
    alp = sieve_rag[lot, 0, :]
    bet = sieve_rag[lot, [1, 2], :]
    if lot in [2, 3]:
        x = [sieve_index+ergo for sieve_index in sieve_indices]
        axe[row, col].errorbar(x = x, y = alp, yerr = bet, fmt = fore, linewidth = lite-1, color = coco, markerfacecolor = 'w', markersize = font_size_bet, capsize = cape, capthick = lite)
    else:
        y = [alp[0]-bet[0, 0], alp[0], alp[0]+bet[1, 0]]
        axe[row, col].hlines(y = y, xmin = np.min(x_ticks), xmax = np.max(x_ticks), colors = coco, linestyles = ['--', '-', '--'], linewidths = [lite-1, lite, lite-1], alpha = alma)
    axe[row, col].plot(appraisal_opal_sieve+ergo, alp[appraisal_opal_sieve], color = coco, marker = mabe, linestyle = '', linewidth = lite, markersize = font_size_chi, alpha = alma, zorder = 3)
    axe[row, col].set_xticks(ticks = x_ticks, labels = [], fontsize = font_size_bet)
    axe[row, col].set_yticks(ticks = y_ticks, labels = y_tick_labels, fontsize = font_size_bet)
    # axe[row, col].set_xlabel(xlabel = 'Meta-Score >= X', fontsize = font_size_bet)
    axe[row, col].set_ylabel(ylabel = 'Time Ensemble Average Score\n(Error Bar)', fontsize = font_size_bet)
    axe[row, col].set_xlim(np.min(x_ticks)-1, 1+np.max(x_ticks))
    axe[row, col].set_ylim(np.min(y_ticks), np.max(y_ticks))
    axe[row, col].grid(axis = 'x', linestyle = ':', color = 'tab:gray', alpha = 0.25)
    axe[row, col].set_title(label = f'ITWT · Time Interval [{tau_mini}, {tau_maxi}] Hours · L{L} Combination', fontsize = font_size_alp)
    axe[row, col].set_title(label = '[C]', loc = 'left', pad = 11, fontsize = font_size_alp, fontweight = 'bold', position = (-0.1, 1))
    
    row = 1
    col = 0
    axe[row, col].plot(sieve_indices, appraisal, color = coco, markerfacecolor = 'w', marker = mark, linestyle = lisp, linewidth = lite, markersize = font_size_bet, alpha = alma, label = pat)
    axe[row, col].plot(appraisal_opal_sieve, appraisal_opal, color = coco, marker = mabe, linestyle = '', linewidth = lite, markersize = font_size_chi, alpha = alma, zorder = 3)
    axe[row, col].set_xticks(ticks = x_ticks, labels = x_tick_labels, fontsize = font_size_bet)
    axe[row, col].set_yticks(ticks = y_ticks, labels = y_tick_labels, fontsize = font_size_bet)
    axe[row, col].set_xlabel(xlabel = r'Meta Score $\geq$ Threshold', fontsize = font_size_bet)
    axe[row, col].set_ylabel(ylabel = 'Optimality Score', fontsize = font_size_bet)
    axe[row, col].set_xlim(np.min(x_ticks)-1, 1+np.max(x_ticks))
    axe[row, col].set_ylim(np.min(y_ticks), np.max(y_ticks))
    axe[row, col].grid(axis = 'x', linestyle = ':', color = 'tab:gray', alpha = 0.25)
    axe[row, col].legend(fontsize = font_size_chi, loc = 'upper right', ncols = 2)

# fig.suptitle(t = f'ITWT · Time Interval [{tau_mini}, {tau_maxi}] Hours · L{L} Combination', fontsize = font_size_alp)
plt.show()

#%%# Save Fig!

fig_path = os.path.dirname(os.path.realpath(__file__))
fig_nick = 'Fig13[C]'
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
