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
    print(f'Observe {observe} Geom Med')
    gem_q = geometric_median(q, list(para_set_raw.keys()), para_set_true)
    gem_q_lit.append(gem_q)

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

#%%# Comparison (ANN, SA) [Plot!]

para_selection = ['N_N', 'G_G', 'G_N', 'N_G'] # ['N_N', 'G_G', 'FC_N', 'G_EA', 'G_N', 'N_G', 'FC_G', 'N_EA', 'MRNA', 'PRO', 'td_FC', 'td_FM', 'tau_C', 'tau_M', 'tau_ef_EA', 'tau_eb_EA', 'tau_pf_NP', 'tau_pb_NP', 'chi_auto']
para_selection_index = [paras.index(para) for para in para_selection]

bins = 100
rows = 2 # len(paras) // 4 + np.sign(len(paras) // 4)
cols = 2 # 4
# fig_size_base = 10 # {2.5, 3.75, 5, 7.5, 10}
fig_size = (cols*fig_size_base, rows*fig_size_base)
fig = plt.figure(figsize = fig_size, layout = "constrained")
axe_mat = fig.subplots(nrows = rows, ncols = cols, squeeze = False)
axe = axe_mat

sieve_opals = [sieve_lit[appraisal_opal_discoveries[lot][0]] for lot in lots]
_pats = pats # _pats = [pat+'_'+f'R{acts[pats.index(pat)]}' for pat in pats] if exam else pats
cocos = ['tab:red', 'tab:purple', 'tab:olive', 'tab:green']
q_index = 0
a_index = 0
q = q_lit[q_index] # {'ANN [L1]', 'ANN [L2]'}
a = a_lit[a_index] # {'SA [L1]', 'SA [L2]'}

for i in range(0, len(para_selection)):
    row = i // cols
    col = i % cols
    index = para_selection_index[i]
    t = axe[row, col].hist(x = a[:, index], bins = bins, range = para_set_true[paras[index]][1], histtype = 'stepfilled', align = 'mid', color = cocos[a_index+2], density = True, weights = a[:, -1], alpha = 0.25, linewidth = 2)
    p = axe[row, col].hist(x = q[:, index], bins = bins, range = para_set_true[paras[index]][1], histtype = 'stepfilled', align = 'mid', color = cocos[q_index], density = True, weights = None, alpha = 0.25, linewidth = 2)
    axe[row, col].axvline(x = mapes[0][index], ymin = 0, ymax = 1, linestyle = '--', color = 'tab:red', label = _pats[0], linewidth = 2)
    axe[row, col].axvline(x = mapes[1][index], ymin = 0, ymax = 1, linestyle = '--', color = 'tab:purple', label = _pats[1], linewidth = 2, alpha = 0.75)
    axe[row, col].axvline(x = gem_a_lit[0][index], ymin = 0, ymax = 1, linestyle = '--', color = 'tab:olive', label = _pats[2], linewidth = 2, alpha = 0.75)
    axe[row, col].axvline(x = gem_a_lit[1][index], ymin = 0, ymax = 1, linestyle = '--', color = 'tab:green', label = _pats[3], linewidth = 2)
    axe[row, col].set_yticks(ticks = [], labels = [])
    axe[row, col].set_xlim(para_set_true[paras[index]][1][0], para_set_true[paras[index]][1][1])
    axe[row, col].set_title(paras[index])
    axe[row, col].legend(fontsize = 13)
plt.show()

#%%# Compare Parameter Sets! [Separate]

from seaborn import boxplot, violinplot

kinds = ['boxer', 'violinist', None]
kind = kinds[0]
pick = 0 # lots
whisk = (2.5, 97.5) # [0, 100]

_mape_norm_1 = np.array([mapes[0][i]/(para_set_true[paras[i]][1][1]-para_set_true[paras[i]][1][0]) for i in range(0, len(paras))])
_mape_norm_2 = np.array([mapes[1][i]/(para_set_true[paras[i]][1][1]-para_set_true[paras[i]][1][0]) for i in range(0, len(paras))])
_gem_q_norm_1 = np.array([gem_q_lit[0][i]/(para_set_true[paras[i]][1][1]-para_set_true[paras[i]][1][0]) for i in range(0, len(paras))])
_gem_q_norm_2 = np.array([gem_q_lit[1][i]/(para_set_true[paras[i]][1][1]-para_set_true[paras[i]][1][0]) for i in range(0, len(paras))])

para_key_label_set = [ # para_key_label_set
    r'$\mathit{Nanog}$_NANOG', r'$\mathit{Gata6}$_GATA6', r'$\mathit{Fgf4}$_NANOG', r'$\mathit{Gata6}$_A-ERK', r'$\mathit{Gata6}$_NANOG', r'$\mathit{Nanog}$_GATA6', r'$\mathit{Fgf4}$_GATA6', r'$\mathit{Nanog}$_A-ERK',
    'mRNA', 'PROTEIN',
    r'$\tau_{\mathrm{d},\mathrm{FGF4}}$', r'$\tau_{\mathrm{d},\mathrm{M‐FGFR‐FGF4}}$',
    r'$\tau_{\mathrm{escape}}$', r'$\tau_{\mathrm{exchange}}$',
    r'$\tau_{\mathrm{pho},\mathrm{ERK}}$', r'$\tau_{\mathrm{doh},\mathrm{ERK}}$',
    r'$\tau_{\mathrm{pho},\mathrm{NANOG}}$', r'$\tau_{\mathrm{doh},\mathrm{NANOG}}$',
    r'$\chi_{\mathrm{auto}}$'
]
para_selection_lit = [ # ['N_N', 'G_G', 'FC_N', 'G_EA', 'G_N', 'N_G', 'FC_G', 'N_EA', 'MRNA', 'PRO', 'td_FC', 'td_FM', 'tau_C', 'tau_M', 'tau_ef_EA', 'tau_eb_EA', 'tau_pf_NP', 'tau_pb_NP', 'chi_auto']
    ['N_N', 'G_G', 'G_N', 'N_G'],
    ['N_N', 'G_G', 'G_N', 'N_G', 'FC_N', 'G_EA', 'FC_G', 'N_EA'],
    ['tau_ef_EA', 'tau_eb_EA', 'tau_pf_NP', 'tau_pb_NP'],
    ['td_FC', 'td_FM', 'tau_C', 'tau_M', 'MRNA', 'PRO', 'chi_auto'],
]
# para_selection_where_lit = [[paras.index(para) for para in para_selection] for para_selection in para_selection_lit] # para_selection_index

rows = 2 # len(paras) // 4 + np.sign(len(paras) // 4)
cols = 2 # 4
# fig_size_base = 10 # {2.5, 3.75, 5, 7.5, 10}
fig_size = (cols*fig_size_base, rows*fig_size_base)
fig = plt.figure(figsize = fig_size, layout = "constrained")
axe_mat = fig.subplots(nrows = rows, ncols = cols, squeeze = False)
axe = axe_mat

font_size_alp = 29 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_bet = 23 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_chi = 19 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_ate = 17 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53} # Annotate!

cocos = ['tab:red', 'tab:purple', 'darkorange', 'darkturquoise'] # lots
marks = ['', '.', 'o', 'D', 'X']
lisps = ['', '-', '--', ':', '-.']
lites = [1, 2, 3, 5, 7]
almas = [0, 0.25, 0.375, 0.5, 1]
capprops = {'color': 'tab:gray', 'linewidth': lites[1], 'alpha': almas[3]}
capwidths = 0.25
boxprops = {'edgecolor': 'tab:gray', 'linewidth': lites[1], 'alpha': almas[2]}
whiskerprops = {'color': 'tab:gray', 'linewidth': lites[1], 'alpha': almas[3]}
medianprops = {'color': 'tab:gray', 'linewidth': lites[3], 'alpha': almas[3]}
_pats = [f'{pats[0]}_MAPE', f'{pats[1]}_MAPE', f'{pats[0]}_SGM', f'{pats[1]}_SGM']
tits = ['[A]', '[B]', '[C]', '[D]']

y_limes = (0, 1, 0.05)
y_ticks = np.round(np.arange(y_limes[0], y_limes[1]+y_limes[2], y_limes[2]), 2)
y_tick_labels = [f'{int(100*y_tick)}%' if (10*y_tick) % (10*2*y_limes[2]) == 0 else None for y_tick in y_ticks]

for para_selection in para_selection_lit:
    
    index = para_selection_lit.index(para_selection)
    row = index // rows
    col = index % cols
    coco = cocos[pick]
    para_selection_index = [paras.index(para) for para in para_selection]
    para_selection_label = [para_key_label_set[index] for index in para_selection_index]
    mape_norm_1 = _mape_norm_1[para_selection_index]
    mape_norm_2 = _mape_norm_2[para_selection_index]
    gem_q_norm_1 = _gem_q_norm_1[para_selection_index]
    gem_q_norm_2 = _gem_q_norm_2[para_selection_index]
    
    if kind == kinds[0]:
        data = np.copy(q_lit[pick][:, para_selection_index]) if pick in [0, 1] else np.copy(a_lit[pick-2][:, para_selection_index])
        for para in para_selection:
            data[:, para_selection.index(para)] /= (para_set_true[para][1][1]-para_set_true[para][1][0])
        boxplot(data = data, color = coco, saturation = 1, width = 0.5, whis = whisk, showfliers = False, ax = axe[row, col], capprops = capprops, capwidths = capwidths, boxprops = boxprops, whiskerprops = whiskerprops, medianprops = medianprops, zorder = 1)
    elif kind == kinds[1]:
        raise NotImplementedError('Oops!')
    else:
        print('Nothing!')
    
    axe[row, col].plot(mape_norm_1, marker = marks[1], markerfacecolor = 'w', markersize = font_size_bet, linestyle = lisps[1], linewidth = lites[2], color = cocos[0], label = _pats[0], alpha = almas[4])
    axe[row, col].plot(mape_norm_2, marker = marks[1], markerfacecolor = 'w', markersize = font_size_bet, linestyle = lisps[1], linewidth = lites[2], color = cocos[1], label = _pats[1], alpha = almas[4])
    axe[row, col].plot(gem_q_norm_1, marker = marks[1], markerfacecolor = 'w', markersize = font_size_bet, linestyle = lisps[1], linewidth = lites[2], color = cocos[2], label = _pats[2], alpha = almas[4])
    axe[row, col].plot(gem_q_norm_2, marker = marks[1], markerfacecolor = 'w', markersize = font_size_bet, linestyle = lisps[1], linewidth = lites[2], color = cocos[3], label = _pats[3], alpha = almas[4])
    axe[row, col].set_xticks(ticks = range(0, len(para_selection)), labels = para_selection_label, fontsize = font_size_bet-4 if row == 0 else font_size_bet, rotation = 0 if row == 0 and col == 0 else 20)
    axe[row, col].set_yticks(ticks = y_ticks, labels = y_tick_labels, fontsize = font_size_bet) if col == 0 else axe[row, col].set_yticks(ticks = y_ticks, labels = [])
    # axe[row, col].set_xlabel(xlabel = 'Parameter', fontsize = font_size_bet)
    if col == 0: axe[row, col].set_ylabel(ylabel = 'Parameter Value [Prior Range]', fontsize = font_size_bet)
    axe[row, col].set_ylim(0, 1)
    axe[row, col].set_xlim(-0.5, len(para_selection)-0.5)
    axe[row, col].grid(axis = 'y', linestyle = lisps[3], alpha = almas[2])
    if row == 0 and col == 0: axe[row, col].legend(fontsize = font_size_chi+4, loc = 'upper right', ncols = 2)
    axe[row, col].set_title(label = 'ITWT', fontsize = font_size_alp)
    axe[row, col].set_title(label = tits[index], loc = 'left', pad = 11, fontsize = font_size_alp, fontweight = 'bold', position = (0, 1))

fig.suptitle(f'Parameter Value Set (Comparison) - {pats[pick]} [Posterior Range]', fontsize = font_size_alp)
plt.show()

#%%# Save Fig!

fig_path = os.path.dirname(os.path.realpath(__file__))
fig_nick = 'Fig11[B][C][D]'
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
