######################################
######## Optimize Exploration ########
######################################

#%%# Catalyzer

sup_comp = False # Super Computer?
cellulate = (0, 0) # (cell_layers, layer_cells)
sieve = 0 # {0.75, 0.8, 0.85, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 1}
reiterate = f'Reiterate_{cellulate[0]}_{cellulate[1]}_Sieve_{int(100*sieve)}'
data_path = 'Optimize_Exploration_1' # 'Shallow_Grid_1_N_Link'
acts = list(range(0, 8)) # [0, 1, ...]
act = max(acts) # Default!
_observe = 1 # {0, 1, ...} # L ~ {1, 2}
observe = _observe if act != 0 else None
curb = 'Mid' # {'Weak', 'Mid', 'Strong'}
restrict = {
    'Weak': {'G_EA': (750, 1500), 'N_EA': (750, 1500)},
    'Mid': {'G_EA': (0, 1000), 'N_EA': (0, 1000)},
    'Strong': {'G_EA': (0, 750), 'N_EA': (0, 750)}
}

import os
import sys
if sup_comp:
    path = None # Careful! The user must provide the path to its own HPC directory!
else:
    path = os.path.dirname(os.path.realpath(__file__)) + '/../Resources'
sys.path.append(path)
from BiochemStem import BiochemStem
from BiochemSimulUltimate import BiochemSimulSwift
import numpy as np
import torch
import time
import pickle
if not sup_comp:
    import matplotlib.pyplot as plt
    plt.rcParams['figure.dpi'] = 250
    plt.rcParams['savefig.dpi'] = 250

#%%# Biochemical System Construction [Preparation]

def construct_stem(parameter_set, parameter_set_true, para_fun, sup_comp = False, verbose = False):
    
    if sup_comp: verbose = False # Enforce!
    
    # Transcription Regulation [Components AND Interactions]
    
    regulation_transcription = { # ([0|1] = [Up|Down]-Regulate, Positive Integer = Transcription Cooperativity)
        'N': {'N': (0, 4), 'G': (1, 4), 'FC': (0, 2)},
        'G': {'N': (1, 4), 'G': (0, 4), 'FC': (1, 2)},
        'EA': {'N': (1, 3), 'G': (0, 3)}
    }
    
    # Species [Specs] # Transmembrane
    
    _species_promoter_state = ['I', 'A'] # {'I': 'Inactive', 'A': 'Active'}
    _species_transcription = ['N', 'G', 'FC']
    _species_translation = _species_transcription.copy()
    _species_translation.extend(['EI'])
    
    # Species [Descript]
    
    _species = {
        'promoter_state': [S + '_' + _ for S in _species_translation for _ in _species_promoter_state], # Promoter State Dynamics
        'transcription': [S + '_MRNA' for S in _species_transcription], # Explicit Transcription
        'translation': _species_translation, # Explicit Translation
        'exportation': ['FM'],
        'jump_diffuse': ['FM'],
        'dimerization': ['FD'],
        'enzymatic': ['EA'],
        'phosphorylation' : ['NP']
    }
    
    # Rate Constants
    
    diffusion_coefficients = {'N': 10e-12, 'G': 10e-12, 'NP': 10e-12, 'FC': 10e-12, 'FM': 10e-12, 'EA': 10e-12} # pow(microns, 2)/seconds # pow(length, 2)/time # Protein diffusion constant
    protein_cross_sections = {'N': 10e-9, 'G': 10e-9, 'NP': 10e-9, 'FC': 10e-9, 'FM': 10e-9, 'EA': 10e-9} # Nanometers
    binding_sites = list(regulation_transcription.keys()) # Transcription Factors!
    
    _cell_radius = 10 # Micrometers
    cell_radius = 1000e-9*_cell_radius
    cell_volume = 4*np.pi*pow(cell_radius, 3)/3 # pow(meters, 3)
    
    half_activation_thresholds = {'N_N': para_fun('N_N'), 'G_G': para_fun('G_G'), 'FC_N': para_fun('FC_N'), 'G_EA': para_fun('G_EA')} # {P}_{B} = {Promoter}_{Binding_Site}
    half_repression_thresholds = {'G_N': para_fun('G_N'), 'N_G': para_fun('N_G'), 'FC_G': para_fun('FC_G'), 'N_EA': para_fun('N_EA')} # {P}_{B} = {Promoter}_{Binding_Site}
    _rates_promoter_binding = {S: 4*np.pi*protein_cross_sections[S]*diffusion_coefficients[S]/cell_volume for S in binding_sites}
    tunes = {'N_N': 10.5, 'G_N': 10.5, 'G_G': 10.5, 'N_G': 10.5, 'G_EA': 4.375, 'N_EA': 4.375, 'FC_N': 1.775, 'FC_G': 1.775}
    _rates_promoter_unbinding = {P+'_'+B: tunes[P+'_'+B]*half_activation_thresholds[P+'_'+B]*_rates_promoter_binding[B] if regulation_transcription[B][P][0] == 0 else tunes[P+'_'+B]*half_repression_thresholds[P+'_'+B]*_rates_promoter_binding[B] for B in binding_sites for P in list(regulation_transcription[B].keys())}
    
    _MRNA_lifetime = 4 # {1, ..., 8} # Hours
    MRNA_lifetime = _MRNA_lifetime*pow(60, 2) # Seconds
    MRNA_copy_number = 250
    synthesis_spontaneous = 0.2 # [0, 1] # 1 - synthesis
    synthesis = 1 - synthesis_spontaneous
    _rates_MRNA_synthesis_spontaneous = synthesis_spontaneous*MRNA_copy_number/MRNA_lifetime
    _rates_MRNA_synthesis = synthesis*MRNA_copy_number/MRNA_lifetime
    _rates_MRNA_degradation = 1/MRNA_lifetime
    
    _protein_lifetimes = {'N': 2, 'G': 2, 'NP': 1, 'FC': 2, 'FM': 2, 'FD': 100*2, 'EI': 48} # {1, ..., 8} # Hours
    protein_lifetime = {S: _protein_lifetimes.get(S)*pow(60, 2) for S in list(_protein_lifetimes.keys())} # Seconds
    protein_copy_numbers = {'N': 4, 'G': 4, 'FC': 4, 'EI': 1000} # [100|1000] Proteins # 'P' Proteins Per MRNA
    _rates_protein_synthesis = {S: protein_copy_numbers[S]/protein_lifetime[S] for S in list(protein_copy_numbers.keys())}
    _rates_protein_degradation = {S: 1/protein_lifetime[S] for S in list(protein_lifetime.keys())}
    
    # Reactions [template = {'exes': [], 'props': [], 'deltas': [{}], 'rates': {}, 'initial_state': {}}]
    
    promoter_binding = { # {P = Promoter}_{B = Binding Site}_{C = Cooperativity}
        'exes': [f'{B} + {P}_{B}_{C} -> {P}_{B}_{C+1}' for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1])],
        'props': [f'{B} * {P}_{B}_{C} * kb_{P}_{B}_{C+1}' for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1])],
        'deltas': [{B: -1, f'{P}_{B}_{C}': -1, f'{P}_{B}_{C+1}': 1, P+'_I' if regulation_transcription[B][P][0] == 1 else P+'_A': 1 if C+1 == regulation_transcription[B][P][1] else 0} for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1])],
        'rates': {f'kb_{P}_{B}_{C+1}': _rates_promoter_binding[B] for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1])},
        'initial_state': {f'{P}_{B}_{C}': 0 if C != 0 else 1 for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]+1)}
    }
    
    promoter_binding_pho = {
        'exes': [f'{B}P + {P}_{B}_{C} -> {P}_{B}_{C+1}' for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]) if B == 'N'],
        'props': [f'{B}P * {P}_{B}_{C} * kb_{P}_{B}_{C+1}' for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]) if B == 'N'],
        'deltas': [{f'{B}P': -1, f'{P}_{B}_{C}': -1, f'{P}_{B}_{C+1}': 1, P+'_I' if regulation_transcription[B][P][0] == 1 else P+'_A': 1 if C+1 == regulation_transcription[B][P][1] else 0, f'{P}_{B}P_{C+1}': 1} for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]) if B == 'N'],
        'rates': {f'kb_{P}_{B}_{C+1}': 1*_rates_promoter_binding[B] for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]) if B == 'N'},
        'initial_state': {f'{P}_{B}P_{C+1}': 0 for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]) if B == 'N'}
    }
    
    pub = 5 # Promoter Unbinding [Coefficient]
    promoter_unbinding = { # Careful! Only auto-activation valid: zero unbinding rate when it occupies C sites!
        'exes': [f'{P}_{B}_{C+1} -> {P}_{B}_{C} + {B}' for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]-1, -1, -1)],
        'props': [f'ku_{P}_{B}_{C} * {P}_{B}_{C+1} * (1 - {P}_{B}P_{C+1})' if B == 'N' else f'ku_{P}_{B}_{C} * {P}_{B}_{C+1}' for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]-1, -1, -1)],
        'deltas': [{B: 1, f'{P}_{B}_{C}': 1, f'{P}_{B}_{C+1}': -1, P+'_I' if regulation_transcription[B][P][0] == 1 else P+'_A': -1 if C+1 == regulation_transcription[B][P][1] else 0} for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]-1, -1, -1)],
        'rates': {f'ku_{P}_{B}_{C}': _rates_promoter_unbinding[P+'_'+B]/pow(pub, C) for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]-1, -1, -1)},
        # 'rates': {f'ku_{P}_{B}_{C}': 0 if P == B and C+1 == 4 else _rates_promoter_unbinding[P+'_'+B]/pow(pub, C) for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]-1, -1, -1)},
        'initial_state': {f'{P}_{B}_{C}': 0 if C != 0 else 1 for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1], -1, -1)}
    }
    
    pub = 5 # Promoter Unbinding [Coefficient]
    promoter_unbinding_pho = { # Careful! Only auto-activation valid: zero unbinding rate when it occupies C sites!
        'exes': [f'{P}_{B}_{C+1} -> {P}_{B}_{C} + {B}P' for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]-1, -1, -1) if B == 'N'],
        'props': [f'ku_{P}_{B}_{C} * {P}_{B}_{C+1} * {P}_{B}P_{C+1}' if B == 'N' else f'ku_{P}_{B}_{C} * {P}_{B}_{C+1}' for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]-1, -1, -1) if B == 'N'],
        'deltas': [{f'{B}P': 1, f'{P}_{B}_{C}': 1, f'{P}_{B}_{C+1}': -1, P+'_I' if regulation_transcription[B][P][0] == 1 else P+'_A': -1 if C+1 == regulation_transcription[B][P][1] else 0, f'{P}_{B}P_{C+1}': -1} for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]-1, -1, -1) if B == 'N'],
        'rates': {f'ku_{P}_{B}_{C}': _rates_promoter_unbinding[P+'_'+B]/pow(pub, C) for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]-1, -1, -1) if B == 'N'},
        # 'rates': {f'ku_{P}_{B}_{C}': 0 if P == B and C+1 == 4 else _rates_promoter_unbinding[P+'_'+B]/pow(pub, C) for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1]-1, -1, -1) if B == 'N'},
        'initial_state': {f'{P}_{B}P_{C}': 0 for B in binding_sites for P in list(regulation_transcription[B].keys()) for C in range(regulation_transcription[B][P][1], 0, -1) if B == 'N'}
    }
    
    MRNA_synthesis_spontaneous = {
        'exes': [f'0 -> {S}' for S in _species['transcription'] if S != 'FC_MRNA'],
        'props': [f'(1 - np.sign({S}_I)) * kss_{S}_MRNA' for S in _species_transcription if S != 'FC'],
        'deltas': [{S: 1} for S in _species['transcription'] if S != 'FC_MRNA'],
        'rates': {f'kss_{S}': _rates_MRNA_synthesis_spontaneous for S in _species['transcription'] if S != 'FC_MRNA'},
        'initial_state': {S: 0 for S in _species['transcription']}
    }
    
    MRNA_synthesis = {
        'exes': [f'{S}_A -> {S}_MRNA' for S in _species_transcription],
        'props': [f'np.sign({S}_A) * (1 - np.sign({S}_I)) * ks_{S}_MRNA' for S in _species_transcription],
        'deltas': [{S: 1} for S in _species['transcription']],
        'rates': {f'ks_{S}': _rates_MRNA_synthesis for S in _species['transcription']},
        'initial_state': {f'{S}_{_}': 0 for S in _species_transcription for _ in _species_promoter_state}
    }
    
    MRNA_degradation = {
        'exes': [f'{S} -> 0' for S in _species['transcription']],
        'props': [f'{S} * kd_{S}' for S in _species['transcription']],
        'deltas': [{S: -1} for S in _species['transcription']],
        'rates': {f'kd_{S}': _rates_MRNA_degradation for S in _species['transcription']},
        'initial_state': {S: 0 for S in _species['transcription']}
    }
    
    protein_synthesis = {
        'exes': [f'{S}_MRNA -> {S}' if S in _species_transcription else f'{S}_A -> {S}' for S in _species['translation']],
        'props': [f'{S}_MRNA * ks_{S}' if S in _species_transcription else f'{S}_A * (1 - {S}_I) * ks_{S}' for S in _species['translation']],
        'deltas': [{S: 1} for S in _species['translation']],
        'rates': {f'ks_{S}': _rates_protein_synthesis[S] for S in _species['translation']},
        'initial_state': {f'{S}_{_}': 0 if S in _species_transcription or _ == 'I' else 1 for S in _species['translation'] for _ in _species_promoter_state}
    }
    
    protein_degradation = {
        'exes': [f'{S} -> 0' for S in _species['translation']],
        'props': [f'{S} * kd_{S}' for S in _species['translation']],
        'deltas': [{S: -1} for S in _species['translation']],
        'rates': {f'kd_{S}': _rates_protein_degradation[S] for S in _species['translation']},
        'initial_state': {S: 0 for S in _species['translation']}
    }
    
    # Autocrine Signaling := {ke_F_CM} # Paracrine Signaling := {kjd_F_CM} # IN ~ Intrinsic (Start) # EX ~ Extrinsic (Final)
    
    ksig_C = 0.75*10*_rates_promoter_binding['EA']
    ksig_M = 0.75*10*_rates_promoter_binding['EA']
    
    ke_F_CM = ksig_C # (Cytoplasm)(Membrane) # Auto
    kjd_F_CM = ksig_C # (Cytoplasm)(Membrane) # Para
    # ksig_C = ke_F_CM + kjd_F_CM
    kjd_F_MM = ksig_M # (Membrane)(Membrane)
    # ksig_M = kjd_F_MM
    
    exportation = {
        'exes': ['FC -> FM'],
        'props': ['FC * ke_F_CM'],
        'deltas': [{'FC': -1, 'FM': 1}],
        'rates': {'ke_F_CM': ke_F_CM}, # + Fast != + Short # (1|10)*Promoter_Binding
        'initial_state': {'FC': 0, 'FM': 0}
    }
    
    jump_diffuse = { # Two-Step Process? # V ~ Void
        'exes': ['FC_IN -> FM_EX', 'FM_IN -> FM_EX'],
        'props': ['FC * kjd_F_CM', 'FM * kjd_F_MM'],
        'deltas': [{'FC': -1}, {'FM': -1}],
        'jump_diffuse_deltas': [{'FM': 1}, {'FM': 1}],
        'rates': {'kjd_F_CM': kjd_F_CM, 'kjd_F_MM': kjd_F_MM}, # + Fast != + Short # (1|10)*Promoter_Binding
        'initial_state': {'FC': 0, 'FM': 0}
    }
    
    dimerization = { # FD has no direct degradation, but we put it at the end!
        'exes': ['FM + FM -> FD', 'FD -> FM + FM'],
        'props': ['FM * (FM - 1) * kdf_FD', 'FD * kdb_FD'],
        'deltas': [{'FM': -2, 'FD': 1}, {'FM': 2, 'FD': -1}],
        'rates': {'kdf_FD': 1*0.5*2*_rates_promoter_binding['EA']/30, 'kdb_FD': 2*10/(2*pow(60, 2))},
        'initial_state': {'FD': 0}
    }
    
    enzymatic = { # Reverse # [Forward | Backward]
        'exes': ['FD + EI -> FD + EA', 'EA -> EI'],
        'props': ['FD * EI * kef_EA', 'EA * keb_EA'],
        'deltas': [{'EI': -1, 'EA': 1}, {'EI': 1, 'EA': -1}],
        'rates': {'kef_EA': 1/para_fun('tau_ef_EA'), 'keb_EA': 1/para_fun('tau_eb_EA')}, # 2*Promoter_Binding/10
        'initial_state': {'EI': 0, 'EA': 0}
    }
    
    phosphorylation = {
        'exes': ['EA + N -> EA + NP', 'NP -> N'],
        'props': ['EA * N * kpf_NP', 'NP * kpb_NP'],
        'deltas': [{'N': -1, 'NP': 1}, {'N': 1, 'NP': -1}],
        'rates': {'kpf_NP': 1/para_fun('tau_pf_NP'), 'kpb_NP': 1/para_fun('tau_pb_NP')}, # (1|10)*Promoter_Binding
        'initial_state': {'NP': 0}
    }
    
    _species_degradation = ['NP', 'FM', 'FD', 'EA']
    degradation = {
        'exes': [f'{S} -> 0' for S in _species_degradation],
        'props': [f'{S} * kd_{S}' for S in _species_degradation],
        'deltas': [{S: -1} for S in _species_degradation],
        'rates': {f'kd_{S}': _rates_protein_degradation['EI'] if S == 'EA' else _rates_protein_degradation[S] for S in _species_degradation},
        'initial_state': {S: 0 if S in {'EA'} else 0 for S in _species_degradation}
    }
    
    # Assemble!
    
    flags = ['promoter_binding', 'promoter_binding_pho', 'promoter_unbinding', 'promoter_unbinding_pho', 'MRNA_synthesis_spontaneous', 'MRNA_synthesis', 'MRNA_degradation', 'protein_synthesis', 'protein_degradation', 'exportation', 'jump_diffuse', 'dimerization', 'enzymatic', 'phosphorylation', 'degradation']
    initial_state = {}
    rates = {}
    for flag in flags:
        exec(f"initial_state.update({flag}['initial_state'])")
        exec(f"rates.update({flag}['rates'])")
    
    stem = BiochemStem(initial_state, rates)
    
    # flags = ['promoter_binding', 'promoter_binding_pho', 'promoter_unbinding', 'promoter_unbinding_pho', 'MRNA_synthesis_spontaneous', 'MRNA_synthesis', 'MRNA_degradation', 'protein_synthesis', 'protein_degradation', 'exportation', 'jump_diffuse', 'dimerization', 'enzymatic', 'phosphorylation', 'degradation']
    for flag in flags:
        if verbose: print(flag)
        indices = eval(f"range(len({flag}['exes']))")
        for index in indices:
            if verbose: print(index)
            name = eval(f"{flag}['exes'][{index}]")
            prop_fun = eval(f"{flag}['props'][{index}]")
            delta = eval(f"{flag}['deltas'][{index}]")
            if flag == 'jump_diffuse':
                jump_diffuse_delta = eval(f"{flag}['jump_diffuse_deltas'][{index}]") # jump_diffuse_delta = None
                # print("\n@@@@@@@@\n\tCareful! No 'jump_diffuse_delta' available!\n@@@@@@@@\n")
            else:
                jump_diffuse_delta = None
            stem.add_reaction(name, prop_fun, delta, verbose = False, jump_diffuse = jump_diffuse_delta)
    
    stem.assemble()
    
    return stem

#%%# Simulation /\ Inference Procedure! [Preparation]

from Utilities import instate_state_tor, instate_rate_mat
from Utilities import interpolator
from Utilities import plotful
from Cell_Space import cell_placement, cell_distance, cell_neighborhood
from Cell_Space import make_initial_pattern, make_rho_mat
from Utilities import make_paras, make_para_fun
from Utilities import make_simulator_ready

_path = '' if act == 0 else f'Observe_{observe}/'
if sup_comp:
    path = None + _path # Careful! The user must provide the path to its own HPC directory!
else:
    path = os.path.dirname(os.path.realpath(__file__)) + '/../../Data_Bank/Shallow_Grid/Shallow_Grid_1_N_Link/' + _path
post = '_Posterior.pkl'

_tag = f'Act_{act}_Observe_{observe}_{curb.capitalize()}'
with open(path + _tag + post, 'rb') as portfolio:
    posterior = pickle.load(portfolio)

def synopsis_data_load(task_pins_info, path, tag):
    import os
    import re
    if task_pins_info is None:
        _arcs = os.listdir(path)
        arcs_discovery = [arc for arc in _arcs if re.findall(tag+'_Synopsis_', arc)]
        _arcs_discovery_identify = [re.findall('(\d+)_(\d+)\.pt', arc)[0] for arc in arcs_discovery]
        arcs_discovery_identify = [(int(ident[0]), int(ident[1])) for ident in _arcs_discovery_identify]
        arcs_discovery_identify.sort()
        task_pins = list(set([ident[0] for ident in arcs_discovery_identify]))
        task_pins.sort()
        tasks_info = {task_pin: [ident[1] for ident in arcs_discovery_identify if task_pin == ident[0]] for task_pin in task_pins}
        mess = 'Oops! Something went wrong!'
        check = task_pins == list(tasks_info.keys())
        assert check, mess
        task_pins_mini = min(task_pins)
        task_pins_maxi = max(task_pins)
    elif type(task_pins) is tuple:
        task_pins_mini = task_pins_info[0]
        task_pins_maxi = task_pins_info[1]
        task_pins = range(task_pins_mini, task_pins_maxi+1)
    else:
        mess = "The format is invalid! The variable 'task_pins_info' must be either 'None' or 'tuple'!"
        raise RuntimeError(mess)
    print(f"Load Synopsis Data! '{tag}' Task Pins! {task_pins_mini} : {task_pins_maxi} Total {len(task_pins)}")
    for task_pin in task_pins:
        tasks = tasks_info[task_pin]
        if task_pin == task_pins_mini:
            synopses = list()
        for task in tasks:
            label = path+tag+'_Synopsis_'+str(task_pin)+'_'+str(task)+'.pt'
            _synopses = torch.load(label)
            synopses.append(_synopses)
            print(f'Task Pin! {task_pin} Tasks! {task+1} ~ {len(tasks)} Index {task_pin} {task}')
    print(f"Load Synopsis Data! '{tag}' Task Pins! {task_pins_mini} : {task_pins_maxi} Total {len(task_pins)}")
    collect = (synopses, tasks_info)
    return collect

def geometric_median(mate_zero, paras, para_set_true, capacity = 50e3):
    from scipy.spatial.distance import pdist, cdist, squareform
    mate = np.full(shape = mate_zero.shape, fill_value = np.nan)
    for i in range(0, len(paras)):
        mate[:, i] = mate_zero[:, i]/(para_set_true[paras[i]][1][1]-para_set_true[paras[i]][1][0])
    if mate.shape[0] < capacity:
        _mate_dist = pdist(mate)
        mate_dist = squareform(_mate_dist)
        mate_suma = np.sum(mate_dist, 0)
        print('Matrix!')
    else:
        mate_suma = np.zeros(shape = tuple([mate.shape[0]]))
        for j in range(mate.shape[0]):
            mate_cure = mate[j, :].reshape((1, -1))
            mate_dist = cdist(mate_cure, mate)
            mate_suma[j] = np.sum(mate_dist)
        print('Loop!')
    _mate_para = np.isclose(a = mate_suma, b = np.min(mate_suma), atol = 1e-13)
    mate_para = mate_zero[_mate_para]
    if len(mate_para) == 0:
        mess = 'Oops! Something went wrong!'
        raise RuntimeError(mess)
    elif len(mate_para) > 1:
        print('Oops!\n\tWe must choose one parameter set!\n\tWe take the parameter set at position ZERO!')
    ret = mate_para[0, :]
    return ret

if sup_comp:
    path = None # Careful! The user must provide the path to its own HPC directory!
else:
    path = os.path.dirname(os.path.realpath(__file__)) + '/../../Data_Bank/Optimize_Exploration/Optimize_Exploration_1/'

para_set_raw = {
    'N_N': (100, (0, 1000)), 'G_G': (200, (0, 1000)), 'FC_N': (500, (0, 1000)), 'G_EA': (750, restrict[curb]['G_EA']),
    'G_N': (50, (0, 1000)), 'N_G': (400, (0, 1000)), 'FC_G': (500, (0, 1000)), 'N_EA': (750, restrict[curb]['N_EA']),
    'MRNA': (50, (0, 250)), 'PRO': (200, (0, 1000)),
    'td_FC': (7200, (300, 28800)), 'td_FM': (7200, (300, 28800)),
    'tau_C': (450, (30, 5*900)), 'tau_M': (450, (30, 2*2100)),
    'tau_ef_EA': (17100, (10*30, 43200)), 'tau_eb_EA': (300, (30, 43200)), 'tau_pf_NP': (1710, (10*30, 43200)), 'tau_pb_NP': (171, (30, 43200)),
    'chi_auto': (0.5, (0, 1))
}
para_set_mode = 0 # {0, 1} # {'No Remap', 'Remap: [A, B] ---->>>> [0, 1]'}
para_set, para_set_true = make_paras(para_set_raw = para_set_raw, para_set_mode = para_set_mode, verbose = False)

#%%# Simulation /\ Inference Procedure! [Extract]

paras = list(para_set_raw.keys())
act_range = (min(acts), max(acts)+1)
# sieve = 1 # [0, 1]

for act in range(act_range[0], act_range[1]):
    tag = f'Act_{act}_Observe_{observe}_{curb.capitalize()}'
    task_pins_info = None
    synopses, tasks_info = synopsis_data_load(task_pins_info, path, tag)
    for index in range(0, len(synopses)):
        _synopsis = synopses[index][0, :, :]
        if index == 0:
            synopsis = _synopsis
        else:
            synopsis = np.concatenate((synopsis, _synopsis))
    _sade = synopsis
    if act == np.min(act_range):
        sade_temp = _sade
    else:
        sade_temp = np.concatenate((sade_temp, _sade))
sade = sade_temp[sade_temp[:, -1] >= sieve, :]
print(f'\nObserve {observe} {sade_temp.shape} Sieve {sieve} {sade.shape} Acts {acts}\n')

_germ_sade = geometric_median(sade[:, 0:-1], paras, para_set_true)
germ_sade = [int(round(_germ_sade[index], 0)) if _germ_sade[index] > 1 else round(_germ_sade[index], 2) for index in range(_germ_sade.size)]
_mape = posterior.map().numpy()
mape = [int(round(_mape[index], 0)) if _mape[index] > 1 else round(_mape[index], 2) for index in range(_mape.size)]

print(f'Map!\n{_tag}\n\t{mape}\nGeometric Median!\n{tag}\nSieve {sieve}\n\t{germ_sade}')
span = 5e-7
doze = [(germ_sade_para, (round(germ_sade_para*(1-span), 7), round(germ_sade_para*(1+span), 7))) for germ_sade_para in germ_sade]

para_set_raw = {
    'N_N': doze[0], 'G_G': doze[1], 'FC_N': doze[2], 'G_EA': doze[3],
    'G_N': doze[4], 'N_G': doze[5], 'FC_G': doze[6], 'N_EA': doze[7],
    'MRNA': doze[8], 'PRO': doze[9],
    'td_FC': doze[10], 'td_FM': doze[11],
    'tau_C': doze[12], 'tau_M': doze[13],
    'tau_ef_EA': doze[14], 'tau_eb_EA': doze[15], 'tau_pf_NP': doze[16], 'tau_pb_NP': doze[17],
    'chi_auto': doze[18]
}
para_set_mode = 0 # {0, 1} # {'No Remap', 'Remap: [A, B] ---->>>> [0, 1]'}
para_set, para_set_true = make_paras(para_set_raw = para_set_raw, para_set_mode = para_set_mode, verbose = False)
print(f'\nPara Set True!\n{para_set_true}')

#%%# Simulation [Function]

def simulator(parameter_set, parameter_set_true, parameter_set_mode, parameter_set_rules = None, species = ['N', 'G', 'N_MRNA', 'G_MRNA', 'NP', 'FC', 'FC_MRNA', 'FM', 'FD', 'EI', 'EA'], steps = 10000, cells = 3, time_mini = 0, time_maxi = 48, time_unit = 'Hours', time_delta = 0.25, cell_layers = 2, layer_cells = 1, faces = 2*3, seed = None, sup_comp = False, verbose = False):
    
    assert cells == cell_layers * layer_cells, f"We have an ill-defined number of 'cells'!\n\t{cells} ~ {cell_layers * layer_cells}"
    
    if sup_comp: verbose = False # Enforce!
    
    for rule in parameter_set_rules:
        
        para_fun = make_para_fun(parameter_set, parameter_set_true, parameter_set_mode)
        stem = construct_stem(parameter_set, parameter_set_true, para_fun, sup_comp, verbose = False)
        pat = {'N_MRNA': (para_fun('MRNA'), 0.5), 'G_MRNA': (para_fun('MRNA'), 0.5), 'N': (para_fun('PRO'), 0.5), 'G': (para_fun('PRO'), 0.5)}
        pat_mode = 'Fish_Bind'
        initial_pattern = make_initial_pattern(pat, pat_mode, verbose, species = list(stem.assembly['species'].values()), cells = cells, seed = seed)
        
        cell_location_mat, cell_size_mat = cell_placement(cell_layers, layer_cells, verbose)
        cell_hood_dit, dot_mat = cell_distance(cell_location_mat, verbose)
        cell_hoods = cell_neighborhood(cell_hood_dit, dot_mat, cell_layers, layer_cells, verbose)
        comm_classes_portrait = { # Jump-Diffuse Reactions
            0: (['FC_IN -> FM_EX', 'FM_IN -> FM_EX'], cell_hoods[0])
        }
        
        state_tor_PRE = {'N': 0, 'G': 0, 'N_MRNA': 0, 'G_MRNA': 0, 'NP': 0, 'FC': 0, 'FC_MRNA': 0, 'FM': 0, 'FD': 0, 'EI': 1000, 'EA': 0}
        state_tor_EPI = {'N': 0, 'G': 0, 'N_MRNA': 0, 'G_MRNA': 0, 'NP': 0, 'FC': 0, 'FC_MRNA': 0, 'FM': 0, 'FD': 0, 'EI': 1000, 'EA': 0}
        state_tors = {'state_tor_PRE': state_tor_PRE, 'state_tor_EPI': state_tor_EPI}
        state_tor = instate_state_tor(stem, cell_layers, layer_cells, state_tors, pat_mode, initial_pattern, blank = False)
        
        rates_exclude = list() # No Void (Lumen AND Sink_Up AND Sink_Do)
        rho_mat = make_rho_mat(cell_hood_dit, faces, cell_layers, layer_cells, verbose = False)
        kd_FC = 1/para_fun('td_FC')
        kd_FM = 1/para_fun('td_FM')
        ksig_C = rule*1/para_fun('tau_C')
        ksig_M = rule*1/para_fun('tau_M')
        chi_auto = para_fun('chi_auto')
        rate_mat = instate_rate_mat(stem, cells, parameter_set, parameter_set_true, para_fun, rates_exclude, rho_mat, blank = False, kd_FC = kd_FC, kd_FM = kd_FM, ksig_C = ksig_C, ksig_M = ksig_M, chi_auto = chi_auto)
        
        instate = {'state_tor': state_tor, 'rate_mat': rate_mat}
        simul = BiochemSimulSwift(stem, instate, steps, cells, species, seed, verbose = False)
        jump_diffuse_tor = simul.jump_diffuse_assemble(comm_classes_portrait)
        epoch_halt_tup = (time_maxi, time_unit) # Stopping Time!
        if sup_comp or verbose: _a = time.time()
        simul.meth_direct(jump_diffuse_tor, epoch_halt_tup)
        if sup_comp or verbose: _b = time.time()
        if sup_comp or verbose: print(f'Simul\nCells = {simul.cells}\tSteps = {simul.steps}\t', _b-_a)
        
        data_inter = interpolator(simul, species, time_mini, time_maxi, time_unit, time_delta, kind = 0, sup_comp = sup_comp, verbose = verbose, err = False, fill_value = 'extrapolate')
        if verbose: plotful(data_inter, species, time_unit)
        _trajectory_set = data_inter[1].flatten()
        rule_index = parameter_set_rules.index(rule)
        if rule_index == 0:
            trajectory_set = _trajectory_set
        else:
            trajectory_set = np.concatenate((trajectory_set, _trajectory_set), 0)
    
    return trajectory_set

#%%# Simulation [Arguments]

parameter_set = para_set
parameter_set_true = para_set_true
parameter_set_mode = para_set_mode
parameter_set_rules = [0, 1] # None
species = ['N', 'G', 'NP'] # ['N', 'G', 'N_MRNA', 'G_MRNA', 'NP', 'FC', 'FC_MRNA', 'FM', 'FD', 'EI', 'EA']
steps = 7500000 if sup_comp else 2500000
cells = cellulate[0] * cellulate[1] # 'Late ICM' Plus No 'Void (Lumen /\ Sink)'
time_mini = 0
time_maxi = 48
time_unit = 'Hours'
time_delta = 0.25
cell_layers = cellulate[0]
layer_cells = cellulate[1]
faces = 2*3
seed = None # None
# sup_comp = False # True
verbose = True

arguments = (parameter_set, parameter_set_true, parameter_set_mode, parameter_set_rules, species, steps, cells, time_mini, time_maxi, time_unit, time_delta, cell_layers, layer_cells, faces, seed, sup_comp, verbose)

argument_values = arguments[1:len(arguments)]
simulator_ready = make_simulator_ready(simulator, argument_values)

#%%# Simulation [Local Computer Test]

if not sup_comp:
    trajectory_set = simulator(*arguments)

#%%# Inference Procedure [Preparation]

from sbi.inference import prepare_for_sbi
from Utilities import make_prior
from Utilities import simul_procedure_loa_comp, simul_procedure_sup_comp

prior = make_prior(para_set_true, para_set_mode, para_set_sieve = None, verbose = not sup_comp)
simulator_ready, prior = prepare_for_sbi(simulator_ready, prior)

proposal = prior

inference_prod_activate = False # Activate Inference Procedure?

#%%# Simulation Procedure: Simul Data Save!

tag = f'Act_{act}_Observe_{observe}_{curb.capitalize()}_{reiterate}'

if not inference_prod_activate:
    if sup_comp: # Super Computer
        task = int(sys.argv[1])
        task_simulations = 250
        safe = True
        collect_zero = simul_procedure_sup_comp(simulator_ready, proposal, task, task_simulations, path, tag, safe)
    else: # Local Computer
        tasks = 2
        task_simulations = 5
        safe = True
        collect_last = simul_procedure_loa_comp(simulator_ready, proposal, tasks, task_simulations, path, tag, safe)
