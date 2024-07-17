######################################
######## Optimize Exploration ########
######################################

#%%# Catalyzer

import os
import sys
path = os.path.dirname(os.path.realpath(__file__)) + '/../Resources'
sys.path.append(path)
import numpy as np
import torch
import pickle
import time
import re

#%%# Retrieve Data [Cellulate /\ Memo]

def _simul_data_load(tasks, path, tag):
    if tasks is None:
        _arcs = os.listdir(path)
        arcs = [arc for arc in _arcs if re.findall(tag+'_'+'(Theta|Trajectory)_Set', arc)]
        _arcs_alp = [arc for arc in arcs if re.findall('_Theta_Set_', arc)]
        _arcs_bet = [arc for arc in arcs if re.findall('_Trajectory_Set_', arc)]
        arcs_alp = [int(re.findall('(\d+)\.pt', arc)[0]) for arc in _arcs_alp]
        arcs_bet = [int(re.findall('(\d+)\.pt', arc)[0]) for arc in _arcs_bet]
        arcs_alp.sort()
        arcs_bet.sort()
        mess = 'Oops! Something went wrong!'
        check = arcs_alp == arcs_bet
        assert check, mess
        _tasks = arcs_alp # _tasks = arcs_bet
        tasks_mini = min(_tasks)
        tasks_maxi = max(_tasks)
    elif type(tasks) is int:
        _tasks = range(tasks)
        tasks_mini = 0
        tasks_maxi = tasks - 1
    else:
        mess = "The format is invalid! The variable 'tasks' must be either 'None' or 'integer'!"
        raise RuntimeError(mess)
    print(f'Load Task Data! {tag}\t{tasks_mini} : {tasks_maxi}\tTotal {len(_tasks)}')
    for task in _tasks:
        if task == tasks_mini:
            label = path+tag+'_Theta_Set_'+str(task)+'.pt'
            theta_set = torch.load(label)
            label = path+tag+'_Trajectory_Set_'+str(task)+'.pt'
            trajectory_set = torch.load(label)
        else:
            label = path+tag+'_Theta_Set_'+str(task)+'.pt'
            _theta_set = torch.load(label)
            theta_set = torch.cat((theta_set, _theta_set), 0)
            label = path+tag+'_Trajectory_Set_'+str(task)+'.pt'
            _trajectory_set = torch.load(label)
            trajectory_set = torch.cat((trajectory_set, _trajectory_set), 0)
        print(f'Load Task Data! {tag}\t{task+1} ~ {len(_tasks)}\tIndex {task}')
    print(f'Load Task Data! {tag}\t{tasks_mini} : {tasks_maxi}\tTotal {len(_tasks)}')
    collect = (theta_set, trajectory_set)
    return collect

def simul_data_load(tasks, path, tag, acts = None, verbose = False):
    if acts is None or len(acts) == 1:
        collect = _simul_data_load(tasks, path, tag)
    else:
        for act in acts:
            act_index = acts.index(act)
            print(f"{'~'*8} Act {act} {'~'*8} {act_index+1} : {len(acts)} {'~'*8} Start {'~'*8}")
            _tag = re.sub('Act_(\d+)_', f'Act_{act}_', tag)
            _path = path
            if verbose: print(f"{' '*8} {_tag}\n{' '*2*8} {_path}")
            if act == 0:
                _tag = re.sub('Observe_(\d+)_', 'Observe_None_', _tag)
                _path = re.sub('Observe_(\d+)/', '', _path)
                if verbose: print(f"{' '*8} {_tag}\n{' '*2*8} {_path}")
            _collect = _simul_data_load(tasks, _path, _tag)
            if act_index == 0:
                collect = _collect
            else:
                theta_set = torch.cat((collect[0], _collect[0]), 0)
                trajectory_set = torch.cat((collect[1], _collect[1]), 0)
                collect = (theta_set, trajectory_set)
            print(f"{'~'*8} Act {act} {'~'*8} {act_index+1} : {len(acts)} {'~'*8} Final {'~'*8}")
    return collect


def retrieve_data_cellulate(memo, cellulate, reiteration, lot, sieve = None, verbose = False):
    if verbose: print(f"{' '*8}{cellulate}")
    prep_data_path = memo['prep_data_path']
    data_path = memo['data_path']
    act = memo['act']
    observe = memo['observe']
    curb = memo['curb']
    if lot in [0, 1]:
        path = os.path.dirname(os.path.realpath(__file__)) + f'/../../Data_Bank/{prep_data_path}/{data_path}/Observe_{observe}/' # f'/media/mars-fias/MARS/MARS_Data_Bank/{prep_data_path}/{data_path}/Observe_{observe}/'
        tag = f'Act_{act}_Observe_{observe}_{curb}_{reiteration}'
    else:
        path = os.path.dirname(os.path.realpath(__file__)) + f'/../../Data_Bank/{prep_data_path}/{data_path}/' # f'/media/mars-fias/MARS/MARS_Data_Bank/{prep_data_path}/{data_path}/'
        tag = f'Act_{act}_Observe_{observe}_{curb}_{reiteration}_Sieve_{sieve}'
    theta_set, trajectory_set = simul_data_load(None, path, tag, None, verbose)
    data_cellulate = (theta_set, trajectory_set)
    return data_cellulate

def retrieve_data_memo(memo, cellulates, rules, lot, sieve = None, verbose = False):
    if verbose: print(f"{'<'*8+' '*8}{memo['data_path']}{' '*8+'>'*8}")
    data_memo = {cellulate: [None]*len(rules) for cellulate in cellulates}
    nook = memo['nook']
    for cellulate in cellulates:
        reiterations = [f'Reiterate_Nook_{cellulate[0]}_{cellulate[1]}', f'Reiterate_{cellulate[0]}_{cellulate[1]}']
        if nook:
            for rule in rules:
                rule_index = rules.index(rule)
                reiteration = reiterations[rule_index]
                data_cellulate = retrieve_data_cellulate(memo, cellulate, reiteration, verbose)
                data_memo[cellulate][rule_index] = data_cellulate[1]
        else:
            reiteration = reiterations[1]
            data_cellulate = retrieve_data_cellulate(memo, cellulate, reiteration, lot, sieve, verbose)
            splitter = int(data_cellulate[1].shape[1]/len(rules))
            rule_split = [(rules.index(rule)*splitter, (rules.index(rule)+1)*splitter) for rule in rules]
            for rule in rules:
                rule_index = rules.index(rule)
                split = rule_split[rule_index]
                data_memo[cellulate][rule_index] = data_cellulate[1][:, split[0]:split[1]]
    return data_memo

def retrieve_data_cellulate_initiate(memo, cellulate, initiate, reiteration, verbose = False):
    if verbose: print(f"{' '*8}{cellulate}{' '*8}{initiate}")
    data_path = memo['data_path']
    act = memo['act']
    observe = memo['observe']
    curb = memo['curb']
    path = os.path.dirname(os.path.realpath(__file__)) + f'/../../Data_Bank/Shallow_Grid/{data_path}/Observe_{observe}/' # f'/media/mars-fias/MARS/MARS_Data_Bank/Shallow_Grid/{data_path}/Observe_{observe}/'
    tag = f'Act_{act}_Observe_{observe}_{curb}_{reiteration}'
    theta_set, trajectory_set = simul_data_load(None, path, tag, None, verbose)
    data_cellulate_initiate = (theta_set, trajectory_set)
    return data_cellulate_initiate

def retrieve_data_memo_initiate(memo, cellulates, initiate_set, wait, verbose = False):
    if verbose: print(f"{'<'*8+' '*8}{memo['data_path']}{' '*8+'>'*8}")
    keys = list(initiate_set.keys())
    values = list(initiate_set.values())
    initiate_set_temp = list(zip(*values))
    data_memo_initiate = {(cellulate, initiate_temp): None for cellulate in cellulates for initiate_temp in initiate_set_temp}
    for cellulate in cellulates:
        for initiate_temp in initiate_set_temp:
            initiate = {keys[index]: initiate_temp[index] for index in range(len(keys))}
            _initiate = '_'.join(map(str, [f'{key}_{value}' for key, value in initiate.items()]))
            reiteration = f'Reiterate_{cellulate[0]}_{cellulate[1]}_Initiate_{_initiate}_Wait_{wait}' # {Wild, Nook, Cast}
            data_cellulate_initiate = retrieve_data_cellulate_initiate(memo, cellulate, initiate, reiteration, verbose)
            data_memo_initiate[(cellulate, initiate_temp)] = data_cellulate_initiate[1]
    return data_memo_initiate

#%%# Override ObjectiveFun Class! [Sieve]

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

#%%# Objective Fun Counter

def objective_fun_counter(self, **keywords):
    # Data Preprocessor! [Start]
    self.restructure_data() # Step Zero!
    tau_mini = keywords.get('tau_mini', None)
    tau_maxi = keywords.get('tau_maxi', None)
    tau_delta = keywords.get('tau_delta', None)
    self.decimate_data(tau_mini, tau_maxi, tau_delta)
    species_sieve = ['N', 'G', 'NP']
    self.sieve_data(species_sieve)
    species_comb_tram_dit = {'NT': (('N', 'NP'), np.add)} # {'GP': (('G', 0.5), np.power)}
    self.comb_tram_data(species_comb_tram_dit)
    species_sieve = ['NT', 'G']
    self.sieve_data(species_sieve)
    # Data Preprocessor! [Final]
    # Data Processor! [Start]
    threshold_NT_positive = self.positive_NT-self.cusp_NT*np.sqrt(self.positive_NT)
    threshold_NT_negative = self.negative_NT+self.cusp_NT*np.sqrt(self.negative_NT)
    threshold_G_positive = self.positive_G-self.cusp_G*np.sqrt(self.positive_G)
    threshold_G_negative = self.negative_G+self.cusp_G*np.sqrt(self.negative_G)
    NT_positive = self.data_objective[:, self.species_objective.index('NT'), :, :] > threshold_NT_positive
    NT_negative = self.data_objective[:, self.species_objective.index('NT'), :, :] < threshold_NT_negative
    G_positive = self.data_objective[:, self.species_objective.index('G'), :, :] > threshold_G_positive
    G_negative = self.data_objective[:, self.species_objective.index('G'), :, :] < threshold_G_negative
    classification = { # (NT, G) # '(+|-)(+|-)'
        '++': np.logical_and(NT_positive, G_positive),
        '+-': np.logical_and(NT_positive, G_negative),
        '-+': np.logical_and(NT_negative, G_positive),
        '--': np.logical_and(NT_negative, G_negative)
    }
    counter = dict()
    counter.update({'DP': np.count_nonzero(classification['++'], 2)})
    counter.update({'NT': np.count_nonzero(classification['+-'], 2)})
    counter.update({'G': np.count_nonzero(classification['-+'], 2)})
    counter.update({'DN': np.count_nonzero(classification['--'], 2)})
    return counter

#%%#  Make Sieve Rag

def make_sieve_rag(lots, cellulate, tau_mini, tau_maxi, keywords = dict()):
    sieve_lit = keywords.get('sieve_lit')
    memos = keywords.get('memos')
    species = keywords.get('species')
    time_mini = keywords.get('time_mini')
    time_maxi = keywords.get('time_maxi')
    time_unit = keywords.get('time_unit')
    time_delta = keywords.get('time_delta')
    rules = keywords.get('rules')
    aim_NT = keywords.get('aim_NT')
    aim_G = keywords.get('aim_G')
    L = keywords.get('L')
    sieve_rag = keywords.get('sieve_rag')
    ANNA = keywords.get('ANNA', False)
    cell_layers = keywords.get('cell_layers', 5)
    layer_cells = keywords.get('layer_cells', 5)
    act_anna = keywords.get('act_anna', 7)
    theta_sets = dict()
    for lot in lots:
        for sieve in sieve_lit:
            sieve_index = sieve_lit.index(sieve) # sieve_index = sieve # Convenience!    
            memo = memos[lot]
            if ANNA:
                reiteration = f'Reiterate_{cellulate[0]}_{cellulate[1]}' if lot in [0, 1] else f'Reiterate_{cellulate[0]}_{cellulate[1]}_Cellulate_{cell_layers}_{layer_cells}_ANNA_{act_anna}_L_{lot-1}'
            else:
                reiteration = f'Reiterate_{cellulate[0]}_{cellulate[1]}'
            theta_set, trajectory_set = retrieve_data_cellulate(memo, cellulate, reiteration, lot, sieve, verbose = True)
            objective_fun = ObjectiveFunPortionRuleTemp(trajectory_set, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi = 10, verbose = False, show = False, rules = rules, aim_NT = aim_NT, aim_G = aim_G, L = L)
            objective_fun.apply(tau_mini = tau_mini, tau_maxi = tau_maxi)
            percentiles, appraisal = objective_fun.appraise()
            alp = np.mean(percentiles[:, 1])
            _bet = np.mean(percentiles[:, [0, 2]], 0)
            bet = np.abs(alp-_bet)        
            sieve_rag[lot, 0, sieve_index] = alp
            sieve_rag[lot, [1, 2], sieve_index] = bet
            sieve_rag[lot, 3, sieve_index] = appraisal
            theta_sets.update({(lot, sieve_index): np.round(np.mean(theta_set.numpy(), 0), 2)})
    ret = (sieve_rag, theta_sets)
    return ret

#%%# Discover Appraisal Opals

def discover_appraisal_opals(lots, sieve_rag):
    appraisal_opal_discoveries = list()
    for lot in lots:
        appraisal = sieve_rag[lot, 3, :]
        appraisal_opal_sieve = np.argmax(appraisal)
        appraisal_opal = np.max(appraisal)
        appraisal_opal_discovery = (appraisal_opal_sieve, appraisal_opal)
        appraisal_opal_discoveries.append(appraisal_opal_discovery)
    return appraisal_opal_discoveries

#%%# ANN Data

# path = '/home/mars-fias/Documents/Education/Doctorate/Simulator/Projects/Art_Intel/Optimize_Exploration/Resources/'
# sys.path.append(path)

from Utilities_Shallow_Grid import make_paras

def posterior_appraisal_selection(path, _tag, postage = None, verbose = False):
    if postage is None:
        _arcs = os.listdir(path)
        arcs = [arc for arc in _arcs if re.findall(_tag+'_'+'Posterior', arc)]
        tasks = [int(re.findall('(\d+)\.pkl', arc)[0]) for arc in arcs if re.findall('(\d+)\.pkl', arc)]
        tasks.sort()
        if len(tasks) == 0 and len(arcs) == 0:
            mess = 'Oops! Where are the posteriors?'
            raise RuntimeError(mess)
        elif len(tasks) == 0 and len(arcs) != 0:
            _postage = '_Posterior.pkl'
            with open(path + _tag + _postage, 'rb') as portfolio:
                posterior = pickle.load(portfolio)
            mess = f"{'~'*4*8}\nPosterior!\n{' '*8}{path + _tag + _postage}\n{'~'*4*8}"
        else:
            posts = list()
            appraisals = list()
            for task in tasks:
                _postage = f'_Posterior_{task}.pkl'
                with open(path + _tag + _postage, 'rb') as portfolio:
                    post = pickle.load(portfolio)
                appraisal = post.appraisal if hasattr(post, 'appraisal') else 0
                posts.append(post)
                appraisals.append(appraisal)
            where = np.argmax(appraisals)
            posterior = posts[where]
            _mess = f'{path + _tag}_Posterior_{where}.pkl'
            mess = f"{'~'*4*8}\nPosterior Tasks!\n{' '*8}{tasks}\nPosterior Appraisals!\n{' '*8}{appraisals}\nPosterior Where?\n{' '*8}{where}\nPosterior!\n{' '*8}{_mess}\n{'~'*4*8}"
    else:
        with open(path + _tag + postage, 'rb') as portfolio:
            posterior = pickle.load(portfolio)
        mess = f"{'~'*4*8}\nPosterior!\n{' '*8}{path + _tag + postage}\n{'~'*4*8}"
    if verbose: print(mess)
    return posterior

def retrieve_posteriors_mapes(data_paths, acts, observers, curbs, verbose = False):
    if verbose:
        tip = 's' if len(data_paths) > 1 else ''
        print(f"Info!\n\tWe will retrieve {len(data_paths)} {'posterior' + tip} and {len(data_paths)} {'mape' + tip}!")
    posteriors = list()
    mapes = list()
    for index in range(len(data_paths)):
        data_path = data_paths[index]
        act = acts[index]
        observe = observers[index]
        curb = curbs[index]
        path = os.path.dirname(os.path.realpath(__file__)) + f'/../../Data_Bank/Shallow_Grid/{data_path}/Observe_{observe}/' # f'/media/mars-fias/MARS/MARS_Data_Bank/Shallow_Grid/{data_path}/Observe_{observe}/'
        _tag = f'Act_{act}_Observe_{observe}_{curb.capitalize()}'
        postage = None # postage = '_Posterior.pkl'
        posterior = posterior_appraisal_selection(path, _tag, postage, verbose = True)
        _mape = posterior.map().numpy()
        mape = [int(round(_mape[index], 0)) if _mape[index] > 1 else round(_mape[index], 2) for index in range(_mape.size)]
        if verbose: print(f'Posterior MAPE!\n{data_path}\n{_tag}\n\t{mape}')
        posteriors.append(posterior)
        mapes.append(mape)
        ret = (posteriors, mapes)
    return ret

def retrieve_para_set_truths(mapes, para_key_sets, para_value_sets, para_set_modes, verbose = False):
    para_set_truths = list()
    for index in range(len(mapes)):
        mape = mapes[index]
        para_key_set = para_key_sets[index]
        para_value_set = para_value_sets[index]
        para_set_mode = para_set_modes[index]
        para_set_raw = {para_key_set[_]: (mape[_], para_value_set[_]) for _ in range(len(mape))}
        _, para_set_true = make_paras(para_set_raw, para_set_mode, verbose)
        para_set_truths.append(para_set_true)
    return para_set_truths

def synthesizer_post(posterior = None, observation = None, posterior_sample_shape = None, parameter_set_true = None, mape_calculate = False, fig_size = (5, 5), verbose = False, **keywords):
    if posterior is None:
        mess = 'Oops! Please, if we wish to synthesize some simulator samples, then we need a posterior distribution!'
        raise RuntimeError(mess)
    check = isinstance(posterior.map(), torch.Tensor)
    mess = "Please, we must provide a valid 'posterior' object!\First, we must execute 'InferenceProd.synthesizer'!"
    assert check, mess
    if observation is None:
        observation = posterior.default_x
    posterior_samples = keywords.get('posterior_samples', None)
    if posterior_samples is None:
        posterior_samples = posterior.sample(sample_shape = posterior_sample_shape, x = observation)
    theta_median = torch.quantile(posterior_samples, 0.5, 0)
    theta_mape = posterior.map()
    theta = {'median': theta_median, 'mape': theta_mape}
    ret = (posterior_samples, theta)
    return ret

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

#%%# Posterior Synthesizer

from sbi import analysis

def synthesizer_post_coda(posterior = None, observation = None, posterior_sample_shape = None, parameter_set_true = None, mape_calculate = False, fig_size = (5, 5), verbose = False, **keywords):
    if posterior is None:
        mess = 'Oops! Please, if we wish to synthesize some simulator samples, then we need a posterior distribution!'
        raise RuntimeError(mess)
    check = isinstance(posterior.map(), torch.Tensor)
    mess = "Please, we must provide a valid 'posterior' object!\First, we must execute 'InferenceProd.synthesizer'!"
    assert check, mess
    paras = list(parameter_set_true.keys())
    para_span = torch.tensor([0, 1])
    if observation is None:
        observation = posterior.default_x
    posterior_samples = keywords.get('posterior_samples', None)
    if posterior_samples is None:
        posterior_samples = posterior.sample(sample_shape = posterior_sample_shape, x = observation)
    card = len(paras)
    theta_median = torch.quantile(posterior_samples, 0.5, 0)
    if mape_calculate:
        posterior.set_default_x(observation)
        posterior.map(num_init_samples = posterior_sample_shape[0])
    theta_mape = posterior.map()
    theta = {'median': theta_median, 'mape': theta_mape}
    minis = torch.floor(torch.min(posterior_samples, 0).values)
    maxis = torch.ceil(torch.max(posterior_samples, 0).values)
    check = torch.tensor([para_span[0] <= minis[index] and para_span[1] >= maxis[index] for index in range(posterior_samples.shape[1])])
    para_key_caste = keywords.get('para_key_caste', ...)
    para_key_elect = keywords.get('para_key_elect', None)
    para_key_label = keywords.get('para_key_label', paras) # para_key_label = [f"{int(np.round(theta['mape'].tolist()[index], 0)) if theta['mape'][index] >= 1 else np.round(theta['mape'].tolist()[index], 2)}\n{_para_key_label[index]}" for index in range(posterior_samples.shape[1])]
    if verbose:
        print(f'Caste\n\t{para_key_caste}\nElect\n\t{para_key_elect}\nLabel\n\t{[para_key_label[para_key_index] for para_key_index in para_key_caste]}')
    if torch.all(check):
        limes = [para_span.tolist()]*card
    else:
        para_value_set = keywords.get('para_value_set', None)
        if para_value_set is None:
            limes = torch.tensor([[minis[index], maxis[index]] for index in range(posterior_samples.shape[1])])
        else:
            limes = torch.tensor([[para_value_set[index][0], para_value_set[index][1]] for index in range(posterior_samples.shape[1])])
    spots = [[lime[0], lime[1], (lime[1]+lime[0])/2] for lime in limes]
    fig = keywords.get('fig', None)
    axes = keywords.get('axes', None)
    mark_size = keywords.get('mark_size', 7)
    theta_set_opals = keywords.get('theta_set_opals', posterior.sample())
    theta_set_picks = keywords.get('theta_set_picks', {'post': 0, 'coda': 1})
    theta_set_cocos = keywords.get('theta_set_cocos', ['magenta']*len(theta_set_opals))
    pick_post = theta_set_picks.get('post')
    pick_coda = theta_set_picks.get('coda')
    points = theta_set_opals.tolist()
    points_colors = theta_set_cocos
    chart = analysis.pairplot(
        samples = posterior_samples, points = points, limits = limes, subset = para_key_caste, offdiag = 'hist', diag = 'hist', figsize = fig_size, labels = para_key_label, ticks = spots, fig = fig, axes = axes,
        samples_colors = theta_set_cocos[slice(pick_post, pick_post+1)], points_colors = points_colors, points_diag = {'linewidth': 2}, points_offdiag = {'markersize': mark_size}, hist_diag = {'histtype': 'stepfilled', 'bins': 100, 'alpha': 0.5}, hist_offdiag = {'bins': 100}
    )
    if pick_coda != len(theta_set_opals):
        condition = theta_set_opals[[pick_coda], :] # posterior.sample() # torch.reshape(theta_mape, (1, theta_mape.size()[0]))
        points_coda = points[slice(pick_coda, pick_coda+1)]
    else:
        condition = torch.reshape(torch.mean(torch.as_tensor(limes, dtype = torch.float), dim = 1, keepdim = True), (1, len(limes)))
        points_coda = condition
    fig_coda = keywords.get('fig_coda', None)
    axes_coda = keywords.get('axes_coda', None)
    chart_coda = analysis.conditional_pairplot(
        density = posterior, condition = condition, limits = limes, points = points_coda, subset = para_key_caste, resolution = 250, figsize = fig_size, labels = para_key_label, ticks = spots, points_colors = points_colors[slice(pick_coda, pick_coda+1)], fig = fig_coda, axes = axes_coda,
        samples_colors = ['tab:gray'], points_diag = {'linewidth': 2}, points_offdiag = {'markersize': mark_size}, kde_diag = {'bins': 100}, hist_offdiag = {'bins': 100}
    )
    core_coda = analysis.conditional_corrcoeff(density = posterior, limits = limes, condition = condition, subset = np.sort(para_key_caste), resolution = 250)
    if verbose:
        print(posterior)
        print(f"MAPE\n\t{theta['mape'][para_key_caste]}")
        print(f'Condition\n\t{condition[:, para_key_caste]}')
        print(f'\n\t{para_key_elect}')
    ret = (chart, chart_coda, posterior_samples, core_coda)
    return ret

def cure_axe_mat(axe_mat = None, para_key_caste = None, para_value_set = None, mape = None, **keywords):
    cure = keywords.get('cure', set())
    verbose = keywords.get('verbose', False)
    axe_mat_rows, axe_mat_cols = axe_mat.shape
    _ave = [(para_value[1]+para_value[0])/2 for para_value in para_value_set]
    ave = [int(np.round(para_value, 0)) if para_value > 1 else np.round(para_value, 2) for para_value in _ave]
    if 'tick' in cure:
        font_size_tick = keywords.get('font_size_tick', 13)
        for axe_mat_row in range(axe_mat_rows):
            for axe_mat_col in range(axe_mat_cols):
                if axe_mat_row != axe_mat_col:
                    continue
                else:
                    axe_mat_index = axe_mat_row # axe_mat_index = axe_mat_col
                    axe_mat_caste = para_key_caste[axe_mat_index]
                    spots = [para_value_set[axe_mat_caste][0], ave[axe_mat_caste], para_value_set[axe_mat_caste][1]]
                    axe_mat[axe_mat_row, axe_mat_col].set_xticks(ticks = spots, labels = spots, fontsize = font_size_tick)
    if 'label' in cure:
        font_size_label = keywords.get('font_size_label', 17)
        for axe_mat_row in range(axe_mat_rows):
            for axe_mat_col in range(axe_mat_cols):
                if axe_mat_col > axe_mat_row:
                    axe_mat_index = axe_mat_col # axe_mat_index = axe_mat_row
                    axe_mat_caste = para_key_caste[axe_mat_index]
                    x_label = mape[axe_mat_caste]
                    y_label = mape[para_key_caste[axe_mat_row]]
                    axe_mat[axe_mat_row, axe_mat_col].set_xlabel(xlabel = x_label, fontsize = font_size_label)
                    axe_mat[axe_mat_row, axe_mat_col].set_ylabel(ylabel = y_label, fontsize = font_size_label)
                else:
                    continue
    if 'legend' in cure:
        font_size_loge = keywords.get('font_size_loge', 17)
        theta_set_opals = keywords.get('theta_set_opals', None)
        theta_set_picks = keywords.get('theta_set_picks', None)
        theta_set_cocos = keywords.get('theta_set_cocos', None)
        theta_set_pats = keywords.get('theta_set_pats', None)
        coda = keywords.get('coda', False)
        if not coda:
            index = theta_set_picks['post']
            theta_set_opal = theta_set_opals[index].tolist()
            theta_set_coco = theta_set_cocos[slice(0, len(theta_set_cocos)-1)]
            theta_set_pat = theta_set_pats[slice(0, len(theta_set_pats)-1)]
        else:
            index = theta_set_picks['coda']
            theta_set_opal = theta_set_opals[index].tolist() if index != len(theta_set_opals) else mape
            theta_set_coco = theta_set_cocos[slice(index, index+1)]
            theta_set_pat = theta_set_pats[slice(index, index+1)]
        x_limes = (-0.5, len(theta_set_coco)-0.5)
        y_limes = (0, 1)
        for axe_mat_row in range(axe_mat_rows):
            for axe_mat_col in range(axe_mat_cols):
                if axe_mat_col == 0 and axe_mat_row == axe_mat_rows-1:
                    for guide in range(0, len(theta_set_coco)):
                        axe_mat[axe_mat_row, axe_mat_col].axvline(
                            x = guide, ymin = y_limes[0]+(y_limes[1]-y_limes[0])*0.5, ymax = y_limes[1]-(y_limes[1]-y_limes[0])*0.5,
                            color = theta_set_coco[guide], linestyle = '-', label = theta_set_pat[guide], linewidth = 2, marker = '.', markersize = 13
                        )
                    axe_mat[axe_mat_row, axe_mat_col].set_xlim(x_limes[0], x_limes[1])
                    axe_mat[axe_mat_row, axe_mat_col].set_ylim(y_limes[0], y_limes[1])
                    axe_mat[axe_mat_row, axe_mat_col].legend(
                        loc = 'center', fontsize = font_size_loge, frameon = True, framealpha = 0.975, facecolor = 'white',
                        title = ['Parameter Sets', 'Parameter Condition'][coda], title_fontproperties = {'weight': 'book', 'size': font_size_loge}, borderpad = 0.25, handleheight = 1.25
                    )
                else:
                    continue
        if verbose:
            mess = f"{['Post', 'Coda'][coda]}\n\t{theta_set_opal}"
            print(mess)
    return axe_mat
