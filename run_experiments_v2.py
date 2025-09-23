import os
from time import time
from collections import defaultdict
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import quapy as qp
from methods_v2 import EMQ, EMQPosteriorSmoothing, EMQTempScaling, EMQDamping,EMQDirichletMAP, EMQConfidentSubset,EMQEntropyReg
from methods_v2 import EMQTempScaling_DirichletMAP, EMQTempScaling_EntropyReg, EMQTempScaling_Damping, EMQPosteriorSmoothing_DirichletMAP, EMQPosteriorSmoothing_EntropyReg, EMQPosteriorSmoothing_Damping
from quapy.model_selection import GridSearchQ
from quapy.protocol import UPP
from pathlib import Path
from functools import partial


SEED = 1


def newClassifier(type='LR'):
    if type == 'LR':
        return LogisticRegression(max_iter=3000, random_state=SEED)
    elif type == 'NN':
        return MLPClassifier(hidden_layer_sizes=(100,), max_iter=3000,early_stopping=True,validation_fraction=0.15,n_iter_no_change=20,tol=1e-4,random_state=SEED)
    else:
        raise ValueError(f"Unknown classifier type: {type}")

    
def gridsearch_params(type = 'LR'):
    if type == 'LR':
        return {'C': np.logspace(-3, 3, 7),'class_weight': ['balanced', None]}
    elif type == 'NN':
        return {'learning_rate_init': [0.001, 0.01]}

def wrap_hyper(classifier_hyper_grid:dict):
    return {'classifier__'+k:v for k, v in classifier_hyper_grid.items()}

def callback(trajectory,method,dataset):
    with open('trajectories/trajectory_{}_{}.pkl'.format(method,dataset), 'ab') as f:
        pickle.dump(trajectory, f)


def show_results(result_path):
    import pandas as pd
    df = pd.read_csv(result_path+'.csv', sep='\t')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pv = df.pivot_table(index='Dataset', columns="Method", values=["MAE", "MRAE", "t_train"], margins=True)
    print(pv)

def load_timings(result_path):
    import pandas as pd
    timings = defaultdict(lambda: {})
    if not Path(result_path + '.csv').exists():
        return timings

    df = pd.read_csv(result_path+'.csv', sep='\t')
    return timings | df.pivot_table(index='Dataset', columns='Method', values='t_train').to_dict()

def get_heuristic_parameters(heuristic):
    if heuristic == 'PSEM':
        return {'epsilon_smoothing': (0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)}
    elif heuristic == 'TSEM':
        return {'tau': (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0)}
    elif heuristic == 'DEM':
        return {'lambd': np.linspace(0.1, 1, 10)}
    elif heuristic == 'EREM':
        return {'eta' : (0.0,0.0001,0.001)}
    elif heuristic == 'DMAPEM':
        return {'alpha': (1.0, 2.0, 5.0, 10.0)}
    elif heuristic == 'CSEM':
        return {'kappa': (0.3,0.5,0.8,1)}
    else:
        raise ValueError(f"Unknown heuristic: {heuristic}")


def run_experiments(classifier_types, datasets, fetch_function, sample_size,result_dir, experiment_type='simpleheuristics'):
    #delete all pkl files in trajectories directory
    # for file in os.listdir('trajectories'):
    #     if file.startswith('trajectory') and file.endswith('.pkl'):
    #         os.remove(os.path.join('trajectories', file))

    METHODS = []
    for classifier_type in classifier_types:
        grid = gridsearch_params(classifier_type)

        if experiment_type=='simpleheuristics':
            methods_for_classifier = [
                ('EM',  EMQ(newClassifier(classifier_type)), wrap_hyper(grid)),
                ('EM_BCTS',  EMQ(newClassifier(classifier_type),recalib='bcts'), wrap_hyper(grid)),
                ('PSEM',  EMQPosteriorSmoothing(newClassifier(classifier_type)), {**wrap_hyper(grid), **get_heuristic_parameters('PSEM')}),
                ('TSEM',  EMQTempScaling(newClassifier(classifier_type)), {**wrap_hyper(grid), **get_heuristic_parameters('TSEM')}),
                ('DEM',  EMQDamping(newClassifier(classifier_type)), {**wrap_hyper(grid), **get_heuristic_parameters('DEM')}),
                ('EREM',  EMQEntropyReg(newClassifier(classifier_type)), {**wrap_hyper(grid),**get_heuristic_parameters('EREM')}),
                ('DMAPEM', EMQDirichletMAP(newClassifier(classifier_type)), {**wrap_hyper(grid), **get_heuristic_parameters('DMAPEM')}),
                ('CSEM', EMQConfidentSubset(newClassifier(classifier_type)), {**wrap_hyper(grid), **get_heuristic_parameters('CSEM')}),
            ]
        elif experiment_type=='combinations':
            methods_for_classifier = [
                ('EM',  EMQ(newClassifier(classifier_type)), wrap_hyper(grid)),
                ('EM_BCTS',  EMQ(newClassifier(classifier_type),recalib='bcts'), wrap_hyper(grid)),
                ('PSEM',  EMQPosteriorSmoothing(newClassifier(classifier_type)), {**wrap_hyper(grid), **get_heuristic_parameters('PSEM')}),
                ('TSEM',  EMQTempScaling(newClassifier(classifier_type)), {**wrap_hyper(grid), **get_heuristic_parameters('TSEM')}),
                ('DEM',  EMQDamping(newClassifier(classifier_type)), {**wrap_hyper(grid), **get_heuristic_parameters('DEM')}),
                ('EREM',  EMQEntropyReg(newClassifier(classifier_type)), {**wrap_hyper(grid),**get_heuristic_parameters('EREM')}),
                ('DMAPEM', EMQDirichletMAP(newClassifier(classifier_type)), {**wrap_hyper(grid), **get_heuristic_parameters('DMAPEM')}),
                ('CSEM', EMQConfidentSubset(newClassifier(classifier_type)), {**wrap_hyper(grid), **get_heuristic_parameters('CSEM')}),
                ('TSEM_DEM', EMQTempScaling_Damping(newClassifier(classifier_type)), {**wrap_hyper(grid), **get_heuristic_parameters('TSEM'),**get_heuristic_parameters('DEM')}),
                ('TSEM_EREM', EMQTempScaling_EntropyReg(newClassifier(classifier_type)), {**wrap_hyper(grid), **get_heuristic_parameters('TSEM'),**get_heuristic_parameters('EREM')}),
                ('TSEM_DMAPEM', EMQTempScaling_DirichletMAP(newClassifier(classifier_type)), {**wrap_hyper(grid), **get_heuristic_parameters('TSEM'),**get_heuristic_parameters('DMAPEM')}),
                ('PSEM_DEM', EMQPosteriorSmoothing_Damping(newClassifier(classifier_type)), {**wrap_hyper(grid), **get_heuristic_parameters('PSEM'),**get_heuristic_parameters('DEM')}),
                ('PSEM_EREM', EMQPosteriorSmoothing_EntropyReg(newClassifier(classifier_type)), {**wrap_hyper(grid), **get_heuristic_parameters('PSEM'),**get_heuristic_parameters('EREM')}),
                ('PSEM_DMAPEM', EMQPosteriorSmoothing_DirichletMAP(newClassifier(classifier_type)), {**wrap_hyper(grid), **get_heuristic_parameters('PSEM'),**get_heuristic_parameters('DMAPEM')}),
            ]
        methods_for_classifier = [(name + '_' + classifier_type, quant,grid) for name, quant, grid in methods_for_classifier]
        METHODS.extend(methods_for_classifier)

    qp.environ['SAMPLE_SIZE'] = sample_size
    qp.environ['N_JOBS'] = -1
    n_bags_val = 250
    n_bags_test = 1000

    os.makedirs(result_dir, exist_ok=True)

    global_result_path = f'{result_dir}/allmethods'
    timings = load_timings(global_result_path)
    with open(global_result_path + '.csv', 'wt') as csv:
        csv.write(f'Method\tDataset\tMAE\tMRAE\tt_train\n')

    for method_name, quantifier, param_grid in METHODS:

        print('Init method', method_name)

        with open(global_result_path + '.csv', 'at') as csv:

            for dataset in datasets:

                print('init', dataset)

                local_result_path = os.path.join(Path(global_result_path).parent, method_name + '_' + dataset + '.dataframe')

                if os.path.exists(local_result_path):
                    print(f'result file {local_result_path} already exist; skipping')
                    report = qp.util.load_report(local_result_path)

                else:
                    with qp.util.temp_seed(SEED):

                        #data = qp.datasets.fetch_UCIMulticlassDataset(dataset, verbose=True)
                        data = fetch_function(dataset, verbose=True)

                        # model selection
                        train, test = data.train_test
                        train, val = train.split_stratified(random_state=SEED)
                        quantifier.log_trajectory = False
                        protocol = UPP(val, repeats=n_bags_val)
                        model = GridSearchQ(
                            quantifier, param_grid, protocol, refit=True, n_jobs=-1, verbose=1, error='mae'
                        )
                        
                        t_init = time()
                        try:
                            model.fit(train)

                            print(f'best params {model.best_params_}')
                            with open('bestparams/{}_{}.pkl'.format(method_name,dataset), 'wb') as f:
                                pickle.dump(model.best_params_, f)
                            print(f'best score {model.best_score_}')

                            quantifier = model.best_model()
                        except Exception as e:
                            print(e)
                            print('something went wrong... trying to fit the default model')
                            quantifier.fit(train)
                        timings[method_name][dataset] = time() - t_init
                        

                        protocol = UPP(test, repeats=n_bags_test)
                        #Set callback to log the iterations of EM variations
                        quantifier.log_trajectory = True
                        quantifier.callback = partial(callback, method=method_name,dataset=dataset)
                        report = qp.evaluation.evaluation_report(
                            quantifier, protocol, error_metrics=['mae', 'mrae'], verbose=True
                        )
                        report.to_csv(local_result_path)

                means = report.mean(numeric_only=True)
                if method_name not in timings or dataset not in timings[method_name]:
                    print("entry does not exist in timings, setting to 0")
                    timings[method_name][dataset] = 0
                csv.write(f'{method_name}\t{dataset}\t{means["mae"]:.5f}\t{means["mrae"]:.5f}\t{timings[method_name][dataset]:.3f}\n')
                csv.flush()

    show_results(global_result_path)

if __name__ == '__main__':
    run_experiments(('LR',),qp.datasets.UCI_MULTICLASS_DATASETS,qp.datasets.fetch_UCIMulticlassDataset, 500, 'results/ucimulti2','simpleheuristics')
    run_experiments(('LR',),qp.datasets.UCI_MULTICLASS_DATASETS,qp.datasets.fetch_UCIMulticlassDataset, 500, 'results/ucimulti2','combinations')