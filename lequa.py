import numpy as np
from sklearn.linear_model import LogisticRegression
import quapy as qp
import quapy.functional as F
from quapy.data.datasets import LEQUA2022_SAMPLE_SIZE, fetch_lequa2022, fetch_lequa2024, LEQUA2024_SAMPLE_SIZE
from quapy.evaluation import evaluation_report
from quapy.method.aggregative import EMQ
from quapy.model_selection import GridSearchQ
from methods_v2 import EMQ, EMQPosteriorSmoothing, EMQTempScaling, EMQDamping,EMQDirichletMAP, EMQConfidentSubset,EMQEntropyReg
from methods_v2 import EMQTempScaling_DirichletMAP, EMQTempScaling_EntropyReg, EMQTempScaling_Damping, EMQPosteriorSmoothing_DirichletMAP, EMQPosteriorSmoothing_EntropyReg, EMQPosteriorSmoothing_Damping
from run_experiments_v2 import gridsearch_params, get_heuristic_parameters, wrap_hyper, newClassifier, load_timings
import pandas as pd
import os
from pathlib import Path
from time import time


def run_experiment(classifier_types=['LR'], experiment_type='simpleheuristics'):
    qp.environ['N_JOBS'] = -1
    datasets = ['T1A', 'T1B', 'T1', 'T2']

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

    result_dir = "results/lequa"
    os.makedirs(result_dir, exist_ok=True)
    global_result_path = f'{result_dir}/allmethods'
    timings = load_timings(global_result_path)
    with open(global_result_path + '.csv', 'wt') as csv:
        csv.write(f'Method\tDataset\tMAE\tMRAE\tt_train\n')

    for method_name, quantifier, param_grid in METHODS:
        print('Init method', method_name)
        with open(global_result_path + '.csv', 'at') as csv:
            for dataset in datasets:
                if dataset == 'T1A' or dataset == 'T1B':
                    training, val_generator, test_generator = fetch_lequa2022(task=dataset, data_home='data/lequa2022')
                    qp.environ['SAMPLE_SIZE'] = LEQUA2022_SAMPLE_SIZE[dataset]
                elif dataset == 'T1' or dataset == 'T2':
                    training, val_generator, test_generator = fetch_lequa2024(task=dataset, data_home='data/lequa2024')
                    qp.environ['SAMPLE_SIZE'] = LEQUA2024_SAMPLE_SIZE[dataset]
                local_result_path = os.path.join(Path(global_result_path).parent, method_name + '_' + dataset + '.dataframe')
                model = GridSearchQ(quantifier, param_grid, protocol=val_generator, error='mae', refit=False, verbose=True)
                t_init = time()
                quantifier = model.fit(training)
                timings[method_name][dataset] = time() - t_init
                report = evaluation_report(quantifier, protocol=test_generator, error_metrics=['mae', 'mrae', 'mkld'], verbose=True)
                report.to_csv(local_result_path)

                means = report.mean(numeric_only=True)
                if method_name not in timings or dataset not in timings[method_name]:
                    print("entry does not exist in timings, setting to 0")
                    timings[method_name][dataset] = 0
                csv.write(f'{method_name}\t{dataset}\t{means["mae"]:.5f}\t{means["mrae"]:.5f}\t{timings[method_name][dataset]:.3f}\n')
                csv.flush()

if __name__ == '__main__':
    run_experiment(classifier_types=('LR','NN',),experiment_type='simpleheuristics')
    