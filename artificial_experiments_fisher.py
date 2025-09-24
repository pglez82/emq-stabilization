import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tabulate

from sklearn.metrics import zero_one_loss, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from quapy.error import ae, rae, se
from quapy.data import LabelledCollection

from methods_v2 import EMQ, EMQTempScaling, EMQPosteriorSmoothing, EMQDamping, EMQEntropyReg, EMQConfidentSubset, EMQDirichletMAP
#from qbase import UsingClassifiers, CV_estimator, CC, AC, PAC, DFy, QUANTy, SORDy
#from utils import absolute_error, relative_absolute_error, squared_error, l1, l2
import quapy as qp
from run_experiments_v2 import get_heuristic_parameters

import time

def indices_to_one_hot(data, n_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(n_classes)[targets]

def plot_histogram(ax,hist,subtitle):
    ax.set_ylim(0, 1)
    bins=np.arange(len(hist))
    ax.bar(bins-0.25,height=hist[:,0],color='b',width=0.5,label="negativos")
    ax.bar(bins+0.25,height=hist[:,1],color='r',width=0.5,label="positivos")
    ax.legend(loc="upper left")
    ax.set_xticks(bins)
    ax.title.set_text(subtitle)

def plot_combined_histogram(ax,hist,subtitle):
    ax.set_ylim(0, 1)
    bins=np.arange(len(hist))
    ax.bar(bins,height=hist,color='g',width=1,label="all")
    ax.legend(loc="upper left")
    ax.set_xticks(bins)
    ax.title.set_text(subtitle)

class PerfectProbClassifier:
    def __init__(self, mu1, std1, mu2, std2):
        self.mu1 = mu1
        self.std1 = std1
        self.mu2 = mu2
        self.std2 = std2
        self.classes_ = np.array([0, 1])

    def _compute_probabilities(self, x, mu1, std1, mu2, std2):
        den1 = 1.0 / (std1 * np.sqrt(2 * np.pi))
        den2 = 1.0 / (std2 * np.sqrt(2 * np.pi))
        pdf1 = den1 * np.exp(-(x - mu1) ** 2 / (2 * std1 ** 2))
        pdf2 = den2 * np.exp(-(x - mu2) ** 2 / (2 * std2 ** 2))
        p0 = pdf1 / (pdf1 + pdf2)
        return np.column_stack([p0, 1 - p0])
    
    def fit(self, X, y):
        pass

    def predict_proba(self, x):
        return self._compute_probabilities(x, self.mu1, self.std1, self.mu2, self.std2)
    
    
    def predict(self, x):
        probs = self.predict_proba(x)
        return np.argmax(probs, axis=1)
    

def run_experiment(est_name, seed, param, ntrain, ntest, nreps, nbags, save_all):
    """ Run a single experiment

        Parameters
        ----------
        est_name: str
            Name of the estimator. 'LR' or 'SVM-RBF'

        seed: int
            Seed of the experiment

        param: int, str
            Extra oarameter for the definition of the problem.
            If dim==1, this value is the std.
            If dim=2 this value is an string to indicate if the dataset is the one designed to test HDX

        ntrain : list
            Training examples that must be tested

        ntest: int
            List with the number of testing instances in each bag e.g.,[50, 100, 200]

        nreps: int
            Number of training datasets created

        nbags: int
            Number of testing bags created for each training datasets.
            The total number of experiments will be nreps * nbags

        nfolds: int
            Number of folds used to estimate the training distributions by the methods AC, HDy and EDy

        save_all: bool
            True if the results of each single experiment must be saved
        
    """

    mu1 = -1
    std1 = param
    mu2 = 1
    std2 = std1

    if est_name == 'PerfectProbClassifier':
        estimator = PerfectProbClassifier(mu1, std1, mu2, std2)
    elif est_name == 'LR':
        estimator = LogisticRegression(C=1, random_state=seed, max_iter=10000, solver='liblinear')
    else:
        estimator = GaussianNB()

    rng = np.random.RandomState(seed)

    #   methods
    pcc = qp.method.aggregative.PCC(classifier=estimator)
    #acc = qp.method.aggregative.ACC(classifier=estimator)
    emq = EMQ(classifier=estimator)


    #   methods
    methods = [pcc,emq]

    methods_names = ['PCC','EMQ']

    heuristics = {'PSEM':"EMQPosteriorSmoothing",'TSEM':'EMQTempScaling','DEM':'EMQDamping','EREM':'EMQEntropyReg','DMAPEM':'EMQDirichletMAP','CSEM':'EMQConfidentSubset'}
    for heuristic,c in heuristics.items():
        hiperparam_values = get_heuristic_parameters(heuristic)
        print(hiperparam_values)
        hiper_name,hiper_values = next(iter(hiperparam_values.items()))
        for hiper_value in hiper_values:
            method = globals()[c](estimator)
            setattr(method, hiper_name,hiper_value)
            methods.append(method)
            
            methods_names.append(heuristic+hiper_name+str(hiper_value))

    print(methods_names)

    #   to store the results
    mae_results = np.zeros((len(methods_names), len(ntest)))
    sqe_results = np.zeros((len(methods_names), len(ntest)))
    mrae_results = np.zeros((len(methods_names), len(ntest)))
    classif_results = np.zeros((2, len(ntest)))

    if save_all:
        name_file = 'results/artificial/timeArtificialBinary-avg-' + str(param) + '-' + est_name + '-rep' + str(nreps) + \
                    '-ntest' + str(ntest) + '.txt'
        file_times = open(name_file, 'w')
        file_times.write('#examples, ')
        for index, m in enumerate(methods_names):
            file_times.write('%s, ' % m)

    name_file = 'results/artificial/artificialBinary-avg-' + str(param) + '-' + est_name + '-rep' + str(nreps) + \
                '-ntrain'+str(ntrain)+'-ntest' + str(ntest)
    plt.rcParams.update({'figure.max_open_warning': 0})
    

    for k in range(len(ntest)):
        # range of testing prevalences
        low = round(ntest[k] * 0.05)
        high = round(ntest[k] * 0.95)
        qp.environ["SAMPLE_SIZE"] = ntest[k]

        all_mae_results = np.zeros((len(methods_names), nreps * nbags))
        all_sqe_results = np.zeros((len(methods_names), nreps * nbags))
        all_mrae_results = np.zeros((len(methods_names), nreps * nbags))

        execution_times = np.zeros(len(methods_names))

        print()
        print('#Test examples ', ntest[k], 'Rep#', end=' ')

        for rep in range(nreps):

            print(rep+1, end=' ')

            x_train = np.vstack(((std1 * rng.randn(ntrain, 1) + mu1), (std2 * rng.randn(ntrain, 1) + mu2)))
            y_train = np.hstack((np.zeros(ntrain, dtype=int), np.ones(ntrain, dtype=int)))

            estimator_train = estimator
            estimator_train.fit(x_train, y_train)
            #predictions_train = estimator_train.predict_proba(x_train)

            for nmethod, method in enumerate(methods):
                method.fit(LabelledCollection(x_train,y_train))

            estimator_test = estimator_train

            for n_bag in range(nbags):
                ps = rng.randint(low, high, 1)
                ps = np.append(ps, [0, ntest[k]])
                ps = np.diff(np.sort(ps))

              
                x_test = np.vstack(((std1 * rng.randn(ps[0], 1) + mu1), (std2 * rng.randn(ps[1], 1) + mu2)))
                

                y_test = np.hstack((np.zeros(ps[0], dtype=int), np.ones(ps[1], dtype=int)))
   
                predictions_test = estimator_test.predict_proba(x_test)

                # Error
                classif_results[0, k] = classif_results[0, k] + zero_one_loss(np.array(y_test),
                                                                              np.argmax(predictions_test, axis=1))
                # Brier loss
                classif_results[1, k] = classif_results[1, k] + brier_score_loss(indices_to_one_hot(y_test, 2)[:, 0],
                                                                                 predictions_test[:, 0])

                prev_true = ps[1] / ntest[k]

                for nmethod, method in enumerate(methods):

                    t = time.process_time()
                    p_predicted = method.quantify(x_test)
                    elapsed_time = time.process_time()
                    execution_times[nmethod] = execution_times[nmethod] + elapsed_time - t

                    all_mae_results[nmethod, rep * nbags + n_bag] = ae(np.array([1-prev_true,prev_true]), p_predicted)
                    all_mrae_results[nmethod, rep * nbags + n_bag] = rae(np.array([1-prev_true,prev_true]), p_predicted)
                    all_sqe_results[nmethod, rep * nbags + n_bag] = se(np.array([1-prev_true,prev_true]), p_predicted)

                    mae_results[nmethod, k] = mae_results[nmethod, k] + all_mae_results[nmethod, rep * nbags + n_bag]
                    mrae_results[nmethod, k] = mrae_results[nmethod, k] + all_mrae_results[nmethod, rep * nbags + n_bag]
                    sqe_results[nmethod, k] = sqe_results[nmethod, k] + all_sqe_results[nmethod, rep * nbags + n_bag]

        execution_times = execution_times / (nreps * nbags)

        if save_all:
            file_times.write('\n%d, ' % ntest[k])
            for i in execution_times:
                file_times.write('%.5f, ' % i)

        if save_all:
            name_file = 'results/artificial/artificialBinary-all-mae-' + str(param) + '-' + est_name + \
                        '-rep' + str(nreps) + '-value' + str(ntrain) + '-ntest' + str(ntest[k]) + '.txt'
            file_all = open(name_file, 'w')

            for method_name in methods_names:
                file_all.write('%s,' % method_name)
            file_all.write('\n')
            for nrep in range(nreps):
                for n_bag in range(nbags):
                    for n_method in range(len(methods_names)):
                        file_all.write('%.5f, ' % all_mae_results[n_method, nrep * nbags + n_bag])
                    file_all.write('\n')
            file_all.close()

            name_file = 'results/artificial/artificialBinary-all-mrae-' + str(param) + '-' + est_name + \
                        '-rep' + str(nreps) + '-value' + str(ntrain) + '-ntest' + str(ntest[k]) + '.txt'
            file_all = open(name_file, 'w')

            for method_name in methods_names:
                file_all.write('%s,' % method_name)
            file_all.write('\n')
            for nrep in range(nreps):
                for n_bag in range(nbags):
                    for n_method in range(len(methods_names)):
                        file_all.write('%.5f, ' % all_mrae_results[n_method, nrep * nbags + n_bag])
                    file_all.write('\n')
            file_all.close()

            name_file = 'results/artificial/artificialBinary-all-sqe-' + str(param) + '-' + est_name + \
                        '-rep' + str(nreps) + '-value' + str(ntrain) + '-ntest' + str(ntest[k]) + '.txt'
            file_all = open(name_file, 'w')

            for method_name in methods_names:
                file_all.write('%s,' % method_name)
            file_all.write('\n')
            for nrep in range(nreps):
                for n_bag in range(nbags):
                    for n_method in range(len(methods_names)):
                        file_all.write('%.5f, ' % all_sqe_results[n_method, nrep * nbags + n_bag])
                    file_all.write('\n')
            file_all.close()

    mae_results = mae_results / (nreps * nbags)
    mrae_results = mrae_results / (nreps * nbags)
    sqe_results = sqe_results / (nreps * nbags)
    classif_results = classif_results / (nreps * nbags)

    file_avg = open('results/artificial/averages.txt', 'w')

    #Create a beautiful table with the results
    columns = ['#examples','Error']+methods_names+['BrierLoss']
    mae = pd.DataFrame(np.zeros((len(ntest),len(columns))),columns=columns)
    mrae = pd.DataFrame(np.zeros((len(ntest),len(columns))),columns=columns)
    sqe = pd.DataFrame(np.zeros((len(ntest),len(columns))),columns=columns)
    for index, number in enumerate(ntest):
        mae.iloc[index,0] = mrae.iloc[index,0] = sqe.iloc[index,0] = number
        mae.iloc[index,1] = mrae.iloc[index,1] = sqe.iloc[index,1] = classif_results[0,index]
        mae.iloc[index,2:2+len(methods_names)]=mae_results[:,index]
        mrae.iloc[index,2:2+len(methods_names)]=mrae_results[:,index]
        sqe.iloc[index,2:2+len(methods_names)]=sqe_results[:,index]
        mae.iloc[index,2+len(methods_names)]=mrae.iloc[index,2+len(methods_names)]=sqe.iloc[index,2+len(methods_names)]=classif_results[1,index]
    fmtformat=[".0f"]+[".4f"]*(len(methods_names)+2)
    file_avg.write('MAE\n')
    file_avg.write(tabulate.tabulate(mae, tablefmt='presto', headers=mae.columns.tolist(), numalign='right', stralign='center',showindex="never",floatfmt=fmtformat))
    file_avg.write('\n\nMRAE\n')
    file_avg.write(tabulate.tabulate(mrae, tablefmt='presto', headers=mrae.columns.tolist(), numalign='right', stralign='center',showindex="never",floatfmt=fmtformat))
    file_avg.write('\n\nSQE\n')
    file_avg.write(tabulate.tabulate(sqe, tablefmt='presto', headers=mrae.columns.tolist(), numalign='right', stralign='center',showindex="never",floatfmt=fmtformat))
    file_avg.close()

    if save_all:
        file_times.close()

    print(mae.mean())

# MAIN
# 1D synthetic experiments
run_experiment(est_name="PerfectProbClassifier", seed=42, param=1.0, ntrain=50, ntest=[100,200,300,400,500,600,700,800,900,1000],
            nreps=20, nbags=50, save_all=True)
