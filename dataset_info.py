import quapy as qp

print("\\begin{table}[ht]")
print("\\centering")
print("\\footnotesize")
print("\\begin{tabular}{lccc}")
print("\\hline")
print("\\textbf{Dataset} & \\textbf{Instances ($n$)} & \\textbf{Features ($d$)} & \\textbf{Classes ($K$)} \\\\")
print("\\hline")

# Iterate over the UCI multiclass datasets
for name in qp.datasets.UCI_MULTICLASS_DATASETS:
    try:
        # Load the dataset
        dataset = qp.datasets.fetch_UCIMulticlassDataset(name)
        
        # Calculate total instances (training + test splits)
        n_instances = len(dataset.training) + len(dataset.test)
        
        # Get dimensionality (features)
        n_features = dataset.training.X.shape[1]
        
        # Get number of classes
        n_classes = len(dataset.classes_)
        
        # Format the dataset name for LaTeX (escaping underscores)
        latex_name = name.replace('_', '\\_')
        
        print(f"{latex_name} & {n_instances} & {n_features} & {n_classes} \\\\")
        
    except Exception as e:
        print(f"% Error processing {name}: {e}")

print("\\hline")
print("\\end{tabular}")
print("\\caption{Characteristics of the UCI multiclass datasets used in the experiments, including total number of instances ($n$), features ($d$), and classes ($K$).}")
print("\\label{tab:datasets_stats}")
print("\\end{table}")