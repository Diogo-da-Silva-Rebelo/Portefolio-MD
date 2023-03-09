from rw.to_csv import *
from data.feature_selection.select_k_best import *
from data.feature_selection.variance_th import *
from system.pycache import delete_cache

def test_numeric_data_with_header() -> None:
    print("="*60 + "\n")
    dataset_name = "iris_missing_data.csv"
    print("\033[93m{}\033[00m".format(f"Lendo Dataset {dataset_name}"))
    ds = read_csv(f"../datasets/{dataset_name}", ',',features=True,label=True)
    print("Leitura completa.\n")
    print("\033[93m{}\033[00m".format("Obtendo estatísticas..."))
    print(f"Mean:   {ds.get_mean()}")
    print(f"Var:    {ds.get_var()}")
    print(f"Median: {ds.get_median()}")
    print(f"Max:    {ds.get_max()}")
    print(f"Min:    {ds.get_min()}")
    print(f"Null:   {ds.count_nulls()}")
    print("\033[93m{}\033[00m".format("\nResumo: "))
    print(ds.summary())

    print("\nReplacing null values...\n")
    ds.replace_nulls()
    print(ds.summary())

def test_numeric_data_without_header() -> None:
    print("="*60 + "\n")
    dataset_name = "breast-bin.data"
    print("\033[93m{}\033[00m".format(f"Lendo Dataset {dataset_name}"))
    ds = read_csv(f"../datasets/{dataset_name}")
    print("Leitura completa.\n")
    print("\033[93m{}\033[00m".format("Obtendo estatísticas..."))
    print(f"Mean:   {ds.get_mean()}")
    print(f"Var:    {ds.get_var()}")
    print(f"Median: {ds.get_median()}")
    print(f"Max:    {ds.get_max()}")
    print(f"Min:    {ds.get_min()}")
    print(f"Null:   {ds.count_nulls()}")
    print("\033[93m{}\033[00m".format("\nResumo: "))
    print(ds.summary())

def test_discrete_data_with_header() -> None:
    print("="*60 + "\n")
    dataset_name = "titanic_dataset.csv"
    print("\033[93m{}\033[00m".format(f"Lendo Dataset {dataset_name}"))
    ds = read_csv(f"../datasets/{dataset_name}",",",features=True,label="Survived")
    print("Leitura completa.\n")
    print("\033[93m{}\033[00m".format("Obtendo estatísticas..."))
    print("\033[93m{}\033[00m".format("\nResumo: "))
    print(ds.summary())

    print("\nReplacing null values...\n")
    ds.replace_nulls()
    print(ds.summary())
    write_csv(filename="../datasets/titanic_no_missing_data.csv",dataset=ds, features=True,label=True)

def test_variance_th_and_k_features() -> None:
    print("="*65 + "\n")
    dataset_name = "iris_missing_data.csv"
    print("\033[93m{}\033[00m".format(f"Lendo Dataset {dataset_name}"))
    ds = read_csv(f"../datasets/{dataset_name}", ',',features=True,label=True)
    print("Leitura completa.\n")
    ds.replace_nulls()

    th = 0.4
    print("\033[93m{}\033[00m".format(f"Seleção de features por Variance Threshold [th = {th}]"))
    varianceth = VarianceThreshold(threshold=th)
    varianceth.fit(dataset=ds)
    transformed_datset = varianceth.transform(dataset=ds,inline=False)

    print(f"Selected Features: {transformed_datset.features}\n")
    write_csv(filename="../datasets/iris_variance_th.csv",dataset=transformed_datset, features=True,label=True)

    k = 2
    print("\033[93m{}\033[00m".format(f"Seleção de features por K Best [k = {k}]"))
    kBest = SelectKBest(k=k)
    kBest.fit(dataset=ds)
    transformed_k_datset = kBest.transform(dataset=ds,inline=False)
    print(f"Selected Features: {transformed_k_datset.features}\n")
    write_csv(filename="../datasets/iris_k_best.csv",dataset=transformed_k_datset, features=True,label=True)
    print("="*65 + "\n")
    delete_cache() # remover arquivos de __pycache__

def main():
    test_variance_th_and_k_features()


if __name__ == "__main__" :
    main()