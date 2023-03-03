from rw.to_csv import read_csv

def test1():
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

def test2():
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

def test3():
    print("="*60 + "\n")
    dataset_name = "titanic_dataset.csv"
    print("\033[93m{}\033[00m".format(f"Lendo Dataset {dataset_name}"))
    ds = read_csv(f"../datasets/{dataset_name}",",",features=True,label="Survived")
    print("Leitura completa.\n")
    print("\033[93m{}\033[00m".format("Obtendo estatísticas..."))
    print("\033[93m{}\033[00m".format("\nResumo: "))
    print(ds.summary())

def main():
    test3()

    

     

if __name__ == "__main__" :
    main()