import pandas as pd

from data.dataset import Dataset

# falta terminar
def export_dataset(self, file_path: str, method: str = 'CSV'):
        """
        Exports the dataset to a CSV or TSV file.

        Parameters
        ----------
        file_path: str
            The file path to the CSV or TSV file.
        method: str (1)
            The export method: 'CSV' or 'TSV'
        """
        if method.upper() == 'CSV':
            delimiter = ','
        elif method.upper() == 'TSV':
            delimiter = '\t'
        else:
            raise ValueError("Invalid method")

        df = pd.DataFrame(self.X, columns=self.features)
        if self.y is not None:
            df[self.label] = self.y.reshape(-1)
            df = df[[*self.features, self.label]]

        df.to_csv(file_path, index=False, sep=delimiter)
        