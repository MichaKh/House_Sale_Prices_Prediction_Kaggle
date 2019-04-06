import pandas as pd
import numpy as np


class DataLoader:

    def __init__(self, data_root_dir, train_path, test_path):
        self.data_root_dir = data_root_dir
        self._test_path = test_path
        self._train_path = train_path

        self._raw_train_df = pd.DataFrame()
        self._raw_test_df = pd.DataFrame()

    def load_csv_data(self):
        if self._train_path:
            try:
                df = pd.read_csv(self._train_path, header=0, keep_default_na=False)
                self._raw_train_df = df
                # self._raw_train_df = df.replace('NA', np.NaN)
            except Exception:
                print(f"Cannot read file in path: {self._train_path}")
        if self._test_path:
            try:
                df = pd.read_csv(self._test_path, header=0, keep_default_na=False)
                self._raw_test_df = df
                # self._raw_test_df = df.replace('NA', np.NaN)
            except Exception:
                print(f"Cannot read file in path: {self._train_path}")
        return self._raw_train_df, self._raw_test_df

    def print_statistics(self):
        # Statistics for train data
        index_col_present = "id" in list(map(str.lower, list(self._raw_train_df.columns)))
        train_n_rows = self._raw_train_df.shape[0]
        train_n_cols = self._raw_train_df.shape[1] - int(index_col_present)
        test_n_cols = self._raw_train_df.shape[1] - int(index_col_present)
        numeric_cols = [col for col in self._raw_train_df.columns if self._raw_train_df[col].dtype not in [str, object]]
        categorical_cols = [col for col in self._raw_train_df.columns if self._raw_train_df[col].dtype in [str, object]]
        print(f"Statistics for dataframe in path: {self._train_path}:")
        print(f"-INFO- Number of rows: {train_n_rows }")
        print(f"-INFO- Number of columns: {train_n_cols}")
        print(f"-INFO- Number of numeric columns (before pre-processing): {len(numeric_cols) - int(index_col_present)}")
        print(f"-INFO- Number of categorical columns (before pre-processing): {len(categorical_cols)}")
        for col in self._raw_train_df.columns:
            col_na_count = self._raw_train_df[col].isnull().sum()
            print(f"$$$$$ Number of NaN values in columns <{col}> is: {col_na_count}")

        if train_n_cols == test_n_cols:
            print(f"--OK-- Train and test files have the same number of columns!")
        else:
            print(f"--WARN-- Train and test files have different number of columns!")
            assert(train_n_cols == test_n_cols)




