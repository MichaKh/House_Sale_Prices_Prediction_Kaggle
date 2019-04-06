import pandas as pd
from FeaturesUtils import *


class PreProcessor:

    def __init__(self, raw_train_df, raw_test_df, cols_to_consider, target_feature):
        self.clean_train_df = raw_train_df
        self.clean_test_df = raw_test_df
        self.target_feature = target_feature
        self.cols_to_consider = cols_to_consider

    @staticmethod
    def fill_missing_values(df: pd.DataFrame):
        if df.isnull().values.any():
            print("##### Filling missing NA values with mean and mode respectively...")
            for col in df.columns:
                if df[col].dtype not in [str, object]:
                    df = df.fillna(df.mean())
                else:
                    df = df.fillna(df[col].mode().iloc[0])
        else:
            print("##### No NaN values have been found...")
        return df

    @staticmethod
    def drop_index_col(df: pd.DataFrame):
        index_col_present = "id" in list(map(str.lower, list(df.columns)))
        if index_col_present:
            print("##### Dropping index column...")
            return df.drop(labels=['Id'], axis=1)
        else:
            return df

    @staticmethod
    def drop_empty_rows(df: pd.DataFrame, na_count_thr: int):
        na_row_count = df.isnull().sum(axis=1)
        dropped_n_rows = len(na_row_count[na_row_count > na_count_thr])
        print(f'##### Dropped {dropped_n_rows} rows...')
        return df.loc[na_row_count[df.index] <= na_count_thr]

    @staticmethod
    def drop_empty_cols(df: pd.DataFrame, na_count_thr: int):
        na_col_count = df.isnull().sum(axis=0)
        cols_to_keep = list(na_col_count[na_col_count <= na_count_thr].index)
        dropped_cols = [col for col in df.columns if col not in cols_to_keep]
        print(f'##### Dropped {len(dropped_cols)} columns: {dropped_cols}...')
        return df.loc[:, cols_to_keep]

    @staticmethod
    def filter_existing_features(df, col_to_consider, target_feature):
        if len(df.columns) == len(col_to_consider):
            print(f'##### Considering all {len(df.columns) - 1} feature columns...')
            return df
        else:
            print(f'##### Considering only specified {len(col_to_consider)} feature columns: {col_to_consider}')
        return df.loc[:, col_to_consider + [target_feature]]

    @staticmethod
    def handle_features(df: pd.DataFrame):
        # HouseStyle: Style of dwelling
        df['NumOfStories'] = df['HouseStyle'].apply(lambda x: get_num_of_stories(x))
        # YearBuilt: Construction year - house age
        df['HouseAge'] = df['YearBuilt'].apply(lambda x: get_house_age(x))
        # Was the house renovated in  the last 10 years
        df['HouseWasRenovated'] = df[['YearRemodAdd', 'YearBuilt']].apply(lambda x: was_the_house_renovated(x['YearBuilt'], x['YearRemodAdd']), axis=1)
        # Does the property have an alley?
        df['HouseHasAlley'] = df['Alley'].apply(lambda x: has_house_feature(x))
        # Adjacent to arterial location (road, transportation facilities)
        df['AdjacentArterial'] = df[['Condition1', 'Condition2']].apply(lambda x: adjacent_to_artery(x['Condition1'], x['Condition2']), axis=1)
        # Adjacent to off-site locations (parks, greenbelts)
        df['AdjacentOffSites'] = df[['Condition1', 'Condition2']].apply(lambda x: adjacent_to_off_sites(x['Condition1'], x['Condition2']), axis=1)
        # Has basement
        df['HasBasement'] = df['BsmtCond'].apply(lambda x: has_house_feature(x))
        # Has Pool
        df['HasPool'] = df['PoolQC'].apply(lambda x: has_house_feature(x))
        # Has fence
        # df['HasFence'] = df['Fence'].apply(lambda x: has_house_feature(x))
        # What time of year the house was sold? (winter, summer..)
        df['HouseSaleSeason'] = df['MoSold'].apply(lambda x: sale_season(x))
        # House sold in the last 3 years?
        df['SoldInLast3Years'] = df['YrSold'].apply(lambda x: sold_in_last_3years(x))

        df = df.drop(labels=['YrSold', 'MoSold', 'YearRemodAdd', 'YearBuilt', 'HouseStyle', 'Alley', 'Condition1', 'Condition2'], axis=1)
        return df

    def pre_process_data(self):
        print('Performing pre-process...')
        return (self.clean_train_df
                .pipe(PreProcessor.drop_index_col)
                .pipe(PreProcessor.drop_empty_cols, int(0.7 * self.clean_train_df.shape[0]))
                .pipe(PreProcessor.drop_empty_rows, int(0.7 * self.clean_train_df.shape[1]))
                .pipe(PreProcessor.fill_missing_values)
                .pipe(PreProcessor.filter_existing_features, self.cols_to_consider, self.target_feature)
                .pipe(PreProcessor.handle_features)
                )
