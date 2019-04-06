from sklearn.ensemble import RandomForestRegressor

from DataLoader import DataLoader
from LabelPredictorUtils import prepare_data
from PreProcessor import PreProcessor

data_dir_root = "./"
data_train_file = "Data/train.csv"
data_test_file = "Data/test.csv"

eval_classifiers = {
    'RandomForestClassifier': RandomForestRegressor(n_estimators=2000, max_depth=4, min_samples_split=2,
                                                    criterion='mse', random_state=42),
}

eval_classifiers_params_grid = {
    'RandomForestClassifier': {'n_estimators': [100, 200, 500, 1000, 2000],
                               'max_depth': [4, 5, 6],
                               'max_features': [0.8, 0.5, 0.2, 0.1]}
}
cols_to_consider = ['MoSold', 'YrSold', 'Fence', 'PoolQC', 'BsmtCond', 'BldgType', 'Condition1', 'Condition2',
                    'OverallCond', 'OverallQual', 'MSZoning', 'Alley', 'LotShape', 'HouseStyle', 'CentralAir',
                    'RoofStyle', 'LandSlope', 'LotArea', 'YearBuilt', 'YearRemodAdd']
data_loader = DataLoader(data_dir_root, data_train_file, data_test_file)
raw_train_df, raw_test_df = data_loader.load_csv_data()
data_loader.print_statistics()

pre_processor = PreProcessor(raw_train_df,
                             raw_test_df,
                             cols_to_consider=cols_to_consider,
                             # cols_to_consider=raw_train_df.columns[0:-1],
                             target_feature='SalePrice')
pre_processor.pre_process_data()
train_X, train_y = prepare_data(pre_processor.clean_train_df,
                                class_col=pre_processor.target_feature,
                                reg_encoding_features=[],
                                one_hot_encoding_features=['SoldInLast3Years', 'HouseSaleSeason', 'HasFence', 'HasPool',
                                                           'HasBasement',
                                                           'AdjacentOffSites', 'AdjacentArterial', 'HouseHasAlley',
                                                           'HouseWasRenovated', 'RoofStyle',
                                                           'CentralAir', 'MSZoning', 'BldgType'],
                                ordinal_encoding_features={'NumOfStories': {'1': 0, '1.5': 1, '2': 2, '2.5': 3, '3': 4},
                                                           'LandSlope': {'Gtl': 0, 'Mod': 1, 'Sev': 2},
                                                           'LotShape': {'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3},
                                                           'BsmtCond': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4,
                                                                        'Ex': 5},
                                                           'Fence': {'NA': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3,
                                                                     'GdPrv': 4},
                                                           'PoolQC': {'NA': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}})
