from sklearn.ensemble import RandomForestClassifier

from DataLoader import DataLoader
from PreProcessor import PreProcessor

data_dir_root = "./"
data_train_file = "Data/train.csv"
data_test_file = "Data/test.csv"

eval_classifiers = {
    'RandomForestClassifier': RandomForestClassifier(n_estimators=2000, max_depth=4, min_samples_split=100,
                                                     random_state=42, class_weight={0: 0.65, 1: 0.35}),
}

eval_classifiers_params_grid = {
    'TreeClassifier': {'max_depth': [4, 5, 6]},
    'AdaBoost': {'n_estimators': [100, 200, 500, 1000, 2000],
                 'learning_rate': [0.2, 0.1, 0.05, 0.01]},
    'LogisticRegression': {'penalty': ['l1', 'l2']},
    'RandomForestClassifier': {'n_estimators': [100, 200, 500, 1000, 2000],
                               'max_depth': [4, 5, 6],
                               'max_features': [0.8, 0.5, 0.2, 0.1]},
    'GBTrees': {'n_estimators': [100, 500, 1000, 2000],
                'max_depth': [4, 5, 6],
                'max_features': [0.8, 0.5, 0.2, 0.1],
                'learning_rate': [0.2, 0.1, 0.05, 0.01]},
    'xgboost': {'n_estimators': [100, 500, 1000, 2000],
                'max_depth': [4, 5, 6],
                'max_features': [0.8, 0.5, 0.2, 0.1],
                'learning_rate': [0.2, 0.1, 0.05, 0.01]},
    'KNN': {'n_neighbors': [2, 3, 4, 5]},
    'SVM': {'gamma': [0.001, 0.01, 0.1, 1],
            'C': [1, 10, 50, 100, 200]},
    'GBC': {'n_estimators': [100, 500, 1000, 2000],
            'max_depth': [4, 5, 6, 8],
            'max_features': [0.8, 0.5, 0.2, 0.1],
            'learning_rate': [0.2, 0.1, 0.05, 0.01]}
}

data_loader = DataLoader(data_dir_root, data_train_file, data_test_file)
raw_train_df, raw_test_df = data_loader.load_csv_data()
data_loader.print_statistics()

pre_processor = PreProcessor(raw_train_df,
                             raw_test_df,
                             cols_to_consider=['MoSold', 'YrSold', 'Fence', 'PoolQC', 'BsmtCond', 'BldgType', 'Condition1', 'Condition2', 'OverallCond', 'OverallQual', 'MSZoning', 'Alley', 'LotShape', 'HouseStyle', 'CentralAir', 'RoofStyle', 'LandSlope', 'LotArea', 'YearBuilt', 'YearRemodAdd'],
                             # cols_to_consider=raw_train_df.columns[0:-1],
                             target_feature='SalePrice')
pre_processor.pre_process_data()
