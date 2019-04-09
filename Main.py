from sklearn.ensemble import RandomForestRegressor

from DataLoader import DataLoader
from Evaluator import Evaluator
from FeaturesUtils import features_ordinal_mappings, one_hot_encod_features
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
                                one_hot_encoding_features=one_hot_encod_features,
                                ordinal_encoding_features=features_ordinal_mappings)

test_X, test_y = prepare_data(pre_processor.clean_test_df,
                              class_col=pre_processor.target_feature,
                              reg_encoding_features=[],
                              one_hot_encoding_features=one_hot_encod_features,
                              ordinal_encoding_features=features_ordinal_mappings)
evaluator = Evaluator(train_X, train_y, test_X, test_y, eval_classifiers, eval_classifiers_params_grid)

all_predictions, final_prediction = evaluator.build_models(grid_search=False)
evaluation_df = evaluator.save_predictions_to_df(all_predictions, final_prediction)
submission_df = evaluator.save_predictions_for_submission(evaluation_df, id_col=pre_processor.raw_test_df['Id'])
evaluation_df.to_csv("test_evaluation_results.csv", index=False)
submission_df.to_csv("test_submission.csv", index=False)
