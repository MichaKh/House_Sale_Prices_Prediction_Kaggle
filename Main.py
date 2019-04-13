from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from DataLoader import DataLoader
from Predictor import Predictor
from FeaturesUtils import features_ordinal_mappings, one_hot_encod_features, no_enc_features
from ModelBuilderUtils import prepare_data
from PreProcessor import PreProcessor

data_dir_root = "./"
data_train_file = "Data/train.csv"
data_test_file = "Data/test.csv"

eval_classifiers = {
    'RandomForestClassifier': RandomForestRegressor(n_estimators=1000, max_depth=6, min_samples_split=2,
                                                    criterion='mse', random_state=42),
    'XGBoost': XGBRegressor(n_estimators=1000, learning_rate=0.08, gamma=0, subsample=0.75,
                            colsample_bytree=1, max_depth=6, colsample_bylevel=0.6, reg_alpha=1),
    'ExtraTreeReggresor': ExtraTreesRegressor(n_estimators=1000, criterion='mse', max_depth=6, min_samples_leaf=4),
    'GradientBoostingRegressor': GradientBoostingRegressor(learning_rate=0.2, n_estimators=1000, max_depth=6, subsample=0.3)
}

eval_classifiers_params_grid = {
    'RandomForestClassifier': {'n_estimators': [100, 200, 500, 1000, 2000],
                               'max_depth': [4, 5, 6],
                               'max_features': [0.8, 0.5, 0.2, 0.1]},
    'XGBoost': {'n_estimators': [100, 200, 500, 1000, 2000],
                'max_depth': [4, 5, 6],
                'learning_rate': [0.01, 0.03, 0.1]},
    'ExtraTreeReggresor': {'n_estimators': [100, 200, 500, 1000, 2000],
                           'max_depth': [4, 5, 6]},
    'GradientBoostingRegressor': {'n_estimators': [100, 200, 500, 1000, 2000],
                                  'max_depth': [4, 5, 6],
                                  'learning_rate': [0.01, 0.03, 0.1]}
}
cols_to_consider = ['MoSold', 'YrSold', 'Fence', 'PoolQC', 'BsmtCond', 'BldgType', 'Condition1', 'Condition2',
                    'OverallCond', 'OverallQual', 'MSZoning', 'Alley', 'LotShape', 'HouseStyle', 'CentralAir',
                    'RoofStyle', 'LandSlope', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'Electrical', 'SaleCondition',
                    'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BsmtQual',
                    'ExterCond', 'BedroomAbvGr', 'BsmtUnfSF', 'TotalBsmtSF', 'Foundation', 'PavedDrive',
                    'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea']


def run():
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
                                    ordinal_encoding_features=features_ordinal_mappings,
                                    no_enc_features=no_enc_features)

    test_X, test_y = prepare_data(pre_processor.clean_test_df,
                                  class_col=pre_processor.target_feature,
                                  reg_encoding_features=[],
                                  one_hot_encoding_features=one_hot_encod_features,
                                  ordinal_encoding_features=features_ordinal_mappings,
                                  no_enc_features=no_enc_features)
    evaluator = Predictor(train_X, train_y, test_X, test_y, eval_classifiers, eval_classifiers_params_grid)

    all_predictions, final_prediction = evaluator.build_models(grid_search=False)
    evaluation_df = evaluator.save_predictions_to_df(all_predictions, final_prediction)
    submission_df = evaluator.save_predictions_for_submission(evaluation_df, id_col=pre_processor.raw_test_df['Id'])
    evaluation_df.to_csv("test_evaluation_results.csv", index=False)
    submission_df.to_csv("test_submission.csv", index=False)


if __name__ == '__main__':
    run()