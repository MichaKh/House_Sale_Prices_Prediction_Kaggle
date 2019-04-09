from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd


def prepare_data(data_df, class_col, reg_encoding_features, one_hot_encoding_features, ordinal_encoding_features):
    """
    Split the data to features and label column, i.e., X and y.
    Encode the categorical features to numeric encoding (to fit "sklearn" implementations)
    Only provided features will be encoded as one-hot-vectors, rest are encoding with regular categorical encoding
    :param ordinal_encoding_features:
    :param one_hot_encoding_features: List of features to encode to one-hot vectors
    :param train_data_df: Input data
    :param class_col: Column name representing the class label
    :param reg_encoding_features: Column names representing the data features
    :return: Dataframes of X and y.
    """
    data_X, data_y = split_features_and_label(data_df, class_col)
    prepared_X = pd.DataFrame()
    if one_hot_encoding_features:
        one_hot_enc = encode_one_hot_features(data_X, one_hot_encoding_features)
        prepared_X = pd.concat([prepared_X, one_hot_enc], axis=1)
    if ordinal_encoding_features:
        ordinal_enc = encode_ordinal_map_features(data_X, ordinal_encoding_features)
        prepared_X = pd.concat([prepared_X, ordinal_enc], axis=1)

    if reg_encoding_features:
        reg_enc = encode_features(data_X, reg_encoding_features)
        prepared_X = pd.concat([prepared_X, reg_enc], axis=1)

    return prepared_X, data_y


def split_features_and_label(data_df, class_col):
    if class_col in data_df.columns:
        train_y = data_df[class_col]
        features_cols = [col for col in data_df.columns if col != class_col]
    else:
        train_y = pd.Series([0])
        features_cols = data_df.columns
    train_X = data_df.loc[:, data_df.columns.isin(features_cols)]
    assert len(train_y) == train_X.shape[0]
    return train_X, train_y


def encode_features(train_X: pd.DataFrame, features_to_encode):
    le = LabelEncoder()
    encoded_df = pd.DataFrame()

    for col in features_to_encode:
        if train_X[col].dtype not in [np.float64, np.int64, np.int32, np.float32]:  # not numeric
            encoded_df[col] = le.fit_transform(train_X[col])
    return encoded_df


def encode_one_hot_features(train_X: pd.DataFrame, features_to_encode):
    encoded_df = pd.DataFrame()
    for col in features_to_encode:
        if train_X[col].dtype not in [np.float64, np.int64, np.int32, np.float32]:  # not numeric
            encoded = pd.get_dummies(train_X[col], prefix=col, drop_first=False)
            encoded_df = pd.concat([encoded_df, encoded], axis=1)
        else:
            encoded_df = pd.concat([encoded_df, train_X[col]], axis=1)
    return encoded_df


def encode_ordinal_map_features(data_df: pd.DataFrame, features_to_encode_map: dict):
    encoded_df = pd.DataFrame()
    for col, mapping in features_to_encode_map.items():
        encoded_df[col] = data_df[col].replace(mapping)
    return encoded_df
