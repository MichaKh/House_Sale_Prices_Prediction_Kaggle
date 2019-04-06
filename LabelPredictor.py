import time
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score


class LabelPredictor:
    eval_classifier = None
    train_X = None
    train_y = None

    def __init__(self, train_X, train_y, eval_classifier):
        self.eval_classifier = eval_classifier
        self.train_X = train_X
        self.train_y = train_y

    def train_classifier_stratified_cv_grid_search(self, classifier_name, classifier, grid_search, params_grid=None):
        print('Training classifier: {}'.format(classifier_name))
        start_time = time.time()
        if classifier is None:
            return None
        else:
            if grid_search and params_grid:
                grid_classifier = GridSearchCV(classifier, param_grid=params_grid, scoring='accuracy', n_jobs=-1, verbose=1, cv=StratifiedKFold(n_splits=5), refit=True)
                grid_classifier.fit(self.train_X, self.train_y)
                classifier = grid_classifier.best_estimator_
            else:
                classifier.fit(self.train_X, self.train_y)
            print("Training {} classifier completed in {} seconds".format(classifier_name, (time.time() - start_time)))
        return classifier

    def train_classifier_nested_stratified_cv_grid_search(self, classifier_name, classifier, grid_search, params_grid=None):
        print('Training classifier: {}'.format(classifier_name))
        start_time = time.time()
        if classifier is None:
            return None
        else:
            if grid_search and params_grid:
                grid_classifier = GridSearchCV(classifier, param_grid=params_grid, scoring='accuracy', n_jobs=-1, verbose=1, cv=StratifiedKFold(n_splits=5), refit=True)
                grid_classifier.fit(self.train_X, self.train_y)
                classifier = grid_classifier.best_estimator_
                print(f'Best grid search score for classifier {classifier_name} is: {grid_classifier.best_score_}')
                cv_scores = cross_val_score(classifier, X=self.train_X, y=self.train_y, cv=StratifiedKFold(n_splits=10))
                print(f'Average accuracy on train is: {np.mean(cv_scores)}')
            else:
                classifier.fit(self.train_X, self.train_y)
            print("Training {} classifier completed in {} seconds".format(classifier_name, (time.time() - start_time)))
        return classifier

    def train_classifier_stratified_cv(self, classifier_name, classifier, cv_folds=10):
        print('Training classifier: {}'.format(classifier_name))
        start_time = time.time()
        if classifier is None:
            return None
        else:
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            print(f'Training classifier: {classifier_name} with {skf.n_splits} stratified folds')
            cv_scores = cross_val_score(classifier, self.train_X, self.train_y, cv=skf, n_jobs=4, verbose=1)
            print(f'Training classifier: {classifier_name} with {skf.n_splits} stratified folds...\n'
                  f'---- VC scores are: {cv_scores}')
            print(f'Average accuracy on train is: {np.mean(cv_scores)}')
            classifier.fit(self.train_X, self.train_y)
            print("Training {} classifier completed in {} seconds".format(classifier_name, (time.time() - start_time)))
        return classifier

    @staticmethod
    def predict_with_classifier(test_X, classifier_name, classifier):
        print('Predicting with classifier: {}'.format(classifier_name))
        start_time = time.time()
        predictions = classifier.predict(test_X)
        print("Predicting with {} classifier completed in {} seconds".format(classifier_name, (time.time() - start_time)))
        return pd.Series(predictions)

    @staticmethod
    def get_grid_search_type(classifier_name):
        ensem_param_grid = {'n_estimators': [100, 200, 500, 1000, 2000],
                            'max_depth': [4, 5, 6, 8],
                            'max_features': [0.8, 0.5, 0.2, 0.1]}
        if 'forest' in classifier_name.lower():
            return ensem_param_grid
        elif 'gb' in classifier_name.lower():
            ensem_param_grid['learning_rate'] = [0.2, 0.1, 0.05, 0.01]
        return ensem_param_grid
