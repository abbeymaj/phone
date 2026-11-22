# Importing packages
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn import set_config
set_config(transform_output='pandas')
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from src.exception import CustomException

# Creating the class to find the best model
class FindBestModel():
    """
    The objective of this class is to find the best xgboost model. The class uses
    Optuna to find the best hyperparameters using stratified k-fold cross validation,
    and then trains a single final model on the entire training set using those
    hyperparameters.
    """
    # Defining the constructor for the class
    def __init__(
        self,
        train_set,
        target_set,
        n_trials=100,
        n_splits=5,
        seed=42
    ):
        """
        This is the constructor for the class. It initializes any
        necessary attributes for the class.
        """
        self.train_set = train_set
        self.target_set = target_set
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.seed = seed

    # Creating a method that returns the fixed (non-tuned) parameters for the model
    def _base_params(self):
        """
        This method returns the parameters that are not tuned by Optuna. These
        parameters are shared by the cross validation models and the final model.
        """
        return {
            'verbosity': 0,
            'eval_metric': 'auc',
            'objective': 'binary:logistic',
            'seed': self.seed
        }

    # Creating a method to suggest the hyperparameters for a trial
    def _suggest_params(self, trial):
        """
        This method uses the trial object to suggest the hyperparameters for the model.
        ================================================================================
        -------------------
        Parameters:
        -------------------
        trial : optuna.trial.Trial - This is the trial object that is used by Optuna.

        -------------------
        Returns:
        -------------------
        params : dict - The xgboost parameters for the trial.
        num_boost_round : int - The number of boosting rounds for the trial.
        ================================================================================
        """
        params = self._base_params()
        params['booster'] = trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart'])
        params['lambda'] = trial.suggest_float('lambda', 1e-8, 1.0, log=True)
        params['alpha'] = trial.suggest_float('alpha', 1e-8, 1.0, log=True)

        if params['booster'] in ('gbtree', 'dart'):
            params['max_depth'] = trial.suggest_int('max_depth', 1, 10)
            params['eta'] = trial.suggest_float('eta', 1e-2, 0.5, log=True)
            params['gamma'] = trial.suggest_float('gamma', 1e-8, 1.0, log=True)
            params['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
        if params['booster'] == 'dart':
            params['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
            params['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
            params['rate_drop'] = trial.suggest_float('rate_drop', 1e-8, 1.0, log=True)
            params['skip_drop'] = trial.suggest_float('skip_drop', 1e-8, 1.0, log=True)

        num_boost_round = trial.suggest_int('num_boost_round', 50, 500)
        return params, num_boost_round

    # Creating the objective function for the class
    def objective(self, trial):
        """
        This method defines the objective function used by Optuna. For the suggested
        hyperparameters, it trains a model on each cross validation fold and returns
        the mean roc auc score across the validation folds. The method does not train
        the final model; that is done once in create_study.
        ================================================================================
        -------------------
        Parameters:
        -------------------
        trial : optuna.trial.Trial - This is the trial object that is used by Optuna.

        -------------------
        Returns:
        -------------------
        mean_auc : float - This is the mean roc auc score across the validation folds.
        ================================================================================
        """
        try:
            params, num_boost_round = self._suggest_params(trial)
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            auc_scores = []
            for fold, (train_idx, val_idx) in enumerate(skf.split(self.train_set, self.target_set)):
                X_trn, X_val = self.train_set.iloc[train_idx], self.train_set.iloc[val_idx]
                y_trn, y_val = self.target_set.iloc[train_idx], self.target_set.iloc[val_idx]
                dtrain = xgb.DMatrix(X_trn, label=y_trn)
                dval = xgb.DMatrix(X_val, label=y_val)

                # Training the xgboost model
                bst = xgb.train(params, dtrain, num_boost_round=num_boost_round)

                # Predicting on the validation set
                y_pred = bst.predict(dval)
                auc_scores.append(roc_auc_score(y_val, y_pred))

                # Reporting the running mean auc after each fold so the pruner can
                # stop unpromising trials early
                trial.report(float(np.mean(auc_scores)), step=fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            # Calculating the mean roc auc score
            return float(np.mean(auc_scores))

        except optuna.TrialPruned:
            # Optuna needs to receive this exception to mark the trial as pruned
            raise
        except Exception as e:
            raise CustomException(e, sys) from e

    # Creating the method to locate the best model using the objective function
    def create_study(self):
        """
        This method uses the study object to find the best hyperparameters, and then
        trains the final model on the entire training set using those hyperparameters.
        ================================================================================
        -------------------
        Returns:
        -------------------
        best_model : xgboost.core.Booster - This is the best model for the dataset.
        best_params : dict - This is the best parameters for the best model.
        ================================================================================
        """
        try:
            # Defining the sampler and the pruner for the Optuna study
            sampler = optuna.samplers.TPESampler(seed=self.seed)
            pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
            
            # Creating the study object and finding the best params
            study = optuna.create_study(sampler=sampler, pruner=pruner, direction='maximize')
            study.optimize(self.objective, n_trials=self.n_trials)
            best_params = study.best_params

            # Training the final model on the entire training set with the best parameters
            final_params = {**self._base_params(), **best_params}
            num_boost_round = final_params.pop('num_boost_round')
            dtrain_full = xgb.DMatrix(self.train_set, label=self.target_set)
            best_model = xgb.train(final_params, dtrain_full, num_boost_round=num_boost_round)
            return best_model, best_params

        except Exception as e:
            raise CustomException(e, sys) from e
