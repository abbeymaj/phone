# Importing packages
import pytest
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.datasets import make_classification
from src.components.find_best_model import FindBestModel

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Creating a fixture to give a small synthetic binary classification dataset
@pytest.fixture(scope="module")
def create_dataset():
    X, y = make_classification(n_samples=200, n_features=6, random_state=42)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y = pd.Series(y, name="target")
    return X, y

# Verifying that the objective function returns a valid mean auc score
@pytest.mark.unit
def test_objective_returns_valid_auc(create_dataset):
    X, y = create_dataset
    finder = FindBestModel(X, y, n_trials=1)
    study = optuna.create_study(direction='maximize')
    study.optimize(finder.objective, n_trials=1)
    assert 0.0 <= study.best_value <= 1.0

# Verifying that the objective function does not train or store a final model
@pytest.mark.unit
def test_objective_does_not_store_model(create_dataset):
    X, y = create_dataset
    finder = FindBestModel(X, y, n_trials=1)
    study = optuna.create_study(direction='maximize')
    study.optimize(finder.objective, n_trials=1)
    assert study.best_trial.user_attrs == {}

# Verifying that a pruned trial is marked as pruned rather than crashing the study
@pytest.mark.unit
def test_objective_lets_pruning_through(create_dataset, monkeypatch):
    X, y = create_dataset
    finder = FindBestModel(X, y, n_trials=1)
    monkeypatch.setattr(optuna.trial.Trial, "should_prune", lambda self: True)
    study = optuna.create_study(direction='maximize')
    study.optimize(finder.objective, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.PRUNED

# Verifying that create_study returns a booster trained on the full dataset with the best params
@pytest.mark.unit
def test_create_study_returns_final_model(create_dataset):
    X, y = create_dataset
    finder = FindBestModel(X, y, n_trials=3)
    best_model, best_params = finder.create_study()
    assert isinstance(best_model, xgb.Booster)
    assert 'booster' in best_params
    assert 'num_boost_round' in best_params
    preds = best_model.predict(xgb.DMatrix(X))
    assert len(preds) == len(X)
    assert ((preds >= 0) & (preds <= 1)).all()
