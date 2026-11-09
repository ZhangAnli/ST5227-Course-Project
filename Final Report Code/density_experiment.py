"""Resumable paired regression workers for Project-YiHern-density.ipynb."""
import os, time, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance
from threadpoolctl import threadpool_limits

NAMES = ['Naive', 'OLS', 'Ridge', 'Lasso', 'GAM', 'DT', 'RF', 'GBT']
CONDITIONS = ['Baseline', 'Density weighted']
from sklearn.base import clone
from scipy.stats import gaussian_kde
from functools import lru_cache

# Fixed before evaluation; these settings are recorded in the run manifest.
WEIGHT_CONFIG = dict(power=1.0, cap=5.0, bandwidth='scott')

@lru_cache(maxsize=64)
def _grade_density(target_bytes, bandwidth):
    # Reuse only identical training-target arrays across candidates and models.
    y = np.frombuffer(target_bytes, dtype=np.float64)
    kde = gaussian_kde(y, bw_method=bandwidth)
    density = kde(y) + kde(-y) + kde(200-y)
    density.setflags(write=False)
    return density

def density_weights(y, power=1.0, cap=5.0, bandwidth='scott'):
    """Return fold-local weights and boundary-reflected density at training grades.

    Cap applies after initial mean normalization, before final normalization.
    Final weights can therefore exceed cap slightly.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    if not len(y) or not np.isfinite(y).all() or (y < 0).any() or (y > 100).any():
        raise ValueError('Grades must be nonempty, finite, and in [0, 100].')
    if power < 0 or cap < 1:
        raise ValueError('power must be nonnegative and cap at least 1.')
    if len(y) < 2 or np.ptp(y) < 1e-10:
        return np.ones(len(y)), np.ones(len(y))
    density = _grade_density(y.astype(np.float64).tobytes(), bandwidth)
    # Work in log space to keep very small densities numerically stable.
    log_w = -power * np.log(np.maximum(density, np.finfo(float).tiny))
    raw = np.exp(log_w - log_w.max())
    weights = np.minimum(raw / raw.mean(), cap)
    return weights / weights.mean(), density

# Setup helper functions for data preparation so that X-train and X-test can be massaged into a format palatable to each model 

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin

class DensityWeightedRegressor(RegressorMixin, BaseEstimator):
    """Compute weights in fit, so GridSearchCV never exposes held-out targets."""
    def __init__(self, estimator, power=1.0, cap=5.0, bandwidth="scott"):
        self.estimator = estimator
        self.power = power
        self.cap = cap
        self.bandwidth = bandwidth

    def fit(self, X, y):
        self.sample_weight_, self.density_ = density_weights(y, self.power, self.cap, self.bandwidth)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, sample_weight=self.sample_weight_)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    @property
    def coef_(self):
        return self.estimator_.coef_

from pygam import LinearGAM, s, l      # pip install pygam
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from scipy.stats import norm

# Columns with no natural order: one-hot encoded
ONEHOT_COLUMNS = ['gender', 'school_type', 'residence_type', 'academic_year',
                  'father_occupation', 'mother_occupation']

# Columns with a clear order (lowest level first): ordinal encoded
ORDINAL_COLUMNS = {
    'father_education_level': [
        'No Formal Education', 'Primary', 'Secondary', 'Higher Secondary',
        "Bachelor's Degree", "Master's Degree", 'Doctorate',
    ],
    'mother_education_level': [
        'No Formal Education', 'Primary', 'Secondary', 'Higher Secondary',
        "Bachelor's Degree", "Master's Degree", 'Doctorate',
    ],
    'family_income_bracket': ['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High'],
    'scholarship_status': ['No Scholarship', 'Partial', 'Full'],
}

def build_prep(scale):
    # one-hot and ordinal encoders are the same for every family. Only scaling differs
    return ColumnTransformer([
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), ONEHOT_COLUMNS),
        ('ordinal', OrdinalEncoder(categories=list(ORDINAL_COLUMNS.values())), list(ORDINAL_COLUMNS)),
        ('num', StandardScaler() if scale else 'passthrough', NUMERIC_COLS),
    ])

def make_linear_prep():  # OLS, Ridge, Lasso
    return build_prep(scale=True)

def make_gam_prep():     # GAM: splines handle the shape, so no scaling
    return build_prep(scale=False)

class GAMRegressor(BaseEstimator, RegressorMixin):
    # Wrapper so pygam's LinearGAM works inside a Pipeline and GridSearchCV
    # Smooth spline term for each continuous column, linear term for every other column
    def __init__(self, n_splines=10, lam=0.6):
        self.n_splines = n_splines
        self.lam = lam      # smoothing penalty, bigger = smoother curves

    def fit(self, X, y, sample_weight=None):
        first_continuous = X.shape[1] - len(NUMERIC_COLS)   # one-hot and ordinal columns come first
        last_continuous = first_continuous + len(CONTINUOUS_COLS)
        terms = l(0)
        for j in range(1, X.shape[1]):
            if first_continuous <= j < last_continuous:
                terms = terms + s(j, n_splines=self.n_splines, lam=self.lam)
            else:
                terms = terms + l(j)
        self.gam_ = LinearGAM(terms).fit(X, y, weights=sample_weight)
        return self

    def predict(self, X):
        return self.gam_.predict(X)

def make_knn_prep():     # KNN: distances need scaled columns
    return build_prep(scale=True)

def make_tree_prep():    # decision tree, random forest
    return build_prep(scale=False)

def make_gbt_prep():     # gradient boosting
    return build_prep(scale=False)

def make_svr_prep():     # SVM: RBF kernel uses distances, so scale
    return build_prep(scale=True)

def compute_test_metrics(fitted_model, X_train, Y_train, X_test, Y_test):
    # Returns (RMSE, MAE, R2, CV RMSE, mean Gaussian log-likelihood) of a fitted model on the test set
    test_predictions = fitted_model.predict(X_test)
    # CV RMSE on the training data: from the grid search if tuned, otherwise 5-fold cross_val_score
    if hasattr(fitted_model, 'best_score_'):
        cv_rmse = -fitted_model.best_score_
    else:
        cv_rmse = -cross_val_score(fitted_model, X_train, Y_train, cv=5, scoring='neg_root_mean_squared_error').mean()
    # Likelihood uses the CV RMSE as the noise sd, so the test set is not used to choose it
    return (root_mean_squared_error(Y_test, test_predictions),
            mean_absolute_error(Y_test, test_predictions),
            r2_score(Y_test, test_predictions),
            cv_rmse,
            norm.logpdf(Y_test, loc=test_predictions, scale=cv_rmse).mean())



GRIDS = {
    'Ridge': {'model__alpha': [.1, 1, 10, 100]},
    'Lasso': {'model__alpha': [.001, .01, .1]},
    'GAM': {'model__lam': [.1, 1, 10, 100]},
    'DT': {'model__max_depth': [3, 6, None], 'model__min_samples_leaf': [5, 20]},
    'RF': {'model__max_features': [.5, 1.], 'model__min_samples_leaf': [5, 20]},
    'GBT': {'model__learning_rate': [.05, .1], 'model__max_depth': [3, 6]},
}

def run_split(experiment, X, y, bands, schema, checkpoint_dir, smoke=False):
    globals().update(schema)
    path = Path(checkpoint_dir) / f'split-{experiment:02d}.joblib'
    if path.exists():
        result = joblib.load(path)
        print(f'Split {experiment + 1}: restored checkpoint', flush=True)
        return result
    result = dict(predictions=[], metrics=[], timings=[], importance=[], selections=[],
                  coefficients=[], parameters=[], models={}, data={})
    with threadpool_limits(limits=1):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.15, random_state=experiment)
        if experiment == 0:
            result['data'] = dict(X_train=Xtr, X_test=Xte, y_train=ytr, y_test=yte)
        for condition in CONDITIONS:
            models = [DummyRegressor(), LinearRegression(), Ridge(), Lasso(max_iter=10000),
                      GAMRegressor(), DecisionTreeRegressor(random_state=experiment),
                      RandomForestRegressor(n_estimators=10 if smoke else 100, random_state=experiment, n_jobs=1),
                      HistGradientBoostingRegressor(max_iter=20 if smoke else 500, early_stopping=True, random_state=experiment)]
            for name, model in zip(NAMES, models):
                prep = (make_linear_prep if name in ['Naive','OLS','Ridge','Lasso'] else
                        make_gam_prep if name == 'GAM' else make_knn_prep if name == 'k-NN' else make_tree_prep)()
                if condition == 'Density weighted':
                    model = DensityWeightedRegressor(model, **WEIGHT_CONFIG)
                estimator = Pipeline([('prep', prep), ('model', model)])
                folds = 2 if smoke else (5 if name == 'RF' else 10)
                cv = KFold(n_splits=folds, shuffle=True, random_state=experiment)
                if name in GRIDS:
                    grid = GRIDS[name]
                    if smoke: grid = {key: [values[0]] for key, values in grid.items()}
                    if condition == 'Density weighted': grid = {key.replace('model__', 'model__estimator__'): value for key,value in grid.items()}
                    estimator = GridSearchCV(estimator, grid, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=1, error_score='raise')
                start = time.perf_counter()
                estimator.fit(Xtr, ytr)
                fit_seconds = time.perf_counter() - start
                start = time.perf_counter()
                pred = estimator.predict(Xte)
                prediction_seconds = time.perf_counter() - start
                fitted = getattr(estimator, 'best_estimator_', estimator)
                assert np.isfinite(pred).all()
                metrics = compute_test_metrics(estimator, Xtr, ytr, Xte, yte)
                row = dict(experiment=experiment, condition=condition, model=name)
                result['metrics'].append(dict(row, **dict(zip(['RMSE','MAE','R2','CV RMSE','LogLik'], metrics))))
                result['parameters'].append(dict(row, parameters=getattr(estimator, 'best_params_', {})))
                result['timings'].append(dict(row, **{'fit seconds':fit_seconds, 'prediction seconds':prediction_seconds,
                    'train size':len(Xtr), 'test size':len(Xte)}))
                result['predictions'].append(pd.DataFrame(dict(row, student_id=Xte.index.to_numpy(),
                    actual=yte.to_numpy(), predicted=pred, **{'grade band':bands.loc[Xte.index].to_numpy()})))
                if experiment == 0: result['models'][(condition,name)] = fitted
                if name in ['Lasso','DT','RF','GBT']:
                    imp = permutation_importance(fitted, Xte, yte, scoring='neg_root_mean_squared_error',
                                                 n_repeats=1 if smoke else 5, random_state=experiment, n_jobs=1)
                    result['importance'].append(dict(row, **dict(zip(PREDICTOR_COLS,imp.importances_mean))))
                if name == 'Lasso':
                    base = fitted
                    coefficients = pd.Series(base.named_steps['model'].coef_, index=base.named_steps['prep'].get_feature_names_out())
                    kept = {max((p for p in PREDICTOR_COLS if f.split('__',1)[1] == p or f.split('__',1)[1].startswith(p+'_')),key=len)
                            for f in coefficients.index[coefficients != 0]}
                    result['selections'].append(dict(row, **{p:p in kept for p in PREDICTOR_COLS}))
                    result['coefficients'].append(dict(row, **coefficients.to_dict()))
            print(f'Split {experiment + 1}: {condition} complete', flush=True)
    temp = path.with_suffix('.tmp')
    joblib.dump(result, temp)
    temp.replace(path)
    return result

def run_experiments(X, y, bands, schema, n_splits=20, n_jobs=4, smoke=False):
    fingerprint = hashlib.sha256(Path(__file__).read_bytes() +
        pd.util.hash_pandas_object(X,index=True).values.tobytes() +
        pd.util.hash_pandas_object(y,index=True).values.tobytes() +
        pd.util.hash_pandas_object(bands,index=True).values.tobytes() +
        json.dumps(dict(schema=schema, smoke=smoke, weight_config=WEIGHT_CONFIG),sort_keys=True).encode()).hexdigest()[:16]
    directory = Path('density-results') / fingerprint
    directory.mkdir(parents=True, exist_ok=True)
    with joblib.parallel_config(backend='loky', inner_max_num_threads=1):
        results = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(run_split)(
            i,X,y,bands,schema,str(directory),smoke) for i in range(n_splits))
    return results, directory
