"""Check bounded weights, exact loss weighting, and training-fold isolation."""
import unittest
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from polynomial_experiment import polynomial_weights, PolynomialWeightedRegressor

class RecordingRegressor(RegressorMixin, BaseEstimator):
    records = []
    def fit(self, X, y, sample_weight=None):
        self.records.append((np.asarray(y).copy(), sample_weight.copy()))
        self.mean_ = np.average(y, weights=sample_weight)
        return self
    def predict(self, X):
        return np.full(len(X), self.mean_)

class PolynomialTests(unittest.TestCase):
    def test_exact_formula_and_direction(self):
        y = np.array([0., 25., 50., 100.])
        np.testing.assert_allclose(polynomial_weights(y), [2, 2.5625, 3.25, 5])
        np.testing.assert_allclose(polynomial_weights(y, power=-2), 1+(1+y/100)**-2)
        self.assertTrue((np.diff(polynomial_weights(y, power=-2)) < 0).all())
        for power in [-3, 0, 2, 5]:
            w = polynomial_weights(np.linspace(0,100,1001), lam=3, power=power)
            self.assertGreaterEqual(w.min(), 1+3*min(1,2.**power))
            self.assertLessEqual(w.max(), 1+3*max(1,2.**power))
        np.testing.assert_array_equal(polynomial_weights(y,lam=0), np.ones(4))

    def test_invalid_values(self):
        for y in [[], [np.nan], [-1,10], [10,101], [np.inf]]:
            with self.assertRaises(ValueError): polynomial_weights(y)
        for kwargs in [dict(lam=-1),dict(lam=np.nan),dict(power=np.inf),dict(power=2000)]:
            with self.assertRaises(ValueError): polynomial_weights([0,100],**kwargs)

    def test_exact_weight_forwarding_and_unweighted_control(self):
        X = np.arange(8).reshape(-1,1)
        y = np.array([0.,4.,30.,45.,50.,55.,80.,100.])
        model = clone(PolynomialWeightedRegressor(DummyRegressor())).fit(X,y)
        np.testing.assert_allclose(model.predict(X), np.average(y,weights=polynomial_weights(y)))
        control = PolynomialWeightedRegressor(LinearRegression(),lam=0).fit(X,y)
        np.testing.assert_allclose(control.predict(X),LinearRegression().fit(X,y).predict(X))

    def test_cv_weights_use_only_training_targets(self):
        RecordingRegressor.records.clear()
        X = np.arange(24).reshape(-1,1)
        y = np.linspace(0,100,24)
        cv = KFold(3,shuffle=True,random_state=4)
        search = GridSearchCV(PolynomialWeightedRegressor(RecordingRegressor(),lam=2,power=-2), {},
                              cv=cv, scoring='neg_root_mean_squared_error',n_jobs=1).fit(X,y)
        expected_targets = [y[train] for train,_ in cv.split(X)] + [y]
        self.assertEqual(len(RecordingRegressor.records),4)
        for (observed, weights), expected in zip(RecordingRegressor.records, expected_targets):
            np.testing.assert_array_equal(observed,expected)
            np.testing.assert_allclose(weights,polynomial_weights(expected,lam=2,power=-2))

if __name__ == '__main__': unittest.main()
