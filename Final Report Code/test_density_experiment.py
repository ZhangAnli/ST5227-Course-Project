"""Focused checks for inverse-density weights and fold-local training."""
import unittest
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GridSearchCV, KFold
from density_experiment import density_weights, DensityWeightedRegressor

class RecordingRegressor(RegressorMixin, BaseEstimator):
    records = []
    def fit(self, X, y, sample_weight=None):
        self.records.append((np.asarray(y).copy(), sample_weight.copy()))
        self.mean_ = np.average(y, weights=sample_weight)
        return self
    def predict(self, X):
        return np.full(len(X), self.mean_)

class DensityTests(unittest.TestCase):
    def test_weights_and_rare_tail(self):
        y = np.r_[np.linspace(65,85,200), 10., 15., 100.]
        weights, density = density_weights(y)
        self.assertTrue(np.isfinite(weights).all() and (weights > 0).all())
        self.assertAlmostEqual(weights.mean(), 1)
        self.assertTrue(np.isfinite(density).all() and (density > 0).all())
        self.assertGreater(weights[-2], np.median(weights[:200]))
        # Cap is applied before final mean normalization.
        raw = 1/density
        expected = np.minimum(raw/raw.mean(), 5)
        np.testing.assert_allclose(weights, expected/expected.mean())

    def test_uniform_and_endpoints(self):
        np.testing.assert_array_equal(density_weights([50]*10)[0], np.ones(10))
        np.testing.assert_array_equal(density_weights([100])[0], [1])
        self.assertTrue(np.isfinite(density_weights([0,1,50,99,100])[0]).all())
        np.testing.assert_allclose(density_weights([0,20,50,100], power=0)[0], 1)

    def test_invalid_targets(self):
        for y in [[], [np.nan], [-1,10], [10,101], [np.inf]]:
            with self.assertRaises(ValueError): density_weights(y)

    def test_weight_forwarding_and_clone(self):
        y = np.r_[np.linspace(70,80,20), 5.]
        X = np.arange(len(y)).reshape(-1,1)
        model = clone(DensityWeightedRegressor(DummyRegressor())).fit(X,y)
        w, _ = density_weights(y)
        np.testing.assert_allclose(model.predict(X), np.average(y, weights=w))
        self.assertFalse(np.isclose(np.mean(y), model.predict(X)[0]))

    def test_cv_fits_density_using_only_fold_targets(self):
        RecordingRegressor.records.clear()
        X = np.arange(24).reshape(-1,1)
        y = np.r_[np.linspace(65,90,21), 0, 10, 100]
        cv = KFold(3,shuffle=True,random_state=4)
        search = GridSearchCV(DensityWeightedRegressor(RecordingRegressor()), {}, cv=cv,
                              scoring='neg_root_mean_squared_error', n_jobs=1)
        search.fit(X,y)
        expected_targets = [y[train] for train,_ in cv.split(X)] + [y]
        self.assertEqual(len(RecordingRegressor.records),4)
        for (observed, weights), expected in zip(RecordingRegressor.records, expected_targets):
            np.testing.assert_array_equal(observed,expected)
            np.testing.assert_allclose(weights,density_weights(expected)[0])

if __name__ == '__main__': unittest.main()
