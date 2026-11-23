"""Focused data-contract tests; no production fields or GPU are required."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np


class PreparationTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).with_name('prepare.py')
        self.assertTrue(path.exists(), 'prepare.py data contracts are not implemented')
        spec = importlib.util.spec_from_file_location('pod_prepare', path)
        self.p = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.p)

    def test_each_time_has_fixed_distinct_spatial_queries(self):
        q = self.p.fixed_query_indices(frames=5, nz=7, nx=9, per_time=8)
        np.testing.assert_array_equal(q, self.p.fixed_query_indices(frames=5, nz=7, nx=9, per_time=8))
        self.assertEqual(q.dtype, np.int64)
        self.assertEqual(len(np.unique(q)), 40)
        np.testing.assert_array_equal(np.bincount(q // 63), np.full(5, 8))
        self.assertTrue(np.all((q >= 0) & (q < 5 * 63)))

    def test_validation_rows_are_fixed_and_source_disjoint(self):
        train = np.array([[1000., 1000., 15., .1, 1.]])
        val = np.array([[float(i), float(i + 1), 15., .1, 1.] for i in range(200)])
        rows, sources = self.p.select_validation_sources(train, val)
        np.testing.assert_array_equal(rows, np.linspace(0, 199, 16, dtype=int))
        np.testing.assert_array_equal(sources, val[rows])
        val[0] = train[0]
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.p.select_validation_sources(train, val)
        val[0] = val[13]
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.p.select_validation_sources(train, val)

    def test_full_rank_codes_reconstruct_normalized_train_only(self):
        y = np.random.default_rng(2).normal(size=(4, 19))
        codes, eig = self.p.pod_source_codes(y, rank=6)
        self.assertEqual(codes.shape, (4, 6))
        self.assertEqual(codes.dtype, np.float64)
        np.testing.assert_array_equal(codes[:, 4:], 0.)
        np.testing.assert_allclose(codes @ codes.T / 4, np.eye(4), atol=2e-15)
        basis = codes.T @ y / 4
        np.testing.assert_allclose(codes @ basis, y, atol=5e-15)
        np.testing.assert_allclose(eig, np.linalg.eigvalsh(y @ y.T / 19)[::-1], rtol=1e-14)

    def test_inherited_normalization_has_no_fit(self):
        raw = np.array([[0., 1.252277344221455e-8, -2.50455468844291e-8]], dtype=np.float64)
        np.testing.assert_array_equal(self.p.normalize_pressure(raw), [[0., 1., -2.]])
        self.assertEqual(self.p.PRESSURE_SCALE, 1.252277344221455e-8)

    def test_validation_field_rejects_nonfinite_and_nonzero_top(self):
        a = np.zeros((4, 3, 3), dtype=np.float32)
        a[:, 1:] = 1.
        self.p.validate_field(a, frames=4, nz=3, nx=3)
        a[0, 0, 0] = 1.
        with self.assertRaises(ValueError):
            self.p.validate_field(a, frames=4, nz=3, nx=3)
        a[0, 0, 0] = 0.
        a[1, 1, 1] = np.nan
        with self.assertRaises(ValueError):
            self.p.validate_field(a, frames=4, nz=3, nx=3)


if __name__ == '__main__':
    unittest.main()
