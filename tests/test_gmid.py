import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from optlocal.gmid import GmIdTable


class GmIdTableTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.axes = {
            "L": np.array([1.0e-7, 2.0e-7]),
            "VSB": np.array([0.0, 0.1]),
            "VDS": np.array([0.5, 1.0]),
            "VGS": np.array([0.2, 0.4, 0.6, 0.8]),
        }

    def _arrays(self, ratios=(25.0, 20.0, 14.0, 10.0)):
        shape = tuple(len(self.axes[name]) for name in ("L", "VSB", "VDS", "VGS"))
        ids = np.ones(shape, dtype=float)
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    ids[i, j, k] *= (i + 1) * (j + 1) * (k + 1) * 1e-6
        gm = ids * np.asarray(ratios, dtype=float)
        return ids, gm

    def _write_npz(self, name="table.npz", polarity="n", ratios=None,
                   ids=None, gm=None, gds=None, cgg=None,
                   include_optional=True):
        path = self.root / name
        if ids is None or gm is None:
            ids, gm = self._arrays(ratios or (25.0, 20.0, 14.0, 10.0))
        payload = dict(self.axes, W=1e-5, polarity=polarity, temp=27.0,
                       vmax=1.0, ids=ids, gm=gm)
        if include_optional:
            payload.update(gds=gm / 20.0 if gds is None else gds,
                           cgg=np.full_like(gm, 1e-14) if cgg is None else cgg)
        np.savez_compressed(path, **payload)
        return path

    def _write_csv(self, name="table.csv", rows=None, fields=None):
        path = self.root / name
        if rows is None:
            ids, gm = self._arrays()
            rows = []
            for i, length in enumerate(self.axes["L"]):
                for j, vsb in enumerate(self.axes["VSB"]):
                    for k, vds in enumerate(self.axes["VDS"]):
                        for q, vgs in enumerate(self.axes["VGS"]):
                            rows.append(dict(
                                L=length, VSB=vsb, VDS=vds, VGS=vgs,
                                W=1e-5, polarity="n", temp=27.0,
                                ids=ids[i, j, k, q], gm=gm[i, j, k, q],
                                gds=gm[i, j, k, q] / 20.0,
                                cgg=1e-14))
        fields = fields or list(rows[0])
        with path.open("w", newline="") as stream:
            stream.write("# synthetic/demo fixture\n")
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_csv_and_npz_have_same_lookup_and_size_result(self):
        npz = self._write_npz()
        csv_path = self._write_csv()
        native = GmIdTable.load(npz)
        flat = GmIdTable.load(csv_path)
        self.assertEqual(native.source(), str(npz))
        self.assertEqual(flat.source(), str(csv_path))
        self.assertEqual(native.width(), flat.width())
        self.assertEqual(native.polarity(), "n")
        self.assertEqual(native.temperature(), 27.0)
        for key in self.axes:
            np.testing.assert_array_equal(native.axes[key], flat.axes[key])
        for key in ("vgs", "gmid", "idw", "ids", "gm", "gain", "ft"):
            np.testing.assert_allclose(
                native.curve(1.5e-7, 0.75, 0.05)[key],
                flat.curve(1.5e-7, 0.75, 0.05)[key],
            )
        a = native.size_for(1.5e-7, 0.75, 0.05, gmid=15.0, ids=4e-5)
        b = flat.size_for(1.5e-7, 0.75, 0.05, gmid=15.0, ids=4e-5)
        self.assertEqual(json.dumps(a, sort_keys=True),
                         json.dumps(b, sort_keys=True))

    def test_size_for_ids_and_gm_scale_width_and_extensive_values(self):
        table = GmIdTable.load(self._write_npz())
        by_ids = table.size_for(1.0e-7, 0.5, gmid=15.0, ids=4.0e-6)
        by_gm = table.size_for(1.0e-7, 0.5, gmid=15.0, gm=6.0e-5)
        self.assertAlmostEqual(by_ids["gmid"], 15.0)
        self.assertAlmostEqual(by_gm["gmid"], 15.0)
        self.assertAlmostEqual(by_ids["ids"], 4.0e-6)
        self.assertAlmostEqual(by_gm["gm"], 6.0e-5)
        self.assertAlmostEqual(by_ids["gm"], 6.0e-5)
        self.assertAlmostEqual(by_gm["ids"], 4.0e-6)
        self.assertAlmostEqual(by_ids["W"], by_gm["W"])
        self.assertAlmostEqual(by_ids["idw"], by_gm["idw"])
        self.assertEqual(by_ids["polarity"], "n")
        json.dumps(by_ids)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            table.size_for(1.0e-7, 0.5, gmid=15.0)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            table.size_for(1.0e-7, 0.5, gmid=15.0, ids=1e-6, gm=1e-5)

    def test_pm_os_magnitude_convention_is_preserved(self):
        table = GmIdTable.load(self._write_npz(polarity="p"))
        curve = table.curve(1.0e-7, 0.5)
        result = table.size_for(1.0e-7, 0.5, gmid=15.0, ids=2e-6)
        self.assertEqual(table.polarity(), "p")
        self.assertTrue(np.all(curve["vgs"] > 0.0))
        self.assertTrue(np.all(curve["ids"] > 0.0))
        self.assertEqual(result["polarity"], "p")
        self.assertGreater(result["vgs"], 0.0)

    def test_curve_and_size_reject_coordinate_extrapolation(self):
        table = GmIdTable.load(self._write_npz())
        for args in ((0.9e-7, 0.5, 0.0), (1.0e-7, 0.4, 0.0),
                     (1.0e-7, 0.5, -0.01)):
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, "outside"):
                table.curve(*args)
        with self.assertRaisesRegex(ValueError, "outside"):
            table.size_for(1.0e-7, 1.1, gmid=15.0, ids=1e-6)

    def test_inverse_solves_linear_gm_over_linear_ids(self):
        ids, _ = self._arrays()
        ratios = np.zeros_like(ids)
        gm = np.zeros_like(ids)
        # On the first VGS segment, ratio interpolation would give 0.5;
        # solving the ratio of the two linearly interpolated quantities gives
        # t=0.25 for target 15.
        ids[..., 0] = 1.0
        ids[..., 1] = 3.0
        ids[..., 2:] = 4.0
        gm[..., 0] = 20.0
        gm[..., 1] = 30.0
        gm[..., 2:] = 40.0
        table = GmIdTable.load(self._write_npz(ids=ids, gm=gm))
        result = table.size_for(1.0e-7, 0.5, gmid=15.0, ids=1.0)
        self.assertAlmostEqual(result["vgs"], 0.25)
        self.assertAlmostEqual(result["gm"] / result["ids"], 15.0)

    def test_exact_internal_nodes_do_not_mix_adjacent_nan_slices(self):
        self.axes = {
            "L": np.array([1.0e-7, 2.0e-7, 3.0e-7]),
            "VSB": np.array([0.0, 0.1, 0.2]),
            "VDS": np.array([0.5, 0.75, 1.0]),
            "VGS": np.array([0.2, 0.4, 0.6, 0.8]),
        }
        shape = tuple(len(self.axes[name]) for name in
                     ("L", "VSB", "VDS", "VGS"))
        ids = np.full(shape, 2e-6, dtype=float)
        gm = ids * np.array([25.0, 20.0, 14.0, 10.0])
        ids[2, :, :, :] = np.nan
        ids[:, 2, :, :] = np.nan
        ids[:, :, 2, :] = np.nan
        gm[2, :, :, :] = np.nan
        gm[:, 2, :, :] = np.nan
        gm[:, :, 2, :] = np.nan
        table = GmIdTable.load(self._write_npz(ids=ids, gm=gm))
        curve = table.curve(2.0e-7, 0.75, 0.1)
        self.assertTrue(np.isfinite(curve["ids"]).all())
        self.assertTrue(np.isfinite(curve["gm"]).all())
        np.testing.assert_allclose(curve["gmid"], [25.0, 20.0, 14.0, 10.0])

    def test_inverse_tolerance_scales_with_tiny_operating_point(self):
        ids, gm = self._arrays()
        ids[..., 0] = 1.0e-17
        ids[..., 1] = 2.5e-17
        ids[..., 2:] = 4.0e-17
        gm[..., 0] = 2.0e-16
        gm[..., 1] = 2.5e-16
        gm[..., 2:] = 3.0e-16
        table = GmIdTable.load(self._write_npz(ids=ids, gm=gm))
        result = table.size_for(1.0e-7, 0.5, gmid=15.0, ids=1.0e-17)
        self.assertAlmostEqual(result["vgs"], 0.2 + 0.2 * (2.0 / 7.0))
        self.assertAlmostEqual(result["gm"] / result["ids"], 15.0)

    def test_endpoint_size_uses_finite_optional_fields_at_t0_and_t1(self):
        ids, gm = self._arrays(ratios=(20.0, 10.0, 8.0, 6.0))
        for index, target, name in ((0, 20.0, "endpoint0.npz"),
                                    (1, 10.0, "endpoint1.npz")):
            gds = np.full_like(gm, np.nan)
            cgg = np.full_like(gm, np.nan)
            gds[..., index] = gm[..., index] / 20.0
            cgg[..., index] = 1.0e-14
            table = GmIdTable.load(self._write_npz(
                name=name, ids=ids, gm=gm, gds=gds, cgg=cgg))
            result = table.size_for(1.0e-7, 0.5, gmid=target, ids=1e-6)
            self.assertTrue(np.isfinite(result["gain"]))
            self.assertTrue(np.isfinite(result["ft"]))

    def test_nan_gap_is_not_crossed(self):
        self.axes["VGS"] = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        shape = tuple(len(self.axes[name]) for name in ("L", "VSB", "VDS", "VGS"))
        ids = np.ones(shape, dtype=float)
        gm = ids * np.array([25.0, 20.0, 15.0, 12.0, 8.0])
        ids[..., 2] = np.nan
        gm[..., 2] = np.nan
        table = GmIdTable.load(self._write_npz(ids=ids, gm=gm))
        with self.assertRaisesRegex(ValueError, "adjacent valid"):
            table.size_for(1.0e-7, 0.5, gmid=15.0, ids=1e-6)

    def test_multiple_distinct_vgs_solutions_are_rejected(self):
        ids, gm = self._arrays(ratios=(20.0, 10.0, 20.0, 10.0))
        table = GmIdTable.load(self._write_npz(ids=ids, gm=gm))
        with self.assertRaisesRegex(ValueError, "multiple distinct"):
            table.size_for(1.0e-7, 0.5, gmid=15.0, ids=1e-6)

    def test_invalid_npz_metadata_and_shape_are_rejected(self):
        path = self.root / "bad.npz"
        ids, gm = self._arrays()
        np.savez(path, L=[2e-7, 1e-7], VSB=self.axes["VSB"],
                 VDS=self.axes["VDS"], VGS=self.axes["VGS"], W=1e-5,
                 polarity="x", ids=ids, gm=gm)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            GmIdTable.load(path)
        np.savez(path, L=self.axes["L"], VSB=self.axes["VSB"],
                 VDS=self.axes["VDS"], VGS=self.axes["VGS"], W=0.0,
                 polarity="n", ids=ids, gm=gm)
        with self.assertRaisesRegex(ValueError, "positive"):
            GmIdTable.load(path)
        np.savez(path, L=self.axes["L"], VSB=self.axes["VSB"],
                 VDS=self.axes["VDS"], VGS=self.axes["VGS"], W=1e-5,
                 polarity="n", ids=ids[:-1], gm=gm)
        with self.assertRaisesRegex(ValueError, "shape"):
            GmIdTable.load(path)

    def test_csv_duplicate_missing_and_inconsistent_metadata_are_rejected(self):
        good = self._write_csv()
        rows = list(csv.DictReader(line for line in good.read_text().splitlines()
                                   if line and not line.startswith("#")))
        duplicate = rows + [dict(rows[0])]
        with self.assertRaisesRegex(ValueError, "missing grid|duplicate"):
            self._write_csv("duplicate.csv", duplicate).exists() and \
                GmIdTable.load(self.root / "duplicate.csv")
        missing = rows[:-1]
        with self.assertRaisesRegex(ValueError, "missing grid"):
            GmIdTable.load(self._write_csv("missing.csv", missing))
        bad_width = [dict(row) for row in rows]
        bad_width[1]["W"] = "2e-5"
        with self.assertRaisesRegex(ValueError, "W must be"):
            GmIdTable.load(self._write_csv("bad_width.csv", bad_width))
        bad_polarity = [dict(row) for row in rows]
        bad_polarity[1]["polarity"] = "p"
        with self.assertRaisesRegex(ValueError, "polarity must"):
            GmIdTable.load(self._write_csv("bad_polarity.csv", bad_polarity))

    def test_demo_csv_is_small_synthetic_and_queryable(self):
        path = Path(__file__).parents[1] / "examples" / "gmid_demo.csv"
        table = GmIdTable.load(path)
        self.assertEqual(table.polarity(), "n")
        result = table.size_for(0.18e-6, 0.6, gmid=15.0, ids=2e-5)
        self.assertAlmostEqual(result["gmid"], 15.0)
        self.assertAlmostEqual(result["ids"], 2e-5)
        self.assertEqual(result["polarity"], "n")

    def test_optional_lut_directory_smoke_without_copying_data(self):
        lut_dir = os.environ.get("VCAL_TEST_LUT_DIR")
        if not lut_dir:
            self.skipTest("VCAL_TEST_LUT_DIR is not set")
        paths = sorted(Path(lut_dir).glob("*.npz"))
        if not paths:
            self.skipTest("VCAL_TEST_LUT_DIR contains no NPZ LUTs")
        polarities = set()
        for path in paths:
            table = GmIdTable.load(path)
            self.assertIn(table.polarity(), ("n", "p"))
            polarities.add(table.polarity())
            curve = table.curve(float(table.axes["L"][0]),
                                float(table.axes["VDS"][len(table.axes["VDS"]) // 2]))
            finite = np.isfinite(curve["gmid"])
            self.assertTrue(finite.any())
            lo = float(np.nanmin(curve["gmid"]))
            hi = float(np.nanmax(curve["gmid"]))
            if lo <= 15.0 <= hi:
                result = table.size_for(float(table.axes["L"][0]),
                                        float(table.axes["VDS"][10]),
                                        gmid=15.0, ids=1e-5)
                self.assertAlmostEqual(result["gmid"], 15.0, places=8)
                self.assertGreater(result["W"], 0.0)
        self.assertTrue(polarities)


if __name__ == "__main__":
    unittest.main()
