"""
Tests for synthetic dataset generation and data checks.
Runs on a tiny grid so CI stays fast and needs no real data.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import pytest
import segyio

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'rockPropCalculator'))
from dataPrep.dataChecks import checkSyntheticInputs, checkDataset, WELL_LOG_COLUMNS, WELL_SEISMIC_COLUMNS, NULL_VALUE
from dataPrep.inputWrapper import standardInputWrapper
from dataPrep.syntheticData import generateSyntheticDataset

IN_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'in_jsons', 'synthetic_small.json')


def tinyInputs(out_dir, **overrides):
    data = {'output_dir': str(out_dir), 'n_inlines': 24, 'n_xlines': 18, 't_end_ms': 1940.0,
            'well_inlines': [1005, 1018], 'well_xlines': [1004, 1012]}
    data.update(overrides)
    return data


@pytest.fixture(scope='module')
def tinyDataset(tmp_path_factory):
    out = tmp_path_factory.mktemp('synthetic')
    manifest, path = generateSyntheticDataset(inputData=tinyInputs(out))
    return manifest, path


# Input checks ##############################################################################
def test_defaults_pass():
    data = checkSyntheticInputs({})
    assert data['n_samples'] == 221
    assert len(data['zone_names']) == len(data['horizon_names']) + 1


def test_example_json_passes():
    with open(IN_JSON) as f: inJson = json.load(f)
    data = checkSyntheticInputs(standardInputWrapper(inJson))
    assert data['name'] == 'Synthetic_Small'


@pytest.mark.parametrize('overrides, message', [
    ({'horizon_mean_time_ms': [1790, 1780, 1865, 1915]}, 'strictly increasing'),
    ({'well_inlines': [5000, 1090]}, 'outside the survey'),
    ({'zone_phi_mean': [0.05, 0.05]}, r'n_horizons \+ 1'),
    ({'t_end_ms': 1700}, 't_end_ms must be greater'),
    ({'zp_min': 70000}, 'zp_min must be less than zp_max'),
    ({'n_inlines': 1277, 'n_xlines': 615}, 'Shrink the grid'),
    ({'target_horizons': ['Horizon_03', 'Horizon_01']}, 'shallower'),
])
def test_bad_inputs_raise(overrides, message):
    with pytest.raises(ValueError, match=message):
        checkSyntheticInputs(overrides)


def test_all_errors_reported_together():
    with pytest.raises(ValueError) as err:
        checkSyntheticInputs({'zp_min': 70000, 'phi_min': 0.5})
    assert 'zp_min' in str(err.value) and 'phi_min' in str(err.value)


# Generated dataset ########################################################################
def test_dataset_passes_checks(tinyDataset):
    _, path = tinyDataset
    report = checkDataset(path)
    assert report['passed'], report['errors']


def test_segy_reads_like_the_calculator(tinyDataset):
    manifest, path = tinyDataset
    root, geo = os.path.dirname(path), manifest['geometry']
    with segyio.open(os.path.join(root, 'seismic.sgy'), iline=geo['iline_byte'], xline=geo['xline_byte']) as s:
        s.mmap()
        cube = s.trace.raw[:].T.reshape(s.samples.size, s.ilines.size, s.xlines.size)
    truth = np.load(os.path.join(root, 'truth.npz'))
    assert np.allclose(np.transpose(cube, (1, 2, 0)), truth['seismic'])


def test_horizons_do_not_cross(tinyDataset):
    manifest, path = tinyDataset
    df = pd.read_excel(os.path.join(os.path.dirname(path), 'horizons.xlsx'))
    assert np.all(np.diff(df[manifest['horizon_names']].to_numpy(), axis=1) >= 0)


def test_well_sheets(tinyDataset):
    manifest, path = tinyDataset
    sheets = pd.read_excel(os.path.join(os.path.dirname(path), 'well_logs.xlsx'), sheet_name=None)
    for well in manifest['wells']:
        assert list(sheets[well['log_sheet']].columns) == WELL_LOG_COLUMNS
        assert list(sheets[well['seismic_sheet']].columns) == WELL_SEISMIC_COLUMNS
        assert (sheets[well['log_sheet']] == NULL_VALUE).to_numpy().any()


def test_porosity_drives_impedance_in_reservoir_zone(tinyDataset):
    _, path = tinyDataset
    truth = np.load(os.path.join(os.path.dirname(path), 'truth.npz'))
    m = truth['zone'] == 2
    assert np.corrcoef(truth['seismic'][m], truth['phi'][m])[0, 1] < -0.5


def test_same_seed_same_data(tmp_path):
    generateSyntheticDataset(inputData=tinyInputs(tmp_path / 'a'))
    generateSyntheticDataset(inputData=tinyInputs(tmp_path / 'b'))
    a, b = np.load(tmp_path / 'a' / 'truth.npz'), np.load(tmp_path / 'b' / 'truth.npz')
    assert np.array_equal(a['seismic'], b['seismic'])


def test_check_catches_wrong_geometry(tinyDataset, tmp_path):
    manifest, path = tinyDataset
    bad = json.loads(json.dumps(manifest))
    bad['geometry']['n_inlines'] += 1
    root = os.path.dirname(path)
    bad['files'] = {k: os.path.join(root, v) for k, v in manifest['files'].items()}
    bad_path = tmp_path / 'manifest.json'
    bad_path.write_text(json.dumps(bad))
    report = checkDataset(str(bad_path))
    assert not report['passed']
    assert any('does not match manifest' in e for e in report['errors'])
