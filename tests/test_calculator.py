###############################################################################
"""
End-to-end test of Calculator.calculate on a tiny synthetic survey.
Uses a SQLite MLflow store so it runs with or without the MLflow container.
"""
import os
import json
import asyncio
import pytest
from os.path import dirname, join, abspath

from Calculator import calculate

IN_JSON = join(dirname(abspath(__file__)), 'in_jsons', 'testCase01.json')

###############################################################################
def loadInJson(overrides={}):
    with open(IN_JSON) as f: inJson = json.load(f)
    params = {p['name']: p for p in inJson['data']['params']}
    for name, value in overrides.items():
        if name in params: params[name]['value'] = value
        else: inJson['data']['params'].append({'name': name, 'value': value, 'valueType': 2})
    tables = {t['name']: t for t in inJson['data']['tables']}
    tables['well_inlines']['values'] = [1005, 1018]
    tables['well_xlines']['values']  = [1004, 1012]
    return inJson

@pytest.fixture(autouse=True)
def mlflowFileStore(tmp_path, monkeypatch):
    monkeypatch.setenv('MLFLOW_TRACKING_URI', 'sqlite:///' + str(tmp_path / 'mlflow.db'))

###############################################################################
def test_synthetic_run(tmp_path):
    inJson = loadInJson({'output_dir': str(tmp_path / 'data'), 'n_inlines': 24, 'n_xlines': 18})
    out = asyncio.run(calculate(inJson=inJson))
    assert not isinstance(out, str), out
    result = out.flatten()
    assert result['data_check_passed'] is True
    assert result['wells'] == ['Well_A', 'Well_B']
    assert os.path.exists(result['manifest_path'])

def test_existing_dataset_run(tmp_path):
    inJson = loadInJson({'output_dir': str(tmp_path / 'data'), 'n_inlines': 24, 'n_xlines': 18})
    first = asyncio.run(calculate(inJson=inJson)).flatten()
    inJson2 = {'data': {'params': [{'name': 'dataset_mode', 'value': 'existing', 'valueType': 3},
                                   {'name': 'manifest_path', 'value': first['manifest_path'], 'valueType': 3}],
                        'tables': []}}
    out = asyncio.run(calculate(inJson=inJson2))
    assert not isinstance(out, str), out
    assert out.flatten()['data_check_passed'] is True

def test_bad_input_returns_error_message(tmp_path):
    inJson = loadInJson({'output_dir': str(tmp_path / 'data'), 'zp_min': 90000})
    out = asyncio.run(calculate(inJson=inJson))
    assert isinstance(out, str) and 'zp_min' in out

def test_bad_mode_returns_error_message():
    out = asyncio.run(calculate(inJson={'data': {'params': [{'name': 'dataset_mode', 'value': 'nope'}], 'tables': []}}))
    assert isinstance(out, str) and 'dataset_mode' in out
###############################################################################
