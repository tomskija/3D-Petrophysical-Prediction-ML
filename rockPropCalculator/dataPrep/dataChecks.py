############################################################################################
"""
Data checks for the 3D petrophysical calculator.

checkSyntheticInputs: fills defaults, casts types and validates the inputs used to build a
                      synthetic dataset (same idea as checkData in Production-Prediction-ML).
checkDataset:         validates any dataset described by a manifest JSON (synthetic or real)
                      before it goes into the model, so nothing about the survey is hard coded.
"""
############################################################################################
import os
import json
import numpy as np
import pandas as pd

############################################################################################
NULL_VALUE = -999.25

WELL_LOG_COLUMNS = [
    'MD_KB_ft', 'TWT_ms', 'X', 'Y', 'P_Velo', 'Zp',
    'DPOR_Low_Res', 'DPOR_High_Res', 'NPHI_Low_Res', 'NPHI_High_Res',
    'TPHI_Low_Res', 'TPHI_High_Res', 'RHOZ', 'TOC'
]
WELL_SEISMIC_COLUMNS = [
    'Time_ms', 'X', 'Y', 'P_Velo',
    'DPOR_Low_Res', 'DPOR_High_Res', 'NPHI_Low_Res', 'NPHI_High_Res',
    'TPHI_Low_Res', 'TPHI_High_Res', 'RHOZ', 'TOC'
]
HORIZON_BASE_COLUMNS = ['Inline', 'Xline', 'X', 'Y']

############################################################################################
# Defaults are generic, not tied to any real survey. Every one can be overridden from the UI.
SYNTHETIC_DEFAULTS = {
    'name':                    'Synthetic_Survey',
    'random_seed':             5195,
    'output_dir':              'Data/Synthetic',
    # Survey geometry
    'n_inlines':               120,
    'n_xlines':                80,
    'inline_start':            1001,
    'xline_start':             1001,
    'bin_size':                110.0,      # ft
    'x_origin':                1000000.0,
    'y_origin':                500000.0,
    'iline_byte':              189,
    'xline_byte':              193,
    # Time axis
    't_start_ms':              1740.0,
    't_end_ms':                1960.0,
    'dt_ms':                   1.0,
    # Structure
    'structural_relief_ms':    12.0,
    'dip_ms':                  15.0,       # total two-way-time change across the inline direction
    'lateral_corr_traces':     15.0,
    'min_thickness_ms':        1.0,
    # Rock physics
    'rho_matrix':              2.71,       # g/cc
    'rho_fluid':               1.05,
    'rho_kerogen':             1.30,
    'vp_phi_sensitivity':      1.6,        # fractional Vp drop per unit porosity
    'vp_kerogen_sensitivity':  0.5,        # fractional Vp drop per unit kerogen volume
    # Ranges (used for clipping and for data checks)
    'phi_min':                 0.0,
    'phi_max':                 0.35,
    'toc_min':                 0.0,
    'toc_max':                 25.0,
    'zp_min':                  20000.0,
    'zp_max':                  65000.0,
    # Inverted-volume look
    'seismic_lateral_smooth':  1.5,        # traces
    'seismic_vertical_smooth': 1.5,        # samples
    'seismic_noise_frac':      0.01,       # noise std as fraction of Zp
    # Wells
    'md_at_t_start_ft':        8500.0,
    'md_step_ft':              0.5,
    'log_noise_frac':          0.03,
    'low_res_smooth_ft':       4.0,
    'null_fraction':           0.01,
    'well_pad_ms':             3.0,
    # Size guard
    'max_segy_mb':             500.0,
    'allow_large':             False,
}

SYNTHETIC_TABLE_DEFAULTS = {
    'horizon_names':          ['Horizon_01', 'Horizon_02', 'Horizon_03', 'Horizon_04'],
    'horizon_mean_time_ms':   [1790.0, 1815.0, 1865.0, 1915.0],
    'horizon_relief_ms':      [0.0, 2.0, 4.0, 4.0],      # thickness variation added below each horizon
    # One entry per zone: above horizon 1, between each pair, below the last (n_horizons + 1)
    'zone_names':             ['Zone_00', 'Zone_01', 'Zone_02', 'Zone_03', 'Zone_04'],
    'zone_phi_mean':          [0.04, 0.07, 0.09, 0.06, 0.05],
    'zone_phi_std':           [0.010, 0.015, 0.020, 0.015, 0.010],
    'zone_toc_mean':          [0.3, 10.0, 1.0, 0.5, 0.3],
    'zone_toc_std':           [0.1, 2.5, 0.4, 0.2, 0.1],
    'zone_vp_matrix_fts':     [19000.0, 15500.0, 17500.0, 18500.0, 19500.0],
    'well_names':             ['Well_A', 'Well_B'],
    'well_inlines':           [1040, 1090],
    'well_xlines':            [1025, 1060],
    'target_horizons':        ['Horizon_01', 'Horizon_03'],   # top, base of the logged interval
}

BOOL_KEYS = ['allow_large']
INT_KEYS  = ['random_seed', 'n_inlines', 'n_xlines', 'inline_start', 'xline_start', 'iline_byte', 'xline_byte']

############################################################################################
def checkSyntheticInputs(inputData={}):
    """
    Fill defaults, cast types and validate synthetic dataset inputs.
    Collects every problem and raises once, so the UI can show them all at the same time.
    :param inputData: Flat dict from standardInputWrapper (may be partial)
    :return:          Complete, validated inputData dict
    """
    ########################################################################################
    data = dict(SYNTHETIC_DEFAULTS)
    data.update({k: list(v) for k, v in SYNTHETIC_TABLE_DEFAULTS.items()})
    data.update({k: v for k, v in inputData.items() if v is not None})
    errors = []
    ########################################################################################
    # Types
    for key in INT_KEYS:
        try: data[key] = int(data[key])
        except (TypeError, ValueError): errors.append(f"{key} must be an integer, got {data[key]!r}")
    for key in BOOL_KEYS:
        data[key] = bool(data[key])
    for key, default in SYNTHETIC_DEFAULTS.items():
        if isinstance(default, float):
            try: data[key] = float(data[key])
            except (TypeError, ValueError): errors.append(f"{key} must be a number, got {data[key]!r}")
    if errors: raise ValueError("Input check failed:\n  - " + "\n  - ".join(errors))
    ########################################################################################
    # Geometry and time axis
    if data['n_inlines'] < 2 or data['n_xlines'] < 2:  errors.append("n_inlines and n_xlines must both be >= 2")
    if data['dt_ms'] <= 0:                             errors.append("dt_ms must be > 0")
    if data['t_end_ms'] <= data['t_start_ms']:         errors.append("t_end_ms must be greater than t_start_ms")
    if data['bin_size'] <= 0:                          errors.append("bin_size must be > 0")
    if data['iline_byte'] == data['xline_byte']:       errors.append("iline_byte and xline_byte must differ")
    if not float(data['t_start_ms']).is_integer():     errors.append("t_start_ms must be a whole number of ms (SEG-Y delay field is an integer)")
    if not (data['dt_ms'] * 1000).is_integer():        errors.append("dt_ms must be a whole number of microseconds")
    ########################################################################################
    # Ranges
    for lo, hi in [('phi_min', 'phi_max'), ('toc_min', 'toc_max'), ('zp_min', 'zp_max')]:
        if data[lo] >= data[hi]: errors.append(f"{lo} must be less than {hi}")
    if not (0.0 <= data['phi_min'] and data['phi_max'] <= 1.0): errors.append("porosity range must sit inside [0, 1]")
    if not (0.0 <= data['null_fraction'] < 0.5):                 errors.append("null_fraction must be in [0, 0.5)")
    if data['md_step_ft'] <= 0:                                  errors.append("md_step_ft must be > 0")
    if not (data['rho_fluid'] < data['rho_matrix']):             errors.append("rho_fluid must be less than rho_matrix")
    ########################################################################################
    # Horizons and zones
    n_h  = len(data['horizon_names'])
    means = np.asarray(data['horizon_mean_time_ms'], dtype=float)
    if n_h < 1: errors.append("need at least one horizon")
    if len(set(data['horizon_names'])) != n_h: errors.append("horizon_names must be unique")
    for key in ['horizon_mean_time_ms', 'horizon_relief_ms']:
        if len(data[key]) != n_h: errors.append(f"{key} needs {n_h} values (one per horizon), got {len(data[key])}")
    if n_h > 1 and np.any(np.diff(means) <= 0): errors.append("horizon_mean_time_ms must be strictly increasing (shallow to deep)")
    if len(means) and (means.min() <= data['t_start_ms'] or means.max() >= data['t_end_ms']):
        errors.append("every horizon mean time must sit inside (t_start_ms, t_end_ms)")
    for key in ['zone_names', 'zone_phi_mean', 'zone_phi_std', 'zone_toc_mean', 'zone_toc_std', 'zone_vp_matrix_fts']:
        if len(data[key]) != n_h + 1: errors.append(f"{key} needs {n_h + 1} values (n_horizons + 1 zones), got {len(data[key])}")
    if len(data['zone_phi_mean']) == n_h + 1:
        phi = np.asarray(data['zone_phi_mean'], dtype=float)
        if np.any(phi < data['phi_min']) or np.any(phi > data['phi_max']): errors.append("zone_phi_mean values must sit inside [phi_min, phi_max]")
        toc = np.asarray(data['zone_toc_mean'], dtype=float)
        if np.any(toc < data['toc_min']) or np.any(toc > data['toc_max']): errors.append("zone_toc_mean values must sit inside [toc_min, toc_max]")
        if np.any(np.asarray(data['zone_phi_std'], dtype=float) < 0) or np.any(np.asarray(data['zone_toc_std'], dtype=float) < 0):
            errors.append("zone standard deviations must be >= 0")
        if np.any(np.asarray(data['zone_vp_matrix_fts'], dtype=float) <= 0): errors.append("zone_vp_matrix_fts must be > 0")
    ########################################################################################
    # Wells
    n_w = len(data['well_names'])
    if n_w < 1: errors.append("need at least one well")
    if len(set(data['well_names'])) != n_w: errors.append("well_names must be unique")
    if any(len(str(w)) > 16 for w in data['well_names']): errors.append("well names must be 16 characters or less (Excel sheet name limit)")
    if len(data['well_inlines']) != n_w or len(data['well_xlines']) != n_w:
        errors.append("well_inlines and well_xlines need one value per well")
    else:
        il_max = data['inline_start'] + data['n_inlines'] - 1
        xl_max = data['xline_start'] + data['n_xlines'] - 1
        for name, il, xl in zip(data['well_names'], data['well_inlines'], data['well_xlines']):
            if not (data['inline_start'] <= int(il) <= il_max and data['xline_start'] <= int(xl) <= xl_max):
                errors.append(f"well {name} at inline {il}, xline {xl} is outside the survey "
                              f"({data['inline_start']}-{il_max}, {data['xline_start']}-{xl_max})")
    targets = data['target_horizons']
    if len(targets) != 2: errors.append("target_horizons needs exactly two names: top and base of the logged interval")
    elif not all(t in data['horizon_names'] for t in targets): errors.append("target_horizons must be names from horizon_names")
    elif data['horizon_names'].index(targets[0]) >= data['horizon_names'].index(targets[1]):
        errors.append("first target horizon must be shallower than the second")
    ########################################################################################
    # Size guard, so nobody accidentally writes a multi-GB file
    n_t = int(round((data['t_end_ms'] - data['t_start_ms']) / data['dt_ms'])) + 1
    segy_mb = data['n_inlines'] * data['n_xlines'] * (240 + 4 * n_t) / 1e6
    data['n_samples'] = n_t
    data['segy_size_mb'] = round(segy_mb, 1)
    if segy_mb > data['max_segy_mb'] and not data['allow_large']:
        errors.append(f"SEG-Y would be about {segy_mb:.0f} MB (limit {data['max_segy_mb']:.0f} MB). "
                      f"Shrink the grid or set allow_large")
    ########################################################################################
    if errors: raise ValueError("Input check failed:\n  - " + "\n  - ".join(errors))
    return data

############################################################################################
def checkDataset(manifestPath='', max_null_fraction=0.2, max_out_of_range_fraction=0.01):
    """
    Validate a dataset (synthetic or real) described by a manifest JSON before modeling.
    :param manifestPath:              Path to the dataset manifest
    :param max_null_fraction:         Max allowed fraction of nulls in any well log column
    :param max_out_of_range_fraction: Max allowed fraction of seismic samples outside [zp_min, zp_max]
    :return:                          dict with passed, errors, warnings, summary
    """
    ########################################################################################
    import segyio
    errors, warnings, summary = [], [], {}
    with open(manifestPath) as f: manifest = json.load(f)
    root = os.path.dirname(os.path.abspath(manifestPath))
    geo, files, ranges = manifest['geometry'], manifest['files'], manifest['ranges']
    path = lambda name: os.path.join(root, name)
    ########################################################################################
    for key, name in files.items():
        if not os.path.exists(path(name)): errors.append(f"missing {key} file: {name}")
    if errors: return {'passed': False, 'errors': errors, 'warnings': warnings, 'summary': summary}
    ########################################################################################
    # Seismic volume
    with segyio.open(path(files['seismic']), iline=geo['iline_byte'], xline=geo['xline_byte']) as seg:
        seg.mmap()
        n_il, n_xl, n_t = len(seg.ilines), len(seg.xlines), len(seg.samples)
        dt = segyio.tools.dt(seg) / 1000.0
        vol = segyio.tools.cube(seg)
        if (n_il, n_xl) != (geo['n_inlines'], geo['n_xlines']):
            errors.append(f"seismic grid {n_il}x{n_xl} does not match manifest {geo['n_inlines']}x{geo['n_xlines']}")
        if seg.ilines[0] != geo['inline_start'] or seg.xlines[0] != geo['xline_start']:
            errors.append("seismic inline/xline numbering does not match manifest")
        if n_t != geo['n_samples'] or not np.isclose(dt, geo['dt_ms']):
            errors.append(f"seismic has {n_t} samples at {dt} ms, manifest says {geo['n_samples']} at {geo['dt_ms']} ms")
        if not np.isclose(seg.samples[0], geo['t_start_ms']):
            errors.append(f"seismic starts at {seg.samples[0]} ms, manifest says {geo['t_start_ms']} ms")
    if not np.all(np.isfinite(vol)): errors.append("seismic volume contains NaN or inf")
    outside = np.mean((vol < ranges['zp_min']) | (vol > ranges['zp_max']))
    if outside > max_out_of_range_fraction:
        errors.append(f"{outside:.1%} of seismic samples sit outside Zp range {ranges['zp_min']}-{ranges['zp_max']}")
    summary['seismic'] = {'shape': list(vol.shape), 'dt_ms': dt, 'zp_p01': float(np.percentile(vol, 1)),
                          'zp_p99': float(np.percentile(vol, 99))}
    ########################################################################################
    # Horizons
    df_h = pd.read_excel(path(files['horizons']), sheet_name=manifest['sheets']['horizons'])
    missing = [c for c in HORIZON_BASE_COLUMNS + manifest['horizon_names'] if c not in df_h.columns]
    if missing: errors.append(f"horizon sheet missing columns: {missing}")
    else:
        if len(df_h) != geo['n_inlines'] * geo['n_xlines']:
            errors.append(f"horizon sheet has {len(df_h)} rows, expected {geo['n_inlines'] * geo['n_xlines']} (one per trace)")
        H = df_h[manifest['horizon_names']].to_numpy(dtype=float)
        t_end = geo['t_start_ms'] + (geo['n_samples'] - 1) * geo['dt_ms']
        if np.any(np.isnan(H)): warnings.append("horizon sheet has gaps (NaN picks)")
        if np.nanmin(H) < geo['t_start_ms'] or np.nanmax(H) > t_end:
            errors.append("horizon picks fall outside the seismic time window")
        if H.shape[1] > 1 and np.any(np.diff(H, axis=1) < 0):
            errors.append("horizons cross (a deeper horizon is shallower than the one above it somewhere)")
    ########################################################################################
    # Wells
    sheets = pd.read_excel(path(files['well_logs']), sheet_name=None)
    for well in manifest['wells']:
        for sheet, cols in [(well['log_sheet'], WELL_LOG_COLUMNS), (well['seismic_sheet'], WELL_SEISMIC_COLUMNS)]:
            if sheet not in sheets:
                errors.append(f"well {well['name']}: sheet {sheet} not found"); continue
            df = sheets[sheet]
            miss = [c for c in cols if c not in df.columns]
            if miss: errors.append(f"well {well['name']} sheet {sheet} missing columns: {miss}"); continue
            nulls = (df[cols] == NULL_VALUE).mean()
            bad = nulls[nulls > max_null_fraction]
            if len(bad): errors.append(f"well {well['name']} sheet {sheet}: too many nulls in {list(bad.index)}")
        if well['log_sheet'] in sheets and 'MD_KB_ft' in sheets[well['log_sheet']].columns:
            df = sheets[well['log_sheet']]
            if np.any(np.diff(df['MD_KB_ft'].to_numpy()) <= 0): errors.append(f"well {well['name']}: MD is not increasing")
            twt = df['TWT_ms'].replace(NULL_VALUE, np.nan).dropna().to_numpy()
            if np.any(np.diff(twt) < 0): errors.append(f"well {well['name']}: TWT is not monotonic with depth")
            tphi = df['TPHI_High_Res'].replace(NULL_VALUE, np.nan)
            if tphi.min() < ranges['phi_min'] - 0.05 or tphi.max() > ranges['phi_max'] + 0.05:
                warnings.append(f"well {well['name']}: TPHI goes outside the porosity range")
            summary[well['name']] = {'log_rows': int(len(df)), 'md_range_ft': [float(df['MD_KB_ft'].min()), float(df['MD_KB_ft'].max())]}
        if not (geo['inline_start'] <= well['inline'] < geo['inline_start'] + geo['n_inlines']
                and geo['xline_start'] <= well['xline'] < geo['xline_start'] + geo['n_xlines']):
            errors.append(f"well {well['name']} sits outside the survey")
    ########################################################################################
    return {'passed': len(errors) == 0, 'errors': errors, 'warnings': warnings, 'summary': summary}
############################################################################################
