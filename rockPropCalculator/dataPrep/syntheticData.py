############################################################################################
"""
Synthetic 3D petrophysics dataset generator.

Builds a layered earth model from user inputs (survey geometry, horizon times, per-zone
porosity / TOC / velocity ranges), converts it to acoustic impedance with a simple rock
physics relation, and writes the same file types the calculator reads:

    seismic.sgy          inverted-style P-impedance volume (SEG-Y, float32)
    horizons.xlsx        one row per trace: Inline, Xline, X, Y, <horizon times>
    well_logs.xlsx       per well: high/low res logs in MD, plus a seismic-scaled sheet
    truth.npz            true porosity, TOC, Zp and zone cubes (for checking model recovery)
    manifest.json        geometry, file names, ranges, wells; read by checkDataset

Nothing here is tied to a real survey. All values come from the input dict.
Output goes to a gitignored folder and stays local.
"""
############################################################################################
import os
import sys
import json
import numpy as np
import pandas as pd
import segyio
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from os.path import dirname, join, isabs, abspath

try:
    from dataPrep.dataChecks import checkSyntheticInputs, NULL_VALUE, WELL_LOG_COLUMNS, WELL_SEISMIC_COLUMNS
    from dataPrep.inputWrapper import standardInputWrapper
except ImportError:
    from .dataChecks import checkSyntheticInputs, NULL_VALUE, WELL_LOG_COLUMNS, WELL_SEISMIC_COLUMNS
    from .inputWrapper import standardInputWrapper

REPO_ROOT = dirname(dirname(dirname(abspath(__file__))))

############################################################################################
def gaussianField(rng, shape, sigma):
    """
    Smooth random field, zero mean and unit std.
    :param rng:   numpy Generator
    :param shape: Output shape
    :param sigma: Gaussian smoothing per axis (in samples)
    :return:      float32 array
    """
    field = gaussian_filter(rng.standard_normal(shape).astype(np.float32), sigma=sigma, mode='reflect')
    return ((field - field.mean()) / (field.std() + 1e-12)).astype(np.float32)

############################################################################################
def buildEarthModel(data={}, rng=None):
    """
    Horizons, zone cube and true rock properties.
    :param data: Checked inputData
    :param rng:  numpy Generator
    :return:     dict of arrays (t, inlines, xlines, X, Y, horizons, zone, phi, toc, rho, vp, zp)
    """
    ########################################################################################
    n_il, n_xl, n_t = data['n_inlines'], data['n_xlines'], data['n_samples']
    t   = data['t_start_ms'] + np.arange(n_t) * data['dt_ms']
    il  = data['inline_start'] + np.arange(n_il)
    xl  = data['xline_start'] + np.arange(n_xl)
    I, J = np.meshgrid(np.arange(n_il), np.arange(n_xl), indexing='ij')
    X   = data['x_origin'] + J * data['bin_size']
    Y   = data['y_origin'] + I * data['bin_size']
    corr = data['lateral_corr_traces']
    ########################################################################################
    # Horizons: shared structure (dip + relief) plus per-interval thickness changes.
    # Each horizon is forced at least min_thickness_ms below the one above, so nothing crosses.
    structure = (data['structural_relief_ms'] * gaussianField(rng, (n_il, n_xl), corr)
                 + data['dip_ms'] * (I / max(n_il - 1, 1) - 0.5))
    horizons = []
    for k, (mean_t, relief) in enumerate(zip(data['horizon_mean_time_ms'], data['horizon_relief_ms'])):
        h = mean_t + structure + relief * gaussianField(rng, (n_il, n_xl), corr)
        if k > 0: h = np.maximum(h, horizons[-1] + data['min_thickness_ms'])
        horizons.append(np.clip(h, t[0], t[-1]))
    horizons = np.asarray(horizons, dtype=np.float32)
    ########################################################################################
    # Zone index per sample: 0 above the first horizon, n_h below the last
    zone = np.zeros((n_il, n_xl, n_t), dtype=np.int8)
    for h in horizons: zone += (t[None, None, :] >= h[:, :, None]).astype(np.int8)
    ########################################################################################
    # Properties: zone mean + std * (lateral trend + vertical detail)
    n_z = len(data['zone_names'])
    def zoneProperty(means, stds):
        lateral  = np.stack([gaussianField(rng, (n_il, n_xl), corr) for _ in range(n_z)])
        vertical = gaussianField(rng, (n_il, n_xl, n_t), (1.0, 1.0, 2.0))
        mu, sd = np.asarray(means, np.float32)[zone], np.asarray(stds, np.float32)[zone]
        return mu + sd * (0.8 * lateral[zone, I[:, :, None], J[:, :, None]] + 0.6 * vertical)
    phi = np.clip(zoneProperty(data['zone_phi_mean'], data['zone_phi_std']), data['phi_min'], data['phi_max'])
    toc = np.clip(zoneProperty(data['zone_toc_mean'], data['zone_toc_std']), data['toc_min'], data['toc_max'])
    ########################################################################################
    # Rock physics: kerogen volume from TOC, bulk density by mixing, Vp drops with porosity and kerogen
    rho_m, rho_f, rho_k = data['rho_matrix'], data['rho_fluid'], data['rho_kerogen']
    vk  = np.clip(toc / 100.0 * rho_m / rho_k, 0.0, 0.6)
    rho = (1.0 - phi) * ((1.0 - vk) * rho_m + vk * rho_k) + phi * rho_f
    vp  = (np.asarray(data['zone_vp_matrix_fts'], np.float32)[zone]
           * (1.0 - data['vp_phi_sensitivity'] * phi) * (1.0 - data['vp_kerogen_sensitivity'] * vk))
    zp  = rho * vp
    ########################################################################################
    return {'t': t, 'inlines': il, 'xlines': xl, 'X': X, 'Y': Y, 'horizons': horizons, 'zone': zone,
            'phi': phi.astype(np.float32), 'toc': toc.astype(np.float32), 'vk': vk.astype(np.float32),
            'rho': rho.astype(np.float32), 'vp': vp.astype(np.float32), 'zp': zp.astype(np.float32)}

############################################################################################
def buildSeismicVolume(data={}, model={}, rng=None):
    """
    Blur true Zp so it looks like an inverted volume, add noise, clip to the Zp range.
    :return: (seismic cube float32, fraction of samples clipped)
    """
    ########################################################################################
    lat, vert = data['seismic_lateral_smooth'], data['seismic_vertical_smooth']
    seis = gaussian_filter(model['zp'], sigma=(lat, lat, vert), mode='nearest')
    seis = seis * (1.0 + data['seismic_noise_frac'] * rng.standard_normal(seis.shape).astype(np.float32))
    clipped = float(np.mean((seis < data['zp_min']) | (seis > data['zp_max'])))
    return np.clip(seis, data['zp_min'], data['zp_max']).astype(np.float32), clipped

############################################################################################
def writeSegy(path='', data={}, model={}, seismic=None):
    """
    Write the cube as SEG-Y with inline/xline at the bytes given in the inputs.
    Traces are inline-major, which is what the calculator's reshape expects.
    """
    ########################################################################################
    spec = segyio.spec()
    spec.ilines, spec.xlines = model['inlines'], model['xlines']
    spec.samples = model['t']
    spec.sorting = segyio.TraceSortingFormat.INLINE_SORTING
    spec.format  = 5  # IEEE float32
    dt_us = int(round(data['dt_ms'] * 1000))
    n_xl  = len(model['xlines'])
    ########################################################################################
    with segyio.create(path, spec) as f:
        for i, il in enumerate(model['inlines']):
            for j, xl in enumerate(model['xlines']):
                k = i * n_xl + j
                f.header[k] = {
                    data['iline_byte']: int(il),
                    data['xline_byte']: int(xl),
                    segyio.TraceField.CDP_X: int(round(model['X'][i, j])),
                    segyio.TraceField.CDP_Y: int(round(model['Y'][i, j])),
                    segyio.TraceField.TRACE_SAMPLE_INTERVAL: dt_us,
                    segyio.TraceField.TRACE_SAMPLE_COUNT: len(model['t']),
                    segyio.TraceField.DelayRecordingTime: int(model['t'][0]),
                }
                f.trace[k] = seismic[i, j, :]
        f.bin.update(hdt=dt_us, hns=len(model['t']), format=5)

############################################################################################
def buildWellLogs(data={}, model={}, well_name='', inline=0, xline=0, rng=None):
    """
    Vertical well at (inline, xline). Converts time to MD with the true velocity, samples the
    target interval at md_step_ft, adds measurement noise, and builds low-res versions.
    :return: (log DataFrame in MD, seismic-scaled DataFrame in time)
    """
    ########################################################################################
    i, j = inline - data['inline_start'], xline - data['xline_start']
    t = model['t']
    trace = {key: model[key][i, j, :].astype(float) for key in ['phi', 'toc', 'vk', 'rho', 'vp']}
    ########################################################################################
    # Time to depth: two-way time, velocity in ft/s
    dz      = trace['vp'] * (data['dt_ms'] / 1000.0) / 2.0
    depth_t = data['md_at_t_start_ft'] + np.concatenate([[0.0], np.cumsum(dz[:-1])])
    ########################################################################################
    # Logged interval: target top to target base, padded
    k_top  = data['horizon_names'].index(data['target_horizons'][0])
    k_base = data['horizon_names'].index(data['target_horizons'][1])
    t_top  = max(t[0], float(model['horizons'][k_top, i, j]) - data['well_pad_ms'])
    t_base = min(t[-1], float(model['horizons'][k_base, i, j]) + data['well_pad_ms'])
    step   = data['md_step_ft']
    md_top = np.ceil(np.interp(t_top, t, depth_t) / step) * step
    md_bot = np.floor(np.interp(t_base, t, depth_t) / step) * step
    md     = np.arange(md_top, md_bot + step / 2, step)
    twt    = np.interp(md, depth_t, t)
    ########################################################################################
    # True values at each MD (nearest sample keeps the sharp layer boundaries), then noise
    idx  = np.clip(np.round((twt - t[0]) / data['dt_ms']).astype(int), 0, len(t) - 1)
    nf   = data['log_noise_frac']
    noisy = lambda x, scale=1.0: x * (1.0 + scale * nf * rng.standard_normal(len(x)))
    phi, toc, vk = trace['phi'][idx], trace['toc'][idx], trace['vk'][idx]
    rhoz   = noisy(trace['rho'][idx], 0.3)
    p_velo = noisy(trace['vp'][idx])
    tphi_h = np.clip(noisy(phi), 0.0, 1.0)
    toc_l  = np.clip(noisy(toc), 0.0, None)
    dpor_h = (data['rho_matrix'] - rhoz) / (data['rho_matrix'] - data['rho_fluid'])  # reads high in organic rock, like real DPOR
    nphi_h = np.clip(phi + 0.35 * vk + nf * 0.05 * rng.standard_normal(len(md)), 0.0, 1.0)
    sigma  = data['low_res_smooth_ft'] / step
    low    = lambda x: gaussian_filter1d(x, sigma=sigma, mode='nearest')
    ########################################################################################
    x_well, y_well = float(model['X'][i, j]), float(model['Y'][i, j])
    df_log = pd.DataFrame({
        'MD_KB_ft': md, 'TWT_ms': twt, 'X': x_well, 'Y': y_well,
        'P_Velo': p_velo, 'Zp': rhoz * p_velo,
        'DPOR_Low_Res': low(dpor_h), 'DPOR_High_Res': dpor_h,
        'NPHI_Low_Res': low(nphi_h), 'NPHI_High_Res': nphi_h,
        'TPHI_Low_Res': low(tphi_h), 'TPHI_High_Res': tphi_h,
        'RHOZ': rhoz, 'TOC': toc_l,
    })[WELL_LOG_COLUMNS]
    ########################################################################################
    # Seismic-scaled: average the logs inside each seismic sample window (upscaling)
    t_bin = data['t_start_ms'] + np.round((twt - data['t_start_ms']) / data['dt_ms']) * data['dt_ms']
    df_seis = (df_log.drop(columns=['MD_KB_ft', 'TWT_ms', 'Zp']).assign(Time_ms=t_bin)
               .groupby('Time_ms', as_index=False).mean())[WELL_SEISMIC_COLUMNS]
    ########################################################################################
    # Random nulls in the measured log curves only (not MD, TWT or coordinates)
    if data['null_fraction'] > 0:
        log_cols = WELL_LOG_COLUMNS[4:]
        mask = rng.random((len(df_log), len(log_cols))) < data['null_fraction']
        df_log[log_cols] = np.where(mask, NULL_VALUE, df_log[log_cols].to_numpy())
    ########################################################################################
    return df_log, df_seis

############################################################################################
def generateSyntheticDataset(inputData={}):
    """
    Build and write a full synthetic dataset.
    :param inputData: Flat dict (from standardInputWrapper). Missing keys fall back to generic defaults.
    :return:          (manifest dict, manifest path)
    """
    ########################################################################################
    data = checkSyntheticInputs(inputData=inputData)
    rng  = np.random.default_rng(data['random_seed'])
    out  = data['output_dir'] if isabs(data['output_dir']) else join(REPO_ROOT, data['output_dir'])
    os.makedirs(out, exist_ok=True)
    files = {'seismic': 'seismic.sgy', 'horizons': 'horizons.xlsx', 'well_logs': 'well_logs.xlsx',
             'truth': 'truth.npz'}
    ########################################################################################
    print(f"Building earth model: {data['n_inlines']} x {data['n_xlines']} x {data['n_samples']} "
          f"(SEG-Y about {data['segy_size_mb']} MB)")
    model = buildEarthModel(data=data, rng=rng)
    seismic, clipped = buildSeismicVolume(data=data, model=model, rng=rng)
    if clipped > 0.01:
        print(f"Warning: {clipped:.1%} of Zp samples were clipped to [{data['zp_min']}, {data['zp_max']}]. "
              f"Widen the Zp range or adjust zone properties")
    writeSegy(path=join(out, files['seismic']), data=data, model=model, seismic=seismic)
    ########################################################################################
    # Horizons, one row per trace, inline-major
    I, J = np.meshgrid(model['inlines'], model['xlines'], indexing='ij')
    df_h = pd.DataFrame({'Inline': I.ravel(), 'Xline': J.ravel(),
                         'X': model['X'].ravel(), 'Y': model['Y'].ravel()})
    for name, h in zip(data['horizon_names'], model['horizons']): df_h[name] = np.round(h.ravel(), 3)
    df_h.to_excel(join(out, files['horizons']), sheet_name='Horizon_Data', index=False)
    ########################################################################################
    wells = []
    with pd.ExcelWriter(join(out, files['well_logs'])) as writer:
        for name, il, xl in zip(data['well_names'], data['well_inlines'], data['well_xlines']):
            df_log, df_seis = buildWellLogs(data=data, model=model, well_name=name, inline=int(il), xline=int(xl), rng=rng)
            df_log.to_excel(writer, sheet_name=name, index=False)
            df_seis.to_excel(writer, sheet_name=f"{name}_Seismic_Scaled", index=False)
            wells.append({'name': name, 'inline': int(il), 'xline': int(xl), 'log_sheet': name,
                          'seismic_sheet': f"{name}_Seismic_Scaled",
                          'md_range_ft': [float(df_log['MD_KB_ft'].min()), float(df_log['MD_KB_ft'].max())],
                          'time_range_ms': [float(df_seis['Time_ms'].min()), float(df_seis['Time_ms'].max())]})
    ########################################################################################
    np.savez_compressed(join(out, files['truth']), phi=model['phi'], toc=model['toc'], zp=model['zp'],
                        zone=model['zone'], horizons=model['horizons'], seismic=seismic)
    ########################################################################################
    manifest = {
        'name': data['name'], 'synthetic': True, 'random_seed': data['random_seed'],
        'files': files,
        'sheets': {'horizons': 'Horizon_Data'},
        'geometry': {key: data[key] for key in ['n_inlines', 'n_xlines', 'inline_start', 'xline_start', 'bin_size',
                                                'x_origin', 'y_origin', 'iline_byte', 'xline_byte',
                                                't_start_ms', 'dt_ms', 'n_samples']},
        'horizon_names': data['horizon_names'],
        'zone_names': data['zone_names'],
        'target_horizons': data['target_horizons'],
        'ranges': {key: data[key] for key in ['phi_min', 'phi_max', 'toc_min', 'toc_max', 'zp_min', 'zp_max']},
        'null_value': NULL_VALUE,
        'zp_clipped_fraction': round(clipped, 5),
        'wells': wells,
        'inputs': {k: v for k, v in data.items() if not isinstance(v, np.ndarray)},
    }
    manifest_path = join(out, 'manifest.json')
    with open(manifest_path, 'w') as f: json.dump(manifest, f, indent=2, default=float)
    print(f"Synthetic dataset written to {out}")
    return manifest, manifest_path

############################################################################################
def main():
    ########################################################################################
    thisDirName = dirname(abspath(__file__))
    fileName = sys.argv[1] if len(sys.argv) > 1 else join(thisDirName, '..', '..', 'tests', 'in_jsons', 'synthetic_small.json')
    with open(fileName) as json_file: inJson = json.load(json_file)
    generateSyntheticDataset(inputData=standardInputWrapper(inJson=inJson))

############################################################################################
if __name__ == "__main__":
    main()
############################################################################################
