############################################################################################
"""
Input form definition for the UI (served at GET /schema).

Defaults come straight from the data check defaults so there's one source of truth.
Field types:
    number, integer, boolean, string, select     sent as params
    list_number, list_string                     sent as tables
show_if hides a field unless another field has one of the listed values.
"""
############################################################################################
try:
    from dataPrep.dataChecks import SYNTHETIC_DEFAULTS as D, SYNTHETIC_TABLE_DEFAULTS as T
except ImportError:
    from .dataChecks import SYNTHETIC_DEFAULTS as D, SYNTHETIC_TABLE_DEFAULTS as T

############################################################################################
def field(name='', label='', type='number', default=None, help='', options=None, min=None, max=None, show_if=None):
    f = {'name': name, 'label': label or name, 'type': type, 'default': default,
         'kind': 'table' if type.startswith('list_') else 'param'}
    if help: f['help'] = help
    if options is not None: f['options'] = options
    if min is not None: f['min'] = min
    if max is not None: f['max'] = max
    if show_if: f['show_if'] = show_if
    return f

############################################################################################
def getSchema():
    existing = {'dataset_mode': ['existing']}
    synthetic = {'dataset_mode': ['synthetic']}
    return {
        'name': 'petro',
        'title': '3D Petrophysical Prediction',
        'description': 'Bayesian Neural Network prediction of porosity and TOC across a 3D seismic volume, with P10/P90 uncertainty.',
        'groups': [
            {'name': 'Run', 'fields': [
                field('name', 'Run name', 'string', 'Petrophysical_Results'),
                field('dataset_mode', 'Dataset', 'select', 'synthetic', options=['synthetic', 'existing'],
                      help='Build a synthetic survey or load one from a manifest'),
                field('manifest_path', 'Manifest path', 'string', '', show_if=existing,
                      help='Path to manifest.json, relative to the repo'),
                field('random_seed', 'Random seed', 'integer', D['random_seed'], show_if=synthetic),
            ]},
            {'name': 'Survey', 'fields': [
                field('n_inlines', 'Inlines', 'integer', D['n_inlines'], min=2, show_if=synthetic),
                field('n_xlines', 'Crosslines', 'integer', D['n_xlines'], min=2, show_if=synthetic),
                field('inline_start', 'First inline', 'integer', D['inline_start'], show_if=synthetic),
                field('xline_start', 'First crossline', 'integer', D['xline_start'], show_if=synthetic),
                field('bin_size', 'Bin size (ft)', 'number', D['bin_size'], min=0, show_if=synthetic),
                field('t_start_ms', 'Start time (ms)', 'number', D['t_start_ms'], show_if=synthetic),
                field('t_end_ms', 'End time (ms)', 'number', D['t_end_ms'], show_if=synthetic),
                field('dt_ms', 'Sample rate (ms)', 'number', D['dt_ms'], min=0, show_if=synthetic),
            ]},
            {'name': 'Horizons and Zones', 'fields': [
                field('horizon_names', 'Horizon names', 'list_string', T['horizon_names'], show_if=synthetic),
                field('horizon_mean_time_ms', 'Horizon mean times (ms)', 'list_number', T['horizon_mean_time_ms'], show_if=synthetic,
                      help='Shallow to deep'),
                field('horizon_relief_ms', 'Horizon relief (ms)', 'list_number', T['horizon_relief_ms'], show_if=synthetic),
                field('zone_names', 'Zone names', 'list_string', T['zone_names'], show_if=synthetic,
                      help='One more zone than horizons'),
                field('zone_phi_mean', 'Zone porosity mean', 'list_number', T['zone_phi_mean'], show_if=synthetic),
                field('zone_phi_std', 'Zone porosity std', 'list_number', T['zone_phi_std'], show_if=synthetic),
                field('zone_toc_mean', 'Zone TOC mean (wt%)', 'list_number', T['zone_toc_mean'], show_if=synthetic),
                field('zone_toc_std', 'Zone TOC std (wt%)', 'list_number', T['zone_toc_std'], show_if=synthetic),
                field('zone_vp_matrix_fts', 'Zone matrix Vp (ft/s)', 'list_number', T['zone_vp_matrix_fts'], show_if=synthetic),
            ]},
            {'name': 'Wells', 'fields': [
                field('well_names', 'Well names', 'list_string', T['well_names'], show_if=synthetic),
                field('well_inlines', 'Well inlines', 'list_number', T['well_inlines'], show_if=synthetic),
                field('well_xlines', 'Well crosslines', 'list_number', T['well_xlines'], show_if=synthetic),
                field('target_horizons', 'Logged interval (top, base)', 'list_string', T['target_horizons'], show_if=synthetic),
            ]},
            {'name': 'Ranges and Noise', 'fields': [
                field('phi_min', 'Porosity min', 'number', D['phi_min'], min=0, max=1, show_if=synthetic),
                field('phi_max', 'Porosity max', 'number', D['phi_max'], min=0, max=1, show_if=synthetic),
                field('toc_min', 'TOC min', 'number', D['toc_min'], show_if=synthetic),
                field('toc_max', 'TOC max', 'number', D['toc_max'], show_if=synthetic),
                field('zp_min', 'Zp min', 'number', D['zp_min']),
                field('zp_max', 'Zp max', 'number', D['zp_max']),
                field('seismic_noise_frac', 'Seismic noise (fraction)', 'number', D['seismic_noise_frac'], min=0, show_if=synthetic),
                field('log_noise_frac', 'Log noise (fraction)', 'number', D['log_noise_frac'], min=0, show_if=synthetic),
                field('null_fraction', 'Log null fraction', 'number', D['null_fraction'], min=0, max=0.5, show_if=synthetic),
            ]},
        ],
    }
############################################################################################
