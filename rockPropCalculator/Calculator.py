###############################################################################
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings
warnings.filterwarnings('ignore')
import json
import asyncio
import mlflow
from os.path import dirname, join, isabs
from dataPrep.inputWrapper import standardInputWrapper
from dataPrep.outputWrapper import standardOutputWrapper
from dataPrep.mlFlowConfig import setupMLFlow
from dataPrep.dataChecks import checkDataset
from dataPrep.syntheticData import generateSyntheticDataset, REPO_ROOT

###############################################################################
def checkData(inputData={}):
    """
    Calculator-level input checks. Synthetic generation inputs are checked
    separately inside generateSyntheticDataset (checkSyntheticInputs).
    """
    ###########################################################################
    inputData['name']         = inputData.get('name', 'Petrophysical_Results')
    inputData['dataset_mode'] = str(inputData.get('dataset_mode', 'synthetic')).lower()
    inputData['max_null_fraction']         = float(inputData.get('max_null_fraction', 0.2))
    inputData['max_out_of_range_fraction'] = float(inputData.get('max_out_of_range_fraction', 0.01))
    ###########################################################################
    if inputData['dataset_mode'] not in ('synthetic', 'existing'):
        raise ValueError(f"dataset_mode must be 'synthetic' or 'existing', got {inputData['dataset_mode']!r}")
    if inputData['dataset_mode'] == 'existing':
        path = inputData.get('manifest_path', '')
        if not path: raise ValueError("dataset_mode 'existing' needs manifest_path")
        inputData['manifest_path'] = path if isabs(path) else join(REPO_ROOT, path)
        if not os.path.exists(inputData['manifest_path']):
            raise ValueError(f"manifest not found: {inputData['manifest_path']}")
    ###########################################################################
    return inputData

###############################################################################
async def calculate(inJson={}):
    ###########################################################################
    setupMLFlow()
    ###########################################################################
    try:
        inputData      = standardInputWrapper(inJson=inJson)
        output_wrapper = standardOutputWrapper()
        inputData      = checkData(inputData=inputData)
        #######################################################################
        with mlflow.start_run(run_name=f"pipeline_{inputData['name']}") as run:
            mlflow.set_tag('dataset_mode', inputData['dataset_mode'])
            ###################################################################
            # Step 1: get a dataset (build synthetic or point at an existing one)
            if inputData['dataset_mode'] == 'synthetic':
                manifest, manifest_path = generateSyntheticDataset(inputData=inputData)
            else:
                manifest_path = inputData['manifest_path']
                with open(manifest_path) as f: manifest = json.load(f)
            ###################################################################
            # Step 2: data checks before anything touches a model
            report = checkDataset(manifestPath=manifest_path,
                                  max_null_fraction=inputData['max_null_fraction'],
                                  max_out_of_range_fraction=inputData['max_out_of_range_fraction'])
            geo = manifest['geometry']
            mlflow.log_params({'n_inlines': geo['n_inlines'], 'n_xlines': geo['n_xlines'],
                               'n_samples': geo['n_samples'], 'dt_ms': geo['dt_ms'],
                               'n_wells': len(manifest['wells']), 'synthetic': manifest.get('synthetic', False)})
            mlflow.log_metrics({'data_check_passed': float(report['passed']),
                                'n_errors': len(report['errors']), 'n_warnings': len(report['warnings']),
                                'zp_p01': report['summary'].get('seismic', {}).get('zp_p01', 0.0),
                                'zp_p99': report['summary'].get('seismic', {}).get('zp_p99', 0.0)})
            mlflow.log_artifact(manifest_path, artifact_path='dataset')
            if not report['passed']:
                raise ValueError("Data check failed:\n  - " + "\n  - ".join(report['errors']))
            ###################################################################
            # Step 3: model training and 3D prediction plug in here (next refactor)
            ###################################################################
            output_wrapper.add_param(name='data_check_passed', value=report['passed'], value_type=1)
            output_wrapper.add_param(name='manifest_path', value=manifest_path, value_type=3)
            output_wrapper.add_param(name='mlflow_run_id', value=run.info.run_id, value_type=3)
            output_wrapper.add_table(name='wells', array=[w['name'] for w in manifest['wells']])
            output_wrapper.add_table(name='warnings', array=report['warnings'])
        #######################################################################
        print("Completed Calculation")
        return output_wrapper
    except Exception as e:
        print("\nError Message: " + str(e) + "\n")
        return str(e)

###############################################################################
async def main():
    ###########################################################################
    thisDirName = dirname(__file__)
    fileDirName = join(thisDirName, '..', 'tests', 'in_jsons', 'testCase01.json')
    ###########################################################################
    with open(fileDirName) as json_file:
        inJson = json.load(json_file)
    output_wrapper = await calculate(inJson=inJson)
    if not isinstance(output_wrapper, str):
        print(json.dumps(output_wrapper.flatten(), indent=2))

###############################################################################
if __name__ == "__main__":
    asyncio.run(main())
###############################################################################
