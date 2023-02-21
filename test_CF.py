#!/usr/bin/python

import sys
import os
import yaml
from LiveCellAnalysis.pipeline_livecell_correlationfunctions import pipeline_correlation_functions
from tllab_common.misc import getConfig
from tllab_common.wimread import imread

fname = os.path.realpath(__file__)
test_files = os.path.join(os.path.dirname(fname), 'test_files')

# This file defines tests to be run to assert the correct working of our scripts
# after updates. Add a test below as a function, name starting with 'test', and 
# optionally using 'assert'.
#
# Place extra files used for these tests in the folder test_files, add imports 
# above this text.
#
# Then navigate to the directory containing this file and run ./test.py directly
# from the terminal. If you see red text then something is wrong and you need to
# fix the code before committing to gitlab.
#
#wp@tl20200124


## ----- add your tests here -----
    
def make_test_pipeline_CF(parameter_file):
    def pipeline_fun(tmp_path):
        tmp_path = str(tmp_path)
        parameters = getConfig(os.path.join(test_files, parameter_file))
        parameters['outputfolder'] = tmp_path
        tmp_parameter_file = os.path.join(tmp_path, 'params.yml')

        with open(tmp_parameter_file, 'w') as f:
            yaml.dump(parameters, f)

        parameters = pipeline_correlation_functions(tmp_parameter_file)

        files = ('pipeline_livecell_correlationfunctions.py', 'autocorrelation_fit_red_uncorrected_analog.pdf',
                 'correlation_functions_analog.pdf', 'Heatmap_analog.pdf')

        assert os.path.exists(parameters['outputfolder']), 'Output folder has not been generated'
        assert len(os.listdir(parameters['outputfolder']))>4, 'There aren''t enough files in the output folder'
        for file in files:
            assert os.path.exists(parameters['file']+'_'+file),\
                'File {} has not been generated'.format(parameters['file']+'_'+file)
    return pipeline_fun

#test_linda  = make_test_pipeline_CF('pipeline_livecell_CF_Linda.yml')
#test_bibi   = make_test_pipeline_CF('pipeline_livecell_CF_Bibi.yml')
test_heta   = make_test_pipeline_CF('pipeline_livecell_correlationfunctions_Heta.yml')

## ----- This part runs the tests -----
    
if __name__ == '__main__':
    if len(sys.argv) < 2:
        py = ['']
    else:
        py = sys.argv[1:]

    for p in py:
        print(f'Testing using python {p}')
        print(f'Loaded tllab_common code from {__file__}')
        os.system(f'python{p} -m pytest -n=12 -p no:warnings --verbose {fname}')
        print('')

    imread.kill_vm()
