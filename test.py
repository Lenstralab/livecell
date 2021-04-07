#!/usr/bin/python

import sys, os, shutil, yaml, copy, tifffile
import numpy as np
from datetime import datetime
from findcells import findcells
from misc import getConfig
from pipeline_livecell_track_movies import pipeline, getPaths, calculate_general_parameters
from wimread import imread

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

## ------ first some tests to explain the principle -----

def test_sum():
    assert sum([1, 2, 3])==6, "Should be 6"

def test_sum_tuple():
    assert sum((1, 2, 2))==5, "Should be 5"
    
def test_sum_np():
    assert np.sum(np.array((1, 2, 3)))==6, 'Should be 6'
    
## ----- Then real tests ----- add your tests here -----
    
def test_findcell_a():
    a = tifffile.imread(os.path.join(test_files, 'findcell.a.tif'))
    c, n = findcells(a[0], a[1], ccdist=150, thres=1, removeborders=True)
    assert np.all(c==a[2]), 'Cellmask wrong'
    assert np.all(n==a[3]), 'Nucleusmask wrong'
    
def make_test_pipeline(parameter_file):
    def pipeline_fun():
        date = datetime.now().strftime('%Y%m%d_%H%M%S')
        outputfolder = os.path.join(test_files, 'test_results_{}_{}'.format(os.path.splitext(parameter_file)[0], date))
        parameters = getConfig(os.path.join(test_files, parameter_file))
        parameters['outputfolder'] = outputfolder
        tmp_parameter_file = os.path.join(outputfolder, 'params.yml')

        if not os.path.exists(outputfolder):
            os.makedirs(outputfolder)
        with open(tmp_parameter_file, 'w') as f:
            yaml.dump(parameters, f)
        if parameters['TSregfile']:
            p = copy.deepcopy(parameters)
            calculate_general_parameters(p)
            for exp in range(0, p['lenExpList']):
                p['pathIn'] = p['expList'][exp]
                getPaths(p)

                if not os.path.exists(p['pathOut']):
                    os.makedirs(p['pathOut'])

                for file in parameters['TSregfile'][exp]:
                    shutil.copyfile(os.path.join(test_files, file+'.txt'),
                                os.path.join(p['pathOut'], file+'.txt'))

        parameters = pipeline(tmp_parameter_file)

        files = ('max.tif', 'pipeline_livecell_track_movies.py', 'sum.tif', 'params.yml')

        assert os.path.exists(parameters['pathOut']), 'Output folder has not been generated'
        assert len(os.listdir(parameters['pathOut']))>4, 'There aren''t enough files in the output folder'
        for file in files:
            assert os.path.exists(parameters['expPathOut']+'_'+file),\
                'File {} has not been generated'.format(parameters['expPathOut']+'_'+file)

        shutil.rmtree(parameters['outputfolder'], True)
    return pipeline_fun

#test_ineke  = make_test_pipeline('pipeline_livecell_track_movies_Ineke.yml') # takes very long
test_linda  = make_test_pipeline('pipeline_livecell_track_movies_Linda.yml')
#test_tineke = make_test_pipeline('pipeline_tineke.yml')
test_bibi   = make_test_pipeline('pipeline_livecell_track_movies_Bibi.yml')
test_heta   = make_test_pipeline('pipeline_livecell_track_movies_Heta.yml')

## ----- This part runs the tests -----
    
if __name__ == '__main__':
    if len(sys.argv)<2:
        py = ['2']
    else:
        py = sys.argv[1:]

    for p in py:
        print('Testing using python {}'.format(p))
        os.system('python{} -m pytest -n=12 -p no:warnings --verbose {}'.format(p, fname))
        print('')

    imread.kill_vm()
