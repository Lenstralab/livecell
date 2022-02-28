# Live Cell Analysis
as used by Gowthaman et al. and Brouwer et al.

## Cloning the repository
Install git: https://git-scm.com/

    git clone https://github.com/Lenstralab/livecell.git
    cd livecell

# Code used for Gowthaman et al.
    git checkout 1edd8b572ebc9944618fe48c23060bb3b6d3b1a3

# Code used for Brouwer et al.
    git checkout d521de19239cd95a4decc091751b47b94542a31c

## Installation
If not done already:
- Install python (at least 3.7): https://www.python.org
- Install pip and git

Then install the livecell script (up to 5 minutes):

    pip install -e livecell_analysis --user

## Usage
### Track Movies
Prepare your parameter file, see pipeline_livecell_track_movies_parameters_template.yml for an example.

From the terminal:

    livecell_track_movies /path/to/parameter_file.yml

or:
    
    cd LiveCellAnalysis
    ./pipeline_livecell_track_movies.py /path/to/parameter_file.yml
or:
   
    cd LiveCellAnalysis
    ipython
    %run pipeline_livecell_track_movies.py /path/to/parameter_file.yml

### Correlation Functions
Prepare you parameter file, see pipeline_livecell_correlationfunctions_parameters_template.yml for an example.

From the terminal:

    livecell_correlationfunctions /path/to/parameter_file.yml
or:

    cd LiveCellAnalysis
    ./pipeline_livecell_correlationfunctions.py /path/to/parameter_file.yml
or:
   
    cd LiveCellAnalysis
    ipython
    %run pipeline_livecell_correlationfunctions.py /path/to/parameter_file.yml

### Bootstrap Testing
Find out how to use from the terminal:

    bootstrap_test --help

### Edit list.py files through a GUI
Find out how to use from the terminal:

    listpyedit --help

### Convert orbital tracking data to tracks and a list.py
Find out how to use from the terminal:

    orbital2listpy --help
