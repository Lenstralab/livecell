# Live Cell Analysis
as used by Gowthaman et al, Brouwer et al, Patel et al and Meeussen et al.

## Installation
If not done already:
- Install python (at least 3.8): https://www.python.org
- Install pip 
- Install git: https://git-scm.com/

## Cloning the repository and installing
    git clone https://github.com/Lenstralab/livecell.git
    cd livecell

# Version of the code used for Meeussen et al.
    git checkout 53bb41a038a7149e6ad910a7405fdb4ea5b188fb
    pip install -e .[tllab_common] --user

# Version of the code used for Patel et al.
    git checkout 13bff5dc90acec9cf1f22cf7711412ee899113c7
    pip install -e . --user

# Version of the code used for Brouwer et al.
    git checkout d521de19239cd95a4decc091751b47b94542a31c
    pip install -e . --user

# Version of the code used for Gowthaman et al.
    git checkout 1edd8b572ebc9944618fe48c23060bb3b6d3b1a3
    pip install -e . --user

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
