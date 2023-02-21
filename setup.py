import setuptools
import os

version = '2023.1.1'
tllab_common_version = '9946c603e833411339b8fb38bc26095ddb756b31'


with open("README.md", "r") as fh:
    long_description = fh.read()

try:
    import tllab_common

    if tllab_common.__git_commit_hash__ == tllab_common_version:
        tllab_common = []
    else:
        tllab_common = ['tllab_common[transforms]@git+https://github.com/Lenstralab/tllab_common.git'
                        f'@{tllab_common_version}']
except (ImportError, AttributeError):
    tllab_common = ['tllab_common[transforms]@git+https://github.com/Lenstralab/tllab_common.git'
                    f'@{tllab_common_version}']

with open(os.path.join(os.path.dirname(__file__), 'LiveCellAnalysis', '_version.py'), 'w') as f:
    f.write(f"__version__ = '{version}'\n")
    try:
        with open(os.path.join(os.path.dirname(__file__), '.git', 'HEAD')) as g:
            head = g.read().split(':')[1].strip()
        with open(os.path.join(os.path.dirname(__file__), '.git', head)) as h:
            f.write("__git_commit_hash__ = '{}'\n".format(h.read().rstrip('\n')))
    except Exception:
        f.write(f"__git_commit_hash__ = 'unknown'\n")


setuptools.setup(
    name="livecellanalysis",
    version=version,
    author="Lenstra lab NKI",
    author_email="t.lenstra@nki.nl",
    description="Live cell analysis code for the Lenstra lab.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.rhpc.nki.nl/LenstraLab/LiveCellAnalysis",
    packages=['LiveCellAnalysis'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    tests_require=['pytest-xdist'],
    install_requires=['ipython', 'numpy', 'scipy', 'tqdm', 'matplotlib', 'parfor', 'pyyaml', 'scikit-image', 'psutil',
                      'Pillow', 'hidden_markov', 'rtree', 'tiffwrite', 'pyimagej', 'stardist', 'csbdeep', 'tensorflow',
                      'lfdfiles; platform_system!="Darwin"'],
    entry_points={'console_scripts': ['bootstrap_test=LiveCellAnalysis.bootstrap_test:main',
                                      'listpyedit=LiveCellAnalysis.listpyedit:main',
                                      'livecell_correlationfunctions=LiveCellAnalysis.'
                                      'pipeline_livecell_correlationfunctions:main',
                                      'livecell_track_movies=LiveCellAnalysis.pipeline_livecell_track_movies:main',
                                      'orbital2list=LiveCellAnalysis.orbital2listpy:main']},
    extras_require={'tllab_common': tllab_common,
                    'pytrackmate': 'pytrackmate@git+https://gitlab.rhpc.nki.nl/LenstraLab/pytrackmate.git'},
    package_data={'': ['*.yml', 'trackmate.jy']},
    include_package_data=True,
)
