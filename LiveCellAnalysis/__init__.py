import sys
import os
from ._version import __version__, __git_commit_hash__
tlpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tllab_common')
if os.path.exists(tlpath):  # Adding local tllab_common to path if it exists
    sys.path.insert(0, tlpath)
