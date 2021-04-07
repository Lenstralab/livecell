import re
import yaml
import numpy as np
from scipy import optimize
import copy

histoF=(lambda hr: np.c_[hr[1][1:],hr[0]])

def leastsqWrap(func, x0, force=None, args=(), Dfun=None, col_deriv=0, ftol=1.49012e-08, xtol=1.49012e-08, gtol=0.0, maxfev=0, epsfcn=0.0, factor=100, diag2=None):
    if force is None: force=np.zeros(x0.shape[0])
    wif=np.where(force==0)[0]
    func2=lambda x: func(np.array([x[np.where(wif==i)[0][0]] if i in wif else x0[i] for i in np.r_[:x0.shape[0]]]))
    resFit=optimize.leastsq(func2, x0[wif], args=args, Dfun=Dfun, full_output=True, col_deriv=col_deriv, ftol=ftol, xtol=xtol, gtol=gtol, maxfev=maxfev, epsfcn=epsfcn, factor=factor, diag=diag2)
    x,err=resFit[0],np.diag(resFit[1])**.5
    x=np.array([x[np.where(wif==i)[0][0]] if i in wif else x0[i] for i in np.r_[:x0.shape[0]]])
    err=np.array([err[np.where(wif==i)[0][0]] if i in wif else 0. for i in np.r_[:x0.shape[0]]])
    return x,err,resFit

loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))


def color(text, fmt):
    """ print colored text: print(color('Hello World!', 'r:b'))
        text: text to be colored/decorated
        fmt: string: 'k': black, 'r': red', 'g': green, 'y': yellow, 'b': blue, 'm': magenta, 'c': cyan, 'w': white
            'b'  text color
            '.r' background color
            ':b' decoration: 'b': bold, 'u': underline, 'r': reverse
            for colors also terminal color codes can be used

        example: >> print(color('Hello World!', 'b.208:b'))
                 << Hello world! in blue bold on orange background

        wp@tl20191122
    """

    if not isinstance(fmt, str):
        fmt = str(fmt)

    decorS = [i.group(0) for i in re.finditer('(?<=\:)[a-zA-Z]', fmt)]
    backcS = [i.group(0) for i in re.finditer('(?<=\.)[a-zA-Z]', fmt)]
    textcS = [i.group(0) for i in re.finditer('((?<=[^\.\:])|^)[a-zA-Z]', fmt)]
    backcN = [i.group(0) for i in re.finditer('(?<=\.)\d{1,3}', fmt)]
    textcN = [i.group(0) for i in re.finditer('((?<=[^\.\:\d])|^)\d{1,3}', fmt)]

    t = ('k', 'r', 'g', 'y', 'b', 'm', 'c', 'w')
    d = {'b': 1, 'u': 4, 'r': 7}

    for i in decorS:
        if i.lower() in d:
            text = '\033[{}m{}'.format(d[i.lower()], text)
    for i in backcS:
        if i.lower() in t:
            text = '\033[48;5;{}m{}'.format(t.index(i.lower()), text)
    for i in textcS:
        if i.lower() in t:
            text = '\033[38;5;{}m{}'.format(t.index(i.lower()), text)
    for i in backcN:
        if 0 <= int(i) <= 255:
            text = '\033[48;5;{}m{}'.format(int(i), text)
    for i in textcN:
        if 0 <= int(i) <= 255:
            text = '\033[38;5;{}m{}'.format(int(i), text)

    return text + '\033[0m'

def getConfig(file):
    """ Open a yml parameter file
    """
    with open(file, 'r') as f:
        return yaml.load(f, loader)

def convertParamFile2YML(file):
    """ Convert a py parameter file into a yml file
    """
    with open(file, 'r') as f:
        lines = f.read(-1)
    with open(re.sub('\.py$', '.yml', file), 'w') as f:
        for line in lines.splitlines():
            if not re.match('^import', line):
                line = re.sub('(?<!#)\s*=\s*', ': ', line)
                line = re.sub('(?<!#);', '', line)
                f.write(line+'\n')

class objFromDict(dict):
    """ Usage: objFromDict(**dictionary).
        Print give the list of attributes.
    """
    def __init__(self, **entries):
        super(objFromDict, self).__init__()
        for key, value in entries.items():
            key = key.replace('-', '_').replace('*', '_').replace('+', '_').replace('/', '_')
            self[key] = value

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        if key not in self:
            raise AttributeError(key)
        return self[key]

    def __repr__(self):
        return '** objFromDict attr. --> '+', '.join(filter((lambda s: (s[:2]+s[-2:])!='____'), self.keys()))

def uniqueFromList(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list