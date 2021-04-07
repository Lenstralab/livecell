import SimpleITK as sitk
import yaml, os, pandas
import numpy as np
from glob import glob

if hasattr(yaml, 'full_load'):
    yamlload = yaml.full_load
else:
    yamlload = yaml.load

""" Uses SimpleElastix https://simpleelastix.readthedocs.io/GettingStarted.html
    wp@tl20190417
"""

def transform_coords(f, T, inverse=False):
    """ Use the affine transformation stored in im to transform x and y columns
          in a dataframe with localisations

        f:  pandas dataframe with columns 'x', 'y' and optionally 'dx', 'dy'
            or tuple with xy, and optionally dxy
            xy: nx2 array with x, y coordinates in columns
            dxy: nx2 array with errors on x and y
        T:  transform or imread instance with transform attribute
        inverse: True/False: perform the inverse transform
          
        wp@tl20190827
    """
    if not isinstance(T, tuple):
        if hasattr(T, 'transform'):
            T = T.transform
        else:
            T = adapt_transform(T)

    if inverse:
        T = invert_transform(T)
    
    if isinstance(f, pandas.DataFrame):
        f = f.copy()
        xy = np.array(f[['x', 'y']])
        dxy = np.array(f[['dx', 'dy']])
        f['x_nt'], f['y_nt'], f['dx_nt'], f['dy_nt'] = f['x'], f['y'], f['dx'], f['dy']
    elif isinstance(f, pandas.Series):
        f = f.copy()
        xy = np.array([f[['x', 'y']],])
        dxy = np.array([f[['dx', 'dy']], ])
        f['x_nt'], f['y_nt'], f['dx_nt'], f['dy_nt'] = f['x'], f['y'], f['dx'], f['dy']
    else:
        xy = np.array(f[0])
        if len(f)>1:
            dxy = np.array(f[1])
        else:
            dxy = np.zeros(xy.shape)
        shape = np.shape(f[0])
    if xy.ndim==1:
        xy = np.expand_dims(xy, 0)
    if dxy.ndim==1:
        dxy = np.expand_dims(dxy, 0)

    offset = np.array([float(i) for i in T[0]['CenterOfRotationPoint']])
    S = np.identity(3)
    s = [float(t) for t in T[0]['TransformParameters']]
    S[0, :2] = s[:2]
    S[1, :2] = s[2:4]
    S[:2, 2] = s[-2:]
    O = np.identity(3)
    O[:2, 2] = offset
    S = np.matmul(O, S)
    O[:2, 2] = -offset
    S = np.matmul(S, O)

    if 'dTransformParameters' in T[0]:
        dS = np.zeros((3, 3))
        ds = [float(t) for t in T[0]['dTransformParameters']]
        dS[0, :2] = ds[:2]
        dS[1, :2] = ds[2:4]
        dS[:2, 2] = ds[-2:]
        S2 = S[:2, :2]**2
        dS2 = dS**2

        # Basically: new_dxy^2 = S^2 * dxy^2 + dS^2 + xy^2, but we need to handle the temporary 3rd element of (d)xy.
        dxy = np.sqrt(np.array([S2.dot(i) for i in dxy**2]) + np.array([dS2.dot((j[0], j[1], 1))[:2] for j in xy**2]))

    # For transformations, the vector gets a third element with value 1, which is discarded after the transformation.
    xy = np.array([S.dot((j[0], j[1], 1))[:2] for j in xy])

    if isinstance(f, (pandas.DataFrame, pandas.Series)):
        f['x'], f['y'], f['dx'], f['dy'] = xy[:,0], xy[:,1], dxy[:,0], dxy[:,1]
    else:
        f = (np.reshape(xy, shape), np.reshape(dxy, shape))
    return f

def affine_registration(fx, mv):
    """ calculate the transformation needed to register the 
        moving image (mv) with the fixed image (fx)
        returns a parametermap
    """
    tfilter = sitk.ElastixImageFilter()
    tfilter.LogToConsoleOff()
    tfilter.SetFixedImage(sitk.GetImageFromArray(fx))
    tfilter.SetMovingImage(sitk.GetImageFromArray(mv))
    tfilter.SetParameterMap(sitk.GetDefaultParameterMap('affine'))
    tfilter.Execute()
    return tfilter.GetTransformParameterMap()

def invert_transform(T):
    """ return an inverted version of a transform
    
        wp@tl20190827
    """
    T = copy_transform(T)
    S = np.identity(3)
    s = [float(t) for t in T[0]['TransformParameters']]
    S[0, :2] = s[:2]
    S[1, :2] = s[2:4]
    S[:2, 2] = s[-2:]
    S = np.linalg.inv(S)
    s[:2] = S[0, :2]
    s[2:4] = S[1, :2]
    s[-2:] = S[:2, 2]
    T[0]['TransformParameters'] = ['{:.99f}'.format(i).rstrip('0').rstrip('.') for i in s]

    if 'dTransformParameters' in T[0]:
        dS = np.zeros((3, 3))
        ds = [float(t) for t in T[0]['dTransformParameters']]
        dS[0, :2] = ds[:2]
        dS[1, :2] = ds[2:4]
        dS[:2, 2] = ds[-2:]

        dV = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                dV[i,j] = np.sqrt(np.sum( [(S[i,k]*dS[k,l]*S[l,j])**2 for k in range(3) for l in range(3)] ))

        ds[:2] = dV[0, :2]
        ds[2:4] = dV[1, :2]
        ds[-2:] = dV[:2, 2]
        T[0]['dTransformParameters'] = ['{:.99f}'.format(i).rstrip('0').rstrip('.') for i in ds]

    return T

def copy_transform(T):
    """ make a dereferenced copy of a transform
    
        wp@tl20190827
    """
    S = (sitk.ParameterMap(),)
    for key, val in T[0].asdict().items():
        S[0][key] = val
    return S

def save_transform(T, fname):
    """ save the parameters of the transform calculated 
        with affine_registration to a yaml file
    """
    if not fname[-3:] == 'yml':
        fname += '.yml'        
    f = open(fname, 'w')
    yaml.safe_dump(T[0].asdict(), f, default_flow_style=None)
    f.close()
    
def load_transform(fname):
    """ load the parameters of a transform from a yaml file
    """
    if not fname[-3:] == 'yml':
        fname += '.yml'
    with open(fname, 'r') as f:
        D = yamlload(f)
    T = (sitk.ParameterMap(),)
    for key, val in D.items():
        T[0][key] = val
    return T

def tfilter_transform(frame, tfilter):
    """ transform a single 2d frame using a tfilter
    """
    dtype = frame.dtype
    frame = frame.astype('float')
    if np.issubdtype(dtype, np.floating):
        set_interpolator(tfilter, 'FinalBSplineInterpolator')
    else:
        set_interpolator(tfilter, 'FinalNearestNeighborInterpolator')

    tfilter.SetMovingImage(sitk.GetImageFromArray(frame))
    tfilter.Execute()
    frame = sitk.GetArrayFromImage(tfilter.GetResultImage())
    frame = frame.astype(dtype)
    return frame

def transform(im, T, inverse=False):
    """ transform a single 2D image (im) using transform T
    """
    if isinstance(T, str):
        T = load_transform(T)
    if inverse:
        T = invert_transform(T)
    tfilter = sitk.TransformixImageFilter()
    tfilter.LogToConsoleOff()
    tfilter.SetTransformParameterMap(T)

    return tfilter_transform(im, tfilter)

def get_transform(im):
    if __package__ is None or __package__=='':
        from wimread import imread
        from tiffwrite import IJTiffWriter
    else:
        from .wimread import imread
        from .tiffwrite import IJTiffWriter
    if im.path.endswith('Pos0'):
        path = os.path.dirname(os.path.dirname(im.path))
    else:
        path = os.path.dirname(im.path)

    tfile = 'transform.yml'
    tpath = os.path.join(path, tfile)
    if os.path.isfile(tpath):
        return load_transform(tpath)
    print('No transform file found, trying to generate one.')
    if im.beadfile is None:
        files = sorted(glob(os.path.join(path, 'beads*')))
        if not files:
            raise Exception('No bead file found!')
        Files = []
        for file in files:
            try:
                if os.path.isdir(file):
                    file = os.path.join(file, 'Pos0')
                with imread(file) as im:
                    pass
                Files.append(file)
            except:
                continue
        if not Files:
            raise Exception('No bead file found!')
    else:
        Files = [im.beadfile]
    T = []
    Km = []
    for file in Files:
        try:
            print('Using {} to calculate a transform.'.format(file))
            t, km = transform_from_beads(file)
            T.append(t)
            Km.append(km)
        except:
            continue

    tifpath = tpath[:-3] + 'tif'
    with IJTiffWriter(tifpath, (3, 1, len(Km))) as f:
        for i, km in enumerate(Km):
            f.save(km, 3*i)

    if not T:
        print('Unable to automatically create a transform.')
        return None
    else:
        T = average_transforms(*T)
        print('Saving transform in {}.'.format(tpath))
        print('Please check the transform in {}.'.format(tifpath))
        save_transform(T, tpath)
        return T

def transform_from_beads(file):
    if __package__ is None or __package__=='':
        from wimread import imread
    else:
        from .wimread import imread

    with imread(file) as jm:
        jmr = jm.maxz(jm.detector.index(jm.slavech))
        jmg = jm.maxz(jm.detector.index(jm.masterch))
        T = affine_registration(jmr, jmg)

        jmr = np.hstack((jmr, jmr))
        jmr -= np.nanmin(jmr)
        jmr *= 255 / np.nanmax(jmr)

        jmg = np.hstack((jmg, transform(jmg, T)))
        jmg -= np.nanmin(jmg)
        jmg *= 255 / np.nanmax(jmg)

        km = np.dstack((jmr, jmg, np.zeros(np.shape(jmr)))).astype('uint8')

        rotcenter = np.array([float(i) for i in T[0]['CenterOfRotationPoint']])
        rotcenter += jm.frameoffset
        jm.close()
        T[0]['CenterOfRotationPoint'] = ['{:.99f}'.format(i).rstrip('0').rstrip('.') for i in rotcenter]
    return T, km

def average_transforms(*Ti):
    N = len(Ti)
    Pi = [[float(i) for i in ti[0]['TransformParameters']] for ti in Ti]
    P = np.vstack(Pi).mean(0)
    dP = np.vstack(Pi).std(0)/np.sqrt(N)
    T = copy_transform(Ti[0])
    T[0]['TransformParameters'] = ['{:.99f}'.format(p).rstrip('0').rstrip('.') for p in P]
    T[0]['dTransformParameters'] = ['{:.99f}'.format(dp).rstrip('0').rstrip('.') for dp in dP]
    return T

def set_interpolator(T, interpolator):
    if isinstance(T, tuple):
        if not T[0]['ResampleInterpolator'] == interpolator:
            T[0]['ResampleInterpolator'] = (interpolator,)
    else:
        if not T.GetTransformParameter(0, 'ResampleInterpolator')[0] == interpolator:
            T.SetTransformParameter(0, 'ResampleInterpolator', interpolator)

def frame_transform(im, c, z, t):
    """ handle transforming a frame of an imread.imread instance
    """
    return tfilter_transform(im.__frame__(c, z, t), im.tfilter)

def adapt_transform(im, T=None):
    """ adapt a transform to an image, by modifying the COPR and Size
        will load a transform if not supplied
    """

    if T is None or T is True:
        T = get_transform(im)
    elif isinstance(T, str):
        T = load_transform(T)

    # modify transform for different ROI's
    rotcenter = np.array([float(i) for i in T[0]['CenterOfRotationPoint']])
    rotcenter -= im.frameoffset + (np.array([float(i) for i in T[0]['Size']]) - im.shape[:2])/2
    T[0]['CenterOfRotationPoint'] = ['{:.99f}'.format(i).rstrip('0').rstrip('.') for i in rotcenter]
    T[0]['Size'] = ['{:.99f}'.format(i).rstrip('0').rstrip('.') for i in im.shape[:2]]
    return T

def init_transform(im, T=None):
    """ initialise a transform in an imread.imread instance
    """
    im.transform = adapt_transform(im, T)
    im.tfilter = sitk.TransformixImageFilter()
    im.tfilter.LogToConsoleOff()
    im.tfilter.SetTransformParameterMap(im.transform)