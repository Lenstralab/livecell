import os
import numpy as np
import tempfile
import pandas
from csbdeep.utils import normalize
from tqdm.auto import trange
from tiffwrite import IJTiffFile
from pytrackmate import trackmate_peak_import
from tllab_common.wimread import imread
from contextlib import ExitStack
from scipy.interpolate import interp1d
from scipy.ndimage import distance_transform_edt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from stardist.models import StarDist2D

try:
    import imagej
    import scyjava

    def kill_vm():
        try:
            scyjava.jimport('java.lang.System').exit(0)
        except Exception:
            pass

except Exception:
    imagej = None
    scyjava = None

    def kill_vm():
        return


def label_dist(labels, lbl, mask=None):
    """ make an array with distances to the edge of the lbl in labels, negative outside, positive inside """
    lbl_mask = (labels == lbl)
    dist = -distance_transform_edt(lbl_mask == 0)
    dist[dist < 0] += 1
    dist += distance_transform_edt(lbl_mask == 1)
    dist[(labels != lbl) * (labels != 0)] = -np.inf
    if mask is not None:
        dist[mask] = -np.inf
    return dist


def interp_label(t, ts, labels, lbl, mask=None):
    """ return a labelfield with lbl at time q interpolated from labels at times ts """
    return lbl * (interp1d(ts, np.dstack([label_dist(label, lbl, mask) for label in labels]),
                           fill_value=np.zeros_like(labels[0]), bounds_error=False)(t) > 0)


def swap_labels(tracks):
    def fun(im, frame_in, c, z, t):
        frame_out = np.zeros_like(frame_in)
        for i, j in tracks.query('t==@t')[['median_intensity', 'label']].to_numpy():
            frame_out[frame_in == i] = j
        return frame_out
    return fun


def sort_labels(tracks):
    """ make labels consistent across different runs """
    relabel_dict = {int(key): value for value, key in
                    enumerate(tracks.groupby('label').aggregate('mean').sort_values('area').index, 1)}
    return tracks.groupby('label').apply(lambda x: x.assign(label=relabel_dict[x['label'].mean()]))


def get_time_points(t, missing):
    t_a = t - 1
    while t_a in missing:
        t_a -= 1
    t_b = t + 1
    while t_b in missing:
        t_b += 1
    return t_a, t_b


def interpolate_missing(tracks, t_len=None):
    """ interpolate the position of the cell in missing frames """
    missing = []
    for cell in tracks['label'].unique():
        h = tracks.query('label==@cell')
        if t_len is None:
            t_missing = list(set(range(int(h['t'].min()), int(h['t'].max()))) - set(h['t']))
        else:
            t_missing = list(set(range(t_len)) - set(h['t']))
        g = pandas.DataFrame(np.full((len(t_missing), tracks.shape[1]), np.nan), columns=tracks.columns)
        g['t'] = t_missing
        g['t_stamp'] = t_missing
        g['x'] = np.interp(t_missing, h['t'], h['x'])
        g['y'] = np.interp(t_missing, h['t'], h['y'])
        g['label'] = cell
        missing.append(g)
    return pandas.concat(missing, ignore_index=True)


def substitute_missing(tracks, missing, distance=1):
    """ relabel rows in tracks if they overlap with a row in missing """
    for _, row in missing.iterrows():
        a = tracks.query(f"t=={row['t']} & (x-{row['x']})**2 + (y-{row['y']})**2 < @distance").copy()
        a['label'] = row['label']
        if len(a) == 1:
            tracks.loc[a.index[0], 'label'] = row['label']
        elif len(a) > 1:
            idx = ((a[['x', 'y']] - row[['x', 'y']].tolist()) ** 2).sum(1).idxmin()
            tracks.loc[idx, 'label'] = row['label']
    return tracks


def trackmate(file_in, file_out, fiji_path=None, **kwargs):
    if fiji_path is None:
        fiji_path = '/DATA/opt/Fiji.app/'
    if os.path.exists(fiji_path):
        ij = imagej.init(fiji_path)
    else:
        ij = imagej.init('sc.fiji:fiji')
    settings = dict(file_in=file_in, file_out=file_out, MIN_AREA=20, MAX_FRAME_GAP=2,
                    ALTERNATIVE_LINKING_COST_FACTOR=1.05,
                    LINKING_MAX_DISTANCE=15.0, GAP_CLOSING_MAX_DISTANCE=15.0, SPLITTING_MAX_DISTANCE=15.0,
                    ALLOW_GAP_CLOSING=True, ALLOW_TRACK_SPLITTING=False, ALLOW_TRACK_MERGING=False,
                    MERGING_MAX_DISTANCE=15.0, CUTOFF_PERCENTILE=0.9)
    settings.update({key.upper(): value for key, value in kwargs.items() if key in settings})
    with open(os.path.join(os.path.dirname(__file__), 'trackmate.jy')) as f:
        ij.py.run_script('py', f.read(), settings)
    ij.dispose()


def stardist(max_im, file_out):
    model = StarDist2D.from_pretrained('2D_versatile_fluo')
    with IJTiffFile(file_out, (1, 1, max_im.shape[4]), pxsize=max_im.pxsize) as tif:
        for t in trange(max_im.shape[4], desc='Running StarDist'):
            tif.save(model.predict_instances(normalize(max_im(0, 0, t)))[0], 0, 0, t)


def stardist_trackmate(image, tiff_out, table_out, **kwargs):
    with tempfile.TemporaryDirectory() as tempdir:
        tif_file = os.path.join(tempdir, 'tm.tif')
        xml_file = os.path.join(tempdir, 'tm.xml')

        with ExitStack() as stack:
            if isinstance(image, str):
                image = stack.enter_context(imread(image))
            stardist(image, tif_file)

        trackmate(tif_file, xml_file, **kwargs)
        tracks = trackmate_peak_import(xml_file, get_tracks=True)
        missing = interpolate_missing(tracks)
        tracks = substitute_missing(tracks, missing)
        tracks = sort_labels(tracks)
        missing = interpolate_missing(tracks)
        tracks = pandas.concat((tracks, missing), ignore_index=True)
        tracks.to_csv(table_out, sep='\t')
        dtype = 'uint8' if tracks['label'].max() < 255 else 'uint16'

        # Relabel the labels according to the tracks and also add missing labels by interpolation
        with imread(tif_file, dtype=int) as im:
            im.frame_decorator = swap_labels(tracks)
            with IJTiffFile(tiff_out, (1, 1, im.shape[4]), pxsize=im.pxsize, colormap='glasbey', dtype=dtype) as tif:
                for t in trange(im.shape[4], desc='Saving stardist/trackmate labelmap'):
                    missing_t = missing.query('t==@t')
                    frame = im(0, 0, t)
                    for cell in missing_t['label'].unique():
                        time_points = get_time_points(t, missing.query('label==@cell')['t'].tolist())
                        a = interp_label(t, time_points, [im(0, 0, i) for i in time_points], int(cell), frame > 0)
                        frame += a
                    tif.save(frame, 0, 0, t)
