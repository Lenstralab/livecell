#!/usr/local/bin/ipython3 -i

import os
import sys
import re
import copy as copy2
import pandas
import scipy
from PIL import Image
from PIL import ImageFilter
from time import time
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm.auto import trange, tqdm
import numpy as np
import shutil
import yaml
import pickle
from itertools import product
from skimage import filters
from matplotlib.backends.backend_pdf import PdfPages
from contextlib import ExitStack
from parfor import parfor, Chunks
from tllab_common.wimread import imread as imr
from tllab_common.transforms import Transform
from tllab_common.findcells import findcells
from tllab_common.misc import ipy_debug
from tiffwrite import IJTiffFile, tiffwrite

if __package__ is None or __package__ == '':  # usual case
    import spotDetection_functions as sdf
    import dataAnalysis_functions as daf
    import fluctuationAnalysis
    import plot_figures
    import misc
    import trackmate
    import _version
else:
    from . import spotDetection_functions as sdf
    from . import dataAnalysis_functions as daf
    from . import fluctuationAnalysis
    from . import plot_figures
    from . import misc
    from . import trackmate
    from . import _version

if not '__file__' in locals():  # when executed using execfile
    import inspect
    __file__ = inspect.getframeinfo(inspect.currentframe()).filename

squareStamp = [np.r_[-5, -5, -5, -5, -5, -5, -5, -5, -4, -3, -2, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 4, 3, 2, -2, -3, -4],
               np.r_[-5, -4, -3, -2, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 4, 3, 2, -2, -3, -4, -5, -5, -5, -5, -5, -5, -5]]


def expandExpList(params, parameter_file):
    """ Completely expand path, including absolute path and extension for every item in params['ExpList']
    """
    extensions = '.ims', '.dv', '.czi'  # add extensions here
    pattern = re.compile('$|'.join(('^Pos\d+', ) + extensions) + '$')
    pat = re.compile('^Pos\d+$')
    pExpList, pCorrectDrift, pTSregfile = [params.get(p, []) for p in ('expList', 'CorrectDrift', 'TSregfile')]
    if l := len(pExpList):
        try:
            if not len(pCorrectDrift):
                pCorrectDrift = l * [False]
        except Exception:
            pCorrectDrift = l * [False if pCorrectDrift is None else pCorrectDrift]
        if not pTSregfile:
            pTSregfile = l * [[]]

    expList, CorrectDrift, TSregfile = [], [], []
    for exp, drift, tsreg in zip(pExpList, pCorrectDrift, pTSregfile):
        folderIn = params['folderIn']
        if not os.path.isabs(folderIn):
            folderIn = os.path.join(os.path.dirname(os.path.abspath(parameter_file)), folderIn)
        pathExp = os.path.join(folderIn, exp)
        if os.path.isdir(pathExp) and not re.search(pat, os.path.basename(exp)):  # just a folder with images, add all
            for item in [re.search(pattern, i) for i in sorted(os.listdir(pathExp))]:
                if item:
                    expList.append(os.path.join(pathExp, item.string))
                    CorrectDrift.append(drift)
                    TSregfile.append(tsreg)
        elif os.path.dirname(pathExp).endswith('.nd'):  # is metamorph file with position
            expList.append(pathExp)
            CorrectDrift.append(drift)
            TSregfile.append(tsreg)
        elif not os.path.exists(pathExp):  # filename without extension, add all we can find, normally just 1 of course
            for ext in extensions:
                if os.path.exists(pathExp + ext):
                    expList.append(pathExp + ext)
                    CorrectDrift.append(drift)
                    TSregfile.append(tsreg)
        else:  # completely defined already, just continue
            expList.append(pathExp)
            CorrectDrift.append(drift)
            TSregfile.append(tsreg)

    if not expList:
        raise FileNotFoundError(f'Could not find/open files in params["folderIn"].')

    params['expList'] = expList
    params['lenExpList'] = len(expList)
    params['CorrectDrift'] = CorrectDrift
    params['TSregfile'] = TSregfile
    return params


def getPaths(params, parameter_file=None):
    """ Update the dict params with necessary keys, and make sure paths exist etc.
    """
    path, ext = os.path.splitext(params['pathIn'])
    path = path.rstrip(os.path.sep)
    params['microscope'] = {'.ims': 'spinningDisk',
                            '.dv': 'deltaVision',
                            '.czi': 'Elyra',
                            '.tif': 'unknown'}.get(ext.lower(), 'AxioObserver')

    if re.search('Pos\d+$', path):
        expName = f'{os.path.basename(os.path.dirname(path))}_{os.path.basename(path)}'
    else:
        expName = os.path.basename(path)

    params['expName'] = re.sub('((?<=.)_(?:\d[_-]?){7}\d|(?:\d[_-]?){7}\d_(?=.))', '', expName)
    date = re.findall('(?:\d[_-]?){7}\d', params['pathIn'])
    if date:
        params['date'] = date[0]
    else:
        params['date'] = path.split(os.path.sep)[-2]
    params['pathOut'] = os.path.join(os.path.abspath(params['outputfolder']), params['date'] + "_" + params['expName'])
    params['expPathOut'] = os.path.join(params['pathOut'], params['date'] + "_" + params['expName'])

    if not parameter_file is None:
        if not os.path.exists(params['pathOut']):
            os.makedirs(params['pathOut'])

        dateTimeObj = datetime.now()
        dateTime = dateTimeObj.strftime("%d-%b-%Y_%Hh%Mm%Ss")
        if os.path.exists(parameter_file):
            shutil.copyfile(os.path.abspath(parameter_file),
                            os.path.join(params['expPathOut'] + "_pipeline_livecell_track_movies_parameters_runtime_"
                                         + dateTime + ".yml"))
        else:
            parameter_file = os.path.join(params['expPathOut'] + "_pipeline_livecell_track_movies_parameters_runtime_"
                                          + dateTime + ".yml")
            with open(parameter_file, 'w') as f:
                yaml.safe_dump(params, f)
        shutil.copyfile(os.path.abspath(__file__), f"{params['expPathOut']}_pipeline_livecell_track_movies.py")
        shutil.copyfile(os.path.abspath(_version.__file__), f"{params['expPathOut']}_livecell_version.py")

    if params['swapColors']:  # define where in image file to find different colors, idx 0: red, 1: green
        params['channels'] = params['ChannelsToAnalyze'][::-1]
    else:
        params['channels'] = range(max(params['ChannelsToAnalyze']) + 1)

    with imr(params['pathIn']) as im:
        if params['ChannelsToAnalyze'] is None:
            params['ChannelsToAnalyze'] = list(range(im.shape[2]))
        if not params['frames'] or params['frames']>im.shape[4]:
            params['frames'] = im.shape[4]
        if not params['zSlices'] or params['zSlices']>im.shape[3]:
            params['zSlices'] = im.shape[3]

        params['nChannels'] = len(params['ChannelsToAnalyze'])
        params['totalImg'] = params['zSlices'] * params['frames'] * params['nChannels']

        if params['sideViews'] is None:
            params['sideViews'] = im.zstack

        if os.path.exists(params['expPathOut'] + "_max.tif"):  # maxfile exists already
            params['domax'] = False
            params['maxFile'] = params['expPathOut'] + "_max.tif"
        elif im.shape[3] == 1:  # maxfile would be the same as raw data
            params['domax'] = False
            params['maxFile'] = params['pathIn']
        else:  # make a maxfile
            params['domax'] = True
            params['maxFile'] = params['expPathOut'] + "_max.tif"

        if os.path.exists(params['expPathOut'] + "_sum.tif"):
            params['dosum'] = False
            params['sumFile'] = params['expPathOut'] + "_sum.tif"
        elif im.shape[3] == 1 and params['frames'] == 1:
            params['dosum'] = False
            params['sumFile'] = params['pathIn']
        else:
            params['dosum'] = True
            params['sumFile'] = params['expPathOut'] + "_sum.tif"

        if params['cellMask']['method'].lower() == 'findcell':
            params['mask_file'] = params['expPathOut'] + "_sum_cells_mask.tif"
        elif params['cellMask']['method'].lower() == 'trackmate':
            params['mask_file'] = params['expPathOut'] + "_cells_mask.tif"
            params['cell_track_file'] = params['expPathOut'] + "_cells_track.tsv"
        else:
            raise ValueError(f"Unknown cell mask method: {params['cellMask']['method']}")

    if params['RegisterChannels']:
        with imr(params['pathIn'], transform=True) as im:
            params['transform'] = im.transform


def correctDrift(params):
    channel = 0
    # TODO: figure out how to use all channels
    if os.path.exists(params['expPathOut'] + "_drift.txt"):
        params['drift'] = np.loadtxt(params['expPathOut'] + "_drift.txt")
    else:
        with imr(params['pathIn'], dtype=float) as im:
            # divide the sequence into groups of about 50 frames,
            # compare each frame to the frame in the middle of the group and compare these middle frames to each other
            t_groups = [list(chunk) for chunk in Chunks(range(im.shape[4]), size=50)]
            t_keys = [int(np.round(np.mean(t_group))) for t_group in t_groups]
            t_pairs = [(int(np.round(np.mean(t_group))), frame) for t_group in t_groups for frame in t_group]
            t_pairs.extend(zip(t_keys, t_keys[1:]))
            fmaxz_keys = {t_key: filters.gaussian(im.max(channel, None, t_key), 5) for t_key in t_keys}

            @parfor(t_pairs, (im, channel, fmaxz_keys), desc='Calculating drift', terminator=imr.kill_vm)
            def drift(t_key_t, im, channel, fmaxz_keys):
                t_key, t = t_key_t
                if t_key == t:
                    return 0, 0
                else:
                    fmaxz = filters.gaussian(im.max(channel, None, t), 5)
                    return Transform(fmaxz_keys[t_key], fmaxz, 'translation').parameters[-2:]

            drift = np.array(drift)
            drift_keys_cum = np.zeros(2)
            for drift_keys, t_group in zip(np.vstack((-drift[0], drift[im.shape[4]:])), t_groups):
                drift_keys_cum += drift_keys
                drift[t_group] += drift_keys_cum
            drift = drift[:im.shape[4]]

        params['drift'] = np.round(drift).astype(int)
        np.savetxt(params['expPathOut'] + "_drift.txt", params['drift'], fmt='%d')


def maxProjection(params):
    """
    do max and sum projections
    will not switch channels, or omit, just save them in the same order as the raw
    """
    with ExitStack() as stack:
        im = imr(params['pathIn'])
        bar = stack.enter_context(tqdm(total=len(params['ChannelsToAnalyze']) * params['frames'],
                                       desc='Max intensity projections'))
        if params['domax']:
            maxfile = stack.enter_context(IJTiffFile(params['maxFile'],
                                          (len(params['ChannelsToAnalyze']), 1, params['frames'])))
        if params['dosum']:
            sumfile = stack.enter_context(IJTiffFile(params['sumFile'], (len(params['ChannelsToAnalyze']), 1, 1)))
        for c in params['ChannelsToAnalyze']:
            imSum = np.zeros(im.shape[:2], 'float')
            for t in range(params['frames']):
                imMax = im.max(c, range(params['zSlices']), t)
                if 'drift' in params:
                    imMax = misc.translateFrame(imMax, params['drift'][t])
                imSum += imMax
                if params['domax']:
                    maxfile.save(imMax.astype('uint16'), c, 0, t)
                bar.update()
            if params['dosum']:
                imSum *= 65536 / np.nanmax(imSum)  # rescale to 16 bit
                sumfile.save(imSum.astype('uint16'), c, 0, 0)


def cellMask(params):
    if params['cellMask']['method'].lower() == 'findcell':
        with imr(params['sumFile']) as im:
            if len(params['channels'])>1:
                km = im(params['channels'][1], 0, 0)  # uses green image
            else:
                km = im(params['channels'][0], 0, 0)
            cells = findcells(km, **params['cellMask'])[0]
            if params['RegisterChannels']:
                #save a mask for each channel, they might be different because of the transforming
                stack = np.dstack([params['transform'](c).frame(cells) for c in range(im.shape[2])])
                stack = stack.astype('uint8') if stack.max() < 256 else stack.astype('uint16')
                tiffwrite(params['mask_file'], stack, 'XYC', colormap='glasbey', pxsize=im.pxsize)
            else:
                cells = cells.astype('uint8') if cells.max() < 256 else cells.astype('uint16')
                tiffwrite(params['mask_file'], cells, colormap='glasbey', pxsize=im.pxsize)
    elif params['cellMask']['method'].lower() == 'trackmate':
        trackmate.stardist_trackmate(params['maxFile'], params['mask_file'], params['cell_track_file'],
                                     **params['cellMask'])


def optimizeTreshold(params):
    imagefile = params['maxFile']
    with imr(imagefile) as imtmp:
        # determine image size and number of stacks
        size = imtmp.shape[:2]
        nbIm = params['frames']

        cellArray, TSreg, maxCellid, listCellnr = TSmask(params)

        threshvals = range(4,14)


        for channel in params['ChannelsToAnalyze']:
            spotslist = np.zeros(len(threshvals))
            spotslistunique = np.zeros(len(threshvals))
            celllist = []

            for j in trange(nbIm, desc='Optimizing threshold'):
                im = copy2.deepcopy(imtmp[:, :, params['channels'][channel], :, j].reshape(size))
                im = sdf.hotpxfilter(im.astype("float32"))
                im = im.astype("uint16")
                imBpass = sdf.bpass(im, params['psfPx'] - 0.2, params['psfPx'] + 0.2)

                if params['TSregfile'] != []:
                    imTS = create_cell_array(params, size, listCellnr, TSreg, j, channel)

                for i, threshval in enumerate(threshvals):
                    celllisttmp = []
                    imBinary = (imBpass > threshval * np.var(imBpass) ** .5) * 1.  # Binary image for high threshold
                    objects = scipy.ndimage.find_objects(scipy.ndimage.label(imBinary)[0])
                    cooGuess = np.array([[np.r_[obj[1]].mean(), np.r_[obj[0]].mean()] for obj in objects])
                    with ExitStack() as stack:
                        if isinstance(cellArray, str):
                            cell_array_imr = stack.enter_context(imr(cellArray))

                        for c in range(cooGuess.shape[0]):
                            if params['TSregfile'] == []:
                                if isinstance(cellArray, str):
                                    cellnr = int(cell_array_imr(channel, 0, j)
                                                 [int(cooGuess[c, 1]), int(cooGuess[c, 0])])
                                else:
                                    cellnr = cellArray[int(cooGuess[c, 1]), int(cooGuess[c, 0]), 0]
                            else:
                                cellnr = int(imTS[int(cooGuess[c, 1]), int(cooGuess[c, 0])])
                            if cellnr > 0 and cellnr != 32767:
                                celllisttmp.append(cellnr)

                    spotslist[i] = spotslist[i] + len(celllisttmp)
                    spotslistunique[i] = spotslistunique[i] + len(set(celllisttmp))
                    try:
                        celllist[i].extend(celllisttmp)
                    except:
                        celllist.append(celllisttmp)

            cellswithspots = [len(set(c)) for c in celllist]

            fig = plt.figure()
            g1 = plt.plot(threshvals[1:],spotslist[1:], '-b')
            g2 = plt.plot(threshvals[1:],spotslistunique[1:], '-r')
            plt.xlabel('Threshold channel ' + str(channel))
            plt.ylabel('Number of spots detected')
            plt.ylim(0,500)
 #           plt.yscale('log')
            plt.legend((g1[0], g2[0]), ("total nr spots","nr unique spots (max 1 spot/cell/frame)"))
            plt.savefig(params['expPathOut'] + "_threshold_optimization_channel" + str(channel) + ".pdf",
                format='pdf')
            plt.close(fig)

            fig = plt.figure()
            plt.plot(threshvals,cellswithspots, '-b')
            plt.title('Total cells (max cellnr) = ' +str(maxCellid))
            plt.xlabel('Threshold channel ' + str(channel))
            plt.ylabel('Number of cells with tracks')
#            plt.legend(loc='upper center')
            plt.savefig(params['expPathOut'] + "_threshold_optimization_nrcellstracked_channel" + str(channel) + ".pdf",
                format='pdf')
            plt.close(fig)


def guessHighThreshold(params, imtmp, channel, nbIm, TSregfile, listCellnr, TSreg, cellArray):
    psfPx = params['psfPx']
    channels = params['channels']
    thresholdSD = params['thresholdSD_' + ('Red', 'Green')[channel]]
    diffThresholdSD = params['diffThresholdSD']
    TSregRadius = params['TSregRadius']

    def guessThreshold(params, imBpass, t, threshold):
        size = imBpass.shape
        if TSregfile:
            cA = create_cell_array(params, size, listCellnr, TSreg, t, channel)
        elif cellArray is None:
            cA = np.ones(size, dtype=int)
        elif cellArray.shape[2] > 1:
            cA = cellArray[:, :, channel]
        else:
            cA = cellArray[:, :, 0]

        # TODO: two spots in two bordering cells might merge
        # TODO: wimread metadata: optovar etc
        imBinary = (imBpass > threshold * np.var(imBpass) ** .5).astype(float)  # Masked binary image

        # # Find all the connex objects and determine their centers as initial guesses for the spot locations
        # R = skimage.measure.regionprops(scipy.ndimage.label(imBinary)[0])
        # cooGuess = np.vstack([(t, cA[tuple([round(i) for i in r.centroid])], *r.centroid[::-1]) for r in R])
        # return cooGuess[cooGuess[:, 1] > 0]

        # Find all the connex objects and determine their centers as initial guesses for the spot locations
        # TODO: regionprops.centroid probably gives a better estimate
        objects = scipy.ndimage.find_objects(scipy.ndimage.label(imBinary)[0])
        R = np.array([[np.mean(np.r_[obj[1]]), np.mean(np.r_[obj[0]])] for obj in objects])
        # TODO: should probably round: round(i)
        cooGuess = np.vstack([(t, cA[tuple([int(i) for i in r[::-1]])], *r) for r in R])
        return cooGuess[cooGuess[:, 1] > 0]

    @parfor(range(nbIm), desc='Localizing spots')
    def fun(t):
        im = copy2.deepcopy(imtmp(channels[channel], 0, t))

        # prevent interference of dead pixels
        im = sdf.hotpxfilter(im.astype("float32"))
        im = im.astype("uint16")

        # bandpass image to find spots
        imBpass = sdf.bpass(im, psfPx - 0.2, psfPx + 0.2)  # Band-passed image
        return guessThreshold(params, imBpass, t, thresholdSD),\
               guessThreshold(params, imBpass, t, (thresholdSD - diffThresholdSD))

    cooGuess1, cooGuess2 = [np.vstack(cooGuess) for cooGuess in zip(*fun)]
    cooGuess2 = [transformCoords(cooGuess2[cooGuess2[:, 0] == t, 2:], channel, params, True) for t in range(nbIm)]

    return cooGuess1, cooGuess2


def localizeHighThresholdPar(params, imtmp, channel, cooGuess_matrix, nbIm, maxCellid):
    psfPx = params['psfPx']
    winSize = params['winSize']
    maxDist = params['maxDist']
    channels = params['channels']
    # minSeparation = params['minSeparation']
    #
    # def rm_duplicates_idxs(points, r):
    #     if len(points) == 0:
    #         return np.array([])
    #
    #     def dist(a, b):
    #         return math.hypot(*[n - m for n, m in zip(a, b)])
    #     keep = []
    #     prop = rtree.index.Property()
    #     prop.dimension = points.shape[1]
    #     index = rtree.index.Index(properties=prop)
    #     for i, p in enumerate(points):
    #         nearby = index.intersection([q - r for q in p] + [q + r for q in p])
    #         if all(dist(p, points[j]) >= r for j in nearby):
    #             keep.append(i)
    #             index.insert(i, (*p, *p))
    #     return np.array(keep)

    @parfor(Chunks(cooGuess_matrix, r=5), ({'RegisterChannels': params['RegisterChannels'],
                                           'transform': params.get('transform')},), desc='Fitting high intensity spots')
    def fun(cgchunk, params0):
        data = []
        for t, c, *x in cgchunk:
            im = imtmp(channels[channel], 0, t).astype('float')
            im = sdf.hotpxfilter(im.astype("float32"))
            im = im.astype("uint16")
            intensity, coo, tilt = sdf.GaussianMaskFit2(im, x, psfPx, winSize=winSize, nbIter=100)
            if intensity != 0 and sum((coo - x) ** 2) < (maxDist * psfPx) ** 2:
                coo = transformCoords(coo, channel, params0, True)
                data.append((intensity, *coo, *tilt, t, c - 1))
        return data

    fitResultsall = np.vstack(sum(fun, []))
    # idx = rm_duplicates_idxs(fitResultsall[:, 1:3], minSeparation * psfPx)  # Disabled, because see note below
    # fitResultsall = fitResultsall[idx]

    data = {}
    for *item, t, c in fitResultsall:
        if not (t, c) in data:
            data[t, c] = []
        data[t, c].append(item)

    dataCellLoc = np.zeros((nbIm, maxCellid, 6))
    dataDigital = np.zeros((nbIm, maxCellid))

    for key in product(range(nbIm), range(maxCellid)):
        if key in data:
            dataCellLoc[key] = max(data[key], key=lambda x: x[0])  # keep the brightest spot in each cell
            dataDigital[key] = 1
    return fitResultsall[:, :-1], dataCellLoc[:, :, 0], dataCellLoc, dataDigital


def localizeHighThreshold(params, im, channel, j, size, cooGuess2_matrix, TSregfile, listCellnr, TSreg, cellArray,
                          fitResultsall, dataCell, dataDigital, dataCellLoc):
    psfPx = params['psfPx']
    thresholdSD = params['thresholdSD_'+('Red', 'Green')[channel]]
    diffThresholdSD = params['diffThresholdSD']
    TSregRadius = params['TSregRadius']
    winSize = params['winSize']
    maxDist = params['maxDist']
    minSeparation = params['minSeparation']

    fitResults = []

    # prevent interference of dead pixels
    im = sdf.hotpxfilter(im.astype("float32"))
    im = im.astype("uint16")

    # bandpass image to find spots
    imBpass = sdf.bpass(im, psfPx - 0.2, psfPx + 0.2)  # Band-passed image
    imBinary1 = (imBpass > thresholdSD * np.var(imBpass) ** .5) * 1.  # Binary image for high threshold
    imBinary2 = (imBpass > (thresholdSD - diffThresholdSD) * np.var(imBpass) ** .5) * 1.  # Binary image for low threshold

    # imBinary=(imBpass>threshold)*1.                        # uncomment this if you want to use a fixed threshold and not a standard deviation threshold)

    # Find all the connex objects
    objects1 = scipy.ndimage.find_objects(scipy.ndimage.label(imBinary1)[0])
    objects2 = scipy.ndimage.find_objects(scipy.ndimage.label(imBinary2)[0])

    # Determine their centers as initial guesses for the spot locations
    cooGuess1 = np.array([[np.r_[obj[1]].mean(), np.r_[obj[0]].mean()] for obj in objects1])
    cooGuess2 = np.array([[np.r_[obj[1]].mean(), np.r_[obj[0]].mean()] for obj in objects2])
    cooGuess2 = transformCoords(cooGuess2, channel, params, True)

    cooGuess2_matrix.append(cooGuess2)
    cooGuess = cooGuess1

    if TSregfile != []:
        imTS = create_cell_array(params, size, listCellnr, TSreg, j, channel)

    # fit each spot with 2D gaussian with tilted plane. The GaussianMaskFit2 function is described in spotDetection_Functions.py
    for i in range(cooGuess.shape[0]):
        if TSregfile == []:
            cellnr = cellArray[int(cooGuess[i, 1]), int(cooGuess[i, 0])]
        else:
            cellnr = int(imTS[int(cooGuess[i, 1]), int(cooGuess[i, 0])])
        if cellnr > 0 and cellnr != 32767:
            intensity, coo, tilt = sdf.GaussianMaskFit2(im, cooGuess[i], psfPx, winSize=winSize, nbIter=100)
            # Keep only if it converged close to the initial guess
            if intensity != 0 and sum((coo - cooGuess[i]) ** 2) < (maxDist * psfPx) ** 2:
                coo = transformCoords(coo, channel, params, True)
                # Remove duplicates;  NOTE: does not work here because fitResults always is []
                if sum([sum((coo - a[1:3]) ** 2) < (minSeparation * psfPx) ** 2 for a in fitResults]) == 0:
   #                 fitResults.append(np.r_[intensity, coo, tilt])  # add results to fitResults (spots for this image only)
                    fitResultsall.append(np.r_[intensity, coo, tilt, j])  # add results to fitResultsall (spots for all images in timeseries)
                # mapping results brightest spots to detected cells
                if intensity > dataCell[j, cellnr - 1]:  # if intensity is higher than intensity already present in array -> selects for highest intensity spots
                    dataCell[j, cellnr - 1] = intensity
                    dataDigital[j, cellnr - 1] = 1
                    dataCellLoc[j, cellnr - 1, :] = np.r_[intensity, coo, tilt]


def transformCoords(coo, channel, params, forward=True):
    """ This function transforms coordinates if needed: if RegisterChannels is True
        We fit only raw data, but save only transformed coordinates. So we need to transform the coordinates forwards
        (raw to transformed) or backwards (transformed to raw) a few times.
    """
    if params['RegisterChannels']:
        ndim = np.ndim(coo)
        coo = np.array(coo, ndmin=2)
        if forward:
            coo = params['transform'](channel, 0).coords(coo)
        else:
            coo = params['transform'].inverse(channel, 0).coords(coo)
        return coo[0] if ndim == 1 else coo
    else:
        return np.array(coo)


def localizeLowThreshold(params, im, t, channel, size, cooGuess2_matrix, TSregfile, listCellnr, TSreg, cellArray,
                         dataCell, dataDigital, dataCell2, dataCellLoc2, fitResultsall2):
    # TODO: paralellize this and take cellnumber from cooGuess2_matrix (amend guessHighThreshold)
    psfPx = params['psfPx']
    TSregRadius = params['TSregRadius']
    winSize = params['winSize']
    maxDist = params['maxDist']
#    writeLocTifFiles = params['writeLocTifFiles']
    dist = params['dist']

    # prevent interference of dead pixels
    im = sdf.hotpxfilter(im.astype("float32"))
    im = im.astype("uint16")

    if TSregfile != []:
        imTS = create_cell_array(params, size, listCellnr, TSreg, t, channel)

    cooGuess = cooGuess2_matrix[t]  # load data from matrix (spots were found above)
    cooGuess = transformCoords(cooGuess, channel, params, False)

    # fit each spot with 2D gaussian with tilted plane. The GaussianMaskFit2 function is described in spotDetection_Functions.py
    for i in range(cooGuess.shape[0]):
        if TSregfile == []:
            cellnr = cellArray[int(cooGuess[i, 1]), int(cooGuess[i, 0])] - 1  # note, different cellnr from above, 1 already subtracted
        else:
            cellnr = int(imTS[int(cooGuess[i, 1]), int(cooGuess[i, 0])] - 1)  # note, different cellnr from above, 1 already subtracted

        if cellnr >= 0:  # only fit if spot is inside cell
            intensity, coo, tilt = sdf.GaussianMaskFit2(im, cooGuess[i], psfPx, winSize=winSize, nbIter=100)  # first try to fit by converging
            if intensity == 0: # if it does not converge, fit at position of initial guess
                intensity, coo, tilt = sdf.GaussianMaskFit2(im, cooGuess[i], psfPx, optLoc=0, winSize=winSize)
            if intensity != 0 and sum((coo - cooGuess[i]) ** 2) < (maxDist * psfPx) ** 2 and cellnr >= 0:  # check if spot is in a cell, and if it converged close to the inital guess
                coo = transformCoords(coo, channel, params, True)

                # check is the array does not already contain a spot of higher intensity and check if there were any spots found in other time frames in that cell
                if dataCell[t, cellnr] == 0 and intensity > dataCell2[t, cellnr] and sum(dataDigital[:, cellnr] > 0) != 0:
                    # determine timeframe of first burst
                    if dataDigital[0, cellnr] > 0:
                        startFirstBurst = 0
                    else:
                        startFirstBurst = (np.where(np.diff(dataDigital[:, cellnr]) > 0)[0][0]) + 1
                    # checks if spot is in timeframe before the first burst (first high intensity spot) -> if so, uses position of first burst to determine if new spot is within the threshold vicinity.
                    if t < startFirstBurst and (coo[0] - dataCellLoc2[startFirstBurst, cellnr, 1]) ** 2 + (
                            coo[1] - dataCellLoc2[startFirstBurst, cellnr, 2]) ** 2 < dist ** 2:
                        # write new spot to arrays
                        dataCell2[t, cellnr] = intensity
                        dataDigital[t, cellnr] = 0.5
                        dataCellLoc2[t, cellnr, :] = np.r_[intensity, coo, tilt]
                        fitResultsall2.append(np.r_[intensity, coo, tilt, t])  # add results to fitResultsall2 (spots for all images in timeseries)

                    # checks if spot is in timeframe after the first burst (first high intensity spot) -> if so, uses position of last burst to determine if new spot is within the threshold vicinity.
                    elif t > startFirstBurst:
                        # determine last burst
                        endBursts = np.where(np.diff(dataDigital[:, cellnr]) < 0)[0]
                        if dataCell2[t, cellnr] != 0:
                            lastBurst = t - 1
                        else:
                            pos = t - endBursts >= 0
                            lastBurst = endBursts[np.where(t - endBursts == min(t - endBursts[pos]))[0][0]]
                        # check if new spot is within the threshold vicinity of last burst
                        if (coo[0] - dataCellLoc2[lastBurst, cellnr, 1]) ** 2 + (coo[1] - dataCellLoc2[lastBurst, cellnr, 2]) ** 2 < dist ** 2:
                            # write new spot to arrays
                            dataCell2[t, cellnr] = intensity
                            dataDigital[t, cellnr] = 0.5
                            dataCellLoc2[t, cellnr, :] = np.r_[intensity, coo, tilt]
                            fitResultsall2.append(np.r_[intensity, coo, tilt, t])  # add results to fitResultsall2 (spots for all images in timeseries)


def coo_guess_gap(params, TSregfile, imTS, dataCellLoc2, t_alt, cell, listCellnr, TSreg, t, channel):
    if TSregfile != []:
        cellnr = int(imTS[int(dataCellLoc2[t_alt, cell, 2]), int(
            dataCellLoc2[t_alt, cell, 1])] - 1)  # note, different cellnr from above, 1 already subtracted
        if cell != cellnr:
            for index in range(len(listCellnr)):
                if listCellnr[index] == cell + 1:
                    ind = index
            return transformCoords([TSreg[ind][t][1], TSreg[ind][t][0]], channel, params, False)
    return transformCoords(dataCellLoc2[t_alt, cell, 1:3], channel, params, False)


def fill_gap(params, im, pxsize, tracks, t, channel, TSregfile, size, listCellnr, TSreg, dataCell2, dataDigital, dataCellLoc2,
             dataCell3, dataCellLoc3, fitResultsall3):
    psfPx = params['psfPx']
    winSize = params['winSize']

    # prevent interference of dead pixels
    im = sdf.hotpxfilter(im.astype("float32"))
    im = im.astype("uint16")

    imTS = create_cell_array(params, size, listCellnr, TSreg, t, channel) if TSregfile else None

    # run through each cell in image
    for cell in range(dataCell2.shape[1]):
        # check if spot was found with high or low threshold (dataDigital = 0)
        # and check if there were any spots found in other time frames in that cell
        if sum(dataDigital[:, cell]) != 0 and sum(dataDigital[:, cell] > 0) != 0:
            # determine first burst
            if dataDigital[0, cell] > 0:
                startFirstBurst = 0
            else:
                startFirstBurst = (np.where(np.diff(dataDigital[:, cell]) > 0)[0][0]) + 1
            if dataCell2[t, cell] == 0:
                if t < startFirstBurst:
                    t_alt = startFirstBurst
                elif t > startFirstBurst:
                    # determine last burst
                    endBursts = np.where(np.diff(dataDigital[:, cell]) < 0)[0]
                    pos = t - endBursts > 0
                    b = min(t - endBursts[pos])
                    minpos = np.where(t - endBursts == b)[0][0]
                    lastBurst = endBursts[minpos]
                    t_alt = lastBurst
                else:
                    raise ValueError(f'Somehow the start of a burst and a gap are both at frame {t}.')

                # fit 2D gaussian at fixed position
                cooGuess = coo_guess_gap(params, TSregfile, imTS, dataCellLoc2, t_alt, cell, listCellnr, TSreg, t,
                                         channel)

                if tracks is not None:
                    # keep the spot fixed relative to the cell instead of relative to the image
                    cooGuess += (tracks.loc[cell + 1, t][['x', 'y']] - tracks.loc[cell + 1, t_alt][['x', 'y']]) / pxsize

                intensity, coo, tilt = sdf.GaussianMaskFit2(im, cooGuess, psfPx, optLoc=0, winSize=winSize)
                coo = transformCoords(coo, channel, params, True)

                # write data to arrays
                dataCell3[t, cell] = intensity
                dataCellLoc3[t, cell, :] = np.r_[intensity, coo, tilt]
                # add results to fitResultsall2 (spots for all images in timeseries)
                fitResultsall3.append(np.r_[intensity, coo, tilt, t])


def TSmask(params):
    TSregfile = params['TSregfile']
    if TSregfile == []:
        # open cell mask and get number of cells
        if 'cell_track_file' in params:
            trackmate_table = pandas.read_table(params['cell_track_file'])
            maxCellid = int(trackmate_table['label'].max()) + 1
            cellArray = params['mask_file']
        else:
            with imr(params['mask_file']) as mask:
                cellArray = np.dstack([mask(c) for c in range(mask.shape[2])]).astype('int')
            maxCellid = np.amax(cellArray)
        TSreg = None
        listCellnr = None
    else:
        cellArray = None
        TSreg = []
        if "trk_results" in TSregfile[0]:  ### check if input locfile from previous localization
            listCellnr = [int(TSregfile[x].split("cellnr_")[1].split("_trk")[0]) for x in range(np.r_[TSregfile].shape[0])]
            TSregfile = [x for _, x in sorted(zip(listCellnr, TSregfile), key=lambda pair: pair[0])]
            listCellnr = np.sort(listCellnr)
            for TS in range(np.r_[TSregfile].shape[0]):
                TSregfiletmp = os.path.join(params['pathOut'], TSregfile[TS])
                if TSregfiletmp[-4:] != ".txt": TSregfiletmp += ".txt"
                TSregtmp = np.loadtxt(TSregfiletmp, delimiter='\t')
                TSregtmp = TSregtmp.astype("float").astype("int")
                TSreg.append(np.array(TSregtmp[0:, np.r_[1, 0]]))  # y and x
            maxCellid = max(listCellnr)
        else:  # if not locfile, it assumes the input is from fiji macro
            for TS in range(np.r_[TSregfile].shape[0]):
                TSregfiletmp = os.path.join(params['pathOut'], TSregfile[TS])
                if TSregfiletmp[-4:] != ".txt": TSregfiletmp += ".txt"
                TSregtmp = np.loadtxt(TSregfiletmp, dtype="str", delimiter='\t')
                TSreg.append(np.array(TSregtmp[1:, np.r_[4, 3]]).astype("int"))  # y and x
            maxCellid = np.r_[TSregfile].shape[0]
            listCellnr = range(1, maxCellid + 1)
    return cellArray, TSreg, maxCellid, listCellnr



def localizeMaster(params):
    ChannelsToAnalyze = params['ChannelsToAnalyze']
    fillIn = params['fillIn']
    fillInRadius = params['fillInRadius']
    fillInColors = params['fillInColors']
    thresholdSD_Red = params['thresholdSD_Red']
    thresholdSD_Green = params['thresholdSD_Green']
    diffThresholdSD = params['diffThresholdSD']
    writeLocTifFiles = params['writeLocTifFiles']
    psfPx = params['psfPx']
    maxDist = params['maxDist']
    minSeparation = params['minSeparation']
    dist = params['dist']
    TSregfile = params['TSregfile']

    pathOut = params['pathOut']
    date = params['date']
    channels = params['channels']

    print("Running Localize")

    if 'cell_track_file' in params:
        tracks = pandas.read_table(params['cell_track_file'], index_col=0)
        tracks = pandas.concat((tracks, trackmate.interpolate_missing(tracks, params['frames'])))
        tracks = tracks.set_index(['label', 't'])
    else:
        tracks = None

    if fillIn == 1 and fillInColors == "red" and fillInRadius > 0:
        ChannelsToAnalyze = ChannelsToAnalyze[::-1]

    # save parameters localize
    par = []
    if 0 in ChannelsToAnalyze:
        par.extend(["ThresholdSD red=" + str(thresholdSD_Red), "psfPx=" + str(psfPx), "MaxDist=" + str(maxDist),
                    "minSeparation=" + str(minSeparation), "dist=" + str(dist)])
    if 1 in ChannelsToAnalyze:
        par.extend(["ThresholdSD green=" + str(thresholdSD_Green), "psfPx=" + str(psfPx), "MaxDist=" + str(maxDist),
                    "minSeparation=" + str(minSeparation), "dist=" + str(dist)])

    parameterfile = os.path.join(pathOut, date + "_" + "localize_parameters.txt")
    np.savetxt(parameterfile, par, delimiter="\n", fmt="%s")

    # Open image
    imagefile = params['maxFile']
    with imr(imagefile) as imtmp:
        # determine image size and number of stacks
        size = imtmp.shape[:2]
        nbIm = params['frames']

        cellArray, TSreg, maxCellid, listCellnr = TSmask(params)

        ### loop over channels
        for channel in ChannelsToAnalyze:
            if (channel == 0 and not os.path.exists(params['expPathOut'] + "_dataLocalize3DArrayRed.npy") or (
                    channel == 1 and not os.path.exists(params['expPathOut'] + "_dataLocalize3DArrayGreen.npy"))):  ### checks if the loc_results file is not there already.

                if channel == 0: color = "red"; thresholdSD = thresholdSD_Red
                if channel == 1: color = "green"; thresholdSD = thresholdSD_Green

                if fillIn == 1 and fillInColors == color and fillInRadius == 0:
                    continue

                if fillIn == 1 and fillInRadius > 0 and fillInColors == color:
                    params['TSregRadius'] = fillInRadius
                    TSregfile = [file for file in os.listdir(params['pathOut'])
                                 if file.endswith(f'_trk_results_{color}.txt')]
                    if TSregfile:
                        cellArray = None
                        TSreg = []
                        if "trk_results" in TSregfile[0]:  ### check if input locfile from previous localization
                            listCellnr = [int(TSregfile[x].split("cellnr_")[1].split("_trk")[0]) for x in
                                          range(np.r_[TSregfile].shape[0])]
                            TSregfile = [x for _, x in sorted(zip(listCellnr, TSregfile), key=lambda pair: pair[0])]
                            listCellnr = np.sort(listCellnr)
                            for TS in range(np.r_[TSregfile].shape[0]):
                                TSregfiletmp = os.path.join(pathOut, TSregfile[TS])
                                if TSregfiletmp[-4:] != ".txt": TSregfiletmp += ".txt"
                                TSregtmp = np.loadtxt(TSregfiletmp, delimiter='\t')
                                TSregtmp = TSregtmp.astype("float").astype("int")
                                TSreg.append(np.array(TSregtmp[0:, np.r_[1, 0]]))  # y and x
                            # maxCellid = max(listCellnr)

                # create empty lists an arrays with rows as timepoints and columns as cellnr.
                fitResultsall = []
                cooGuess2_matrix = []

                dataCell = np.zeros((nbIm, maxCellid))  # 2D array with rows as timepoints and columns as cellnr. If non-zero, records will be intensity of spots
                dataCellLoc = np.zeros((nbIm, maxCellid, 6))  # 3D array with rows as timepoints and columns as cellnr and z as information on spot: integrated intensity, x position, y position, offset of tilted plane, x tilt, y tilt
                dataDigital = np.zeros((nbIm, maxCellid))  # 2D array with rows as timepoints and columns as cellnr. If 1, spot was found with high intensity threshold. If 0.5, spot was found with low intensity threshold. If 0, spot was filled in.

                if params.get('ParallelLocalisation', False):
                    cooGuess1_matrix, cooGuess2_matrix = guessHighThreshold(params, imtmp, channel, nbIm, TSregfile,
                                                                            listCellnr, TSreg, cellArray)
                    fitResultsall, dataCell, dataCellLoc, dataDigital = localizeHighThresholdPar(params, imtmp, channel,
                                                                                      cooGuess1_matrix, nbIm, maxCellid)
                else:
                    # run through each image to determine most intense spots (high threshold) and weaker spots (low threshold)
                    with ExitStack() as stack:
                        if isinstance(cellArray, str):  # each frame has a separate cell array made by trackmate
                            cell_array_imr = stack.enter_context(imr(cellArray, dtype=int))

                        for j in trange(nbIm, desc='Fitting high intensity spots'):
                            im = copy2.deepcopy(imtmp[:, :, channels[channel], :, j].reshape(size))
                            if cellArray is None:
                                cA = None
                            elif isinstance(cellArray, str):
                                cA = cell_array_imr(channels[channel], 0, j)  # TODO: multiple channels and registration
                            elif cellArray.shape[2] > 1:
                                cA = cellArray[:, :, channels[channel]]
                            else:
                                cA = cellArray[:, :, 0]
                            localizeHighThreshold(params, im, channel, j, size, cooGuess2_matrix, TSregfile, listCellnr,
                                                  TSreg, cA, fitResultsall, dataCell, dataDigital, dataCellLoc)

                # Save the results of spots found with high threshold in a text file, save images to tif file: columns are: integrated intensity, x position, y position, offset of tilted plane, x tilt, y tilt, framenumber
                fnTxt = params['expPathOut'] + "_loc_results_" + color + "_threshold_SD" + str(thresholdSD) + ".txt"
                np.savetxt(fnTxt, fitResultsall, delimiter="\t")

                # write text file of highest intensity per cell
                outfileLocCell = params['expPathOut'] + "_intensity_per_cell_" + color + "_threshold_SD" + str(
                    thresholdSD) + ".txt"
                np.savetxt(outfileLocCell, dataCell, delimiter="\t")

                # print results in terminal window
                print("\n\n*** Found %d spots. ***\nResults save in '%s'." % (len(fitResultsall), fnTxt))

                #############################################################################################################
                ##### Finding spot with lower threshold where no spot was found, in vicinity of spots with higher intensity.
                #############################################################################################################

                # Copy 3 arrays to new variables
                dataCell2 = copy2.deepcopy(
                    dataCell)  # 2D array with rows as timepoints and columns as cellnr. If non-zero, records will be intensity of spots
                dataCellLoc2 = copy2.deepcopy(
                    dataCellLoc)  # 3D array with rows as timepoints and columns as cellnr and z as information on spot: integrated intensity, x position, y position, offset of tilted plane, x tilt, y tilt
                fitResultsall2 = [] # all spots found with lower intensity threshold

                # run through each image to find most weaker intensity spots (low threshold)
                with ExitStack() as stack:
                    if isinstance(cellArray, str):  # each frame has a separate cell array made by trackmate
                        cell_array_imr = stack.enter_context(imr(cellArray, dtype=int))
                    for j in trange(nbIm, desc='Fitting low intensity spots'):
                        im = copy2.deepcopy(imtmp[:, :, channels[channel], :, j].reshape(size))
                        if cellArray is None:
                            cA = None
                        elif isinstance(cellArray, str):
                            cA = cell_array_imr(channels[channel], 0, j)  # TODO: multiple channels and registration
                        elif cellArray.shape[2]>1:
                            cA = cellArray[:,:,channels[channel]]
                        else:
                            cA = cellArray[:,:,0]
                        localizeLowThreshold(params, im, j, channel, size, cooGuess2_matrix, TSregfile, listCellnr,
                                             TSreg, cA, dataCell, dataDigital, dataCell2, dataCellLoc2, fitResultsall2)

                # Save the results of spots found with low threshold in a text file: columns are: integrated intensity, x position, y position, offset of tilted plane, x tilt, y tilt, framenumber
                fnTxt2 = params['expPathOut'] + "_loc_results_" + color + "_threshold_SD" + str(thresholdSD-diffThresholdSD) + ".txt"
                np.savetxt(fnTxt2, fitResultsall2, delimiter="\t")

                # write text file of spots intensities, found with high and low threshold per cell
                outfileLocCell2 = params['expPathOut'] + "_intensity_per_cell_" + color + "_threshold2_SD" + str(
                    thresholdSD - diffThresholdSD) + ".txt"
                np.savetxt(outfileLocCell2, dataCell2, delimiter="\t")

                #############################################################################################################
                ##### Find intensity of sites where no spot was found, at position of previous/first spots
                #############################################################################################################

                # Copy 3 arrays to new variables
                dataCell3 = copy2.deepcopy(
                    dataCell2)  # 2D array with rows as timepoints and columns as cellnr. If non-zero, records will be intensity of spots
                dataCellLoc3 = copy2.deepcopy(
                    dataCellLoc2)  # 3D array with rows as timepoints and columns as cellnr and z as information on spot: integrated intensity, x position, y position, offset of tilted plane, x tilt, y tilt
                fitResultsall3 = []

                #### run through each image and if no spot was found for a cell in that timeframe, fill in the intensity at position of previous spot (or first spot in movie)
                for t in trange(nbIm, desc='Filling gaps with no spots'):
                    im = imtmp[:, :, channels[channel], :, t].reshape(size)
                    fill_gap(params, im, imtmp.pxsize, tracks, t, channel, TSregfile, size, listCellnr, TSreg,
                             dataCell2, dataDigital, dataCellLoc2, dataCell3, dataCellLoc3, fitResultsall3)

                # Save the results of filledGap spots in a text file: columns are: integrated intensity, x position, y position, offset of tilted plane, x tilt, y tilt, framenumber
                fnTxt3 = params['expPathOut'] + "_loc_results_" + color + "_filledGap.txt"
                np.savetxt(fnTxt3, fitResultsall3, delimiter="\t")

                # write text file of all spot intensities per cell, found with high + low threshold and filled-in.
                outfileLocCell3 = params['expPathOut'] + "_intensity_per_cell_" + color + "_all.txt"
                np.savetxt(outfileLocCell3, dataCell3, delimiter="\t")

                # write files for each cell, with x position, y position, integrated intensity, frame, dataDigital (1 = found with high threshold, 0.5 = found with low threshold, 0 = not found)
                for cell in range(dataCell3.shape[1]):
                    if sum(dataDigital[:, cell]) != 0:
                        data = np.c_[dataCellLoc3[:, cell, 1], dataCellLoc3[:, cell, 2], dataCellLoc3[:, cell, 0], np.c_[
                            range(dataCellLoc3.shape[0])], dataDigital[:, cell]]
                        outfileLocCellAll = params['expPathOut'] + "_cellnr_" + str(
                            cell + 1) + "_trk_results_" + color + ".txt"
                        np.savetxt(outfileLocCellAll, data, delimiter="\t", fmt="%1.3f")

                # write digital data to text file: rows as timepoints and columns as cellnr. If 1, spot was found with high intensity threshold. If 0.5, spot was found with low intensity threshold. If 0, spot was filled in.
                outfileDataDigital = params['expPathOut'] + "_" + color + "_digital.txt"
                np.savetxt(outfileDataDigital, dataDigital, delimiter="\t")

                if color == "green":
                    np.save(params['expPathOut'] + "_dataLocalize3DArrayGreen.npy",
                         dataCellLoc3)  # save dataCellLoc3 in numpy array
                    params['dataDigitalGreen'] = dataDigital
                if color == "red":
                    np.save(params['expPathOut'] + "_dataLocalize3DArrayRed.npy",
                         dataCellLoc3)  # save dataCellLoc3 in numpy array
                    params['dataDigitalRed'] = dataDigital

                # write image file with 1: original image, 2: spots found with high threshold, 3: low threshold, 4: all spots (including filled-in).
                if writeLocTifFiles:
                    fname = params['expPathOut'] + "_loc_results_" + color + "_thresholdhigh_SD" + str(thresholdSD) \
                            + "+thresholdlow_SD" + str(thresholdSD - diffThresholdSD)+ "+all+mask.tif"

                    # We open the max proj and the new tiff file and transfer the data frame by frame.
                    # IJTiffFile takes the filename of the new tiff file and the shape (C, Z, T) of the new file,
                    #  it creates a new object which we call tiff.
                    # Calling im(c, z, t) gives exactly 1 frame in with 2 dimensions from the max proj.
                    # We save a frame by calling tiff.save(frame, c, z, t), where c, z, t is the position in the tiff
                    #  file where we want this frame to end up. The order in which we write the frames is not important.
                    # We do need to make sure the datatype is supported, im and the imSpotsAll had datatype float64,
                    #  which is not supported.

                    with imr(params['maxFile']) as im, IJTiffFile(fname, (5, 1, nbIm)) as tiff, ExitStack() as stack:
                        cell_array_per_frame = False
                        if TSregfile == []:
                            # open cell mask and make outline
                            if os.path.exists(f"{params['expPathOut']}_cells_mask.tif"):
                                cell_array_per_frame = True
                                cell_array_imr = stack.enter_context(imr(f"{params['expPathOut']}_cells_mask.tif",
                                                                         dtype=int))
                            else:
                                with imr(f"{params['expPathOut']}_sum_cells_mask.tif") as mask:
                                    cellmaskArray = mask(channels[channel]).astype('int')


                        for t in trange(nbIm, desc='Writing image to check thresholds'):
                            tiff.save(im(channels[channel], 0, t).astype('uint16'), 0, 0, t)

                            # add squares to image around spots, remove numbers above 1023 and below 0.
                            def image_with_spots(data, size, t):
                                imSpots = np.zeros(size)  # make empty image
                                if len(data)>0:
                                    for a in data:
                                        if a[6] == t:
                                            x = np.clip(int(a[2]) + squareStamp[1], 0, size[0] - 1)
                                            y = np.clip(int(a[1]) + squareStamp[0], 0, size[1] - 1)
                                            imSpots[x, y] = 1
                                return imSpots

                            #### make image with spots found with high threshold
                            imSpots = image_with_spots(fitResultsall, size, t)
                            tiff.save(imSpots.astype('uint16'), 1, 0, t)

                            #### make image with spots found with low threshold
                            imSpots = image_with_spots(fitResultsall2, size, t)
                            tiff.save(imSpots.astype('uint16'), 2, 0, t)

                            # #### make image with all spots (also filledGap)
                            imSpots = image_with_spots(fitResultsall3, size, t)
                            tiff.save(imSpots.astype('uint16'), 3, 0, t)

                            #### add mask to image
                            if TSregfile != []:
                                cellmaskArray = create_cell_array(params, size, listCellnr, TSreg, t, channel)
                            elif cell_array_per_frame:
                                cellmaskArray = cell_array_imr(channels[channel], 0, t)
                            cellmask = Image.fromarray(cellmaskArray.astype('uint8'), 'L')
                            cellmask = cellmask.filter(ImageFilter.FIND_EDGES)
                            celloutline = np.array(cellmask) >= 1
                            celloutline = celloutline*1

                            tiff.save(celloutline.astype('uint16'), 4, 0, t)
    params['maxCellid'] = maxCellid

def localizeSlave(params):
    fillInColors = params['fillInColors']
    psfPx = params['psfPx']
    winSize = params['winSize']
    channels = params['channels']

    maxCellid = params['maxCellid']
    if 'dataDigitalGreen' in params:
        dataDigitalGreen = params['dataDigitalGreen']
    if 'dataDigitalRed' in params:
        dataDigitalRed = params['dataDigitalRed']

    color = {'red': 'green', 'green': 'red'}[fillInColors]

    outfileLocCell3 = params['expPathOut'] + "_intensity_per_cell_" + color + "_all.txt"
    dataCell3 = np.loadtxt(outfileLocCell3, delimiter="\t", ndmin=2)
    if color=='green':
        dataCellLocGreen = np.load(params['expPathOut'] + "_dataLocalize3DArrayGreen.npy")
    if color=='red':
        dataCellLocRed = np.load(params['expPathOut'] + "_dataLocalize3DArrayRed.npy")

    # Open image
    with imr(params['maxFile']) as imtmp:
        # determine image size and number of stacks
        size = imtmp.shape[:2]
        nbIm = params['frames']

        if ((fillInColors == "red" and not os.path.exists(params['expPathOut']+"_dataLocalize3DArrayRed.npy")) or (fillInColors == "green" and not os.path.exists(params['expPathOut']+"_dataLocalize3DArrayGreen.npy"))):
            redEmpty = []
            greenEmpty = []
            if fillInColors == "red": dataDigitalRed = np.zeros((nbIm,maxCellid)); dataCellLocRed = np.zeros((nbIm,maxCellid,6))
            if fillInColors == "green": dataDigitalGreen = np.zeros((nbIm,maxCellid)); dataCellLocGreen = np.zeros((nbIm,maxCellid,6))
            for c in range(dataCell3.shape[1]):
                if sum(dataDigitalRed[:,c]) == 0 and sum(dataDigitalGreen[:,c])!=0: redEmpty.append(c)
                if sum(dataDigitalRed[:,c]) != 0 and sum(dataDigitalGreen[:,c])==0: greenEmpty.append(c)
            if fillInColors == "red": channel = 0
            if fillInColors == "green": channel = 1
            for t in trange(dataCell3.shape[0], desc='Filling in {} channel'.format(fillInColors)):
                im = copy2.deepcopy(imtmp[:,:,channels[channel],:,t].reshape(size))
                for cell in redEmpty:
                    cooGuess = dataCellLocGreen[t,cell,1:3].copy()
                    cooGuess = transformCoords(cooGuess, channel, params, False)
                    intensity,coo,tilt=sdf.GaussianMaskFit2(im, cooGuess, psfPx, optLoc = 0, winSize = winSize)
                    coo = transformCoords(coo, channel, params, True)
                    dataCellLocRed[t,cell,:] = np.r_[intensity,coo,tilt]

                for cell in greenEmpty:
                    cooGuess = dataCellLocRed[t, cell, 1:3].copy()
                    cooGuess = transformCoords(cooGuess, channel, params, False)
                    intensity,coo,tilt=sdf.GaussianMaskFit2(im, cooGuess, psfPx, optLoc = 0, winSize = winSize)
                    coo = transformCoords(coo, channel, params, True)
                    dataCellLocGreen[t,cell,:] = np.r_[intensity,coo,tilt]

            if fillInColors == "red": np.savetxt(params['expPathOut']+"_"+fillInColors+"_digital.txt", dataDigitalRed, delimiter = "\t")
            if fillInColors == "green": np.savetxt(params['expPathOut']+"_"+fillInColors+"_digital.txt", dataDigitalGreen, delimiter = "\t")

            # write results to text file
            for cell in redEmpty:
                data = np.c_[dataCellLocRed[:,cell,1],dataCellLocRed[:,cell,2], dataCellLocRed[:,cell,0], np.c_[range(dataCellLocRed.shape[0])], dataDigitalRed[:,cell]]
                outfileLocCellAll = params['expPathOut']+"_cellnr_"+str(cell+1)+"_trk_results_red.txt"
                np.savetxt(outfileLocCellAll, data, delimiter = "\t", fmt = "%1.3f")
            for cell in greenEmpty:
                data = np.c_[dataCellLocGreen[:,cell,1],dataCellLocGreen[:,cell,2], dataCellLocGreen[:,cell,0], np.c_[range(dataCellLocGreen.shape[0])], dataDigitalGreen[:,cell]]
                outfileLocCellAll = params['expPathOut']+"_cellnr_"+str(cell+1)+"_trk_results_green.txt"
                np.savetxt(outfileLocCellAll, data, delimiter = "\t", fmt = "%1.3f")

            np.save(params['expPathOut']+"_dataLocalize3DArrayGreen.npy",dataCellLocGreen) # save dataCellLoc3 in numpy array
            np.save(params['expPathOut']+"_dataLocalize3DArrayRed.npy",dataCellLocRed) # save dataCellLoc3 in numpy array

def loadFiles(params):
    fillIn = params['fillIn']
    fillInColors = params['fillInColors']
    ChannelsToAnalyze = params['ChannelsToAnalyze']

    dataCellLoc3 = None
    dataDigitalRed = None
    dataDigitalGreen = None

    # load dataCellLoc3 and dataDigital
    if 1 in ChannelsToAnalyze:
        dataCellLocGreen = np.load(params['expPathOut'] + "_dataLocalize3DArrayGreen.npy")
        dataCellLoc3 = dataCellLocGreen
        outfileDataDigitalGreen = params['expPathOut'] + "_green_digital.txt"
        dataDigitalGreen = np.loadtxt(outfileDataDigitalGreen, ndmin=2)
    if 0 in ChannelsToAnalyze:
        dataCellLocRed = np.load(params['expPathOut'] + "_dataLocalize3DArrayRed.npy")
        dataCellLoc3 = dataCellLocRed
        outfileDataDigitalRed = params['expPathOut'] + "_red_digital.txt"
        dataDigitalRed = np.loadtxt(outfileDataDigitalRed, ndmin=2)
    if fillIn == 1:
        if fillInColors == "red":
            dataDigitalRed = np.loadtxt(params['expPathOut'] + "_green_digital.txt", ndmin=2)
            dataDigitalGreen = np.loadtxt(params['expPathOut'] + "_green_digital.txt", ndmin=2)
        elif fillInColors == "green":
            dataDigitalRed = np.loadtxt(params['expPathOut'] + "_red_digital.txt", ndmin=2)
            dataDigitalGreen = np.loadtxt(params['expPathOut'] + "_red_digital.txt", ndmin=2)

    return dataCellLoc3, dataDigitalRed, dataDigitalGreen

def calcBackground(params):
    pathOut = params['pathOut']
    expPathOut = params['expPathOut']
    ChannelsToAnalyze = params['ChannelsToAnalyze']
    psfPx = params['psfPx']
    winSize = params['winSize']
    channels = params['channels']
    DistBgPx = 5
    dir = [[1, 1], [1, -1], [-1, -1], [-1, 1]]

    # Open image
    imagefile = params['maxFile']
    with imr(imagefile) as imtmp:

        # determine image size and number of stacks
        nbIm = params['frames']

        files = os.listdir(pathOut)
        trk_red = []
        trk_green = []
        for x in range(len(files)):
            if files[x][0] != ".":
                if files[x][-19:] == "trk_results_red.txt":
                    trk_red.append(files[x])
                if files[x][-21:] == "trk_results_green.txt":
                    trk_green.append(files[x])

        for channel in ChannelsToAnalyze:
            if channel == 0:
                color = "red"
                listCellnr = [int(trk_red[x].split("cellnr_")[1].split("_trk")[0]) for x in range(np.r_[trk_red].shape[0])]
                trk_red = [x for _, x in sorted(zip(listCellnr, trk_red), key=lambda pair: pair[0])]
            if channel == 1:
                color = "green"
                listCellnr = [int(trk_green[x].split("cellnr_")[1].split("_trk")[0]) for x in
                              range(np.r_[trk_green].shape[0])]
                trk_green = [x for _, x in sorted(zip(listCellnr, trk_green), key=lambda pair: pair[0])]
            listCellnr = np.sort(listCellnr)

            # 3D array with rows as bg nr and columns as timepoints and z as information on 1st background spot:
            #   integrated intensity, x position, y position, offset of tilted plane, x tilt, y tilt

            # bg = np.zeros((len(dir), nbIm, 6))
            # for cell in trange(len(listCellnr), desc='Calculating background'):
            @parfor(range(len(listCellnr)), desc='Calculating background')
            def fun(cell):
                bg = np.zeros((len(dir), nbIm, 6))
                trk_results = np.loadtxt(os.path.join(pathOut, trk_red[cell] if channel == 0 else trk_green[cell]),
                                         ndmin=2)
                coo = transformCoords(trk_results[:, :2], channel, params, False)
                for j in range(nbIm):
                    im = copy2.deepcopy(imtmp(channels[channel], 0, j))
                    for i in range(len(dir)):
                        try:
                            bg[i, j, :] = np.r_[sdf.GaussianMaskFit2(im, coo[j]+DistBgPx*np.array(dir[i]),
                                                                     psfPx, optLoc=0, winSize=winSize)]
                        except Exception:
                            print(f'Fitting background {i+1} at different location for cell {listCellnr[cell]}')
                            bg[i, j, :] = np.r_[sdf.GaussianMaskFit2(im, coo[j]-2*DistBgPx*np.array(dir[i]),
                                                                     psfPx, optLoc=0, winSize=winSize)]
                        bg[i, j, :2] = transformCoords(bg[i, j, :2], channel, params, True)

                for i in range(len(dir)):
                    np.savetxt(f'{expPathOut}_cellnr_{listCellnr[cell]}_trk_results_bg{i+1}_{color}.txt',
                               np.c_[bg[i, :, 1], bg[i, :, 2], bg[i, :, 0],
                               np.c_[range(bg.shape[1])], np.zeros(nbIm)],
                               delimiter='\t', fmt='%1.3f')


def writeFiles(params):
    ChannelsToAnalyze = params['ChannelsToAnalyze']
    microscope = params['microscope']
    timeInterval = params['timeInterval']
    TSregfile = params['TSregfile']
    nChannels = params['nChannels']
    pathIn = params['pathIn']

    print("Write files")
    # write metadata_head file (first 500 lines metadata)
    if microscope == "AxioObserver":  # TODO: cross compatibility
        if not os.path.exists(params['expPathOut'] + "_metadata_head.txt"):
            os.system('head -n 500 ' + os.path.join(re.escape(pathIn), 'metadata.txt') + ' > ' + re.escape(
                params['expPathOut'] + "_metadata_head.txt"))

    dataCellLoc3, dataDigitalRed, dataDigitalGreen = loadFiles(params)

    #### write listfiles
    listFilesOut = params['expPathOut'] + ".list.py"
    if not os.path.exists(listFilesOut):
        with open(listFilesOut, "w") as f:
            f.write("# coding: utf-8" + "\n" + "listFiles=[" + "\n")
            if TSregfile == []:
                listCellnr = range(1, dataCellLoc3.shape[1] + 1)
            else:
                if "trk_results" in TSregfile[0]:  ### check if input locfile from previous localization
                    listCellnr = [int(TSregfile[x].split("cellnr_")[1].split("_trk")[0]) for x in
                                  range(np.r_[TSregfile].shape[0])]
                    TSregfile = [x for _, x in sorted(zip(listCellnr, TSregfile), key=lambda pair: pair[0])]
                    listCellnr = np.sort(listCellnr)
                else:
                    listCellnr = range(1, dataCellLoc3.shape[1] + 1)

            for i in range(len(listCellnr)):
                GreenTrkExists = os.path.exists(params['expPathOut'] + "_cellnr_" + str(listCellnr[i]) + "_trk_results_green.txt")
                RedTrkExists = os.path.exists(params['expPathOut'] + "_cellnr_" + str(listCellnr[i]) + "_trk_results_red.txt")
                if dataDigitalRed is None:
                    SumDigitalRed = 0
                else:
                    SumDigitalRed = sum(dataDigitalRed.reshape(dataCellLoc3.shape[0], dataCellLoc3.shape[1])[:, i])
                if dataDigitalGreen is None:
                    SumDigitalGreen = 0
                else:
                    SumDigitalGreen = sum(dataDigitalGreen.reshape(dataCellLoc3.shape[0], dataCellLoc3.shape[1])[:, i])


                if (SumDigitalGreen != 0 or SumDigitalRed != 0) and (
                        (nChannels == 2 and GreenTrkExists and RedTrkExists) or (
                        nChannels == 1 and (GreenTrkExists or RedTrkExists))):

                    if microscope == "AxioObserver":
                        f.write("    " + "{'metadata': '" + params['expPathOut'] + "_metadata_head.txt" + "'," + '\n')
                    elif microscope == "spinningDisk" or "unknown":
                        f.write("    " + "{'timeInterval': '" + str(timeInterval) + "'," + '\n')
                    else:
                        with imr(pathIn) as im:
                            f.write("    " + "{'timeInterval': '" + str(im.timeinterval) + "'," + '\n')
                    f.write("    " + "'maxProj': '" + params['maxFile'] + "'," + '\n')
                    f.write("    " + "'rawPath': '" + pathIn + "'," + '\n')
                    if os.path.exists(params['expPathOut'] + "_cellnr_" + str(listCellnr[i]) + "_trk_results_green.txt"):
                        f.write("    " + "'trk_g': '" + params['expPathOut'] + "_cellnr_" + str(
                                listCellnr[i]) + "_trk_results_green.txt" + "'," + '\n')
                    elif 1 not in ChannelsToAnalyze:
                        f.write("    " + "'trk_g': '" + params['expPathOut'] + "_cellnr_" + str(
                                listCellnr[i]) + "_trk_results_red.txt" + "'," + '\n')
                    if os.path.exists(params['expPathOut'] + "_cellnr_" + str(listCellnr[i]) + "_trk_results_red.txt"):
                        f.write("    " + "'trk_r': '" + params['expPathOut'] + "_cellnr_" + str(
                                listCellnr[i]) + "_trk_results_red.txt" + "'," + '\n')
                    elif 0 not in ChannelsToAnalyze:
                        f.write("    " + "'trk_r': '" + params['expPathOut'] + "_cellnr_" + str(
                                listCellnr[i]) + "_trk_results_green.txt" + "'," + '\n')
                    if dataDigitalRed is None:
                        lastRed = dataCellLoc3.shape[0]
                    else:
                        lastRed = np.where(dataDigitalRed.reshape(dataCellLoc3.shape[0], dataCellLoc3.shape[1])[:, i] >= 0.5)[0][-1]
                    if dataDigitalGreen is None:
                        lastGreen = dataCellLoc3.shape[0]
                    else:
                        lastGreen = np.where(dataDigitalGreen.reshape(dataCellLoc3.shape[0], dataCellLoc3.shape[1])[:, i] >= 0.5)[0][-1]
                    f.write("    " + "'frameWindow': [0," + str(
                        min(lastRed, lastGreen)) + "]" + '\n' + '    ' + '},' + '\n' + '\n')
            f.write(']')

    # load data based on list file
    listFilesOutEnd = params['expPathOut'] + ".list.py"
    params['loadedData'] = daf.ExpData(listFilesOutEnd)

    # write PDFs of trace and correlation functions for each cell.
    for i in range(len(params['loadedData'])):
        name = params['loadedData'][i].name.split(".txt")[0]
        trackPDF = daf.showData(params['loadedData'][i], 0)
        trackPDF.savefig(name + "_trace.pdf", format="pdf")

def writeMovies(params):
    print("Making movies")
    if not 'loadedData' in params:
        # load data based on list file
        listFilesOutEnd = os.path.join(params['expPathOut'] + ".list.py")
        params['loadedData'] = daf.ExpData(listFilesOutEnd)

    channels = [params['channels'][c] for c in params['ChannelsToAnalyze']]

    if params['RegisterChannels']:
        transform = params['transform']
    else:
        transform = False

    daf.showTracking(params['loadedData'], channels, params['expPathOut'], params['sideViews'],
                          params['zSlices'], (0, params['frames']), transform=transform, drift=params.get('drift'))

def adjustContrast(params):
    ChannelsToAnalyze = params['ChannelsToAnalyze']
    sideViews = params['sideViews']
    frames = params['frames']

    print("Adjusting contrast movies")
    dataCellLoc3, dataDigitalRed, dataDigitalGreen = loadFiles(params)
    # run macro for each movie
    for cell in range(1, (dataCellLoc3.shape[1] + 1)):
        imPath = params['expPathOut'] + "_cellnr_" + str(cell) + "_track" + "Side" * sideViews + ".tif"
        if os.path.exists(imPath):
            if len(ChannelsToAnalyze) > 1:
                os.system(
                    "/DATA/lenstra_lab/Fiji.app/ImageJ-linux64 -eval '%s'" % daf.macro2color.replace('__imPath__', imPath))
            else:
                os.system("/DATA/lenstra_lab/Fiji.app/ImageJ-linux64 --headless -eval '%s'" % daf.macro1color.replace(
                    '__imPath__', imPath).replace('__frames__', str(frames)))

def computeCorrelFunc(params):
    print("Calculating correlation functions")
    # load data
    listFilesOutEnd = os.path.join(params['expPathOut'] + ".list.py")
    loadedData = daf.ExpData(listFilesOutEnd)

    ss = fluctuationAnalysis.mcSigSet([fluctuationAnalysis.mcSig(dA.t, np.vstack((dA.r, dA.g)), frameWindow=dA.frameWindow) for dA in loadedData])
    ss.alignSignals()
    ss.compAvgCF()
    ss.bootstrap()

    with PdfPages(os.path.join(params['expPathOut'] + "_correlation_functions_no_bg_subtration.pdf")) as pdf:
        plot_figures.showCorrFun(pdf, ss)

def create_circular_mask(h, w, center, radius):
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask


def create_cell_array(params, size, listCellnr, TSreg, t, channel):
    TSregRadius = params['TSregRadius']
    cell_array = np.zeros(size, dtype=int)

    for TS, cell in zip(TSreg, listCellnr):
        center = transformCoords([TS[t][1], TS[t][0]], channel, params, False)
        cell_array = np.fmax(cell_array, create_circular_mask(*size, center=center, radius=TSregRadius) * cell)
    return cell_array


def pipeline(params):
    ''' Runs the pipeline for livecell track movies
        params is either a dictionary containing the parameters for the pipeline
         or a string pointing to the yml file with the parameters
    '''
    if not isinstance(params, dict):
        parameter_file = params
        if parameter_file[-3:] == '.py':
            print(misc.color('Converting py parameter file into yml format', 'g'))
            misc.convertParamFile2YML(parameter_file)
            parameter_file = parameter_file[:-3]+'.yml'
        if not parameter_file[-4:] == '.yml':
            parameter_file += '.yml'
        params = misc.getParams(parameter_file, __file__.replace('.py', '_parameters_template.yml'),
                                ('folderIn', 'expList', 'outputfolder'))
    else:
        parameter_file = ''

    params = expandExpList(params, parameter_file) # get general parameters
    if params['TSregfile'] != [[]] and params['TSregfile'] != []:
        TSregfileList = params['TSregfile'] # store TSregfile
    else:
        params['TSregfile'] = []
        TSregfileList = []

    for exp in range(params['lenExpList']):
        params['pathIn'] = params['expList'][exp]
        if TSregfileList != []:
            params['TSregfile'] = TSregfileList[exp]

        getPaths(params, parameter_file) #### make file list and output directory
        
        if params['CorrectDrift'][exp]: ####correct any x-y drift
            correctDrift(params)

        #if params['MaxProj']:
        if params['domax'] or params['dosum']:
            maxProjection(params)

        if params['CellMask']: ##### Get cell masks
            cellMask(params)

        if params['OptimizeThreshold']:
            optimizeTreshold(params)

        if params['RunLocalize']: #### Finds spots (Localize)
            localizeMaster(params)

        #### fill in red tracks for green-only traces and fill in green track for red-only traces at position TS other color
        if params['RunLocalize'] and params['nChannels'] == 2 and params['fillIn'] and params['fillInRadius'] == 0:
            localizeSlave(params)

        if params['WriteFiles']: ##### Write other files (metadata file, listfiles and PDFs of each trace)
            writeFiles(params)

        if params['CalculateBackground']: #### calculate background if at 4 points at fixed location from TS
            calcBackground(params)

        if params['WriteMovies']: ##### Write movie of each cell
            writeMovies(params)

        if params['WriteMovies'] == 2: ### Let Fiji adjust the contrast, probably not necessary
            adjustContrast(params)

        if params['ComputeCorrelFunc']: ##### Calculate auto/cross correlation
            # Note, it is better to use pipeline_livecell_yeast_autocorrelation_combinedTraces.py
            # This script does not do background subtraction.
            computeCorrelFunc(params)

        if 'transform' in params:
            params.pop('transform')
        print('Saved results as: {}*****'.format(params['expPathOut']))
        with open(params['expPathOut'] + '_params.yml', 'w') as f:
            yaml.dump(params, f)

        with open(params['expPathOut'] + '_params.pkl', 'wb') as f:
            pickle.dump(params, f)

    return params


def main():
    ipy_debug()
    tm = time()
    if len(sys.argv) < 2:
        parameter_files = [os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                      'pipeline_livecell_track_movies_parameters.yml'))]
    else:
        parameter_files = sys.argv[1:]

    if len(parameter_files) == 1:
        parameter_file = parameter_files[0]
        if not os.path.exists(parameter_file):
            raise FileNotFoundError('Could not find the parameter file.')
        print(misc.color('Working on: {}'.format(parameter_file), 'b:b'))
        params = pipeline(parameter_file)
    else:
        for parameter_file in parameter_files:
            print(misc.color('Working on: {}'.format(parameter_file), 'b:b'))
            print('')
            try:
                pipeline(parameter_file)
            except Exception:
                print(misc.color('Exception while working on: {}'.format(parameter_file), 'r:b'))

    print('------------------------------------------------')
    print(misc.color('Pipeline finished, took {} seconds.'.format(time() - tm), 'g:b'))
    imr.kill_vm()  # stop java used for imread, needed to let python exit
    trackmate.kill_vm()


if __name__ == '__main__':
    main()
