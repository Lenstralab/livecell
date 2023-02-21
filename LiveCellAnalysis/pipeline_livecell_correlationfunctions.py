#!/usr/local/bin/ipython3 -i

import sys
import os
import psutil
import numpy as np
import copy as copy2
import yaml
import shutil
from operator import le
from scipy.optimize import curve_fit
from scipy.ndimage import label
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from tllab_common.wimread import imread as imr
from tllab_common.misc import ipy_debug

if __package__ is None or __package__ == '':  # usual case
    import misc
    import fluctuationAnalysis as FA
    import dataAnalysis_functions as daf
    import plot_figures
    import _version
else:  # in case you do from another package:
    from . import misc
    from . import fluctuationAnalysis as FA
    from . import dataAnalysis_functions as daf
    from . import plot_figures
    from . import _version

A4 = (11.69, 8.27)

if '__file__' not in locals():  # when executed using execfile
    import inspect
    __file__ = inspect.getframeinfo(inspect.currentframe()).filename


# This script is meant to calculate autocorrelations on a dataset.
# The dataset is specified in a list.py file (and can be a combination of multiple experiments).


def get_paths(params, parameter_file=None):
    """ specify input list.py and outputfolder """
    if not parameter_file == '':
        parameter_file = os.path.abspath(parameter_file)
        if not os.path.isabs(params['outputfolder']):
            params['outputfolder'] = os.path.abspath(os.path.join(os.path.split(parameter_file)[0],
                                                                  params['outputfolder']))
        if not os.path.isabs(params['PyFile']):
            params['PyFile'] = os.path.abspath(os.path.join(os.path.split(parameter_file)[0], params['PyFile']))

    params['outputfolder'] = os.path.join(params['outputfolder'], '')
    if not os.path.exists(params['outputfolder']):
        os.makedirs(params['outputfolder'])

    params['file'] = os.path.join(params['outputfolder'], os.path.split(params['PyFile'])[1].replace('.list.py', ''))

    date_time = datetime.now().strftime("%d-%b-%Y_%Hh%Mm%Ss")

    if os.path.exists(parameter_file):
        shutil.copyfile(parameter_file, os.path.join(
            f"{params['file']}_pipeline_livecell_correlationfunctions_parameters_runtime_{date_time}.yml"))
    else:
        parameter_file = os.path.join(
            f"{params['file']}_pipeline_livecell_correlationfunctions_parameters_runtime_{date_time}.yml")
        with open(parameter_file, 'w') as f:
            yaml.dump(params, f)
    shutil.copyfile(os.path.abspath(__file__), f"{params['file']}_pipeline_livecell_correlationfunctions.py")
    shutil.copyfile(os.path.abspath(_version.__file__), f"{params['file']}_livecell_version.py")


# MAIN FUNCTIONS
# Process analog data

def bg_sub_traces(data_orig, params):
    # Copy to new variable dataA: Analog data, processed
    data_a = copy2.deepcopy(data_orig)

    # Scale traces to correct for day-to-day effects if necessary
    if not params.get('scaleFactors') is None and len(params['scaleFactors']) == len(data_a):
        for data, scale_factor in zip(data_a, params['scaleFactors']):
            data.r *= scale_factor
            data.g *= scale_factor

    for channel in params['ChannelsToAnalyze']:
        color = ('Red', 'Green')[channel]

        bg_window_color = 'bgWindow{}'.format(color)
        c = color.lower()[0]
        trk_c = 'trk_'+c
        bgc = 'bg'+c
        mc = 'm'+c
        sdc = 'sd'+c

        # background subtract traces
        for i, data in enumerate(data_a):
            if params.get('quantizeTrace') and channel in params.get('quantizeTrace', {}).keys():
                data[bgc] = np.zeros(data[c].shape)
                data[mc] = 0
                data[sdc] = 1
            else:
                name = data.name.split(".txt")[0]

                # load digital data
                # use background for n spots at fixed distance from TS
                if os.path.exists('{}_bg1_{}.txt'.format(data.name, color.lower())):
                    bg = []
                    j = 1
                    found = True
                    while found:
                        bgtrk = np.loadtxt('{}_bg{}_{}.txt'.format(data.name, j, color.lower()))
                        bg.extend(bgtrk[FA.get_mask(data.t, data.frameWindow) == 1, 2])
                        j += 1
                        found = os.path.exists('{}_bg{}_{}.txt'.format(data.name, j, color.lower()))
                    data[bgc] = bg
                # define background manually in parameter file
                elif i in params[bg_window_color]:
                    bg = range(int(params[bg_window_color][i][0]/data.dt),
                               int(params[bg_window_color][i][1]/data.dt))
                    data[bgc] = data[c][bg]
                # define background as part of trace were no strong spots were found (high threshold)
                # only do background subtraction when there actually is a 5th column with (0, 0.5, 1)'s
                elif data[trk_c].shape[1] > 4:
                    bg = np.where(data[trk_c][:, -1] < 1)
                    data[bgc] = data[c][bg]
                else:  # no background subtraction, for example when using orbital tracking data
                    data[bgc] = np.zeros(data[c].shape)

                # log-normal fit
                if np.any(np.asarray(data[bgc]) > 0):
                    m, sd = misc.distfit(np.log(data[bgc]))
                else:
                    m, sd = 0, 0

                # replaced by fitting with the cdf in the lines above (wp)
                # x, y = misc.histoF(np.histogram(dA[bgc], bins=np.r_[0:10:.1]*np.median(dA[bgc]), density=True)).T
                # x -= np.diff(x)[0]/2
                # m, sd = optimize.fmin(lambda a: sum((y-((1/(x*a[1]*(2*np.pi)**0.5))
                #                       * np.exp(-(((np.log(x)-a[0])**2)/(2*a[1]**2)))))**2), np.r_[10000,5000], disp=0)

                data[mc] = m
                data[sdc] = sd
                data[c] -= np.exp(data[mc])

                # use this code if you want to set all the data outside the defined frameWindow to 0
                # dA[c][:dA.frameWindow[0]]=0.; dA[c][dA.frameWindow[1]:]=0.

                if 0 in params['ChannelsToAnalyze']:
                    np.savetxt(name + "_bg_sub_red.txt", data.r)

                else:
                    data.r = data.g
                if 1 in params['ChannelsToAnalyze']:
                    np.savetxt(name + "_bg_sub_green.txt", data.g)
                else:
                    data.g = data.r

#            # write PDF of trace with histogram and digital data trace
#            histFig = write_hist(None, "r", dA, sdThresholdRed*dA.sdr)
#            histFig.savefig(name+"_bg_sub_trace.pdf")
#            np.savetxt(name+"_bg_sub.txt",dA.g)
#            plt.close()
    return data_a


def binary_call_traces(data_a, params):
    """#############################################################
    ### Threshold data based on sdThreshold to make binary data
    #############################################################"""
    data_b = copy2.deepcopy(data_a)
    if params.get('binaryCallMethod', '').lower() == 'markov_ensemble':
        print("Calculating binary data using a hidden Markov model")
        for channel in params['ChannelsToAnalyze']:
            color = 'rgb'[channel]
            mbg = np.mean([np.exp(data['m'+color]) for data in data_a])
            sdbg = np.sqrt(np.sum([np.exp(data['m'+color]) ** 2 * data['sd'+color] ** 2
                                   for data in data_a])) / len(data_a)
            h = daf.hmm(data_a, color, (mbg, params['sdThreshold' + {'r': 'Red', 'g': 'Green'}[color]] * sdbg))
            for d_a, d_b in zip(data_a, data_b):
                d_b[color] = h(d_a[color]) * FA.get_mask(d_a.t, d_a.frameWindow)
    elif params.get('binaryCallMethod', '').lower() == 'markov_individual':
        print("Calculating binary data using hidden Markov models")
        for channel in params['ChannelsToAnalyze']:
            color = 'rgb'[channel]
            for d_a, d_b in zip(data_a, data_b):
                h = daf.hmm(d_a, color, (np.exp(d_a['m'+color]),
                                         params['sdThreshold' + {'r': 'Red', 'g': 'Green'}[color]]
                                         * np.exp(d_a['m'+color]) * d_a['sd'+color]))
                d_b[color] = h(d_a[color]) * FA.get_mask(d_a.t, d_a.frameWindow)
    else:
        print("Calculating binary data using thresholds")
        names = []
        thresh_cell = []
        for d_a, d_b in zip(data_a, data_b):
            if 0 in params['ChannelsToAnalyze']:
                # threshold with sdTreshold above SD-background of trace
                sdbg = ((np.exp(2*d_a.mr+d_a.sdr**2))*(np.exp(d_a.sdr**2)-1))**0.5
                thresh_cell.append(sdbg * params['sdThresholdRed'])
                names.append(d_a.name)
                d_b.r = d_a.r > (sdbg * params['sdThresholdRed'])  # red binary
                # set all data outside define frameWindow to 0
                d_b.r[0:d_b.frameWindow[0]], d_b.r[d_b.frameWindow[1]: len(d_b.r)+1] = 0, 0
            if 1 not in params['ChannelsToAnalyze']:
                d_b.g = d_b.r
            if 1 in params['ChannelsToAnalyze']:
                # threshold with sdTreshold above SD-background of trace
                sdbg = ((np.exp(2 * d_a.mg + d_a.sdg ** 2)) * (np.exp(d_a.sdg ** 2) - 1)) ** 0.5
                thresh_cell.append(sdbg * params['sdThresholdGreen'])
                d_b.g = d_a.g > (sdbg * params['sdThresholdGreen'])  # green binary
                # set all data outside define frameWindow to 0
                d_b.g[0:d_b.frameWindow[0]], d_b.g[d_b.frameWindow[1]: len(d_b.g)+1] = 0, 0
            if 0 not in params['ChannelsToAnalyze']:
                d_b.r = d_b.g
        with open(params['file'] + '_thresholds_per_cell_with_cell_label.txt', 'w') as f:
            f.write('\n'.join([f'{name},{thres}' for name, thres in zip(names, thresh_cell)]))

        for channel in params['ChannelsToAnalyze']:
            color = 'rg'[channel]

            for cell in range(len(data_b)):
                for i in range(len(data_b[cell][color])-2):
                    # remove one-frame gaps:
                    if params['Remove1FrameGaps']:
                        if data_b[cell][color][i] == 1 and data_b[cell][color][i+1] == 0 and \
                                data_b[cell][color][i+2] == 1:
                            data_b[cell][color][i+1] = 1

            for cell in range(len(data_b)):
                for i in range(len(data_b[cell][color])-2):
                    # remove one-frame jumps:
                    if params['Remove1FrameJumps']:
                        if data_b[cell][color][i] == 0 and data_b[cell][color][i+1] == 1 and \
                                data_b[cell][color][i+2] == 0:
                            data_b[cell][color][i+1] = 0
            # writing binary traces
            for cell in range(len(data_b)):
                d_b = data_b[cell]
                name = d_b.name.split(".txt")[0]
                if color == 'r':
                    np.savetxt(name + "_bg_sub_red_digital.txt", d_b.r)
                elif color == 'g':
                    np.savetxt(name + "_bg_sub_green_digital.txt", d_b.g)

    return data_b

# code to select for specific traces, for example to select for traces that have a burst,
# or make a mask based on the distance of two transcripion sites


def filter_traces(data_a, data_b, params):
    # filter traces for tracesToExclude and traces that do not have a first burst
    if params['alignTracesCF'] and params['color2align'] == "red":
        params['retainedTraces'] = [i for i in np.r_[:len(data_b)] if data_b[i].r.sum() > 0 and
                                    i not in params['tracesToExclude']]
    elif params['alignTracesCF'] and params['color2align'] == "green":
        params['retainedTraces'] = [i for i in np.r_[:len(data_b)] if data_b[i].g.sum() > 0 and
                                    i not in params['tracesToExclude']]
    elif not params['alignTracesCF']:
        params['retainedTraces'] = [i for i in np.r_[:len(data_a)] if i not in params['tracesToExclude']]

    # Make mask based on distance
    if params['selectOnDistance']:
        if len(params['ChannelsToAnalyze']) == 2:
            for data in data_a:
                xy_red = data.trk_r[:, :2]
                xy_green = data.trk_g[:, :2]
                data.spotFoundBoth = list(
                    set(np.where(data.trk_r[:, -1] > 0)[0]) & set(np.where(data.trk_g[:, -1] > 0)[0]))

                dist = [np.NaN] * len(xy_red)
                distinv = [0] * len(xy_red)
                dist0 = [0] * len(xy_red)
                for a in range(len(xy_red)):
                    # for a in data.spotFoundBoth:
                    dist[a] = dist0[a] = (
                                ((xy_red[a, 0] - xy_green[a, 0]) ** 2 + (xy_red[a, 1] - xy_green[a, 1]) ** 2) ** 0.5)
                    distinv[a] = 1 / ((((xy_red[a, 0] - xy_green[a, 0]) ** 2 +
                                        (xy_red[a, 1] - xy_green[a, 1]) ** 2) ** 0.5))
                data.xydist = dist
                data.xydistinv = distinv
                data.distmask = (np.r_[data.xydist] < 3) * 1.
                data.distmask[:data.frameWindow[0]] = 0
                data.distmask[data.frameWindow[1]:] = 0

    to_include = []
    for i in params['retainedTraces']:
        d_a = data_a[i]
        d_b = data_b[i]

        if not params['alignTracesCF']:
            start = d_a.frameWindow[0]
            if np.sum(d_b.r) == 0 and np.sum(d_b.g) == 0:
                print(misc.color(f'Rejected {os.path.basename(d_a.rawPath)} because red and green = 0', 208))
                continue
        else:
            color = params['color2align']
            if np.sum(d_b[color[0]]) == 0:
                print(misc.color(f'Rejected {os.path.basename(d_a.rawPath)} because {color} = 0', 208))
                continue
            start = np.where(d_b[color[0]])[0][0]
        if d_a.frameWindow[-1] - start < params['minLengthCF']:
            print(misc.color(f'Rejected {os.path.basename(d_a.rawPath)} because {d_a.frameWindow[-1] - start} <\
                             {params["minLengthCF"]}', 208))
            continue
        to_include.append(i)

    params['retainedTraces'] = to_include
    return data_a, params


def calc_timefraction_on(data_b, params):
    """#####################################################################
    #### calculate time of being in on state, per cell (histogram) and overall
    ######################################################################"""
    for channel in params['ChannelsToAnalyze']:
        col = ('red', 'green')[channel]
        color = col[0]

        frames_on = []
        frames_off = []
        frac_on = []

        for i in params['retainedTraces']:
            if not params['alignTracesCF']:
                start = data_b[i].frameWindow[0]
            elif np.sum(data_b[i][params['color2align'][0]]) != 0:
                start = np.where(data_b[i][params['color2align'][0]])[0][0]
            else:
                start = 0
            digidata = data_b[i][color][start:data_b[i].frameWindow[-1]]

            if len(digidata):
                frames_on.append(np.sum(digidata))
                frames_off.append(len(digidata)-np.sum(digidata))
                frac_on.append(float(np.sum(digidata))/float(len(digidata)))

        total_on = np.sum(frames_on)
        total_off = np.sum(frames_off)

        print(f'Frames on: {total_on} total frames: {total_off+total_on}')
        np.savetxt(f"{params['file']}_framesOn_framesOff_{col}.txt", [total_on, total_off])
        plot_figures.HistogramPlot(frac_on, 20, f'Histogram of fraction of frames in on-state per cell {col}',
                                   'Fraction frames in on-state',
                                   f"{params['file']}_histogram_fraction_on_per_cell_{col}")


def make_plots_traces(data_orig, data_a, data_b, params):
    """ write PDFs of binary data, shows red and green binary data on top """
    print("Plotting histograms for background subtraction")
    for d_a, d_b in zip(data_a, data_b):
        name = d_a.name.split(".txt")[0]
        with PdfPages(name+"_bg_sub_trace.pdf") as pdfTrace:
            if 0 in params['ChannelsToAnalyze']:
                sdbg = ((np.exp(2*d_a.mr+d_a.sdr**2))*(np.exp(d_a.sdr**2)-1))**0.5
                plot_figures.write_hist(pdfTrace, "r", d_a, params['sdThresholdRed']*sdbg)
                plt.close()
                if len(d_a.bgr) == 4 * len(d_a.t):
                    plot_figures.showBackgroundTrace(pdfTrace, d_a, "r", params['sdThresholdRed']*d_a.sdr)
                    plt.close()
            if 1 in params['ChannelsToAnalyze']:
                sdbg = ((np.exp(2 * d_a.mg + d_a.sdg ** 2)) * (np.exp(d_a.sdg ** 2) - 1)) ** 0.5
                plot_figures.write_hist(pdfTrace, "g", d_a, params['sdThresholdGreen']*sdbg)
                plt.close()
                if len(d_a.bgg) == 4 * len(d_a.t):
                    plot_figures.showBackgroundTrace(pdfTrace, d_a, "g", params['sdThresholdGreen']*d_a.sdg)
                    plt.close()
            plot_figures.showBinaryCall(pdfTrace, d_a, d_b)
            plt.close()

    # plot PDFs of individual auto- and cross correlations as heatmaps
    print("Plotting heatmap correlation functions")

    nb_pts1 = min([d.G[1, 0].shape[0] for d in data_a])  # determine minimum number of point in crosscorrelation gr
    nb_pts2 = min([d.G[0, 1].shape[0] for d in data_a])  # determine minimum number of point in crosscorrelation rg

    # order heatmap correlationfunctions
    if params['OrderCFHeatmap'] == 'maxCC':
        if params['maxCCMethod'] == 'max':
            max_val = [np.where(np.hstack((data_a[i].G[1, 0][::-1][-nb_pts1:-1], data_a[i].G[0, 1][:nb_pts2])) ==
                                max(np.hstack((data_a[i].G[1, 0][::-1][-nb_pts1:-1],
                                               data_a[i].G[0, 1][:nb_pts2]))))[0][0] for i in params['retainedTraces']]
            sorted_ids = np.array([x for _, x in sorted(zip(max_val, params['retainedTraces']))])

        # first takes out data that has no positive peak, then sort remaining on peak by gaussian fitting
        elif params['maxCCMethod'] == 'gaussian':
            def gauss_function(x, a, x0, sigma, b):
                return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + b
        
            max_val2 = []
            amp = []
            for d_a in data_a:
                # fit data to gaussion function
                x = np.hstack((-d_a.tau[::-1][-nb_pts1:-1], d_a.tau[:nb_pts2]))  # the number of data
                y = np.hstack((d_a.G[1, 0][::-1][-nb_pts1:-1], d_a.G[0, 1][:nb_pts2]))
                n = nb_pts1 + nb_pts2 - 1
                # calculate mean and standard deviation
                mean = sum(x*y)/n
                sigma = abs((sum(y*(x-mean)**2)/n))**0.5
                try:
                    popt, pcov = curve_fit(gauss_function, x, y, [0.5, mean, sigma, 0], maxfev=100000,
                                           bounds=((0, np.nanmin(x), 0, -np.inf),
                                                   (np.inf, np.nanmax(x), np.inf, np.inf)))
                    max_val2.append(popt[1])
                    amp.append(popt[0])
                except Exception:
                    max_val2.append(mean)
                    amp.append(y.max() - y.min())
            no_peak = np.where(np.r_[amp] < 0)[0]
            peak = np.where(np.r_[amp] >= 0)[0]
            for x in params['retainedTraces']:
                if x in peak:
                    sorted_ids2 = [x for _, x in sorted(zip([max_val2[x] for x in range(len(max_val2)) if x in peak],
                                                            params['retainedTraces']))]
            sorted_ids = np.r_[no_peak, sorted_ids2]
        else:
            raise ValueError(f"Uknown option for maxCCMethod: {params['maxCCMethod']}")

        heatmap_cf = plot_figures.showHeatMapCF(data_a, 3, 3, 3, sortedIds=sorted_ids, Z=None, Normalize=False)
        heatmap_cf.savefig(params['file']+"_Heatmap_correlation_functions.pdf")
        plt.close()

    elif params['OrderCFHeatmap'] in ('ACred', 'ACgreen'):
        i = ['red', 'green'].index(params['OrderCFHeatmap'][2:])
        max_val = []
        for data in data_a:
            if np.sum(data.G) != 0 and len(data.G[i, i] > 1):
                max_val.append(data.G[i, i][1])

        sorted_ids2 = np.flip(np.argsort(max_val))
        sorted_ids = []
        for x in sorted_ids2:
            if x in params['retainedTraces']:
                sorted_ids.append(x)

    elif params.get('OrderCFHeatmap') is None:
        sorted_ids = range(len(data_a))

    else:
        sorted_ids = range(len(data_a))

    heatmap_cf = plot_figures.showHeatMapCF(data_a, 3, 3, 3, sortedIds=sorted_ids, Z=None, Normalize=False)
    heatmap_cf.savefig(params['file'] + "_Heatmap_correlation_functions.pdf")
    plt.close()

    # make figure heatmap
    print("Plotting trace heatmaps")

    data_a_trim = copy2.deepcopy(data_a)
    data_b_trim = copy2.deepcopy(data_b)
    outside_fw = copy2.deepcopy(data_a)
    for d_a, d_b, d_fw in zip(data_a_trim, data_b_trim, outside_fw):
        d_a.g[:d_a.frameWindow[0]], d_a.g[d_a.frameWindow[1]:] = 0, 0
        d_a.r[:d_a.frameWindow[0]], d_a.r[d_a.frameWindow[1]:] = 0, 0
        d_b.g[:d_b.frameWindow[0]], d_b.g[d_b.frameWindow[1]:] = 0, 0
        d_b.r[:d_b.frameWindow[0]], d_b.r[d_b.frameWindow[1]:] = 0, 0
        d_fw.g[:d_fw.frameWindow[0]], d_fw.g[d_fw.frameWindow[1]:] = 1, 0.3
        d_fw.g[d_fw.frameWindow[0]:d_fw.frameWindow[1]] = 0

    if params['trimHeatmaps']:
        data_heatmap_an = data_a_trim
        data_heatmap_dig = data_b_trim
    else:
        data_heatmap_an = data_a
        data_heatmap_dig = data_b

    if params['OrderTraceHeatmap'] == "CF":
        sorted_ids_trace = sorted_ids
    else:
        sorted_ids_trace = params['retainedTraces']

    # write heatmaps of analog and digital data
    if params['trimHeatmaps']:
        heatmap_a = plot_figures.showHeatMap(data_heatmap_an, maxRed=params['heatMapScalingRed'],
                                             maxGreen=params['heatMapScalingGreen'], trimdata=outside_fw,
                                             sortedIds=sorted_ids_trace)
        heatmap_a.savefig(fname=params['file']+"_Heatmap_analog.pdf")
        plt.close()
    else:
        heatmap_a = plot_figures.showHeatMap(data_heatmap_an, maxRed=params['heatMapScalingRed'],
                                             maxGreen=params['heatMapScalingGreen'], sortedIds=sorted_ids_trace)
        heatmap_a.savefig(fname=params['file']+"_Heatmap_analog.pdf")
        plt.close()

    if params['trimHeatmaps']:
        heatmap_b = plot_figures.showHeatMap(data_heatmap_dig, trimdata=outside_fw, sortedIds=sorted_ids_trace)
        heatmap_b.savefig(fname=params['file']+"_Heatmap_digital.pdf")
        plt.close()
    else:
        heatmap_b = plot_figures.showHeatMap(data_heatmap_dig, sortedIds=sorted_ids_trace)
        heatmap_b.savefig(fname=params['file']+"_Heatmap_digital.pdf")
        plt.close()

    if params['CalcIndTimes']:
        print("Plotting ordered trace heatmaps")
        for channel in params['ChannelsToAnalyze']:
            col = ['red', 'green'][channel]
            indtimes = []
            names = []
            for cell in params['retainedTraces']:
                bindata = data_b[cell][col[0]]
                if sum(bindata) != 0:
                    indframe = np.where(bindata > 0)[0][0]
                    indtime = data_b[cell].dt * float(indframe) / 60
                    indtimes.append(indtime)
                    names.append(data_b[cell].name)
            np.savetxt(f"{params['file']}_induction_times_{col}.txt", indtimes)
            np.savetxt(f"{params['file']}_induction_times_{col}_with_cell_label.txt", np.transpose([names, indtimes]),
                       delimiter=",", fmt="%s")

            plot_figures.HistogramPlot(indtimes, 20, f'Histogram of induction times {col}', 'Induction time (min)',
                                       f"{params['file']}_histogram_induction_times_{col}")
            plot_figures.CumHistogramPlot(indtimes, f'Cumulative distribution of induction times {col}',
                                          'Induction time (min)',
                                          f"{params['file']}_cumulative_distribution_induction_times_{col}")

            if params['SortHeatMapIndTimes'] == col:
                indtimessorted_retained_traces = np.flip(np.argsort(indtimes))
                indtimessorted = [params['retainedTraces'][i] for i in indtimessorted_retained_traces]
            else:
                indtimessorted = params['retainedTraces']

        # making heatmaps sorted by induction time
        if params['SortHeatMapIndTimes'] is not None:
            if params['trimHeatmaps']:
                sortedheatmap_a = plot_figures.showHeatMap(data_heatmap_an, maxRed=params['heatMapScalingRed'],
                                                           maxGreen=params['heatMapScalingGreen'],
                                                           sortedIds=indtimessorted, trimdata=outside_fw)
            else:
                sortedheatmap_a = plot_figures.showHeatMap(data_heatmap_an, maxRed=params['heatMapScalingRed'],
                                                           maxGreen=params['heatMapScalingGreen'],
                                                           sortedIds=indtimessorted)
            sortedheatmap_a.savefig(fname=params['file'] + "_Heatmap_analog_sorted_by_induction.pdf")
            plt.close()

        if params['trimHeatmaps']:
            sortedheatmap_b = plot_figures.showHeatMap(data_heatmap_dig, sortedIds=indtimessorted, trimdata=outside_fw)
        else:
            sortedheatmap_b = plot_figures.showHeatMap(data_heatmap_dig, sortedIds=indtimessorted)
            
        sortedheatmap_b.savefig(fname=params['file'] + "_Heatmap_digital_sorted_by_induction.pdf")
        plt.close()

    # Plot area under traces, useful to see if some traces dominate
    print("Plotting area under traces")
    if 0 in params['ChannelsToAnalyze']:
        fig_avg_r = plot_figures.showAreaUnderTraces(data_a, params['retainedTraces'], "r")
        fig_avg_r.savefig(params['file']+"_AreaUnderTraces_red.pdf")
        plt.close()
    if 1 in params['ChannelsToAnalyze']:
        fig_avg_g = plot_figures.showAreaUnderTraces(data_a, params['retainedTraces'], "g")
        fig_avg_g.savefig(params['file']+"_AreaUnderTraces_green.pdf")
        plt.close()

    # Making plot of all non-background corrected intensities
    print("Plotting intensity distribution")
    for channel in params['ChannelsToAnalyze']:
        col = ('red', 'green')[channel]
        color = col[0]
        fig_int = plot_figures.PlotIntensities(None, data_orig, data_a, data_b, params, color, params['file'] +
                                               '_intensities_frames_on.npy')
        fig_int.savefig(params['file'] + '_histogram_intensity_values_' + col + '.pdf')
        plt.close()

    if len(params['ChannelsToAnalyze']) == 2:
        print("Plotting distance distribution")
        fig_dist = plot_figures.PlotDistances(None, data_a, params['retainedTraces'])
        fig_dist.savefig(params['file'] + '_histogram_distances' + '.pdf')
        plt.close()


def calculate_cf(data_a, data_b, params):
    """ compute auto and cross correlation. For full options check Fluctuation_analysis.py script
        mask defines area of trace that correlation functions are calculated for.
        methAvg can be 'arithm' or 'harmo' for arthimetic or harmonic avarege calculation of the correlation functions
        to correct for non-steady state effects during induction, you need to align the traces on the start of the first
        burst. No need for alignment if data was taken at 'steady state'. """
    print('Calculating correlation functions, including {} traces'.format(len(params['retainedTraces'])))

    if params['alignTracesCF']:  # for aligning traces on first green burst, will complain if no first burst in a trace
        t0 = [dB.t[np.where(dB[params['color2align'].lower()[0]])[0][0]]
              for dB in [data_b[i] for i in params['retainedTraces']]]
        fname_ana = f"{params['file']}_correlation_functions_aligned_startBurst_{params['color2align']}_analog.pdf"
        fname_digi = f"{params['file']}_correlation_functions_aligned_startBurst_{params['color2align']}_digital.pdf"

    # TODO: make excludeInduction = True work with the test in test_CF.py
    elif params['excludeInduction']:  # for excluding the induction time in ACF calculation, without burst alignment
        t0 = None
        burst_starts = [dB.t[np.where(dB[params['color2align'].lower()[0]])[0][0]]
                        for dB in [data_b[i] for i in params['retainedTraces']]]
        fname_ana = f"{params['file']}_correlation_functions_excludeInduction_{params['color2align']}_analog.pdf"
        fname_digi = f"{params['file']}_correlation_functions_excludeInduction_{params['color2align']}_digital.pdf"

    else:  # for auto/cross correlation from first until last frameWindow, without alignment
        t0 = None
        fname_ana = f"{params['file']}_correlation_functions_analog.pdf"
        fname_digi = f"{params['file']}_correlation_functions_digital.pdf"

    ss_ana = []
    ss_digi = []

    corr_count = 0
    for i, (d_a, d_b) in enumerate(zip(data_a, data_b)):
        if i not in params['retainedTraces']:
            corr_count += 1
            continue

        if 'r_orig' in d_a or 'g_orig' in d_a:
            v_orig_ana = np.vstack([d_a[c + '_orig'] for c in 'rgb' if c + '_orig' in d_a])
            v_orig_digi = np.vstack([d_b[c + '_orig'] for c in 'rgb' if c + '_orig' in d_b])
        else:
            v_orig_ana = None
            v_orig_digi = None
        if params.get('selectOnDistance', False):
            ss_ana.append(FA.mcSig(d_a.t, np.vstack((d_a.r, d_a.g)), mask=d_a.distmask, v_orig=v_orig_ana))
            ss_digi.append(FA.mcSig(d_b.t, np.vstack((d_b.r, d_b.g)), mask=d_b.distmask, v_orig=v_orig_digi))
        elif params.get('excludeInduction', False):
            frame0 = int(burst_starts[i - corr_count] / d_a.dt)
            ss_ana.append(FA.mcSig(d_a.t, np.vstack((d_a.r, d_a.g)), frameWindow=[frame0, d_a.frameWindow[1]],
                                   v_orig=v_orig_ana))
            ss_digi.append(FA.mcSig(d_b.t, np.vstack((d_b.r, d_b.g)), frameWindow=[frame0, d_b.frameWindow[1]],
                                    v_orig=v_orig_digi))
        else:
            ss_ana.append(FA.mcSig(d_a.t, np.vstack((d_a.r, d_a.g)), frameWindow=d_a.frameWindow, v_orig=v_orig_ana))
            ss_digi.append(FA.mcSig(d_b.t, np.vstack((d_b.r, d_b.g)), frameWindow=d_b.frameWindow, v_orig=v_orig_digi))

    ss_ana = FA.mcSigSet(ss_ana)
    ss_digi = FA.mcSigSet(ss_digi)

    ss_ana.alignSignals(t0)
    ss_digi.alignSignals(t0)
    if params.get('ccfFunction') is not None and params['ccfFunction'].lower() in ('linefit', 'sc'):
        ss_ana.compAvgCF_SC(mT=params.get('mTauOrder'), fitWindow=params.get('fitWindow'),
                            pearson=params.get('ccfPearson'), subtract_mean=params.get('ccfSubtractMean'))
        ss_digi.compAvgCF_SC(mT=params.get('mTauOrder'), fitWindow=params.get('fitWindow'),
                             pearson=params.get('ccfPearson'), subtract_mean=params.get('ccfSubtractMean'))
    else:
        ss_ana.compAvgCF(mT=params.get('mTauOrder'))
        ss_digi.compAvgCF(mT=params.get('mTauOrder'))

    ss_ana.bootstrap(nBs=params.get('nBs'))
    ss_digi.bootstrap(nBs=params.get('nBs'))

    with PdfPages(fname_ana) as pdf:
        plot_figures.showCorrFun(pdf, ss_ana)

    with PdfPages(fname_digi) as pdf:
        plot_figures.showCorrFun(pdf, ss_digi)

    """ ---- Explanation of results from ss ---
        ss.G = correlation functions: ss.G[0,0] = green, ss.G[1,1] = red,
                                      ss.G[0,1] = red to green, ss.G[1,0] = green to red
        ss.P = corrected correlation functions. Ordering the same
        ss.N = correlation function of average trace. Ordering the same
        *r, *g: correlation functions normalized wrt. red or green signal
        d*: error on correlation function
    """

    # display average trace and individual background subtracted traces
    print("Plotting average trace")
    with PdfPages(params['file']+"_average_trace.pdf") as pdf:
        plot_figures.showAvTrace(pdf, ss_ana, names=[data_a[i].name for i in params['retainedTraces']])
        plt.close()

    # write average trace to file
    if 0 in params['ChannelsToAnalyze']:
        np.savetxt(params['file']+'_average_trace_red.txt', np.vstack((ss_ana.t, ss_ana.v[0], ss_ana.dv[0])).T)
    if 1 in params['ChannelsToAnalyze']:
        np.savetxt(params['file'] + '_average_trace_green.txt', np.vstack((ss_ana.t, ss_ana.v[1], ss_ana.dv[1])).T)
    return ss_ana, ss_digi


def make_plots_cf(ss_ana, ss_digi, params):
    """ fit and display autocorrelation functions """
    print("Plotting individual autocorrelations")
    with PdfPages(params['file']+"_individual_correlation_functions_analog.pdf") as pdfTrace:
        plot_figures.showCorrelFunAll(pdfTrace, ss_ana.sigsAlign, params['ChannelsToAnalyze'], params)
    with PdfPages(params['file']+"_individual_correlation_functions_digital.pdf") as pdfTrace:
        plot_figures.showCorrelFunAll(pdfTrace, ss_digi.sigsAlign, params['ChannelsToAnalyze'], params)

    print("Fitting and plotting correlation functions")
    # calculate fit for (non-) corrected red autocorrelation.
    # The autocorrelation can be shifted up or down before the fit by changing the y the shiftACFRed parameter.
    for channel in params['ChannelsToAnalyze']:
        for correctCF in (False, True):
            color_name = ['Red', 'Green'][channel]
            color = color_name.lower()
            col = color[0]
            fit_frames_color = 'fitFrames' + color_name
            shift_acf_color = 'shiftACF' + color_name

            n1 = params[fit_frames_color][0]  # start frame of data for fitting
            n2 = params[fit_frames_color][1]  # end frame of data for fitting

            if correctCF:
                G_ana = (ss_ana.P.copy(), ss_ana.Pr.copy(), ss_ana.Pg.copy())
                dG_ana = (ss_ana.dP.copy(), ss_ana.dPr.copy(), ss_ana.dPg.copy())

                G_digi = (ss_digi.P.copy(), ss_digi.Pr.copy(), ss_digi.Pg.copy())
                dG_digi = (ss_digi.dP.copy(), ss_digi.dPr.copy(), ss_digi.dPg.copy())

            else:
                G_ana = (ss_ana.G.copy(), ss_ana.Gr.copy(), ss_ana.Gg.copy())
                dG_ana = (ss_ana.dG.copy(), ss_ana.dGr.copy(), ss_ana.dGg.copy())

                G_digi = (ss_digi.G.copy(), ss_digi.Gr.copy(), ss_digi.Gg.copy())
                dG_digi = (ss_digi.dG.copy(), ss_digi.dGr.copy(), ss_digi.dGg.copy())
                
            G_ana = [g[channel, channel] for g in G_ana]
            dG_ana = [g[channel, channel] for g in dG_ana]
            G_ana[0] += params[shift_acf_color]

            G_digi = [g[channel, channel] for g in G_digi]
            dG_digi = [g[channel, channel] for g in dG_digi]
            G_digi[0] += params[shift_acf_color]

            with PdfPages(f"{params['file']}_autocorrelation_fit_{color}_"
                          f"{(1-correctCF) * 'un'}corrected_analog.pdf") as pdf, \
                 PdfPages(f"{params['file']}_autocorrelation_fit_{color}_"
                          f"{(1-correctCF)*'un'}corrected_zoom_analog.pdf") as zoompdf:
                for i, (g, dg, t) in enumerate(zip(G_ana, dG_ana, ('G', 'Normalized r', 'Normalized g'))):
                    fitp, dfitp = misc.fit_line(ss_ana.tau[n1:n2], g[n1:n2], dg[n1:n2])
                    plot_figures.showAutoCorr(pdf, col, t, ss_ana.tau, g, dg, fitp, dfitp)
                    plot_figures.showAutoCorr(zoompdf, col, t, ss_ana.tau, g, dg, fitp, dfitp, params['ACFxmax'])
                    if i == 0:
                        np.savetxt(f"{params['file']}_autocorrelation_{color}_{(1-correctCF)*'un'}corrected_analog.txt",
                                   np.vstack((ss_ana.tau, g, dg)).T, delimiter="\t", fmt="%1.5f")

            with PdfPages(f"{params['file']}_autocorrelation_fit_{color}_"
                          f"{(1-correctCF)*'un'}corrected_digital.pdf") as pdf, \
                 PdfPages(f"{params['file']}_autocorrelation_fit_{color}_"
                          f"{(1-correctCF)*'un'}corrected_zoom_digital.pdf") as zoompdf:
                for i, (g, dg, t) in enumerate(zip(G_digi, dG_digi, ('G', 'Normalized r', 'Normalized g'))):
                    fitp, dfitp = misc.fit_line(ss_digi.tau[n1:n2], g[n1:n2], dg[n1:n2])
                    plot_figures.showAutoCorr(pdf, col, t, ss_digi.tau, g, dg, fitp, dfitp)
                    plot_figures.showAutoCorr(zoompdf, col, t, ss_digi.tau, g, dg, fitp, dfitp, params['ACFxmax'])
                    if i == 0:
                        np.savetxt(f"{params['file']}_autocorrelation_{color}_"
                                   f"{(1-correctCF)*'un'}corrected_digital.txt",
                                   np.vstack((ss_digi.tau, g, dg)).T, delimiter="\t", fmt="%1.5f")

    acf_ana_red = ss_ana.P[1, 1][params['rangeACFResidual'][0]:params['rangeACFResidual'][1]]
    acf_digi_red = ss_digi.P[1, 1][params['rangeACFResidual'][0]:params['rangeACFResidual'][1]]
    acf_ana_green = ss_ana.P[0, 0][params['rangeACFResidual'][0]:params['rangeACFResidual'][1]]
    acf_digi_green = ss_digi.P[0, 0][params['rangeACFResidual'][0]:params['rangeACFResidual'][1]]

    sum_res_red = sum([(ana - digi) ** 2 for ana, digi in zip(acf_ana_red, acf_digi_red)])
    sum_res_green = sum([(ana - digi) ** 2 for ana, digi in zip(acf_ana_green, acf_digi_green)])
    np.savetxt(params['file']+'_sum_sq_residuals_ACF_corrected.txt', [sum_res_red, sum_res_green])

    if len(params['ChannelsToAnalyze']) == 2:
        # calculate fit for (non-)corrected cross-correlation
        for correctCF in (False, True):
            if correctCF:
                Gt = [ss_ana.P.copy(), ss_ana.Pr.copy(), ss_ana.Pg.copy()]
                dGt = [ss_ana.dP.copy(), ss_ana.dPr.copy(), ss_ana.dPg.copy()]
            else:
                Gt = [ss_ana.G.copy(), ss_ana.Gr.copy(), ss_ana.Gg.copy()]
                dGt = [ss_ana.dG.copy(), ss_ana.dGr.copy(), ss_ana.dGg.copy()]
            Gt[0] += params['shiftCC']
            G = [g[0, 1] for g in Gt]
            G2 = [g[1, 0, ::-1] for g in Gt]
            dG = [g[0, 1] for g in dGt]
            dG2 = [g[1, 0, ::-1] for g in dGt]

            tau = ss_ana.tau
            tau2 = -tau[::-1]

            ml, nl = params.get('fitFramesCCLeft', [1, len(tau)])
            mr, nr = params.get('fitFramesCCRight', [0, len(tau)])
            nl = min(nl, len(tau))
            nr = min(nr, len(tau))
            ml = max(ml, 1)
            mr = max(mr, 0)
            # n = nl + nr - ml - mr + 2  # total length for fit

            def gauss_function(x, a, x0, sigma, b):
                """ gaussian function """
                return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + b

            with PdfPages('{}_crosscorrelation_fit_{}corrected.pdf'.format(params['file'], (1-correctCF)*'un')) as pdf:
                for i, (g, g2, dg, dg2, t) in enumerate(zip(G, G2, dG, dG2, ('G', 'Normalized r', 'Normalized g'))):
                    # get data for fit in array
                    x = np.hstack((tau2[-nl-1:-ml], tau[mr:nr+1]))  # the number of data
                    y = np.hstack((g2[-nl-1:-ml], g[mr:nr+1]))
                    yerr = np.hstack((dg2[-nl-1:-ml], dg[mr:nr+1]))
    
                    # calculate initial parameters for fit
                    y0 = np.nanmin(y)
                    mean = np.sum(x*(y-y0))/np.sum(y-y0)
                    sigma = np.sqrt(np.sum((y-y0)*(x-mean)**2)/np.sum(y-y0))
                    A = np.trapz(y-y0, x)/sigma/np.sqrt(2*np.pi)

                    if params.get('CCxlim') is None:
                        params['CCxlim'] = [-ss_ana.tau.max() / 1.5, ss_ana.tau.max() / 1.5]
                    try:  # fit data to gaussian function
                        popt, pcov = curve_fit(gauss_function, x, y, sigma=yerr, p0=[A, mean, sigma, y0])
                        perr = np.sqrt(np.diag(pcov))
                        plot_figures.showCrossCorr(pdf, t, ss_ana.tau, g, g2, dg, dg2, None, perr[0], perr[1], *popt,
                                                   xlim=params['CCxlim'])
                        with open(f"{params['file']}_crosscorrelation_{(1-correctCF)*'un'}corrected_fit_params.txt",
                                  'a' if i else 'w') as f:
                            np.savetxt(f, np.expand_dims(np.hstack((popt, perr)), 0), delimiter='\t', fmt='%1.5f')
                    except Exception:
                        plot_figures.showCrossCorr(pdf, t, ss_ana.tau, g, g2, dg, dg2, xlim=params['CCxlim'])
                        print(misc.color('Error fitting cross correlation', 208))
                        with open(f"{params['file']}_crosscorrelation_{(1-correctCF)*'un'}corrected_fit_params.txt",
                                  'a' if i else 'w') as f:
                            np.savetxt(f, np.full((1, 8), np.nan), delimiter='\t', fmt='%1.5f')
            np.savetxt(f"{params['file']}_crosscorrelation_{(1-correctCF)*'un'}corrected.txt",
                       np.hstack([np.c_[np.r_[tau2[:-1], tau]]] + [np.c_[np.r_[g2[:-1], g], np.r_[dg2[:-1], dg]]
                                                                   for g, g2, dg, dg2 in zip(G, G2, dG, dG2)]),
                       delimiter='\t', fmt='%1.5f')


def durations(labeled_trace):
    """ calculate burst duration and time between bursts from thresholded data """
    lbl = np.unique(labeled_trace)
    durations_list = []
    min_durations_list = []  # start and/or end not in trace, so duration is 'at least'
    for l in lbl[lbl > 0]:
        if labeled_trace[0] != l and labeled_trace[-1] != l:  # start and end both in trace
            durations_list.append(sum(labeled_trace == l))
        else:
            min_durations_list.append(sum(labeled_trace == l))
    return np.array(durations_list), np.array(min_durations_list)


def survival_ratio(bounded_times, unbounded_times):
    """ calculate data for survival ratio graph
        arguments:
            bounded_times: times / durations which are known
            unbounded_times: times / durations for which only the lower bound is known
        returns:
            t: time
            s / r: survival ratio
    """
    t = np.sort(np.hstack((0, bounded_times, unbounded_times)))
    n = np.sort(unbounded_times)
    s = (len(t) - np.arange(len(t)) - 1)
    r = np.max((s, (len(t) - le(*np.meshgrid(n, t)).sum(1) - 1)), 0)
    return t, s / r


def make_burst_histograms(data_b, params, save=True):
    # _BurstDuration+Freq_after_threshold_
    print('Plotting burst histograms')
    for channel in params['ChannelsToAnalyze']:
        color_cap = ('Red', 'Green')[channel]
        color = color_cap.lower()
        sd_threshold = params['sdThreshold' + color_cap]

        burst_durations = []
        min_burst_durations = []
        times_between_bursts = []
        min_times_between_bursts = []

        for idx in params['retainedTraces']:
            # load digital data trace
            data_cell = data_b[idx][color[0]][slice(*data_b[idx].frameWindow[:2])]
            burst_duration = durations(label(data_cell)[0])
            burst_durations.extend(data_b[idx].dt * burst_duration[0])
            min_burst_durations.extend(data_b[idx].dt * burst_duration[1])
            time_between_bursts = durations(label(1 - data_cell)[0])
            times_between_bursts.extend(data_b[idx].dt * time_between_bursts[0])
            min_times_between_bursts.extend(data_b[idx].dt * time_between_bursts[1])

        if save:
            np.savetxt(f"{params['file']}_minTimeBetweenBursts_checked_threshold_{color}_{sd_threshold}.txt",
                       min_times_between_bursts, fmt='%u')
            np.savetxt(f"{params['file']}_TimeBetweenBursts_checked_threshold_{color}_{sd_threshold}.txt",
                       times_between_bursts, fmt='%u')
            np.savetxt(f"{params['file']}_BurstDuration_checked_threshold_{color}_{sd_threshold}.txt",
                       burst_durations, fmt='%u')
            np.savetxt(f"{params['file']}_minBurstDuration_checked_threshold_{color}_{sd_threshold}.txt",
                       min_burst_durations, fmt='%u')

        # make histogram of burst duration and time between bursts
        bin_size = params['binSizeHistogram']  # time interval
        max_xaxis = params['maxXaxisHistogram']  # range histogram
        if len(burst_durations) != 0:
            bootstr_duration = plot_figures.CalcBootstrap(burst_durations, 1000)
        else:
            bootstr_duration = np.nan, np.nan
        burst_duration_mean = bootstr_duration[0]
        burst_duration_mean_err = bootstr_duration[1]
        if times_between_bursts:
            bootstr_time_betw_burst = plot_figures.CalcBootstrap(times_between_bursts, 1000)
        else:
            bootstr_time_betw_burst = np.nan, np.nan

        bins = np.arange(bin_size / 2, max_xaxis, bin_size)
        time_between_burst_mean = bootstr_time_betw_burst[0]
        time_between_burst_mean_err = bootstr_time_betw_burst[1]
        hist_burst_duration = np.histogram(burst_durations, bins)[0]
        hist_burst_duration = hist_burst_duration / np.sum(hist_burst_duration)
        if not burst_durations:
            hist_burst_duration = np.zeros_like(hist_burst_duration)
        hist_burst_freq = np.histogram(times_between_bursts, bins)[0]
        hist_burst_freq = hist_burst_freq / np.sum(hist_burst_freq)
        if not times_between_bursts:
            hist_burst_freq = np.zeros_like(hist_burst_freq)

        def fit_exp(x, y, x_max=None):
            if not len(x):
                return (np.nan, np.nan), (np.nan, np.nan), ((np.nan, np.nan), (np.nan, np.nan))
            x_max = x_max or np.nanmax(x)
            x, y = x[y > 0], y[y > 0]
            p, cov = np.polyfit(x, np.log(y), 1, cov=True)
            dp = np.sqrt(np.diag(cov))
            x = np.linspace(0, x_max)
            return (np.exp(p[1]), -1 / p[0]), (np.exp(p[1]) * dp[1], dp[0] / (p[0] ** 2)), (x, np.exp(np.polyval(p, x)))

        print('Fitting A*exp(-t/t0)')
        p_duration, p_duration_err, fit_duration = fit_exp(bins[:len(hist_burst_duration)][hist_burst_duration > 0][1:],
                                                           hist_burst_duration[hist_burst_duration > 0][1:], max_xaxis)
        print('Fit parameters burst duration: A = {0:.3g} ± {2:.3g}, t = {1:.3g} ± {3:.3g}'
              .format(*p_duration, *p_duration_err))
        p_freq, p_freq_err, fit_freq = fit_exp(bins[:len(hist_burst_freq)][hist_burst_freq > 0][1:],
                                               hist_burst_freq[hist_burst_freq > 0][1:], max_xaxis)
        print('Fit parameters time between bursts: A = {0:.3g} + {2:.3g}, t = {1:.3g} ± {3:.3g}'
              .format(*p_freq, *p_freq_err))

        t_on, s_on = survival_ratio(burst_durations, min_burst_durations)
        p_on, p_on_err, fit_on = fit_exp(t_on[1:], (s_on[1:] + s_on[:-1]) / 2)
        print('Fit parameters burst survival: A = {0:.3g} ± {2:.3g}, t = {1:.3g} ± {3:.3g}'.format(*p_on, *p_on_err))
        t_off, s_off = survival_ratio(times_between_bursts, min_times_between_bursts)
        p_off, p_off_err, fit_off = fit_exp(t_off[1:], (s_off[1:] + s_off[:-1]) / 2)
        print('Fit parameters off time survival: A = {0:.3g} ± {2:.3g}, t = {1:.3g} ± {3:.3g}'
              .format(*p_off, *p_off_err))

        with PdfPages(f'{params["file"]}_BurstDuration+Freq_after_threshold_{color}.pdf') as pdf:
            fig = plt.figure(figsize=A4)
            gs = GridSpec(2, 1, figure=fig)

            ax_on = fig.add_subplot(gs[0, 0])
            plt.bar(bins[:-1], hist_burst_duration, color='blue', width=bin_size-2)
            plt.plot(*fit_duration, color='black')
            plt.ylim(0, 1.1)
            plt.title('Burst duration')
            plt.xlabel('Burst duration (s)')
            plt.ylabel('Frequency')
            plt.text(0.9, 0.9,
                     f'burst duration, mean: {burst_duration_mean:.2f} +/- {burst_duration_mean_err:.2f} s\n'
                     f'Exp fit burst duration, tau: {p_duration[1]:.2f} +/- {p_duration_err[1]:.2f} s',
                     horizontalalignment='right', verticalalignment='top', transform=ax_on.transAxes)

            ax_off = fig.add_subplot(gs[1, 0])
            plt.bar(bins[:-1], hist_burst_freq, color='gray', width=bin_size-2)
            plt.plot(*fit_freq, color='black')
            plt.ylim(0, 1.1)
            plt.title('Burst frequency')
            plt.xlabel('Time between bursts (s)')
            plt.ylabel('Frequency')
            plt.text(0.9, 0.9,
                     f'time between bursts, mean: {time_between_burst_mean:.2f} +/- {time_between_burst_mean_err:.2f} s'
                     f'\nExp fit time between bursts, tau: {p_freq[1]:.2f} +/- {p_freq_err[1]:.2f} s',
                     horizontalalignment='right', verticalalignment='top', transform=ax_off.transAxes)

            plt.tight_layout()
            pdf.savefig(fig)

            ax_on.set_yscale('log')
            ax_on.set_ylim(hist_burst_duration[hist_burst_duration > 0].min(initial=1) / 2, 1.1)
            ax_off.set_yscale('log')
            ax_off.set_ylim(hist_burst_freq[hist_burst_freq > 0].min(initial=1) / 2, 1.1)
            pdf.savefig(fig)
            plt.close()

            fig = plt.figure(figsize=A4)
            gs = GridSpec(2, 1, figure=fig)
            ax_on = fig.add_subplot(gs[0, 0])
            plt.step(t_on, s_on, where='post', color='blue')
            plt.plot(*fit_on, 'k')
            plt.ylim(0, 1.1)
            plt.title('Burst duration survival ratio')
            plt.xlabel('Burst duration (s)')
            plt.ylabel('Frequency')
            plt.text(0.9, 0.9,
                     f'Exp fit burst duration, tau: {p_on[1]:.2f} +/- {p_on_err[1]:.2f} s',
                     horizontalalignment='right', verticalalignment='top', transform=ax_on.transAxes)

            ax_off = fig.add_subplot(gs[1, 0])
            plt.step(t_off, s_off, where='post', color='gray')
            plt.plot(*fit_off, 'k')
            plt.ylim(0, 1.1)
            plt.title('Off time survival ratio')
            plt.xlabel('Time between bursts (s)')
            plt.ylabel('Frequency')
            plt.text(0.9, 0.9,
                     f'Exp fit time between bursts, tau: {p_off[1]:.2f} +/- {p_off_err[1]:.2f} s',
                     horizontalalignment='right', verticalalignment='top', transform=ax_off.transAxes)

            plt.tight_layout()
            pdf.savefig(fig)

            ax_on.set_yscale('log')
            ax_on.set_ylim(s_on[s_on > 0].min(initial=1) / 2, 2)
            ax_off.set_yscale('log')
            ax_off.set_ylim(s_off[s_off > 0].min(initial=1) / 2, 2)
            pdf.savefig(fig)
            plt.close()


def make_burst_histograms_singlecells(data_orig, data_a, data_b, params):
    """ Calculating single-cell parameters """
    print("Plotting burst histograms per single cell")
    for channel in params['ChannelsToAnalyze']:
        col = ('red', 'green')[channel]
        color = col[0]

        burst_duration_sec = {}
        time_between_bursts_sec = {}
        for cc in params['retainedTraces']:
            # load digital data trace
            datacell = data_b[cc][color][slice(*data_b[cc].frameWindow[:2])]
            burst_duration_sec[cc] = durations(label(datacell)[0])[0] * data_b[cc].dt
            time_between_bursts_sec[cc] = durations(label(1 - datacell)[0])[0] * data_b[cc].dt

        # burst durations, individual and per cell
        mean_durs_sec = [np.mean(durs) for durs in burst_duration_sec.values() if len(durs)]
        all_durs_sec = np.hstack(list(burst_duration_sec.values()))

        np.savetxt(f"{params['file']}_mean_burst_duration_per_cell_{col}.txt", mean_durs_sec)
        plot_figures.HistogramPlot(mean_durs_sec, 20, f'Histogram of average burst duration per cell {col}',
                                   'Average burst duration (s)',
                                   f"{params['file']}_histogram_average_burst_duration_per_cell_{col}")
        
        np.savetxt(f"{params['file']}_burst_duration_{col}.txt", all_durs_sec)
        plot_figures.HistogramPlot(all_durs_sec, 20, f'Burst durations {col}',
                                   'Average burst duration (s)', f"{params['file']}_histogram_burst_duration_{col}")
        
        # time between bursts
        mean_time_betw_sec = [np.mean(times) for times in time_between_bursts_sec.values() if len(times)]
        all_time_betw_sec = np.hstack(list(time_between_bursts_sec.values()))

        np.savetxt(f"{params['file']}_mean_time_between_bursts_per_cell_{col}.txt", mean_time_betw_sec)
        plot_figures.HistogramPlot(mean_time_betw_sec, 20, f'Histogram of average time between bursts per cell {col}',
                                   'Average time between bursts (s)',
                                   f"{params['file']}_histogram_average_time_between_bursts_per_cell_{col}")

        np.savetxt(f"{params['file']}_time_between_bursts_{col}.txt", all_time_betw_sec)
        plot_figures.HistogramPlot(all_time_betw_sec, 20, f'Time between bursts {col}',
                                   'Average time between bursts (s)',
                                   f"{params['file']}_histogram_time_between_bursts_{col}")
        
        # number of bursts per cell
        nbr_bursts = [len(b) for b in burst_duration_sec.values()]
        np.savetxt(f"{params['file']}_number_of_bursts_per_cell_{col}.txt", nbr_bursts)
        plot_figures.HistogramPlot(nbr_bursts, 20, f'Histogram of number of bursts per cell {col}', 'Number of bursts',
                                   f"{params['file']}_histogram_number_of_bursts_per_cell_{col}")
            
        # intensity in on state per cell and individual frames
        av_int_cell = []
        all_ints = []
        for cc in params['retainedTraces']:
            datacell_ints = data_orig[cc][col[0]][data_orig[cc].frameWindow[0]:data_orig[cc].frameWindow[1]]
            datacell_digi = data_b[cc][col[0]][data_b[cc].frameWindow[0]:data_b[cc].frameWindow[1]]
            on_ints = datacell_ints * datacell_digi
            if sum(datacell_digi) != 0:
                av_int = np.sum(on_ints)/np.sum(datacell_digi)
                av_int_cell.append(av_int)
            for i in range(len(on_ints)):
                if on_ints[i] != 0:
                    all_ints.append(on_ints[i])
        
        np.savetxt(f"{params['file']}_intensity_frames_on_per_cell_{col}.txt", av_int_cell)
        np.savetxt(f"{params['file']}_intensity_frames_on_{col}.txt", all_ints)
        
        plot_figures.HistogramPlot(av_int_cell, 20,
                                   f'Histogram of average intensity of frames in on-state per cell {col}',
                                   'Intensity (AU)', f"{params['file']}_intensity_frames_on_per_cell_{col}")
        plot_figures.HistogramPlot(all_ints, 20, f'Histogram of intensity of all frames in on-state {col}',
                                   'Intensity (AU)', f"{params['file']}_intensity_frames_on_{col}")

        # Second value of ACF plot (measure of ACF amplitude)
        acf_ampl = []
        for cell in params['retainedTraces']:
            i = ['red', 'green'].index(col)
            if np.sum(data_a[cell].G != 0) and len(data_a[cell].G[i, i] > 1):
                acf_ampl.append(data_a[cell].G[i, i][1])

        np.savetxt(f"{params['file']}_ACF_amplitudes_per_cell_{col}.txt", acf_ampl)
        plot_figures.HistogramPlot(acf_ampl, 20, f'Histogram of ACF amplitude per cell {col}', 'ACF amplitude',
                                   f"{params['file']}_ACF_amplitude_per_cell_{col}")

        # Correlation of ACF with induction time
        plot_figures.corrACFAmplToIndPlot(data_a, data_b, col, params)


def smooth_or_quantize(data, params):
    # Better not smooth and quantize the same trace!
    if params.get('quantizeTrace') is not None:
        for channel, q in params['quantizeTrace'].items():
            daf.getMolNumber(data, q, channel)
    if params.get('smoothTrace') is not None:
        for channel, (window_length, polyorder) in params['smoothTrace'].items():
            daf.smoothData(data, channel, window_length, polyorder)


def set_global_frame_window(data, params):
    # change frameWindow if necessary
    if not params.get('globalFrameWindow') is None:
        global_frame_window = params['globalFrameWindow']
        if global_frame_window[0] == misc.none():
            global_frame_window[0] = -np.inf
        if global_frame_window[1] == misc.none():
            global_frame_window[1] = np.inf
        for d in data:
            d.frameWindow = np.clip(d.frameWindow, *global_frame_window).astype(int).tolist()


def pipeline_correlation_functions(params):
    """ pipeline for correlation functions script
        params is either a dictionary containing the parameters for the pipeline or a string pointing to the yml file
        with the parameters """

    if not isinstance(params, dict):
        parameter_file = params
        if parameter_file[-3:] == '.py':
            print('Converting py parameter file into yml format')
            misc.convertParamFile2YML(parameter_file)
            parameter_file = parameter_file[:-3]+'.yml'
        if os.path.isdir(parameter_file):
            parameter_file = os.path.join(parameter_file, 'parameters.yml')
        if not parameter_file[-4:] == '.yml':
            parameter_file += '.yml'
        params = misc.getParams(parameter_file, __file__.replace('.py', '_parameters_template.yml'),
                                ('PyFile', 'outputfolder'))
    else:
        parameter_file = ''

    get_paths(params, parameter_file)
    
    if params['processTraces']:
        # Load original raw data
        data_orig = daf.ExpData(params['PyFile'])
        set_global_frame_window(data_orig, params)
        smooth_or_quantize(data_orig, params)
        data_a = bg_sub_traces(data_orig, params)
        data_b = binary_call_traces(data_a, params)

        if params.get('filterTraces', True):
            data_a, params = filter_traces(data_a, data_b, params)
        else:
            params['retainedTraces'] = list(range(len(data_a)))

        calc_timefraction_on(data_b, params)

        if params['makePlots']:
            make_plots_traces(data_orig, data_a, data_b, params)

        if params['calcCF']:
            ss_ana, ss_digi = calculate_cf(data_a, data_b, params)  # INEKE
            params['ssAna'] = ss_ana
            params['ssDigi'] = ss_digi
            for i, s in zip(params['retainedTraces'], ss_ana.sigsAlign):
                s.name = data_orig[i].name
            for i, s in zip(params['retainedTraces'], ss_digi.sigsAlign):
                s.name = data_orig[i].name

            if params['makeCFplot']:
                make_plots_cf(ss_ana, ss_digi, params)

        if params['makeHistogram']:
            make_burst_histograms(data_b, params)

        if params['SingleCellParams']:
            make_burst_histograms_singlecells(data_orig, data_a, data_b, params)

    return params


def main():
    ipy_debug()
    if len(sys.argv) < 2:
        parameter_files = [os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                        'pipeline_livecell_correlationfunctions_parameters.yml'))]
    else:
        parameter_files = sys.argv[1:]

    if len(parameter_files) == 1:
        parameter_file = parameter_files[0]
        if not os.path.exists(parameter_file):
            raise FileNotFoundError('Could not find the parameter file.')
        print(misc.color('Working on: {}'.format(parameter_file), 'b:b'))
        params = pipeline_correlation_functions(parameter_file)
    else:
        for parameter_file in parameter_files:
            print(misc.color('Working on: {}'.format(parameter_file), 'b:b'))
            print('')
            try:
                pipeline_correlation_functions(parameter_file)
            except Exception:
                print(misc.color('Exception while working on: {}'.format(parameter_file), 'r:b'))

    # this only runs when this script is run from command-line with ./pipeline..., not when run from ipython
    # if we do not kill the java vm, (i)python will not exit after completion
    # be sure to call imr.kill_vm() at the end of your script/session, note that you cannot use imread afterwards
    if os.path.basename(__file__) in [os.path.basename(i) for i in psutil.Process(os.getpid()).cmdline()]:
        imr.kill_vm()  # stop java used for imread, needed to let python exit
        print('Stopped the java vm used for imread.')

    print('------------------------------------------------')
    print(misc.color('Pipeline finished.', 'g:b'))


if __name__ == '__main__':
    main()
