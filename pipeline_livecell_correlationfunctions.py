#!/usr/local/bin/ipython2

import sys
import os
import re
import psutil
import misc
import numpy as np
import copy as copy2
import dataAnalysis_functions as daf
import plot_figures
import yaml
import shutil
from scipy import optimize
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime
from wimread import imread as imr

### if you want to execute this pipeline from a different file than pipeline_livecell_correlationfunctions_parameters, first run: sys.argv = ['pipeline_livecell_correlationfunctions.py','yourfilename'], which should be in the same folder, and then run: execfile("pipeline_livecell_correlationfunctions.py")

A4 = (11.69, 8.27)
from matplotlib.backends.backend_pdf import PdfPages
from scipy.cluster.hierarchy import dendrogram, linkage

if not '__file__' in locals(): #when executed using execfile
    import inspect
    __file__ = inspect.getframeinfo(inspect.currentframe()).filename


# This script is meant to calculate autocorrelations on a dataset. The dataset is specified in a list.py file (and can be a combination of multiple experiments).

#### specify input list.py and outputfolder

def get_paths(params, parameter_file=None):
    if not parameter_file == '':
        parameter_file = os.path.abspath(parameter_file)
        if not os.path.isabs(params['outputfolder']):
            params['outputfolder'] = os.path.join(os.path.split(parameter_file)[0], params['outputfolder'])
        if not os.path.isabs(params['PyFile']):
            params['PyFile'] = os.path.join(os.path.split(parameter_file)[0], params['PyFile'])

    if params['outputfolder'][-1] != "/": params['outputfolder'] += "/"

    params['file'] = params['outputfolder'] + "/".join((params['PyFile']).split("/")[-1:]).split(".list")[0]

    dateTimeObj = datetime.now()
    dateTime = dateTimeObj.strftime("%d-%b-%Y_%Hh%Mm%Ss")

    if os.path.exists(parameter_file):
        shutil.copyfile(parameter_file,
            os.path.join(params['file'] + "_pipeline_livecell_correlationfunctions_parameters_runtime_" + dateTime + ".yml"))
    else:
        parameter_file = os.path.join(params['file'] + "_pipeline_livecell_correlationfunctions_parameters_runtime_" + dateTime + ".yml")
        with open(parameter_file, 'w') as f:
            yaml.dump(params, f)
    shutil.copyfile(os.path.abspath(__file__), os.path.join(params['file'] + "_pipeline_livecell_correlationfunctions.py"))

    return params

def parseParams(params):
    """
    add defaults for missing parameters in the parameterfile
    """
    required  = ['PyFile', 'outputfolder']
    optionals = {'processTraces': True, 'makePlots': True, 
                 'makeHistogram': True, 'SingleCellParams': True, 'scaleFactors': None, 'ChannelsToAnalyze': [0,1],
                 'sdThresholdRed': 1, 'sdThresholdGreen': 1, 'bgWindowRed': 1, 'bgWindowGreen': 1,
                 'heatMapScalingRed': None, 'heatMapScalingGreen': None, 'trimHeatmaps': False, 'CalcIndTimes': True,
                 'SortHeatMapIndTimes': None, 'OrderTraceHeatmap': None, 
                 'maxCCMethod': 'gaussian', 'tracesToExclude': [], 'selectOnDistance': False, 'alignTracesCF': False,
                 'color2align': 'green', 'binSizeHistogram': 10, 'maxXaxisHistogram': 100, 'Remove1FrameJumps': True,
                 'Remove1FrameGaps': True, 'mTauOrder': 0}

    for p in required:
        if not p in params:
            raise Exception('Parameter {} not given in parameter file'.format(p))

    for k, v in optionals.items():
        if not k in params:
            print(misc.color('Parameter {} missing in parameter file, adding with value: {}'.format(k, v), 'r'))
            params[k] = v

#################
#################
### MAIN FUNCTIONS

######################################
### Process analog data
######################################

def bg_sub_traces(dataOrig, params):

    # Copy to new variable dataA: Analog data, processed
    dataA=copy2.deepcopy(dataOrig)

    # Scale traces to correct for day-to-day effects if necessary
    if params['scaleFactors'] != 'None' and len(params['scaleFactors']) == len (dataA):
        for i in range(len(dataA)): dataA[i].r = dataA[i].r * params['scaleFactors'][i];
        for i in range(len(dataA)): dataA[i].g = dataA[i].g * params['scaleFactors'][i];

    for channel in params['ChannelsToAnalyze']:
        if channel == 0:
            color = "red"
        elif channel == 1:
            color = "green"
    # background subtract traces
        for i in range(len(dataA)):
            dA=dataA[i]

            # define cellnr and name trace
            cellnr = int(dA.name.split("cellnr_")[-1].split("_")[0])-1
            name =dA.name.split(".txt")[0]

            # load digital data
            if channel == 0:
                # use background for 4 spots at fixed distance from TS
                if os.path.exists(dA.name+"_bg1_red.txt"):
                    bg = []
                    for i in range(1,5):
                        bgtrk = np.loadtxt(dA.name+"_bg"+str(i)+"_red.txt")
                        bg.extend(bgtrk[:,2])
                    dA.bgr = bg
                # define background manually in parameter file
                elif i in params['bgWindowRed']:
                    bg= range(int(params['bgWindowRed'][i][0]/dA.dt),int(params['bgWindowRed'][i][1]/dA.dt))
                    dA.bgr = dA.r[bg]
                # define background as part of trace were no strong spots were found (high threshold)
                else:
                    bg = np.where(dA.trk_r[:,-1] < 1)
                    dA.bgr = dA.r[bg]

                # log-normal fit
                x,y=misc.histoF(np.histogram(dA.bgr,bins=np.r_[0:10:.1]*np.median(dA.bgr),density = True)).T; x-=np.diff(x)[0]/2; m,sd=optimize.fmin(lambda a: sum((y-((1/(x*a[1]*(2*np.pi)**0.5))*np.exp(-(((np.log(x)-a[0])**2)/(2*a[1]**2)))))**2),np.r_[10000,5000],disp=0); dA.mr=m; dA.sdr=sd
                expVal = np.exp(dA.mr)
                dA.r=dA.r-expVal

                # use this code if you want to set all the data outside the defined frameWindow to 0
                #dA.r[:dA.frameWindow[0]]=0.; dA.r[dA.frameWindow[1]:]=0.

                if 1 not in params['ChannelsToAnalyze']:
                    dA.g = dA.r
            elif channel == 1:
                # use background for 4 spots at fixed distance from TS
                if os.path.exists(dA.name+"_bg1_green.txt"):
                    bg2 = []
                    for i in range(1,5):
                        bg2trk = np.loadtxt(dA.name+"_bg"+str(i)+"_green.txt")
                        bg2.extend(bg2trk[:,2])
                    dA.bgg = bg2
                # define background manually in parameter file
                elif i in params['bgWindowGreen']:
                    bg2= range(int(params['bgWindowGreen'][i][0]/dA.dt),int(params['bgWindowGreen'][i][1]/dA.dt))
                    dA.bgg = dA.g[bg2]
                # define background as part of trace were no strong spots were found (high threshold)
                else:
                    bg2 = np.where(dA.trk_g[:,-1] < 1)
                    dA.bgg = dA.g[bg2]

                # log-normal fit
                x,y=misc.histoF(np.histogram(dA.bgg,bins=np.r_[0:10:.1]*np.median(dA.bgg),density = True)).T; x-=np.diff(x)[0]/2; m,sd=optimize.fmin(lambda a: sum((y-((1/(x*a[1]*(2*np.pi)**0.5))*np.exp(-(((np.log(x)-a[0])**2)/(2*a[1]**2)))))**2),np.r_[10000,5000],disp=0); dA.mg=m; dA.sdg=sd
                expVal = np.exp(dA.mg)
                dA.g=dA.g-expVal

                # use this code if you want to set all the data outside the defined frameWindow to 0
                #dA.g[:dA.frameWindow[0]]=0.; dA.g[dA.frameWindow[1]:]=0.

                
                if 0 not in params['ChannelsToAnalyze']:
                    dA.r = dA.g
            
            # write text file of green trace and red background subtracted trace
            if 0 in params['ChannelsToAnalyze']:
                np.savetxt(name+"_bg_sub_red.txt",dA.r)
            if 1 in params['ChannelsToAnalyze']:
                np.savetxt(name+"_bg_sub_green.txt",dA.g)
            
#            # write PDF of trace with histogram and digital data trace
#            histFig = write_hist(None, "r", dA, sdThresholdRed*dA.sdr)
#            histFig.savefig(name+"_bg_sub_trace.pdf")
#            np.savetxt(name+"_bg_sub.txt",dA.g)
#            plt.close()
    return dataA

def calc_correlfunc(dataA,dataB,params):

    ### Compute correlation functions on background subtracted data
    for i in range(len(dataA)):
        if params['alignTracesCF'] == 0:
            start = dataB[i].frameWindow[0]
        if params['alignTracesCF'] == 1:
            if params['color2align'] == "red":
                if np.sum(dataB[i].r) == 0: continue
                start = np.where(dataB[i].r)[0][0]
            if params['color2align'] == "green":
                if np.sum(dataB[i].g) == 0: continue
                start = np.where(dataB[i].g)[0][0]
        dataA[i].G, dataA[i].tau = daf.compG_multiTau(np.c_[dataA[i].r, dataA[i].g][start:dataA[i].frameWindow[1]].T,
                                                      dataA[i].t[start:dataA[i].frameWindow[1]], 0, 0)
        # Write .fcs4 files
        fn=re.escape(dataA[i].name+'bg_sub.fcs4')
        np.savetxt('tmp.txt', np.c_[dataA[i].tau,dataA[i].G[0,0],dataA[i].G[1,1],dataA[i].G[0,1],dataA[i].G[1,0]],'%12.5e  ')
        os.system('echo "#Tau (in s)     Grr            Ggg            Grg            Ggr" > '+fn+'; cat tmp.txt >> '+fn)

    return dataA

#############################################################
### Threshold data based on sdThreshold to make binary data
#############################################################

def binary_call_traces(dataA, params):
    print("Calculating binary data")
    # binary data
    dataB=copy2.deepcopy(dataA)

    for i in range(len(dataB)):
        dA=dataA[i]; dB=dataB[i]

        if 0 in params['ChannelsToAnalyze']:
        # threshold with sdTreshold above SD-background of trace
            sdbg = ((np.exp(2*dA.mr+dA.sdr**2))*(np.exp(dA.sdr**2)-1))**0.5
            dB.r=(dA.r/sdbg>params['sdThresholdRed'])*1  # red binary
            # set all data outside define frameWindow to 0
            dB.r[0:dB.frameWindow[0]] = 0; dB.r[dB.frameWindow[1]: len(dB.r)+1] = 0
        if 1 not in params['ChannelsToAnalyze']:
                dB.g = dB.r
        if 1 in params['ChannelsToAnalyze']:
        # threshold with sdTreshold above SD-background of trace
            sdbg = ((np.exp(2 * dA.mg + dA.sdg ** 2)) * (np.exp(dA.sdg ** 2) - 1)) ** 0.5
            dB.g=(dA.g/sdbg>params['sdThresholdGreen'])*1  # green binary
            # set all data outside define frameWindow to 0
            dB.g[0:dB.frameWindow[0]] = 0; dB.g[dB.frameWindow[1]: len(dB.g)+1] = 0
        if 0 not in params['ChannelsToAnalyze']:
                 dB.r = dB.g

    for channel in params['ChannelsToAnalyze']:
        if channel == 0: color = "r"
        elif channel == 1: color = "g"

        for cell in range(0,len(dataB)):
            for i in range(0,len(dataB[cell][color])-2):
                #remove one-frame gaps:
                if params['Remove1FrameGaps'] == 1:
                    if dataB[cell][color][i] == 1 and dataB[cell][color][i+1] == 0 and dataB[cell][color][i+2] == 1:
                        dataB[cell][color][i+1] = 1

        for cell in range(0,len(dataB)):
            for i in range(0,len(dataB[cell][color])-2):
                #remove one-frame jumps:
                if params['Remove1FrameJumps'] == 1:
                    if dataB[cell][color][i] == 0 and dataB[cell][color][i+1] == 1 and dataB[cell][color][i+2] == 0:
                        dataB[cell][color][i+1] = 0

    return dataB

###################################
### code to select for specific traces, for example to select for traces that have a burst, or make a mask based on the distance of two transcripion sites
###################################

def filter_traces(dataA, dataB, params):
    # filter traces for tracesToExclude and traces that do not have a first burst
    if params['alignTracesCF'] == 1 and params['color2align'] == "red":
        params['retainedTraces'] = [i for i in np.r_[:len(dataB)] if dataB[i].r.sum() > 0 and i not in params['tracesToExclude']]
    elif params['alignTracesCF'] == 1 and params['color2align'] == "green":
        params['retainedTraces'] = [i for i in np.r_[:len(dataB)] if dataB[i].g.sum() > 0 and i not in params['tracesToExclude']]
    elif params['alignTracesCF'] == 0:
        params['retainedTraces']=[i for i in np.r_[:len(dataA)] if i not in params['tracesToExclude'] ]

    ##### Make mask based on distance distance
    if params['selectOnDistance']:
        if len(params['ChannelsToAnalyze']) == 2 :
            for data in dataA:
                xyRed = data.trk_r[:, :2]
                xyGreen = data.trk_g[:, :2]
                data.spotFoundBoth = list(
                    set(np.where(data.trk_r[:, -1] > 0)[0]) & set(np.where(data.trk_g[:, -1] > 0)[0]))

                dist = [np.NaN] * len(xyRed)
                distinv = [0] * len(xyRed)
                dist0 = [0] * len(xyRed)
                for a in range(len(xyRed)):
                    #          for a in data.spotFoundBoth:
                    dist[a] = dist0[a] = (
                                ((xyRed[a, 0] - xyGreen[a, 0]) ** 2 + (xyRed[a, 1] - xyGreen[a, 1]) ** 2) ** 0.5)
                    distinv[a] = 1 / (
                    (((xyRed[a, 0] - xyGreen[a, 0]) ** 2 + (xyRed[a, 1] - xyGreen[a, 1]) ** 2) ** 0.5))
                data.xydist = dist
                data.xydistinv = distinv
                data.distmask = (np.r_[data.xydist] < 5) * 1.
                data.distmask[:data.frameWindow[0]] = 0
                data.distmask[data.frameWindow[1]:] = 0
            #    data.distmask = (np.r_[data.xydist] < 4)*1.

                
    return dataA, params
    #####################################################################
    #### calculate time of being in on state, per cell (histogram) and overall
    ######################################################################
def calc_timefraction_on(dataB,params):

    for channel in params['ChannelsToAnalyze']:
        if channel == 0:
            color = "r"
            col = 'red'
        elif channel == 1:
            color = "g"
            col = 'green'

        framesOn = []
        framesOff = []
        fracOn = []

        for i in params['retainedTraces']:
            if params['alignTracesCF'] == 0:
                start = dataB[i].frameWindow[0]
            if params['alignTracesCF'] == 1:
                if params['color2align'] == 'red':
                    if np.sum(dataB[i].r) == 0: continue
                    start = np.where(dataB[i].r)[0][0]
                if params['color2align'] == 'green':
                    if np.sum(dataB[i].g) == 0: continue
                    start = np.where(dataB[i].g)[0][0]

            if color == "r":
                digidata = dataB[i].r[start:dataB[i].frameWindow[1]]
            elif color == "g":
                digidata = dataB[i].g[start:dataB[i].frameWindow[1]]


            framesOn.append(np.sum(digidata))
            framesOff.append(len(digidata)-np.sum(digidata))
            fracOn.append(float(np.sum(digidata))/float(len(digidata)))

        totalOn = np.sum(framesOn)
        totalOff = np.sum(framesOff)

        print('Frames on: '+str(totalOn)+' total frames: '+str(totalOff+totalOn))
        np.savetxt(params['file'] + "_framesOn_framesOff_" + col + ".txt", [totalOn, totalOff])
        plot_figures.HistogramPlot(fracOn, 20, 'Histogram of fraction of frames in on-state per cell ' + col, 'Fraction frames in on-state',
                                   params['file'] + '_histogram_fraction_on_per_cell_' + col)

def make_plots_traces(dataOrig, dataA, dataB, params):

    #####################################################################
    #### write PDFs of binary data, shows red and green binary data on top
    ######################################################################

    print("Plotting histograms for background subtraction")
    for i in range(len(dataA)):
        dA=dataA[i]
        cellnr = int(dA.name.split("_trk_results")[0][-1])-1
        name =dA.name.split(".txt")[0]
        with PdfPages(name+"_bg_sub_trace.pdf") as pdfTrace:
            if 0 in params['ChannelsToAnalyze']:
                sdbg = ((np.exp(2*dA.mr+dA.sdr**2))*(np.exp(dA.sdr**2)-1))**0.5
                plot_figures.write_hist(pdfTrace, "r", dA, params['sdThresholdRed']*sdbg)
                plt.close()
                if len(dA.bgr) == 4* len(dA.t):
                    plot_figures.showBackgroundTrace(pdfTrace, dA,"r", params['sdThresholdRed']*dA.sdr)
                    plt.close()
            if 1 in params['ChannelsToAnalyze']:
                sdbg = ((np.exp(2 * dA.mg + dA.sdg ** 2)) * (np.exp(dA.sdg ** 2) - 1)) ** 0.5
                plot_figures.write_hist(pdfTrace, "g", dA, params['sdThresholdGreen']*sdbg)
                plt.close()
                if len(dA.bgg) == 4* len(dA.t):
                    plot_figures.showBackgroundTrace(pdfTrace, dA,"g", params['sdThresholdGreen']*dA.sdg)
                    plt.close()
            plot_figures.showBinaryCall(pdfTrace, dA, dataB[i])
            plt.close()




    #####################################################################
    #### make figure heatmap
    ######################################################################
    
    print("Plotting trace heatmaps")

    dataAtrim=copy2.deepcopy(dataA)
    dataBtrim=copy2.deepcopy(dataB)
    outsideFW=copy2.deepcopy(dataA)
    for i in range(len(dataAtrim)):
        dA=dataAtrim[i]
        dA.g[:dA.frameWindow[0]]=0.; dA.g[dA.frameWindow[1]:]=0.
        dA.r[:dA.frameWindow[0]]=0.; dA.r[dA.frameWindow[1]:]=0.
        
        dB=dataBtrim[i]
        dB.g[:dB.frameWindow[0]]=0.; dB.g[dB.frameWindow[1]:]=0.
        dB.r[:dB.frameWindow[0]]=0.; dB.r[dB.frameWindow[1]:]=0.
        
        dFW=outsideFW[i]
        dFW.g[:dFW.frameWindow[0]]=1.; dFW.g[dFW.frameWindow[1]:]=0.3
        dFW.g[dFW.frameWindow[0]:dFW.frameWindow[1]]=0.

    if params['trimHeatmaps'] == 1:
        dataHeatmapAn = dataAtrim
        dataHeatmapDig = dataBtrim
    else:
        dataHeatmapAn = dataA
        dataHeatmapDig = dataB

    if params['OrderTraceHeatmap'] == "CF":
        sortedIdsTrace = sortedIds
    else:
        sortedIdsTrace = params['retainedTraces']

    # write heatmaps of analog and digital data
    if params['trimHeatmaps'] == 1:
        heatmapA = plot_figures.showHeatMap(dataHeatmapAn, maxRed = params['heatMapScalingRed'], maxGreen = params['heatMapScalingGreen'],trimdata = outsideFW, sortedIds = sortedIdsTrace)
        heatmapA.savefig(fname = params['file']+"_Heatmap_analog.pdf")
        plt.close()
    else:
        heatmapA = plot_figures.showHeatMap(dataHeatmapAn, maxRed = params['heatMapScalingRed'], maxGreen = params['heatMapScalingGreen'], sortedIds = sortedIdsTrace)
        heatmapA.savefig(fname = params['file']+"_Heatmap_analog.pdf")
        plt.close()

    if params['trimHeatmaps'] == 1:
        heatmapB = plot_figures.showHeatMap(dataHeatmapDig,trimdata = outsideFW, sortedIds = sortedIdsTrace)
        heatmapB.savefig(fname = params['file']+"_Heatmap_digital.pdf")
        plt.close()
    else:
        heatmapB = plot_figures.showHeatMap(dataHeatmapDig, sortedIds = sortedIdsTrace)
        heatmapB.savefig(fname = params['file']+"_Heatmap_digital.pdf")
        plt.close()

    if params['CalcIndTimes'] == 1:
        print("Plotting ordered trace heatmaps")
        for channel in params['ChannelsToAnalyze']:
            if channel == 0:
                col = "red"
            elif channel == 1:
                col = "green"

            indtimes = []
            for cell in params['retainedTraces']:
                if col == "red": bindata = dataB[cell].r
                if col == "green": bindata = dataB[cell].g
                if sum(bindata) == 0: continue
                indframe = np.where(bindata > 0)[0][0]
                indtime = (dataB[cell].dt) * float(indframe) / 60
                indtimes.append(indtime)
            np.savetxt(params['file'] + "_induction_times_" + col + ".txt", indtimes)

            plot_figures.HistogramPlot(indtimes, 20, 'Histogram of induction times ' + col, 'Induction time (min)',
                          params['file'] + '_histogram_induction_times_' + col)
            plot_figures.CumHistogramPlot(indtimes, 'Cumulative distribution of induction times ' + col, 'Induction time (min)',
                             params['file'] + '_cumulative_distribution_induction_times_' + col)

            if params['SortHeatMapIndTimes'] == col:
                indtimessortedRetainedTraces = np.flip(np.argsort(indtimes))
                indtimessorted = [params['retainedTraces'][i] for i in indtimessortedRetainedTraces]
            else:
                indtimessorted = params['retainedTraces']

        # making heatmaps sorted by induction time
        if params['SortHeatMapIndTimes'] != None:
            if params['trimHeatmaps'] == 1:
                sortedheatmapA = plot_figures.showHeatMap(dataHeatmapAn, maxRed=params['heatMapScalingRed'],
                                                          maxGreen=params['heatMapScalingGreen'], sortedIds=indtimessorted,
                                                          trimdata=outsideFW)
            else:
                sortedheatmapA = plot_figures.showHeatMap(dataHeatmapAn, maxRed=params['heatMapScalingRed'],
                                                          maxGreen=params['heatMapScalingGreen'], sortedIds=indtimessorted)
            sortedheatmapA.savefig(fname=params['file'] + "_Heatmap_analog_sorted_by_induction.pdf")
            plt.close()

        if params['trimHeatmaps'] == 1:
            sortedheatmapB = plot_figures.showHeatMap(dataHeatmapDig, sortedIds=indtimessorted, trimdata=outsideFW)
        else:
            sortedheatmapB = plot_figures.showHeatMap(dataHeatmapDig, sortedIds=indtimessorted)
            
        sortedheatmapB.savefig(fname=params['file'] + "_Heatmap_digital_sorted_by_induction.pdf")
        plt.close()


    #####################################################################
    #### Plot area under traces, useful to see if some traces dominate
    ######################################################################
   
    print("Plotting area under traces")
    if 0 in params['ChannelsToAnalyze']:
        figAvgR = plot_figures.showAreaUnderTraces(dataA, params['retainedTraces'], "r")
        figAvgR.savefig(params['file']+"_AreaUnderTraces_red.pdf")
        plt.close()
    if 1 in params['ChannelsToAnalyze']:
        figAvgG = plot_figures.showAreaUnderTraces(dataA, params['retainedTraces'], "g")
        figAvgG.savefig(params['file']+"_AreaUnderTraces_green.pdf")
        plt.close()

    #####################################################################
    #### Making plot of all non-background corrected intensities
    ######################################################################

    print("Plotting intensity distribution")
    for channel in params['ChannelsToAnalyze']:
        if channel == 0:
            col = "red";
            color = "r"
        elif channel == 1:
            col = "green";
            color = "g"
        figInt = plot_figures.PlotIntensities(None, dataOrig, dataA, dataB, params, color, params['file']+'_intensities_frames_on.npy')
        figInt.savefig(params['file'] + '_histogram_intensity_values_' + col + '.pdf')
        plt.close()

    #if len(params['ChannelsToAnalyze']) == 2 :
     #   print("Plotting distance distribution")
      #  figDist = plot_figures.PlotDistances(None,dataA,params['retainedTraces'])
       # figDist.savefig(params['file'] + '_histogram_distances' + '.pdf')
        #plt.close()



#####################################################################
#### calculate burst duration and time between bursts from thresholded data
######################################################################

def make_burst_histograms(dataB, params):
    print("Plotting burst histograms")
    for channel in params['ChannelsToAnalyze']:
        if channel == 0: col = "red"; color = "r"
        elif channel == 1: col = "green"; color = "g"

        BurstDuration = []
        TimeBetweenBursts = []

        for cc in params['retainedTraces']:
            ## load digital data trace
            if channel == 0: datacell = dataB[cc].r[dataB[cc].frameWindow[0]:dataB[cc].frameWindow[1]]; sdThreshold = params['sdThresholdRed']
            elif channel == 1: datacell = dataB[cc].g[dataB[cc].frameWindow[0]:dataB[cc].frameWindow[1]]; sdThreshold = params['sdThresholdGreen']
            if sum(datacell) > 0:
                start = np.where(np.diff(datacell)==1)[0] # define where bursts start
                end = np.where(np.diff(datacell)==-1)[0] # define where bursts end
                if len(start)!= 0 and len(end)!=0: # if start and end not the same length (for example if burst is found in first or last frame)
                    if start [0] > end [0]: endAdj = end[1:];startAdj = start[:len(endAdj)];BurstDuration.extend(endAdj-startAdj); endAdj2 = end[:-1]; TimeBetweenBursts.extend(startAdj-endAdj2) # if first burst starts before first frame, remove first burst.
                    elif len(start) > len(end): startAdj = start[:len(end)];BurstDuration.extend(end-startAdj); start2 = start[1:]; TimeBetweenBursts.extend(start2-end) # if last burst does not end, remove last burst
                    else: BurstDuration.extend(end-start); start2 = start[1:];end2 = end[:-1]; TimeBetweenBursts.extend(start2-end2) # all other burst

        np.savetxt((params['file']+"_TimeBetweenBursts_checked_threshold_"+col+"_"+str(sdThreshold)+".txt"), TimeBetweenBursts, fmt = '%u')
        np.savetxt((params['file']+"_BurstDuration_checked_threshold_"+col+"_"+str(sdThreshold)+".txt"), BurstDuration, fmt = '%u')

        # make histogram of burst duration and time between bursts
        binSize = params['binSizeHistogram'] # time interval
        maxXaxis = params['maxXaxisHistogram'] # range histogram
        dt=dataB[0].dt*1.
        if len(BurstDuration) != 0:
            bootstrDuration = plot_figures.CalcBootstrap(BurstDuration, 1000)
        else: bootstrDuration = [np.nan,np.nan]
        BurstDurationMean = bootstrDuration[0]
        BurstDurationMeanErr = bootstrDuration[1]
        if len(TimeBetweenBursts) != 0:
            bootstrTimeBetwBurst = plot_figures.CalcBootstrap(TimeBetweenBursts, 1000)
        else: bootstrTimeBetwBurst = [np.nan, np.nan]
        TimeBetweenBurstMean = bootstrTimeBetwBurst[0]
        TimeBetweenBurstMeanErr = bootstrTimeBetwBurst[1]
        hBD=np.histogram(BurstDuration,bins=np.r_[:maxXaxis/dt:binSize/dt]-.5, density = True)
        hBF = np.histogram(TimeBetweenBursts, bins=np.r_[:maxXaxis / dt:binSize / dt] - .5, density=True)
        if len(TimeBetweenBursts) == 0:
            hBF = [np.zeros(len(hBF[0])),hBF[1]]
        if len(BurstDuration) == 0:
            hBD = [np.zeros(len(hBD[0])),hBD[1]]

        bins = (hBD[1]+0.5)[:-1]*dt

        # define exponential function
        def expofunc(x, a, b):
            return a * np.exp(-x/b)

        # find where histogram burst duration is non-zeo, do not use first point for fitting
        lox = np.where(hBD[0] !=0)[0][1:]
        # fit histogram burst duration to exponential function
        if len(lox) == 0 or np.isnan(np.sum(hBD[0])):
            poptDur = [0,0]
            pcovDur = [[0,0],[0,0]]
        else:
            poptDur, pcovDur = curve_fit(expofunc, bins[lox], hBD[0][lox], p0 = [400, 30], bounds = [0,np.inf])
        print('Fit parameters burst duration: '+str(poptDur))
        perr = np.sqrt(np.diag(pcovDur))
        fit = expofunc(bins, *poptDur)

        # find where histogram burst frequency is non-zeo, do not use first point for fitting
        lox2 = np.where(hBF[0] !=0)[0][1:]
        # fit histogram burst frequency to exponential function
        if len(lox2) == 0 or np.isnan(np.sum(hBF[0])):
            poptFreq = [0,0]
            pcovFreq = [[0,0],[0,0]]
        else:
            poptFreq, pcovFreq = curve_fit(expofunc, bins[lox2], hBF[0][lox2], p0 = [400, 200], bounds = [0,np.inf])
        print('Fit parameters time between bursts: ' +str(poptFreq))
        perrFreq = np.sqrt(np.diag(pcovFreq))
        fit2 = expofunc(bins, *poptFreq)

        fig = plt.figure(figsize=A4)
        gs = GridSpec(2,1, figure = fig)
        
        fig.add_subplot(gs[0,0])
        plt.bar(hBD[1][:-1]*dt, hBD[0], color ='blue', width = dt-2)
        plt.plot(hBD[1][:-1]*dt,fit, color='black')
        plt.ylim(0,1.1)
        plt.title('Burst duration')
        plt.xlabel('Burst duration (s)')
        plt.ylabel('Frequency')
        plt.text(hBD[1][-1]*dt*0.5,0.9,'burst duration, mean = '+str(round(BurstDurationMean*dt,2))+' +/- '+str(round(BurstDurationMeanErr*dt,2))+' s\nExp fit burst duration: tau = '+str(round(poptDur[1],2))+' +/- '+str(round(perr[1],2))+' s')

        fig.add_subplot(gs[1,0])
        plt.bar(hBF[1][:-1]*dt, hBF[0], color='gray', width = dt-2)
        plt.plot(hBF[1][:-1]*dt,fit2, color='black')
        plt.ylim(0,1.1)
        plt.title('Burst frequency')
        plt.xlabel('Time between bursts (s)')
        plt.ylabel('Frequency')
        
        plt.text(hBD[1][-1]*dt*0.5,0.9,'time between bursts, mean = '+str(round(TimeBetweenBurstMean*dt,2))+' +/- '+str(round(TimeBetweenBurstMeanErr*dt,2))+' s\nExp fit time between bursts: tau = '+str(round(poptFreq[1],2))+' +/- '+str(round(perrFreq[1],2))+' s')

        plt.tight_layout()
        
        fig.savefig(params['file']+ '_BurstDuration+Freq_after_threshold_'+col+'.pdf')
        plt.close()


##############################################################################
### Calculating single-cell parameters #######################################
##############################################################################
def make_burst_histograms_singlecells(dataOrig,dataA,dataB, params):
    print("Plotting burst histograms per single cell")
    for channel in params['ChannelsToAnalyze']:
        if channel == 0: col = "red"; color = "r"
        elif channel == 1: col = "green"; color = "g"

        BurstDuration = []
        TimeBetweenBursts = []

        for cc in params['retainedTraces']:
            ## load digital data trace
            if channel == 0: datacell = dataB[cc].r[dataB[cc].frameWindow[0]:dataB[cc].frameWindow[1]]
            elif channel == 1: datacell = dataB[cc].g[dataB[cc].frameWindow[0]:dataB[cc].frameWindow[1]]
            if sum(datacell) > 0:
                start = np.where(np.diff(datacell)==1)[0] # define where bursts start
                end = np.where(np.diff(datacell)==-1)[0] # define where bursts end
                if len(start)!= 0 and len(end)!=0: # if start and end not the same length (for example if burst is found in first or last frame)
                    if start [0] > end [0]: endAdj = end[1:];startAdj = start[:len(endAdj)];BurstDuration.append([cc,endAdj-startAdj]); endAdj2 = end[:-1]; TimeBetweenBursts.append([cc,startAdj-endAdj2]) # if first burst starts before first frame, remove first burst.
                    elif len(start) > len(end): startAdj = start[:len(end)];BurstDuration.append([cc,end-startAdj]); start2 = start[1:]; TimeBetweenBursts.append([cc,start2-end]) # if last burst does not end, remove last burst
                    else: BurstDuration.append([cc,end-start]); start2 = start[1:];end2 = end[:-1]; TimeBetweenBursts.append([cc,start2-end2]) # all other burst
            
        ### burst durations, individual and per cell
        meanDurs = []
        allDurs = []
        for i in range(0,len(BurstDuration)):
            durs = BurstDuration[i][1]
            if len(durs!=0):
                meanDurs.append(np.mean(durs))
                for j in range(0,len(durs)):
                    allDurs.append(durs[j])

        meanDurssec = [i * 15 for i in meanDurs]
        np.savetxt(params['file']+"_mean_burst_duration_per_cell_"+col+".txt",meanDurssec)
        plot_figures.HistogramPlot(meanDurssec, 20, 'Histogram of average burst duration per cell '+col, 'Average burst duration (s)', params['file']+'_histogram_average_burst_duration_per_cell_'+col)
        
        allDurssec = [i * 15 for i in allDurs]
        np.savetxt(params['file']+"burst_duration_"+col+".txt",allDurssec)
        plot_figures.HistogramPlot(allDurssec, 20, 'Burst durations '+col, 'Average burst duration (s)', params['file']+'_histogram_burst_duration_'+col)
        
        ### time between bursts
        meanTimeBetw = []
        allTimeBetw = []
        for i in range(0,len(TimeBetweenBursts)):
            times = TimeBetweenBursts[i][1]
            if len(times!=0):
                meanTimeBetw.append(np.mean(times))
                for j in range(0,len(times)):
                    allTimeBetw.append(times[j])
                
        meanTimeBetwsec = [i * 15 for i in meanTimeBetw]
        np.savetxt(params['file']+"_mean_time_between_bursts_per_cell_"+col+".txt",meanTimeBetwsec)
        plot_figures.HistogramPlot(meanTimeBetwsec, 20, 'Histogram of average time between bursts per cell '+col, 'Average time between bursts (s)', params['file']+ '_histogram_average_time_between_bursts_per_cell_'+col)

        allTimeBetwsec = [i * 15 for i in allTimeBetw]
        np.savetxt(params['file']+"_time_between_bursts_"+col+".txt",allTimeBetwsec)
        plot_figures.HistogramPlot(allTimeBetwsec, 20, 'Time between bursts '+col, 'Average time between bursts (s)', params['file']+ '_histogram_time_between_bursts_'+col)
        
        ##number of bursts per cell
        nbrbursts = []
        for i in range(0,len(BurstDuration)):
            nbrbursts.append(len(BurstDuration[i][1]))
        np.savetxt(params['file']+"_number_of_bursts_per_cell_"+col+".txt",nbrbursts)
        plot_figures.HistogramPlot(nbrbursts, 20, 'Histogram of number of bursts per cell '+col, 'Number of bursts', params['file']+ '_histogram_number_of_bursts_per_cell_'+col)
            
        ##intensity in on state per cell and individual frames
        avIntCell = []
        allInts = []
        for cc in params['retainedTraces']:
            if col == "red": datacellInts = dataOrig[cc].r[dataOrig[cc].frameWindow[0]:dataOrig[cc].frameWindow[1]]
            datacellDigi = dataB[cc].r[dataB[cc].frameWindow[0]:dataB[cc].frameWindow[1]]
            if col == "green": datacellInts = dataOrig[cc].g[dataOrig[cc].frameWindow[0]:dataOrig[cc].frameWindow[1]]
            datacellDigi = dataB[cc].g[dataB[cc].frameWindow[0]:dataB[cc].frameWindow[1]]
            onints = datacellInts * datacellDigi
            if sum(datacellDigi)!=0:
                avInt = np.sum(onints)/np.sum(datacellDigi)
                avIntCell.append(avInt)
            for i in range(0,len(onints)):
                if onints[i]!=0: allInts.append(onints[i])
        
        np.savetxt(params['file']+"_intensity_frames_on_per_cell_"+col+".txt",avIntCell)
        np.savetxt(params['file']+"_intensity_frames_on_"+col+".txt",allInts)
        
        plot_figures.HistogramPlot(avIntCell, 20, 'Histogram of average intensity of frames in on-state per cell '+col, 'Intensity (AU)', params['file']+ '_intensity_frames_on_per_cell_'+col)
        plot_figures.HistogramPlot(allInts, 20, 'Histogram of intensity of all frames in on-state '+col, 'Intensity (AU)', params['file']+ '_intensity_frames_on_'+col)

        ##Second value of ACF plot (measure of ACF amplitude)
        ACFampl = []
        for cell in params['retainedTraces']:
            if np.sum(dataA[cell].G) == 0: continue
            if col == 'red':
                if len(dataA[cell].G[0, 0]) <= 1: continue
                ACFampl.append(dataA[cell].G[0,0][1])
            if col == 'green':
                if len(dataA[cell].G[1, 1]) <= 1: continue
                ACFampl.append(dataA[cell].G[1,1][1])
        np.savetxt(params['file']+"_ACF_amplitudes_per_cell_"+col+".txt", ACFampl)
        plot_figures.HistogramPlot(ACFampl, 20, 'Histogram of ACF amplitude per cell '+col, 'ACF amplitude', params['file']+ '_ACF_amplitude_per_cell_'+col)

        ##Correlation of ACF with induction time
        plot_figures.corrACFAmplToIndPlot(dataA, dataB, col, params)

def pipeline_correlation_functions(params):
    # pipeline for correlation functions script
    # params is either a dictionary containing the parameters for the pipeline or a string pointing to the yml file with the parameters

    if not isinstance(params, dict):
        parameter_file = params
        if parameter_file[-3:] == '.py':
            print('Converting py parameter file into yml format')
            misc.convertParamFile2YML(parameter_file)
            parameter_file = parameter_file[:-3]+'.yml'
        if not parameter_file[-4:] == '.yml':
            parameter_file += '.yml'
        params = misc.getConfig(parameter_file)
    else:
        parameter_file = ''

    parseParams(params)  #### add default values if missing
    get_paths(params, parameter_file)
    
    if params['processTraces'] == 1:
    
        # Load original raw data
        dataOrig=daf.loadExpData(params['PyFile'])

        # Better not smooth and quantize the same trace
        if 'quantizeTrace' in params:
            for channel, q in params['quantizeTrace'].items():
                daf.getMolNumber(dataOrig, q, channel)
        if 'smoothTrace' in params:
            for channel, (window_length, polyorder) in params['smoothTrace'].items():
                daf.smoothData(dataOrig, channel, window_length, polyorder)

        dataA = bg_sub_traces(dataOrig, params)
        dataB = binary_call_traces(dataA, params)
        dataA = calc_correlfunc(dataA, dataB, params)
        dataA, params = filter_traces(dataA, dataB, params)

    calc_timefraction_on(dataB,params)

    if params['makePlots'] == 1:
        make_plots_traces(dataOrig, dataA, dataB, params)
    
    if params['makeHistogram'] == 1:
        make_burst_histograms(dataB, params)
    
    if params['SingleCellParams'] == 1:
        make_burst_histograms_singlecells(dataOrig,dataA,dataB, params)

    return params

if __name__ == '__main__':
    if len(sys.argv)<2:
        if os.path.exists('pipeline_livecell_correlationfunctions_parameters.yml'):
            parameter_files = ['pipeline_livecell_correlationfunctions_parameters.yml']
        elif os.path.exists('pipeline_livecell_correlationfunctions_parameters.py'):
            parameter_files = ['pipeline_livecell_correlationfunctions_parameters.py']
        else:
            raise IOError('Could not find the parameter file.')
    else:
        parameter_files = sys.argv[1:]

    if len(parameter_files)==1:
        params = pipeline_correlation_functions(parameter_files[0])
    else:
        for parameter_file in parameter_files:
            print(misc.color('Working on: {}'.format(parameter_file), 'b:b'))
            print('')
            try:
                params = pipeline_correlation_functions(parameter_file)
            except:
                print(misc.color('Exception while working on: {}'.format(parameter_file), 'r:b'))

    # this only runs when this script is run from command-line with ./pipeline..., not when run from ipython
    # if we do not kill the java vm, (i)python will not exit after completion
    # be sure to call imr.kill_vm() at the end of your script/session, note that you cannot use imread afterwards
    if os.path.basename(__file__) in [os.path.basename(i) for i in psutil.Process(os.getpid()).cmdline()]:
        imr.kill_vm() #stop java used for imread, needed to let python exit
        print('Stopped the java vm used for imread.')

    print('------------------------------------------------')
    print(misc.color('Pipeline finished.', 'g:b'))
