import numpy as np
import matplotlib.pyplot as plt
import copy as copy2
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import dendrogram
import dataAnalysis_functions as daf
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.max_open_warning'] = 250
A4 = (11.69, 8.27)

# This script defines functions used in the pipeline_livecell_correlationfunctions to plot figures


##################
##################
### Misc functions


def write_hist(pdf, color, dA, sdThreshold):
    # color: g or r
    if color=='g':
        m = dA.mg
        sd = dA.sdg
        dataDigital = dA.trk_g[:,-1]
        bg = dA.bgg
    else:
        m = dA.mr
        sd = dA.sdr
        dataDigital = dA.trk_r[:,-1]
        bg = dA.bgr
    
    fig = plt.figure(figsize=A4)
    gs = GridSpec(2, 1, figure=fig)

    fig.add_subplot(gs[0,0])
    plt.hist(bg, np.clip(int(np.ceil(np.sqrt(len(bg)))), 1, 100), density=True, color=color)
    xl = plt.xlim()
    X = np.linspace(xl[0], xl[1], 1000)
    plt.plot(X, ((1/(X*sd*(2*np.pi)**0.5))*np.exp(-(((np.log(X)-m)**2)/(2*sd**2)))), '-k')
    expVal = np.exp(m-sd**2)
    plt.axvline(expVal,linestyle = '--')
    plt.xlabel('background counts')
    plt.ylabel('frequency')
    plt.title(dA.name.split("/")[-1] + ' ' + color)

    fig.add_subplot(gs[1,0])
    plt.plot((dA.t.min(), dA.t.max()), np.zeros(2), '--k')
    sdt = plt.plot((dA.t.min(), dA.t.max()), sdThreshold*np.ones(2), '--'+color)
    t0 = plt.plot(dA.t, dA[color], '-'+color)
    s = dA[color].max()
    d = plt.plot(dA.t,s*(dataDigital-1)/10, '-c')
    plt.plot(np.r_[0,np.array(dA.frameWindow),dA.r.shape[0]][[1,0,0,1,1,2,2,3,3,2]]*dA.dt,np.r_[0,1,0,1,0,0,1,0,1,0]*max(np.r_[dA.r,dA.g]), color = 'gray')
    plt.xlim(dA.t.min(), dA.t.max())
    plt.legend((t0[0], sdt[0], d[0]), ('Trace #0', 'sdThreshold', 'Digital'), loc='upper right')
    plt.xlabel('time (s)')
    plt.ylabel('peak intensity (counts)')
    if not pdf is None:
        pdf.savefig(fig)
    else:
        return fig

### Display results of the binary thresholding
def showBinaryCall(pdf, dA, dB):
    fig = plt.figure(figsize=A4)
    gs = GridSpec(2, 1, figure=fig)

    fig.add_subplot(gs[0,0])
    plt.plot(dB.t, dB.g*.8+1.1, '-g')
    plt.plot(dB.t, dB.r*.8+.1,  '-r')
    plt.ylim(0,2)
    plt.xlabel('time (s)')
    plt.title(dA.name + ' binary thresholding')

    fig.add_subplot(gs[1,0])
    plt.plot(dA.t, dA.g/dA.g.max(), '-g')
    plt.plot(dA.t, dA.r/dA.r.max(), '-r')
    plt.plot((0, dA.t.max()), (0, 0), '--', color='gray')
    plt.plot(np.r_[0,np.array(dA.frameWindow),dA.r.shape[0]][[1,0,0,1,1,2,2,3,3,2]]*dA.dt,np.r_[0,1,0,1,0,0,1,0,1,0], color = 'gray')
    plt.xlabel('time (s)')
    plt.ylabel('peak intensity (AU)')
    if not pdf is None:
        pdf.savefig()
    else:
        return fig

def showBackgroundTrace(pdf, dA, color, sdThreshold):
    fig = plt.figure(figsize=A4)
    gs = GridSpec(2, 1, figure=fig)

    if color == 'g':
        bg = dA.bgg
        m = dA.mg

    else:
        bg = dA.bgr
        m = dA.mr

    fig.add_subplot(gs[0,0])
    plt.plot(dA.t, dA[color], '-'+color)
    plt.plot(dA.t, np.r_[bg[:len(dA.t)]]-np.exp(m), '-b')
    plt.plot(dA.t, np.r_[bg[len(dA.t) : 2*len(dA.t)]]-np.exp(m), '-c')
    plt.plot(dA.t, np.r_[bg[2*len(dA.t) : 3*len(dA.t)]]-np.exp(m), '-m')
    plt.plot(dA.t, np.r_[bg[3*len(dA.t) : 4*len(dA.t)]]-np.exp(m), '-y')
    plt.plot((dA.t.min(), dA.t.max()), sdThreshold * np.ones(2), '--', color = 'gray')
    plt.plot((0, dA.t.max()), (0, 0), '--', color='gray')
    plt.plot(np.r_[0,np.array(dA.frameWindow),dA.r.shape[0]][[1,0,0,1,1,2,2,3,3,2]]*dA.dt,np.r_[0,1,0,1,0,0,1,0,1,0]*dA[color].max(), color = 'gray')
    plt.xlabel('time (s)')
    plt.ylabel('intensity (AU)')

    fig.add_subplot(gs[1,0])
    plt.plot(dA.t, np.c_[np.r_[bg[:len(dA.t)]],np.r_[bg[len(dA.t) : 2*len(dA.t)]], np.r_[bg[2*len(dA.t) : 3*len(dA.t)]],np.r_[bg[3*len(dA.t) : 4*len(dA.t)]] ].mean(axis = 1)-m , '-b')
    plt.plot((dA.t.min(), dA.t.max()), sdThreshold * np.ones(2), '--', color = 'gray')
    plt.plot((0, dA.t.max()), (0, 0), '--', color='gray')
    plt.plot(np.r_[0,np.array(dA.frameWindow),dA.r.shape[0]][[1,0,0,1,1,2,2,3,3,2]]*dA.dt,np.r_[0,1,0,1,0,0,1,0,1,0]*max(bg), color = 'gray')
    plt.xlabel('time (s)')
    plt.ylabel('mean background intensity (AU)')


    if not pdf is None:
        pdf.savefig()
    else:
        return fig

def showAvTrace(pdf, ss, names):
    fig = plt.figure(figsize=A4)
    gs = GridSpec(2, 1, figure=fig)

    fig.add_subplot(gs[0,0])
    plt.fill_between(ss.t, ss.v[0,:,0]-ss.v[0,:,1], ss.v[0,:,0]+ss.v[0,:,1], facecolor='r', edgecolor=None, alpha=0.5)
    plt.fill_between(ss.t, ss.v[1,:,0]-ss.v[1,:,1], ss.v[1,:,0]+ss.v[1,:,1], facecolor='g', edgecolor=None, alpha=0.5)
    plt.plot(ss.t, ss.v[0,:,0], '-r')
    plt.plot(ss.t, ss.v[1,:,0], '-g')
    plt.plot(np.r_[0,0],np.r_[ss.v.min(),ss.v.max()], color = 'black' )
    plt.xlim(-300, ss.t.max())
    plt.xlabel('time (s)')
    plt.ylabel('average peak intensity (counts)')
    plt.title('average peak intensity')

    idx = len(ss.sigsAlign)%2
    nameInd = 0
    for s in ss.sigsAlign:
        if not idx:
            pdf.savefig(fig)
            fig = plt.figure(figsize=A4)
            gs = GridSpec(2, 1, figure=fig)
        fig.add_subplot(gs[idx,0])
        plt.plot(s.t,s.v[0], '-r')
        plt.plot(s.t,s.v[1], '-g')
        plt.plot(s.t,(1-s.mask[0])*s.v.max(), '--', color = 'gray')
        plt.plot(np.r_[0,0],np.r_[getylim(s.v, np.tile(s.t, (2, 1)), (-300, ss.t.max()))], '--',color = 'black' )
        plt.text(0.5*(-300 + ss.t.max()),s.v.max(),names[nameInd].split("/")[-1], horizontalalignment='center')
        plt.xlim(-300, ss.t.max())
        plt.xlabel('time (s)')
        plt.ylabel('peak intensity (counts)')
        if not idx:
            plt.title('peak intensity for individual traces')
        
        idx = not idx
        nameInd +=1
    if not pdf is None:
        pdf.savefig(fig)
    else:
        return fig
    
def getylim(y, x=None, xlim=None, margin=0.05):
    """ get limits for plots according to data
        copied from matplotlib.axes._base.autoscale_view

        y: the y data
        optional, for when xlim is set manually on the plot
            x: corresponding x data
            xlim: limits on the x-axis in the plot, example: xlim=(0, 100)
            margin: what fraction of white-space to have at all borders
        y and x can be lists or tuples of different data in the same plot

        wp@tl20191220
    """
    y = np.array(y).flatten()
    if not x is None and not xlim is None:
        x = np.array(x).flatten()
        y = y[(np.nanmin(xlim)<x)*(x<np.nanmax(xlim))*(np.abs(x)>0)]
        if len(y)==0:
            return -margin, margin
    y0t, y1t = np.nanmin(y), np.nanmax(y)
    if (np.isfinite(y1t) and np.isfinite(y0t)):
        delta = (y1t - y0t) * margin
    else:
        # If at least one bound isn't finite, set margin to zero
        delta = 0
    return y0t-delta, y1t+delta

def showCorrFun(pdf, ss):
    fig = plt.figure(figsize=A4)
    gs = GridSpec(2, 3, figure=fig)
    
    xlim = ss.tau.max()/2

    fig.add_subplot(gs[0,0])
    plt.fill_between(ss.tau, ss.G[0,0,:,0]-ss.G[0,0,:,1], ss.G[0,0,:,0]+ss.G[0,0,:,1], facecolor='r', edgecolor=None, alpha=0.5)
    plt.fill_between(ss.tau, ss.Gns[0,0,:,0]-ss.Gns[0,0,:,1], ss.Gns[0,0,:,0]+ss.Gns[0,0,:,1], facecolor='k', edgecolor=None, alpha=0.5)
    g = plt.plot(ss.tau, ss.G[0,0,:,0], '-r')
    gns = plt.plot(ss.tau, ss.Gns[0,0,:,0], '-k')
    plt.xlim(0, xlim)
    plt.ylim(getylim((ss.G[0,0,:,0]-ss.G[0,0,:,1], ss.G[0,0,:,0]+ss.G[0,0,:,1],
                      ss.Gns[0,0,:,0]-ss.Gns[0,0,:,1], ss.Gns[0,0,:,0]+ss.Gns[0,0,:,1]),
                     4*(ss.tau,), (0, xlim)))
    plt.legend((g[0], gns[0]), ('G (tau)', 'G ns (tau)'))
    plt.ylabel(r'$\mathrm{G}(\tau)$')
    plt.title('Non-stationary\nred auto-correlation')
    
    fig.add_subplot(gs[0,1])
    plt.fill_between(ss.tau, ss.G[0,1,:,0]-ss.G[0,1,:,1], ss.G[0,1,:,0]+ss.G[0,1,:,1], facecolor='b', edgecolor=None, alpha=0.5)
    plt.fill_between(-ss.tau, ss.G[1,0,:,0]-ss.G[1,0,:,1], ss.G[1,0,:,0]+ss.G[1,0,:,1], facecolor='y', edgecolor=None, alpha=0.5)
    plt.fill_between(ss.tau, ss.Gns[0,1,:,0]-ss.Gns[0,1,:,1], ss.Gns[0,1,:,0]+ss.Gns[0,1,:,1], facecolor='k', edgecolor=None, alpha=0.5)
    plt.fill_between(-ss.tau, ss.Gns[1,0,:,0]-ss.Gns[1,0,:,1], ss.Gns[1,0,:,0]+ss.Gns[1,0,:,1], facecolor='k', edgecolor=None, alpha=0.5)
    grg = plt.plot(ss.tau,ss.G[0,1,:,0], '-b')
    ggr = plt.plot(-ss.tau,ss.G[1,0,:,0], '-y')
    grgns = plt.plot(ss.tau,ss.Gns[0,1,:,0], '-k')
    ggrns = plt.plot(-ss.tau,ss.Gns[1,0,:,0], '-k')
    plt.xlim(-xlim, xlim)    
    plt.ylim(getylim((ss.G[0,1,:,0]-ss.G[0,1,:,1], ss.G[0,1,:,0]+ss.G[0,1,:,1],
                      ss.G[1,0,:,0]-ss.G[1,0,:,1], ss.G[1,0,:,0]+ss.G[1,0,:,1],
                      ss.Gns[0,1,:,0]-ss.Gns[0,1,:,1], ss.Gns[0,1,:,0]+ss.Gns[0,1,:,1],
                      ss.Gns[1,0,:,0]-ss.Gns[1,0,:,1], ss.Gns[1,0,:,0]+ss.Gns[1,0,:,1]),
                     (ss.tau, ss.tau, -ss.tau, -ss.tau, ss.tau, ss.tau, -ss.tau, -ss.tau), (-xlim, xlim)))    
    # plt.ylim(getylim((ss.G[:2,:2,:,0], ss.Gns[:2,:2,:,0]), (np.tile(ss.tau, (2, 2, 1)),)*2, (-xlim, xlim)))
    plt.legend((grg[0], grgns[0], ggr[0], ggrns[0]), ('G[rg] (tau)', 'G[rg] ns (tau)', 'G[gr] (tau)', 'G[gr] ns (tau)'))
    plt.title('Non-stationary\ncross-correlations')
    
    fig.add_subplot(gs[0,2])
    plt.fill_between(ss.tau, ss.G[1,1,:,0]-ss.G[1,1,:,1], ss.G[1,1,:,0]+ss.G[1,1,:,1], facecolor='g', edgecolor=None, alpha=0.5)
    plt.fill_between(ss.tau, ss.Gns[1,1,:,0]-ss.Gns[1,1,:,1], ss.Gns[1,1,:,0]+ss.Gns[1,1,:,1], facecolor='k', edgecolor=None, alpha=0.5)
    g = plt.plot(ss.tau, ss.G[1,1,:,0], '-g')
    gns = plt.plot(ss.tau, ss.Gns[1,1,:,0], '-k')
    plt.xlim(0, xlim)
    plt.ylim(getylim((ss.G[1,1,:,0]-ss.G[1,1,:,1], ss.G[1,1,:,0]+ss.G[1,1,:,1],
                      ss.Gns[1,1,:,0]-ss.Gns[1,1,:,1], ss.Gns[1,1,:,0]+ss.Gns[1,1,:,1]),
                     4*(ss.tau,), (0, xlim)))
    plt.legend((g[0], gns[0]), ('G (tau)', 'G ns (tau)'))
    plt.title('Non-stationary\ngreen auto-correlation')
    
    fig.add_subplot(gs[1,0])
    plt.fill_between(ss.tau, ss.Gps[0,0,:,0]-ss.Gps[0,0,:,1], ss.Gps[0,0,:,0]+ss.Gps[0,0,:,1], facecolor='r', edgecolor=None, alpha=0.5)
    plt.plot(ss.tau, ss.Gps[0,0,:,0], '-r')
    plt.xlim(0, xlim)
    plt.ylim(getylim((ss.Gps[0,0,:,0]-ss.Gps[0,0,:,1], ss.Gps[0,0,:,0]+ss.Gps[0,0,:,1]), 2*(ss.tau,), (0, xlim)))
    plt.xlabel('time delay (s)')
    plt.ylabel(r'$\mathrm{G}(\tau)$')
    plt.title('Pseudo-stationary red')
    
    fig.add_subplot(gs[1,1])
    plt.fill_between(ss.tau, ss.Gps[0,1,:,0]-ss.Gps[0,1,:,1], ss.Gps[0,1,:,0]+ss.Gps[0,1,:,1], facecolor='b', edgecolor=None, alpha=0.5)
    plt.fill_between(-ss.tau, ss.Gps[1,0,:,0]-ss.Gps[1,0,:,1], ss.Gps[1,0,:,0]+ss.Gps[1,0,:,1], facecolor='y', edgecolor=None, alpha=0.5)
    plt.plot(ss.tau,ss.Gps[0,1,:,0], '-b')
    plt.plot(-ss.tau,ss.Gps[1,0,:,0], '-y')
    plt.xlim(-xlim, xlim)
    plt.ylim(getylim((ss.Gps[0,1,:,0]-ss.Gps[0,1,:,1], ss.Gps[0,1,:,0]+ss.Gps[0,1,:,1],
                      ss.Gps[1,0,:,0]-ss.Gps[1,0,:,1], ss.Gps[1,0,:,0]+ss.Gps[1,0,:,1]),
                     (ss.tau, ss.tau, -ss.tau, -ss.tau), (-xlim, xlim)))
    plt.xlabel('time delay (s)')
    plt.title('Pseudo-stationary cross-correlations')
    
    fig.add_subplot(gs[1,2])
    plt.fill_between(ss.tau, ss.Gps[1,1,:,0]-ss.Gps[1,1,:,1], ss.Gps[1,1,:,0]+ss.Gps[1,1,:,1], facecolor='g', edgecolor=None, alpha=0.5)
    plt.plot(ss.tau, ss.Gps[1,1,:,0], '-g')
    plt.xlim(0, xlim)
    plt.ylim(getylim((ss.Gps[1,1,:,0]-ss.Gps[1,1,:,1], ss.Gps[1,1,:,0]+ss.Gps[1,1,:,1]), 2*(ss.tau,), (0, xlim)))
    plt.xlabel('time delay (s)')
    plt.title('Pseudo-stationary green')

    if not pdf is None:
        pdf.savefig(fig)
        plt.close()
        
def showAutoCorr(color, ss, G, l1a, l1t, l2a, l2t, l1aSEM, l1tSEM):
    ### color should be "g" or "r"
    fig = plt.figure(figsize=(A4[0]/2, A4[1]/2))
    fill = plt.fill_between(ss.tau, G[:,0]-G[:,1], G[:,0]+G[:,1], facecolor=color, edgecolor=None, alpha=0.5)
    corr = plt.plot(ss.tau, G[:,0], '-'+color)
    xlim = ss.tau.max()/2

    x = np.linspace(ss.tau.min(), ss.tau.max(),int(ss.tau.max()))
    H2 = lambda x: -x*(x<0)
    if l1a != None and l1t != None and l2a != None and l2t != None and l1aSEM != None and l1tSEM != None:
        fit = plt.plot(x, l1a/l1t*H2(x-l1t) + l2a*(1-x/l2t), '-k')
        plt.plot(x, l2a*(1-x/l2t), '-k')
    plt.plot((0, xlim), (0,0), '--k')
    plt.xlabel('time delay (s)')
    plt.ylabel(r'$\mathrm{G}(\tau)$')
        
    plt.xlim(0, xlim)
    plt.ylim(-1,1.5*np.max(G[:,0][0]))
    if l1a != None and l1t != None and l2a != None and l2t != None and l1aSEM != None and l1tSEM != None:
        plt.legend((corr[0], fill, fit[0]), ('auto correlation', 'error', u'fit:\namplitude = {:.2f} \261 {:.2f}\ndwell time = {:.2f} \261 {:.2f} s'.format(l1a, l1aSEM, l1t, l1tSEM)))
    else:
        plt.legend((corr[0], fill ), ('auto correlation, error in fit', 'error'))
    plt.title('ACF '+color)
    return (plt)

def showAutoCorrZoom(color, ss, G, l1a, l1t, l2a, l2t, l1aSEM, l1tSEM,xmax):
    ### color should be "g" or "r"
    fig = plt.figure(figsize=(A4[0]/2, A4[1]/2))
    fill = plt.fill_between(ss.tau, G[:,0]-G[:,1], G[:,0]+G[:,1], facecolor=color, edgecolor=None, alpha=0.5)
    #corr = plt.plot(ss.tau, G[:,0], '-'+color)
    corr = plt.plot(ss.tau, G[:,0], 'o-'+color)

    x = np.linspace(ss.tau.min(), ss.tau.max(),int(ss.tau.max()))
    H2 = lambda x: -x*(x<0)
    if l1a != None and l1t != None and l2a != None and l2t != None and l1aSEM != None and l1tSEM != None:
        fit = plt.plot(x, l1a/l1t*H2(x-l1t) + l2a*(1-x/l2t), '-k')
        plt.plot(x, l2a*(1-x/l2t), '-k')
    plt.plot((0, xmax), (0,0), '--k')
    plt.xlabel('time delay (s)')
    plt.ylabel(r'$\mathrm{G}(\tau)$')
        
    plt.xlim(0, xmax)
    plt.ylim(-0.2,1.5*np.max(G[:,0][0]))
    if l1a != None and l1t != None and l2a != None and l2t != None and l1aSEM != None and l1tSEM != None:
        plt.legend((corr[0], fill, fit[0]), ('auto correlation', 'error', u'fit:\namplitude = {:.2f} \261 {:.2f}\ndwell time = {:.2f} \261 {:.2f} s'.format(l1a, l1aSEM, l1t, l1tSEM)))
    else:
        plt.legend((corr[0], fill ), ('auto correlation, error in fit', 'error'))
    plt.title('ACF '+color)
    return (plt)

 
def showCrossCorr(ss, G, G2, ylim=None, perr0=None, perr1=None, popt0=None, popt1=None, popt2=None, popt3=None, xlim=None):
    fig = plt.figure(figsize=(A4[0]/2, A4[1]/2))
    plt.fill_between(ss.tau, G[:,0]-G[:,1], G[:,0]+G[:,1], facecolor='b', edgecolor=None, alpha=0.5)
    plt.fill_between(-ss.tau[::-1], G2[:,0]-G2[:,1], G2[:,0]+G2[:,1], facecolor='y', edgecolor=None, alpha=0.5)

    grg = plt.plot(ss.tau,G[:,0], '-b')
    ggr = plt.plot(-ss.tau[::-1],G2[:,0], '-y')
#    xlim = ss.tau.max()/1.5
    if ylim is None:
        ylim = getylim((G[:,0]-G[:,1], G[:,0]+G[:,1], G2[:,0]-G2[:,1], G2[:,0]+G2[:,1]), (ss.tau, ss.tau, -ss.tau[::-1], -ss.tau[::-1]), xlim)
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim)
    
    def gauss_function(x,a,x0,sigma,b):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))+b
    
    if not perr0 is None:
        fit = gauss_function(np.hstack((-ss.tau[::-2],ss.tau)), popt0, popt1, popt2, popt3)
        fitp = plt.plot(np.hstack((-ss.tau[::-2],ss.tau)),fit, '-k' )
    plt.plot((xlim[0], xlim[1]), (0,0), '--k')
    plt.plot((0,0), (ylim), '--k')
    plt.xlabel('time delay (s)')
    plt.ylabel(r'$\mathrm{G}(\tau)$')
    if perr0 != None:
        plt.legend((grg[0], ggr[0], fitp[0]), ('G[rg] (tau)',  'G[gr] (tau)',  u'Gauss fit: \nmean = {:.2f} \261 {:.2f}\namplitude = {:.2f} \261 {:.2f}'.format(popt1,perr1,popt0,perr0)))
    else:
        plt.legend((grg[0], ggr[0], ), ('G[rg] (tau)', 'G[gr] (tau), error in fit'))
    plt.title('Cross correlation')
    return (plt)

### Display area under trace for all traces (this function can be helpful to see which traces dominate your autocorrelation function).
def showAreaUnderTraces(dataA, retainedTraces, color):
    # color should be "g" or "r"
    fig = plt.figure(figsize=(A4[0]/2, A4[1]/2))
          
    meanR = []
    meanRh = []
    for i in retainedTraces:
        if color == "g":
            meanR.append(dataA[i].g.mean())
            meanRh.append(1./dataA[i].g.mean())
        elif color == "r":
            meanR.append(dataA[i].r.mean())
            meanRh.append(1./dataA[i].r.mean())
    plt.plot(retainedTraces, meanR, color+'.')
    
    arith = plt.plot((0, len(retainedTraces)),(np.sum(meanR)/len(meanR),np.sum(meanR)/len(meanR)), '--', color = 'blue')
    harm = plt.plot((0, len(retainedTraces)),(len(meanRh)/np.sum(meanRh),len(meanRh)/np.sum(meanRh)), '--', color = 'gray')
    
    plt.legend((arith[0],harm[0]),('arithmetic mean', 'harmonic mean'))
    plt.title('Area under trace '+color)
    plt.xlabel('Trace nr')
    plt.ylabel('Mean of trace')
    return (fig)

#### Histograms of burst duration, code not tested
#def showBurstDurationHistogram(binSize=30,maxT=1200):
#    dt=dataOrig[0].dt*1.
#    burstTime=[concatenate([diff((lambda a: a[:(a.shape[0]/2)*2].reshape(-1,2))(where(abs(diff(dB.r)))[0][dB.r[0]:]),1) for dB in dataB]), concatenate([diff((lambda a: a[:(a.shape[0]/2)*2].reshape(-1,2))(where(abs(diff(dB.g[0:])))[0][dB.g[0]:]),1) for dB in dataB])]
#    hr=histogram(burstTime[0],bins=r_[:maxT/dt:binSize/dt]-.5,density=1);
#    hg=histogram(burstTime[1],bins=r_[:maxT/dt:binSize/dt]-.5,density=1);
#    fig = plt.figure()
#    plt.bar(hr[1][:-1]*dt, hr[0], '-r')
#    plt.bar(hg[1][:-1]*dt, hg[0], '-g')
#    plt.title('Burst duration after binary thresholding')
#    plt.xlabel('Burst duration (s)')
#    plt.ylabel('Frequency')
#    plt.legend((hr[0], hg[0]), ('red','green'))
#    return(fig)

   
def showHeatMap(data, maxRed=None, maxGreen=None, trimdata=None, sortedIds=None):
    fig = plt.figure(figsize=(A4[0], A4[1]))
    if maxRed == 'None' or isinstance(maxRed, type(None)): maxRed = np.nanmax(np.hstack([d.r for d in data]))
    if maxGreen == 'None' or isinstance(maxGreen, type(None)): maxGreen = np.nanmax(np.hstack([d.g for d in data]))
    if isinstance(sortedIds, type(None)): sortedIds=range(len(data))
    nbPts = np.max([d.r.shape[0] for d in data])
#    heatMap = array([r_[c_[data[i].r/maxRed,data[i].g/maxGreen,data[i].g*0],zeros((nbPts-data[i].t.shape[0],3))] for i in sortedIds]).clip(0,1);
    if isinstance(trimdata, type(None)):
        heatMap = np.array([np.r_[np.c_[data[i].r/maxRed,data[i].g/maxGreen,data[i].g*0],np.zeros((nbPts-data[i].t.shape[0],3))] for i in sortedIds]).clip(0,1);
    else: heatMap = np.array([np.r_[np.c_[data[i].r/maxRed,data[i].g/maxGreen,trimdata[i].g],np.zeros((nbPts-data[i].t.shape[0],3))] for i in sortedIds]).clip(0,1);
    if len(heatMap) != 0: plt.imshow(heatMap)
    lab = np.arange(0, len(data[0].t), step = 6)
    plt.xticks(lab, data[0].t[lab].astype('int16'), rotation = 90, fontsize = 6)
    plt.yticks(np.arange(len(sortedIds)), sortedIds, fontsize = 6)
    plt.xlabel('time (s)')
    plt.ylabel('experiment #')
    plt.tight_layout()
    return(plt)

def showHeatMapCF(data, maxRed=None, maxGreen=None, maxCF = None, sortedIds=None, Z = None, Normalize = False):
    fig = plt.figure(figsize=(A4[0], A4[1]))
    gs = GridSpec(1, 5, figure=fig)
    if isinstance(sortedIds, type(None)): sortedIds=range(len(data))

    fig.add_subplot(gs[0,0])
    maxRed = maxRed or np.nanmax(np.hstack([d.G[0,0] for d in data]))
    maxtaulen = np.max(np.hstack([len(d.tau) for d in data]))
    for m in range(len(data)):
        if len(data[m].tau) == maxtaulen:
            tau = data[m].tau
    nbPts = np.max([d.G[0,0].shape[0] for d in data])
    #heatMap = array([r_[c_[data[i].r/maxRed,data[i].g/maxGreen,data[i].g*0],zeros((nbPts-data[i].t.shape[0],3))] for i in sortedIds]).clip(0,1);
    if Normalize == False: heatMap = np.array([np.r_[np.c_[data[i].G[0,0]/maxRed],np.zeros((nbPts - data[i].G[0,0].shape[0],1))] for i in sortedIds]).clip(0,1);
    elif Normalize == True: heatMap = np.array([np.r_[np.c_[data[i].G[0,0]/data[i].G[0,0][0]],np.zeros((nbPts - data[i].G[0,0].shape[0],1))] for i in sortedIds]).clip(0,1);
    plt.imshow(heatMap[:,:,0])
    plt.xlabel('tau (s)')
    plt.ylabel('experiment #')
    lab = np.arange(0, len(tau), step = 2)
    plt.xticks(lab, tau[lab].astype('int16'), rotation = 90, fontsize = 6)
    plt.yticks(np.arange(len(sortedIds)), sortedIds, fontsize = 6)
    plt.title('Heatmap ACF red')
    
    fig.add_subplot(gs[0,1])
    maxGreen = maxGreen or np.nanmax(np.hstack([d.G[1,1] for d in data]))
    nbPts = np.max([d.G[1,1].shape[0] for d in data])
    if Normalize == False: heatMap = np.array([np.r_[np.c_[data[i].G[1,1]/maxGreen],np.zeros((nbPts - data[i].G[1,1].shape[0],1))] for i in sortedIds]).clip(0,1);
    elif Normalize == True: heatMap = np.array([np.r_[np.c_[data[i].G[1,1]/data[i].G[1,1][0]],np.zeros((nbPts - data[i].G[1,1].shape[0],1))] for i in sortedIds]).clip(0,1);
    plt.imshow(heatMap[:,:,0])
    plt.xlabel('tau (s)')
    plt.xticks(lab, tau[lab].astype('int16'), rotation = 90, fontsize = 6)
    plt.yticks(np.arange(len(sortedIds)), sortedIds, fontsize =6)
    plt.title('Heatmap ACF green')
 
    fig.add_subplot(gs[0,2:4])
    xx = []
    for x in range(len(data)):
        xx.extend(data[x].G[0,1])
        xx.extend(data[x].G[1,0])
    maxCF = maxCF or np.nanmax(xx)
    nbPts1 = np.max([d.G[1, 0][1:].shape[0] for d in data])
    nbPts2 = np.max([d.G[0, 1].shape[0] for d in data])
    if Normalize == False: heatMap = np.array([np.r_[np.zeros((nbPts1 - data[i].G[1, 0][::-1][:-1].shape[0],1)),np.c_[(np.hstack((data[i].G[1, 0][::-1][:-1], data[i].G[0, 1]))/maxCF)],np.zeros((nbPts2 - data[i].G[0, 1].shape[0],1))] for i in sortedIds]).clip(0,1);
    elif Normalize == True: heatMap = np.array([np.r_[np.zeros((nbPts1 - data[i].G[1, 0][::-1][:-1].shape[0],1)),np.c_[(np.hstack((data[i].G[1, 0][::-1][:-1], data[i].G[0, 1]))/max(np.hstack((data[i].G[1, 0][::-1][:-1], data[i].G[0, 1]))))],np.zeros((nbPts2 - data[i].G[0, 1].shape[0],1))] for i in sortedIds]).clip(0,1);
    plt.imshow(heatMap[:,:,0])
    plt.xlabel('tau (s)')
    plt.title('Heatmap crosscorrelation')
    lab2 = np.arange(0, np.hstack((-tau[::-1][:-1], tau)).shape[0], step = 2)
    plt.xticks(lab2, np.hstack((-tau[::-1][:-1], tau))[lab2].astype('int16'), rotation = 90, fontsize = 6)
    plt.yticks(np.arange(len(sortedIds)), sortedIds, fontsize = 6)
    
    if Z is not None:
        fig.add_subplot(gs[0,4])
        dendrogram(Z, orientation = "right")
        plt.xticks([])
    
    plt.tight_layout()
    return(plt)



# Function to align traces on the start of the first burst
def alignTraces(dataIn,startFrames):
    dataOut=copy2.deepcopy(dataIn);
    nbPts=np.max([d.r.shape[0] for d in dataIn])*2
    for i in np.r_[:len(dataIn)]:
        d=dataOut[i]; d.t=np.r_[:nbPts]*np.diff(d.t).mean() #d.t=r_[d.t,d.t+d.t[-1]+d.t[1]];
        d.r=np.roll(np.r_[d.r,np.zeros(nbPts-d.r.shape[0])],nbPts/2-startFrames[i])
        d.g=np.roll(np.r_[d.g,np.zeros(nbPts-d.g.shape[0])],nbPts/2-startFrames[i])
    return dataOut




def showCorrelFunAll(pdf, data, ChannelsToAnalyze, params):

    for channel in ChannelsToAnalyze:
        color = ["red", "green"]
        fig = plt.figure(figsize=A4)
        gs = GridSpec(4, 5, figure=fig)

        x, y = [], []
        for i in params['retainedTraces']:
            x.extend(data[i].tau[1:])
            y.extend(data[i].G[channel, channel, 1:])
        xlim = (0, params['IACxlim'][channel])
        ylim = np.clip(getylim(y, x, xlim), -0.2, np.inf)

        names = [data[i].name.split("/")[-1].split("_trk")[0] for i in params['retainedTraces']]

        # adds enter in name if it is longer than 19 characters (to fit in plot frame)
        make_enter = 0
        for pname in range(len(names)):
             for letter in range(len(names[pname])):
                 letter2 = letter % 20
                 if letter2 == 19:
                     make_enter += 1
                 if names[pname][letter] == "_" and make_enter > 0:
                     names[pname] = "\n".join([names[pname][:letter], names[pname][letter:]])
                     make_enter = 0


        for i, r in enumerate(params['retainedTraces']):
            idx = i%20
            if idx == 0:
                plt.suptitle('Autocorrelation functions ' + color[channel])
                if i>0:
                    pdf.savefig(fig)
                    fig = plt.figure(figsize=A4)
                    gs = GridSpec(4, 5, figure=fig)
            d = data[r]
            figall = fig.add_subplot(gs[idx//5,idx%5])
            if channel == 0: plt.plot(d.tau, d.G[0, 0], '-r')
            if channel == 1: plt.plot(d.tau, d.G[1, 1], '-g')
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.text(np.mean(xlim), 0.98*ylim[1], names[i], fontsize = 6,horizontalalignment='center', verticalalignment = "top" )
            figall.set_xlabel('tau')
            figall.set_ylabel('G(tau)')
            for figall in fig.get_axes():
                figall.label_outer()
        pdf.savefig(fig)
        plt.close()

    if len(ChannelsToAnalyze) == 2:
        
        fig = plt.figure(figsize=A4)
        gs = GridSpec(4, 5, figure=fig)

        x, y = [], []
        for i in params['retainedTraces']:
            for _ in range(2):
                x.extend(data[i].tau)
            y.extend(data[i].G[0,1])
            y.extend(data[i].G[1,0])
        xlim = params['ICCxlim']
        ylim = getylim(y, x, xlim)

        for i, r in enumerate(params['retainedTraces']):
            idx = i%20
            if idx == 0:
                plt.suptitle('Crosscorrelation functions')
                if i>0:
                    pdf.savefig(fig)
                    fig = plt.figure(figsize=A4)
                    gs = GridSpec(4, 5, figure=fig)
            d = data[r]
            figall = fig.add_subplot(gs[idx//5,idx%5])
            plt.plot(d.tau, d.G[0, 1], '-b')
            plt.plot(-d.tau[::-1], d.G[1, 0][::-1], '-y')
            plt.plot((0,0), (ylim), '--k', linewidth=0.5)
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.text(np.mean(xlim), 0.98*ylim[1], names[i], fontsize = 6, horizontalalignment='center', verticalalignment = "top")

            figall.set_xlabel('tau')
            figall.set_ylabel('G(tau)')
            for figall in fig.get_axes():
                figall.label_outer()
    
        pdf.savefig(fig)
        plt.close()

#function to calculate bootstrap errors for non-normal distributions
def CalcBootstrap(input, repeats, samplesize=None):
    if samplesize is None:
        samplesize = len(input)
    if len(input) == 0:
        return np.nan, np.nan
    else:
       bootmeans = []
       for i in range(repeats):
           bootsample = np.random.choice(input, size=samplesize, replace=True)
           bootmeans.append(np.mean(bootsample))
       bootstrapmean = np.mean(bootmeans)
       bootstrapSD = np.std(bootmeans)
       return bootstrapmean, bootstrapSD

def HistogramPlot(input, nbrbins, titletext, xaxlabel, outname):
    hist = np.histogram(input, bins = nbrbins, density = True)

    binedges = hist[1]
    plotbins = []
    for i in range(0,len(binedges)-1):
        plotbins.append((0.5*(binedges[i+1]+binedges[i])))

    binwidth = plotbins[1]-plotbins[0]
    fig = plt.figure(figsize=A4)
    plt.bar(plotbins,hist[0],width = 0.8*binwidth)
    plt.title(titletext)
    plt.xlabel(xaxlabel)
    plt.ylabel('Frequency')
    if len(input) == 0:
        stats = [np.nan,np.nan]
    else:
        stats = CalcBootstrap(input, 1000)
    meanvalue = stats[0]
    errorvalue = stats[1]
    plt.text(0.5*np.max(plotbins),0.8*np.max(hist[0]),'Mean: '+str(round(meanvalue,2))+'+/- '+str(round(errorvalue,2)))
    fig.savefig(outname+'.pdf')
    plt.close()

def CumHistogramPlot(input, titletext, xaxlabel, outname):
    sortedvals = np.sort(input)
    xToPlot = []
    yToPlot = []
    if len(input) == 0:
        xToPlot = 0
        yToPlot = 0
    else:
        for i in range(0,len(sortedvals)):
            xToPlot.append(sortedvals[i])
            yToPlot.append(i+1)
    fig = plt.figure(figsize=A4)
    plt.plot(xToPlot,yToPlot)
    plt.title(titletext)
    plt.xlabel(xaxlabel)
    plt.ylabel('Frequency')
    stats = CalcBootstrap(input, 1000)
    meanvalue = stats[0]
    errorvalue = stats[1]
    plt.text(0.5*np.max(xToPlot),0.8*np.max(yToPlot),'Mean: '+str(round(meanvalue,2))+'+/- '+str(round(errorvalue,2)))
    fig.savefig(outname+'.pdf')
    plt.close()

def PlotIntensities(pdf, dataOrig, dataA, dataB, params, color, outfile):
    nonbgcorrints = []
    bgcorrintsOn = []
    bgcorrintsOff = []
    bgcorrintsAll = []
    for cell in params['retainedTraces']:
        if params['alignTracesCF'] == 0:
            start = dataB[cell].frameWindow[0]
            if np.sum(dataB[cell].r) == 0 and np.sum(dataB[cell].g) == 0: continue
        if params['alignTracesCF'] == 1:
            if params['color2align'] == 'red':
                if np.sum(dataB[cell].r) == 0: continue
                start = np.where(dataB[cell].r)[0][0]
            if params['color2align'] == 'green':
                if np.sum(dataB[cell].g) == 0: continue
                start = np.where(dataB[cell].g)[0][0]

        for val in range(start, dataB[cell].frameWindow[1]):
            nonbgcorrints.append(dataOrig[cell][color][val])
            bgcorrintsAll.append(dataA[cell][color][val])
            if dataB[cell][color][val] == 1:
                bgcorrintsOn.append(dataA[cell][color][val])
            elif dataB[cell][color][val] == 0:
                bgcorrintsOff.append(dataA[cell][color][val])

    np.save(outfile, bgcorrintsOn)
    histInts = np.histogram(nonbgcorrints, bins=100, density=True)
    histIntsbg = np.histogram(bgcorrintsAll, bins=100, density=True)
    histIntsbgOn = np.histogram(bgcorrintsOn, bins=histIntsbg[1], density=True)
    histIntsbgOff = np.histogram(bgcorrintsOff, bins=histIntsbg[1], density=True)
    binedges = histInts[1]
    binedgesbg = histIntsbg[1]
    plotbins = []
    plotbinsbg = []
    for i in range(0, len(binedges) - 1):
        plotbins.append(0.5 * (binedges[i + 1] + binedges[i]))
    for i in range(0, len(binedgesbg) - 1):
        plotbinsbg.append(0.5 * (binedgesbg[i + 1] + binedgesbg[i]))
    binwidth = plotbins[1] - plotbins[0]
    binwidthbg = plotbinsbg[1] - plotbinsbg[0]

    statsInts = CalcBootstrap(nonbgcorrints, 1000)
    statsIntsbg = CalcBootstrap(bgcorrintsAll, 1000)
    statsIntsbgOn = CalcBootstrap(bgcorrintsOn, 1000)
    statsIntsbgOff = CalcBootstrap(bgcorrintsOff, 1000)

    fig = plt.figure(figsize=A4)
    gs = GridSpec(1, 2, figure=fig)

    fig.add_subplot(gs[0,0])
    labelInts = 'Mean: '+str(round(statsInts[0],2))+'+/- '+str(round(statsInts[1], 2))
    plt.bar(np.asarray(plotbins), histInts[0], width=0.8 * binwidth, label = labelInts)
    plt.title('Histogram of non bg corrected intensity values ' + color)
    plt.xlabel('Intensity value (AU)')
    plt.ylabel('Frequency')
    plt.legend(loc = 'upper right')

    fig.add_subplot(gs[0,1])
    labelIntsbg  = 'All; mean: '+str(round(statsIntsbg[0],2))+'+/- '+str(round(statsIntsbg[1], 2))
    labelIntsbgOn = 'On; mean: '+str(round(statsIntsbgOn[0],2))+'+/- '+str(round(statsIntsbgOn[1], 2))
    labelIntsbgOff = 'Off; mean: '+str(round(statsIntsbgOff[0],2))+'+/- '+str(round(statsIntsbgOff[1], 2))
    if len(bgcorrintsAll) != 0:
        plt.bar(np.asarray(plotbinsbg), histIntsbg[0], width=0.8 * binwidthbg, label = labelIntsbg, alpha = 0.3)
        plt.bar(np.asarray(plotbinsbg), histIntsbgOn[0]*(len(bgcorrintsOn)*1./len(bgcorrintsAll)), width=0.8 * binwidthbg, alpha = 0.6, label =labelIntsbgOn)
        plt.bar(np.asarray(plotbinsbg), histIntsbgOff[0]*(len(bgcorrintsOff)*1./len(bgcorrintsAll)), width=0.8 * binwidthbg, alpha = 0.6, label=labelIntsbgOff)

    plt.title('Histogram of bg corrected intensity values ' + color)
    plt.xlabel('Intensity value (AU)')
    plt.ylabel('Frequency')
    plt.legend(loc = 'upper right')

    if not pdf is None:
        pdf.savefig(fig)
    else:
        return fig

def PlotDistances(pdf, dataA, retainedTraces):
    distAll = []
    distOn = []
    for cell in retainedTraces:
        data = dataA[cell]
        minframe = dataA[cell].frameWindow[0]
        maxframe = dataA[cell].frameWindow[1]
        xyRed = data.trk_r[:,:2]
        xyGreen = data.trk_g[:,:2]
        data.spotFoundBoth = list(set(np.where(data.trk_r[:,-1]>0)[0])& set( np.where(data.trk_g[:,-1]>0)[0]))
        data.spotFoundBoth = [val for val in data.spotFoundBoth if val >= minframe and val <= maxframe]
        data.spotFoundBoth.sort()
        dist0 = [0] * len(xyRed)
        for a in range(minframe,maxframe+1):
            dist0[a] = (((xyRed[a,0]-xyGreen[a,0])**2 + (xyRed[a,1]-xyGreen[a,1])**2)**0.5)
        distAll.append(dist0)
        dist0 = [0] * len(xyRed)
        for a in data.spotFoundBoth:
            dist0[a] = (((xyRed[a,0]-xyGreen[a,0])**2 + (xyRed[a,1]-xyGreen[a,1])**2)**0.5)
        distOn.append(dist0)
    
    distAll = [l for lst in distAll for l in lst] #flatten multidimensional list to 1D
    distOn = [l for lst in distOn for l in lst]
    distAll = list(filter(lambda a: a != 0, distAll)) #remove zeroes
    distOn = list(filter(lambda a: a != 0, distOn))

    histdistAll = np.histogram(distAll, bins=100, density=True)
    histdistOn = np.histogram(distOn, bins=histdistAll[1], density=True)
    binedges = histdistAll[1]
    plotbins = []
    for i in range(0, len(binedges) - 1):
        plotbins.append(0.5 * (binedges[i + 1] + binedges[i]))
    binwidth = plotbins[1] - plotbins[0]
    
    statsdistAll = CalcBootstrap(distAll, 1000)
    statsdistOn = CalcBootstrap(distOn, 1000)
    
    fig = plt.figure(figsize=(5,6))
    gs = GridSpec(1, 1, figure=fig)
    
    fig.add_subplot(gs[0,0])
    labeldistAll  = 'All; mean: '+str(round(statsdistAll[0],2))+'+/- '+str(round(statsdistAll[1], 2))
    labeldistOn = 'On; mean: '+str(round(statsdistOn[0],2))+'+/- '+str(round(statsdistOn[1], 2))
    plt.bar(np.asarray(plotbins), histdistAll[0], width=0.8 * binwidth, label = labeldistAll, alpha = 0.3)
    plt.bar(np.asarray(plotbins), histdistOn[0]*(len(distOn)*1./len(distAll)), width=0.8 * binwidth, alpha = 0.6, label =labeldistOn)
    
    plt.title('Histogram of distances between alleles ')
    plt.xlabel('Distance (pixels)')
    plt.ylabel('Frequency')
    plt.legend(loc = 'upper right')
    
    if not pdf is None:
        pdf.savefig(fig)
    else:
        return fig

def corrACFAmplToIndPlot(dataA, dataB, col, params):
    amplAfterInd = []
    indTimes = []
    nrcells = len(dataA)
    for i in range(0, nrcells):
        # read data, digital data and framewindow
        dataCell = dataA[i]
        dataDig = dataB[i]
        fW = dataCell.frameWindow

        # get induction frame of this cell
        if col == 'red':
            bindata = dataDig.r
        elif col == 'green':
            bindata = dataDig.g
        if sum(bindata) != 0:
            indframe = np.where(bindata > 0)[0][0]
        else:
            indframe = fW[1]

        if indframe < fW[0]: indframe = fW[0]

        if indframe > fW[1] - 2: continue

        CFAfterInd = daf.compG_multiTau(np.c_[dataA[i].r, dataA[i].g][indframe:fW[1]].T, dataA[i].t[indframe:fW[1]], 8)[0][0]
        if col == 'red':
            ACFAfterInd = CFAfterInd[0]
        elif col == 'green':
            ACFAfterInd = CFAfterInd[1]

        ampl = ACFAfterInd[1]
        amplAfterInd.append(ampl)
        indTimes.append(indframe * dataCell.dt)

    fig = plt.figure()
    plt.scatter(indTimes, amplAfterInd)
    plt.xlabel('Induction time (s)')
    plt.ylabel('Amplitude ACF (AU)')
    plt.title('Correlation of ACF amplitude with induction time')
    fig.savefig(params['file'] + '_correlation_induction_time_ACF_amplitude_' + col + '.pdf')
    plt.close()
