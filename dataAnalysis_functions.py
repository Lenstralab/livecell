import re, imp, os, sys, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm.auto import trange

if __package__ is None or __package__=='': #usual case
    import misc
    from wimread import imread as imr
    from tiffwrite import IJTiffWriterMulti
else: #in case you do from another package: from LiveCellAnalysis import dataAnalysis_functions
    from . import misc
    from .wimread import imread as imr
    from .tiffwrite import IJTiffWriterMulti

if 'verbose' not in globals(): verbose=1;


##################################
### Numerical computations

### Compute all crosscorrelations G
"""Antoine Coulon, 2020. <antoine.coulon@curie.fr>"""
def compG_multiTau(v,t,n=8,ctr=1):
    """v: data vector (channels=rows), t: time, n: bin every n steps.\n--> Matrix of G, time vector"""
    def compInd(v1,v2):
        if len(t)<2:
            return np.array([[], []]).T
        tau=[]; G=[]; t0=t*1.; i=0; dt=t0[1]-t0[0]
        while i<t0.shape[0]:
            tau.append(i*dt); G.append(np.mean(v1[:int(v1.shape[0]-i)]*v2[int(i):]))
            if i==n:
                i=i/2
                dt*=2
                t0,v1,v2=np.c_[t0,v1,v2][:int(t0.shape[0]/2)*2].T.reshape(3,-1,2).mean(2)
            i+=1
        return np.array([tau,G]).T
    if ctr: vCtr=((v.T-np.mean(v,1)).T);
    else: vCtr=v
    res=np.array([[ compInd(v1,v2) for v2 in vCtr] for v1 in vCtr])
    return ( res[:,:,:,1].T /(np.dot(np.mean(v,1).reshape(-1,1),np.mean(v,1).reshape(1,-1)))).T, res[0,0,:,0]

#################################
#### Read experimental data

# Read head of metadata file
def readMetadata(pathToFile):
    if pathToFile[-3:] == 'czi':
        with imr(pathToFile) as im:
            return misc.objFromDict(Interval_ms=im.timeinterval*1000)
    with open(pathToFile, 'r') as f:
        content = f.read(-1)
    if len(content)>5:
        # fix missing }
        if content[-2:] == ',\n':
            content = content[:-2]
        elif content[-1] == ',':
            content = content[:-1]
        lines = content.replace('\\\"', '').splitlines() #2. but ignore \"
        lines = ''.join([re.sub('\"[^\"]*\"', '', l) for l in lines]) #1. exclude {} between ""
        content += '}' * (lines.count('{') - lines.count('}'))
        tmpMD = json.loads(content)
        if 'Summary' in tmpMD:
            return misc.objFromDict(**tmpMD['Summary'])
    return misc.objFromDict(**{})

def loadExpData(fn, nMultiTau=8):
    global poolName
    if fn[-3:]=='.py': fn=fn[:-3]
    if fn[-5:]!='.list': fn=fn+'.list'
    poolName=fn[:-5]
    fn=os.path.expanduser(fn)
    if fn in sys.modules: imp.reload(sys.modules[fn])
    inFiles=imp.load_source('',fn+'.py')
    lf=[misc.objFromDict(**a) for a in inFiles.listFiles]
    data=[]
    if verbose: (sys.stdout.write('** Loading experimental data:\n'),sys.stdout.flush());
    for i in range(len(lf)):
        a=lf[i]
        if verbose: (sys.stdout.write('     file %d/%d "%s..."\n'%(i+1,len(lf),a.trk_r.replace('_green.trk',''))),sys.stdout.flush());
        data.append(misc.objFromDict(**{}))
        #    data[-1].path=procDataPath
        if 'trk_r' in a:  data[-1].trk_r=np.loadtxt(a.trk_r)
        if 'trk_g' in a:  data[-1].trk_g=np.loadtxt(a.trk_g)

        if 'detr' in a: # Columns of detr: frame, red mean, red sd, green mean, green sd, red correction raw, red correction polyfit, green correction raw, green correction polyfit
            data[-1].detr=np.loadtxt(a.detr)
            rn=data[-1].detr[:,2]/data[-1].detr[0,2]; x=np.where(abs(np.diff(rn))<.1)[0]; pf=np.polyfit(x,np.log(rn[x]),8)
            rf=np.exp(np.sum([data[-1].detr[:,0]**ii*pf[-1-ii] for ii in range(len(pf))],0))
            gn=data[-1].detr[:,4]/data[-1].detr[0,4]; x=np.where(abs(np.diff(gn))<.1)[0]; pf=np.polyfit(x,np.log(gn[x]),8)
            gf=np.exp(np.sum([data[-1].detr[:,0]**ii*pf[-1-ii] for ii in range(len(pf))],0))
            data[-1].detr=np.c_[data[-1].detr,rn,rf,gn,gf]

        if 'frameWindow' in a:  data[-1].frameWindow=a.frameWindow
        if 'actualDt' in a: data[-1].actualDt=data[-1].dt=a.actualDt
        else: data[-1].dt=0
        if 'hrsTreat' in a: data[-1].hrsTreat=a.hrsTreat

        if 'rawPath' in a: data[-1].rawPath=a.rawPath
        if 'rawTrans' in a: data[-1].rawTrans=a.rawTrans

        if 'fcs_rr' in a: data[-1].fcs_rr=np.loadtxt(a.fcs_rr,skiprows=7)
        if 'fcs_gg' in a: data[-1].fcs_gg=np.loadtxt(a.fcs_gg,skiprows=7)
        if 'fcs_rg' in a: data[-1].fcs_rg=np.loadtxt(a.fcs_rg,skiprows=7)
        if 'fcs_gr' in a: data[-1].fcs_gr=np.loadtxt(a.fcs_gr,skiprows=7)

        if 'trk_r' in a:  data[-1].name=a.trk_r.replace('_red.txt','').replace('_green.txt','').replace('.txt','')
        if 'trk_g' in a:  data[-1].name=a.trk_g.replace('_green.txt','').replace('_red.txt','').replace('.txt','')

        if 'ctrlOffset' in a:  data[-1].ctrlOffset=np.array(a.ctrlOffset)
        if 'transfLev' in a:  data[-1].transfLev=np.array(a.transfLev)

        if 'maxProj' in a:
            data[-1].maxProj=a.maxProj
            if not os.path.exists(data[-1].maxProj): print("!! Warning: file '%s' does not exist."%(a.maxProj))

        if 'metadata' in a:
            data[-1].metadata=readMetadata(a.metadata)
            if data[-1].dt==0: data[-1].dt=data[-1].metadata.Interval_ms/1000.
            #else: print "Using provided dt=%fs, not %fs from metadata."%(data[-1].dt,data[-1].metadata.Interval_ms/1000.)
        elif 'timeInterval' in a:
            data[-1].dt=int(float(a.timeInterval))
        else:
            print("!! Warning: No metadata and no dt provided. Using dt=1."); data[-1].dt=1.

        data[i].t=data[i].trk_r[:,3]*data[-1].dt
        data[i].r=data[i].trk_r[:,2]
        data[i].g=data[i].trk_g[:,2]
        if 'detr' in data[i]: # Detrending from s.d. polyfit
            data[i].r=data[i].r/data[i].detr[:,6]
            data[i].g=data[i].g/data[i].detr[:,8]

        if not 'frameWindow' in data[-1]: data[-1].frameWindow=[0,data[-1].t.shape[0]]

    if verbose: (sys.stdout.write('     Done.\n'),sys.stdout.flush());

    # Recompute correlations
    if nMultiTau!=0:
        if verbose: (sys.stdout.write('** Recomputing correlations functions... '),sys.stdout.flush());
        for d in data:
            if d.frameWindow[1]-d.frameWindow[0]:
                d.fcsRecomp=True
                d.G,d.tau=compG_multiTau(np.c_[d.r,d.g][d.frameWindow[0]:d.frameWindow[1]].T,d.t[d.frameWindow[0]:d.frameWindow[1]],nMultiTau)
                # Write .fcs4 files
                fn=re.escape(d.name+'.fcs4')
                np.savetxt('tmp.txt',np.c_[d.tau,d.G[0,0],d.G[1,1],d.G[0,1],d.G[1,0]],'%12.5e  ')
                os.system('echo "#Tau (in s)     Grr            Ggg            Grg            Ggr" > '+fn+'; cat tmp.txt >> '+fn)
            else:
                d.fcsRecomp = False
                d.G = np.zeros((2,2,0))
                d.tau = np.zeros(0)
        if verbose: (sys.stdout.write('Done.\n'),sys.stdout.flush());
    else:
        for i in range(len(data)):
            data[i].fcsRecomp=False
            data[i].tau=data[i].fcs_rr[:,0]*data[i].dt/data[i].fcs_rr[0,0]
            data[i].G=np.array([[data[i].fcs_rr[:,1],data[i].fcs_rg[:,1]],[data[i].fcs_gr[:,1],data[i].fcs_gg[:,1]]])

    return data




#################
### Displays

def showData(data, *args, **kwargs):
    if isinstance(data, misc.objFromDict):
        data = [data]
    d = data[0]
    fig = plt.figure(figsize=(11.69, 8.27))
    gs = GridSpec(2, 3, figure=fig)

    fig.add_subplot(gs[0,:2])
    plt.plot(d.t, d.g/d.g.max(), 'g')
    plt.plot(d.t, d.r/d.r.max(), 'r')
    plt.plot((0, d.t.max()), (0, 0), '--', color='gray')
    plt.xlim(0, d.t.max())
    plt.ylim(-0.5, 1.1)
    plt.xlabel('time (s)')
    plt.ylabel('fluorescence (AU)')

    fig.add_subplot(gs[0,2])
    colors = ['-or', '-og', '-ob', '-oy']
    for i in range(4):
        a = np.c_[d.tau, d.G[i%2, int(i%3!=0)]]
        plt.plot(a[:,0], a[:,1], colors[i])
    plt.legend(('G_rr(t)', 'G_gg(t)', 'G_rg(t)', 'G_gr(t)'))
    if len(d.tau):
        plt.plot((0, d.tau.max()), (0, 0), '--', color='gray')
        plt.xlim(0, d.tau.max())
    plt.xlabel('time lag (s)')
    plt.ylabel('G(t)')

    fig.add_subplot(gs[1,:2])
    plt.plot(d.t, np.sqrt((d.trk_r[:,0]-d.trk_g[:,0])**2+(d.trk_r[:,1]-d.trk_g[:,1])**2)+10, 'k')
    plt.plot(d.t[1:], np.sqrt(np.diff(d.trk_r[:,0])**2+np.diff(d.trk_r[:,1])**2)+5, 'r')
    plt.plot(d.t[1:], np.sqrt(np.diff(d.trk_g[:,0])**2+np.diff(d.trk_g[:,1])**2), 'g')
    plt.plot((0, d.t.max()), (0, 0), '--', color='gray')
    plt.plot((0, d.t.max()), (5, 5), '--', color='gray')
    plt.plot((0, d.t.max()), (10, 10), '--', color='gray')
    plt.xlim(0, d.t.max())
    plt.xlabel('time (s)')
    plt.ylabel('distance (pixels)')

    fig.add_subplot(gs[1,2])
    plt.plot(d.trk_r[:,0], d.trk_r[:,1], 'r')
    plt.plot(d.trk_g[:,0], d.trk_g[:,1], 'g')
    plt.gca().invert_yaxis()
    #fig.subtitle(d.name)
    plt.tight_layout()
    return fig


def showTracking(Data, channels=None, expPathOut=None, sideViews=None, zSlices=None, frameRange=None, rgb=False, transform=False):
    """ saves tiff with localisations
        data:       one item of data as loaded by LoadExp
        channels:   which channels to consider, [0] for 1 color,
                     [0,1] or [1,0] for 2 color,                   default: all channels
        pathOut:    in which path to save the tiff                 default: path of raw data
        cell:       cell number for inclusion in filename,         default: 0
        sideViews:  True/False: yes/no,                            default: True if raw is z-stack
        zSlices:    list: which zSlices to take from the stack,    default: all slices
        frameRange: (start, end): which frames (time) to include,  default: all frames
        rgb:        True: data is saved in 8bit format and will
                     be opened by Fiji as rgb image
                    False: data is saved in 16bit format           default: False

        wp@tl20200310
    """
    # add squares around the spots in seperate channel (from empty image)
    squareStamp = [
        np.r_[-5, -5, -5, -5, -5, -5, -5, -5, -4, -3, -2, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 4, 3, 2, -2, -3, -4],
        np.r_[-5, -4, -3, -2, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 4, 3, 2, -2, -3, -4, -5, -5, -5, -5, -5, -5, -5]]
    nbPixPerZ = 2  # nr of pixels per z stack
    pathIn = Data[0].rawPath

    expPathOut = expPathOut or pathIn

    DataXY = [[data['trk_' + c][:, :2] for c in 'rgb' if 'trk_' + c in data] for data in Data]
    D = [(data, dataXY) for data, dataXY in zip(Data, DataXY) if np.sum(dataXY)]
    if not D:
        print('Warning: no localisations found!')
        return
    Data, DataXY = [d for d in zip(*D)]

    Cell = [int(re.findall('(?<=cellnr_)\d+', data.name)[0]) for data in Data]

    Outfile = [expPathOut + "_cellnr_" + str(cell) + "_track" + "Side" * sideViews + ".tif" for cell in Cell]

    if os.path.exists(Data[0].maxProj):
        maxFile = Data[0].maxProj
        maxFileExists = True
    else:
        maxFile = np.zeros(0)
        maxFileExists = False

    with imr(pathIn, transform=transform) as raw, imr(maxFile, transform=transform) as mx:
        mx.masterch = raw.masterch
        mx.slavech  = raw.slavech
        mx.detector = raw.detector

        channels = channels or np.arange(raw.shape[2])
        nCh = min(len(channels), 3)
        if sideViews is None:
            sideViews = raw.zstack
        if zSlices is None:
            zSlices = np.arange(raw.shape[3])
        else:
            try:
                zSlices = np.arange(zSlices)
            except:
                pass

        frameRange = frameRange or (0, raw.shape[4])
        # nbPixPerZ = int(np.round(raw.deltaz/raw.pxsize))
        shape = (1, 1, 4 * (frameRange[1] - frameRange[0])) if rgb else (nCh + 1, 1, frameRange[1] - frameRange[0])

        with IJTiffWriterMulti(Outfile, [shape]*len(Outfile)) as out:
            if rgb:
                depth = 4
            else:
                depth = nCh + 1

            Box = []
            Loc_xy = []
            Width = []
            Height = []
            for dataXY in DataXY:
                box = np.hstack((np.floor(np.vstack(dataXY).min(0)),
                                 np.ceil(np.vstack(dataXY).max(0)))).astype('int') + [-20, -20, 20, 20]
                box = np.maximum(np.minimum(box * [1, 1, -1, -1], (box.reshape(2, 2).mean(0).repeat(2).reshape(2, 2).astype(int)
                                + [-50, 50]).T.flatten() * [1, 1, -1, -1]), -np.array((0, 0) + raw.shape[:2])) * [1, 1, -1, -1]
                loc_xy = [np.round(d - box[:2] + 0.5).astype('int') for d in dataXY]
                width = (box[2] - box[0]) + sideViews * len(zSlices) * nbPixPerZ
                height = (box[3] - box[1]) + sideViews * len(zSlices) * nbPixPerZ

                Box.append(box)
                Loc_xy.append(loc_xy)
                Width.append(width)
                Height.append(height)

            Minimum = len(Box) * [[np.inf, np.inf]]
            Maximum = len(Box) * [[0, 0]]
            if rgb:
                for t in trange(*frameRange, desc='Calculating dynamic range.'):
                    if maxFileExists and not sideViews:
                        CroppedIm = [mx(c, 0, t) for c in channels]
                    else:
                        CroppedIm = [raw[c, zSlices, t] for c in channels]
                    for i, box in enumerate(Box):
                        croppedIm = [im[box[1]:box[3], box[0]:box[2], ...].flatten() for im in CroppedIm]
                        Maximum[i] = [np.max((im.max(), m)) for im, m in zip(croppedIm, Maximum[i])]
                        Minimum[i] = [np.min((im.min(), m)) for im, m in zip(croppedIm, Minimum[i])]

            for t in trange(*frameRange, desc='Saving tracking tiffs'):
                if maxFileExists and not sideViews:
                    CroppedIm = [mx(c, 0, t) for c in channels]
                else:
                    CroppedIm = [raw[c, zSlices, t] for c in channels]

                for outfile, box, loc_xy, width, height, minimum, maximum in zip(Outfile, Box, Loc_xy, Width, Height, Minimum, Maximum):
                    frame = np.zeros((height, (len(channels) + (nCh > 1)) * width, depth), 'uint16')
                    for i in range(len(channels)):
                        if sideViews:
                            # Make xz and yz projection images. Projects 11 pixels around spot
                            croppedIm = CroppedIm[i][box[1]:box[3], box[0]:box[2], ...]
                            xyIm = np.nanmax(croppedIm, 3).squeeze((2, 3))
                            xzIm = np.nanmax(croppedIm, 0).squeeze((1, 3)).repeat(nbPixPerZ, 1).T
                            yzIm = np.nanmax(croppedIm, 1).squeeze((1, 3)).repeat(nbPixPerZ, 1)

                            # Make blank square for right bottom corner
                            blankSq = np.ones((xzIm.shape[0], yzIm.shape[1])) * np.mean(xyIm)
                            im = np.vstack((np.hstack((xyIm, yzIm)), np.hstack((xzIm, blankSq))))
                        elif maxFileExists:
                            im = CroppedIm[i][box[1]:box[3], box[0]:box[2]]
                        else:
                            croppedIm = CroppedIm[i][box[1]:box[3], box[0]:box[2], ...]
                            im = np.nanmax(croppedIm, 3).squeeze((2, 3))

                        frame[:, i * width:(i + 1) * width, i] = im.astype('uint16')
                        try:
                            if nCh > 1:
                                frame[:, -width:, i] = im.astype('uint16')
                                frame[loc_xy[i][t, 1] + squareStamp[1], loc_xy[i][t, 0] + squareStamp[0] + nCh * width, -1] = 65535
                            frame[loc_xy[i][t, 1] + squareStamp[1], loc_xy[i][t, 0] + squareStamp[0] + i * width, -1] = 65535
                        except:
                            pass

                    if rgb:
                        # reduce to 8bit
                        frame = frame.astype('float')
                        for c in range(nCh):
                            frame[:, :, c] = 255*(frame[:, :, c]-minimum[c])/(maximum[c]-minimum[c])
                        out.save(outfile, np.clip(frame, 0, 255).astype('uint8'), 0, 0, 4 * t)
                    else:
                        for c in range(nCh + 1):
                            out.save(outfile, frame[:, :, c], c, 0, t)


macro1color="""open("__imPath__");
    run("Stack to Hyperstack...", "order=xyczt(default) channels=2 slices=1 frames=__frames__ display=Color");
    Stack.setDisplayMode("color");
    Stack.setChannel(2);
    AUTO_THRESHOLD = 5000;
    getRawStatistics(pixcount);
    limit = pixcount/10;
    threshold = pixcount/AUTO_THRESHOLD;
    nBins = 256;
    getHistogram(values, histA, nBins);
    i = -1;
    found = false;
    do {
    counts = histA[++i];
    if (counts > limit) counts = 0;
    found = counts > threshold;
    } while ((!found) && (i < histA.length-1))
    hmin = values[i];
    i = histA.length;
    do {
    counts = histA[--i];
    if (counts > limit) counts = 0;
    found = counts > threshold;
    } while ((!found) && (i > 0))
    hmax = values[i];
    setMinAndMax(hmin, hmax);
    //print(hmin, hmax);
    Stack.setChannel(1);
    AUTO_THRESHOLD = 5000;
    getRawStatistics(pixcount);
    limit = pixcount/10;
    threshold = pixcount/AUTO_THRESHOLD;
    nBins = 256;
    getHistogram(values, histA, nBins);
    i = -1;
    found = false;
    do {
    counts = histA[++i];
    if (counts > limit) counts = 0;
    found = counts > threshold;
    } while ((!found) && (i < histA.length-1))
    hmin = values[i];
    i = histA.length;
    do {
    counts = histA[--i];
    if (counts > limit) counts = 0;
    found = counts > threshold;
    } while ((!found) && (i > 0))
    hmax = values[i];
    setMinAndMax(440, hmax+20);
    Stack.setDisplayMode("composite");
    run("Save");
    run("Quit");"""


macro2color ="""open("__imPath__");
    run("Make Composite");
    Stack.setDisplayMode("color");
    Stack.setChannel(2);
    AUTO_THRESHOLD = 5000;
    getRawStatistics(pixcount);
    limit = pixcount/10;
    threshold = pixcount/AUTO_THRESHOLD;
    nBins = 256;
    getHistogram(values, histA, nBins);
    i = -1;
    found = false;
    do {
    counts = histA[++i];
    if (counts > limit) counts = 0;
    found = counts > threshold;
    } while ((!found) && (i < histA.length-1))
    hmin = values[i];
    i = histA.length;
    do {
    counts = histA[--i];
    if (counts > limit) counts = 0;
    found = counts > threshold;
    } while ((!found) && (i > 0))
    hmax = values[i];
    setMinAndMax(hmin, hmax);
    //print(hmin, hmax);
    Stack.setChannel(1);
    AUTO_THRESHOLD = 5000;
    getRawStatistics(pixcount);
    limit = pixcount/10;
    threshold = pixcount/AUTO_THRESHOLD;
    nBins = 256;
    getHistogram(values, histA, nBins);
    i = -1;
    found = false;
    do {
    counts = histA[++i];
    if (counts > limit) counts = 0;
    found = counts > threshold;
    } while ((!found) && (i < histA.length-1))
    hmin = values[i];
    i = histA.length;
    do {
    counts = histA[--i];
    if (counts > limit) counts = 0;
    found = counts > threshold;
    } while ((!found) && (i > 0))
    hmax = values[i];
    setMinAndMax(440, hmax+20);
    Stack.setChannel(1);
    AUTO_THRESHOLD = 5000;
    getRawStatistics(pixcount);
    limit = pixcount/10;
    threshold = pixcount/AUTO_THRESHOLD;
    nBins = 256;
    getHistogram(values, histA, nBins);
    i = -1;
    found = false;
    do {
    counts = histA[++i];
    if (counts > limit) counts = 0;
    found = counts > threshold;
    } while ((!found) && (i < histA.length-1))
    hmin = values[i];
    i = histA.length;
    do {
    counts = histA[--i];
    if (counts > limit) counts = 0;
    found = counts > threshold;
    } while ((!found) && (i > 0))
    hmax = values[i];
    setMinAndMax(hmin, hmax);
    //print(hmin, hmax);
    Stack.setChannel(1);
    AUTO_THRESHOLD = 5000;
    getRawStatistics(pixcount);
    limit = pixcount/10;
    threshold = pixcount/AUTO_THRESHOLD;
    nBins = 256;
    getHistogram(values, histA, nBins);
    i = -1;
    found = false;
    do {
    counts = histA[++i];
    if (counts > limit) counts = 0;
    found = counts > threshold;
    } while ((!found) && (i < histA.length-1))
    hmin = values[i];
    i = histA.length;
    do {
    counts = histA[--i];
    if (counts > limit) counts = 0;
    found = counts > threshold;
    } while ((!found) && (i > 0))
    hmax = values[i];
    setMinAndMax(440, hmax+20);
    Stack.setDisplayMode("composite");
    saveAs("Tiff", "__imPath__");
    run("Quit");
    """
