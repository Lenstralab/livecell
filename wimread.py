# -*- coding: utf-8 -*-

import untangle, os, sys, javabridge, bioformats, re, json, pandas, psutil
import numpy as np
from tqdm.auto import tqdm
from tifffile import imread as tiffread
from datetime import datetime

py2 = sys.version_info[0] == 2

if not py2:
    unicode = str

if __package__ is None or __package__=='':
    from misc import getConfig
else:
    from .misc import getConfig

javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)
log4j = javabridge.JClassWrapper("loci.common.Log4jTools")
log4j.enableLogging()
log4j.setRootLevel("ERROR")

def tolist(item):
    if hasattr(item, 'items'):
        return item
    elif isinstance(item, (str, unicode)):
        return [item]
    try:
        iter(item)
        return list(item)
    except TypeError:
        return list((item,))


class xmldata(dict):
    def __init__(self, elem):
        super(xmldata, self).__init__()
        if elem:
            if isinstance(elem, dict):
                self.update(elem)
            else:
                self.update(xmldata._todict(elem)[1])

    def re_search(self, reg, default=None, *args, **kwargs):
        return tolist(xmldata._output(xmldata._search(self, reg, True, default, *args, **kwargs)[1]))

    def search(self, key, default=None):
        return tolist(xmldata._output(xmldata._search(self, key, False, default)[1]))

    def re_search_all(self, reg, *args, **kwargs):
        K, V = xmldata._search_all(self, reg, True, *args, **kwargs)
        return {k: xmldata._output(v) for k, v in zip(K, V)}

    def search_all(self, key):
        K, V = xmldata._search_all(self, key, False)
        return {k: xmldata._output(v) for k, v in zip(K, V)}

    @staticmethod
    def _search(d, key, regex=False, default=None, *args, **kwargs):
        if hasattr(d, 'items'):
            for k, v in d.items():
                if isinstance(k, (str, unicode)):
                    if (not regex and k == key) or (regex and re.findall(key, k, *args, **kwargs)):
                        return True, v
                    elif isinstance(v, dict):
                        found, value = xmldata._search(v, key, regex, default, *args, **kwargs)
                        if found:
                            return True, value
                    elif isinstance(v, (list, tuple)):
                        for w in v:
                            found, value = xmldata._search(w, key, regex, default, *args, **kwargs)
                            if found:
                                return True, value
                else:
                    found, value = xmldata._search(v, key, regex, default, *args, **kwargs)
                    if found:
                        return True, value
        return False, default

    @staticmethod
    def _search_all(d, key, regex=False, *args, **kwargs):
        K = []
        V = []
        if hasattr(d, 'items'):
            for k, v in d.items():
                if isinstance(k, (str, unicode)):
                    if (not regex and k == key) or (regex and re.findall(key, k, *args, **kwargs)):
                        K.append(k)
                        V.append(v)
                    elif isinstance(v, dict):
                        q, w = xmldata._search_all(v, key, regex, *args, **kwargs)
                        K.extend([str(k) + '|' + i for i in q])
                        V.extend(w)
                    elif isinstance(v, (list, tuple)):
                        for j, val in enumerate(v):
                            q, w = xmldata._search_all(val, key, regex, *args, **kwargs)
                            K.extend([str(k) + '|' + str(j) + '|' + i for i in q])
                            V.extend(w)
                else:
                    q, w = xmldata._search_all(v, key, regex, *args, **kwargs)
                    K.extend([str(k) + '|' + i for i in q])
                    V.extend(w)
        return K, V

    @staticmethod
    def _enumdict(d):
        d2 = {}
        for k, v in d.items():
            idx = [int(i) for i in re.findall('(?<=:)\d+$', k)]
            if idx:
                key = re.findall('^.*(?=:\d+$)', k)[0]
                if not key in d2:
                    d2[key] = {}
                d2[key][idx[0]] = d['{}:{}'.format(key, idx[0])]
            else:
                d2[k] = v
        rec = False
        for k, v in d2.items():
            if [int(i) for i in re.findall('(?<=:)\d+$', k)]:
                rec = True
                break
        if rec:
            return xmldata._enumdict(d2)
        else:
            return d2

    @staticmethod
    def _unique_children(l):
        if l:
            keys, values = zip(*l)
            d = {}
            for k in set(keys):
                value = [v for m, v in zip(keys, values) if k == m]
                if len(value) == 1:
                    d[k] = value[0]
                else:
                    d[k] = value
            return d
        else:
            return {}

    @staticmethod
    def _todict(elem):
        d = {}
        if hasattr(elem, 'Key') and hasattr(elem, 'Value'):
            name = elem.Key.cdata
            d = elem.Value.cdata
            return name, d

        if hasattr(elem, '_attributes') and not elem._attributes is None and 'ID' in elem._attributes:
            name = elem._attributes['ID']
            elem._attributes.pop('ID')
        elif hasattr(elem, '_name'):
            name = elem._name
        else:
            name = 'none'

        if name == 'Value':
            if hasattr(elem, 'children') and len(elem.children):
                return xmldata._todict(elem.children[0])

        if hasattr(elem, 'children'):
            children = [xmldata._todict(child) for child in elem.children]
            children = xmldata._unique_children(children)
            if children:
                d = dict(d, **children)
        if hasattr(elem, '_attributes'):
            children = elem._attributes
            if children:
                d = dict(d, **children)

        return name, xmldata._enumdict(d)

    @staticmethod
    def _output(s):
        if isinstance(s, dict):
            return xmldata(s)
        elif isinstance(s, (tuple, list)):
            return [xmldata._output(i) for i in s]
        elif not isinstance(s, (str, unicode)):
            return s
        elif len(s) > 1 and s[0] == '[' and s[-1] == ']':
            return [xmldata._output(i) for i in s[1:-1].split(', ')]
        elif re.search('^[-+]?\d+$', s):
            return int(s)
        elif re.search('^[-+]?\d?\d*\.?\d+([eE][-+]?\d+)?$', s):
            return float(s)
        elif s.lower() == 'true':
            return True
        elif s.lower() == 'false':
            return False
        elif s.lower() == 'none':
            return None
        else:
            return s


class imread:
    ''' class to read image files, while taking good care of important metadata,
            currently optimized for .czi files, but can open anything that bioformats can handle
        path: path to the image file
        optional:
        series: in case multiple experiments are saved in one file, like in .lif files
        transform: automatically correct warping between channels, need transforms.py among others
        meta: define metadata, used for pickle-ing
        beadfile: image file with beads which can be used for correcting warp

        NOTE: run imread.kill_vm() at the end of your script/program, otherwise python will not terminate

        modify images on the fly with a decorator function:
            define a function which takes an instance of this object, one image frame,
            and the coordinates c, z, t as arguments, and one image frame as return
            >> imread.frame_decorator = fun
            then use imread as usually

        Examples:
            >> im = imread('/DATA/lenstra_lab/w.pomp/data/20190913/01_YTL639_JF646_DefiniteFocus.czi')
            >> im
             << shows summary
            >> im.shape
             << (256, 256, 2, 1, 600)
            >> plt.imshow(im(1, 0, 100))
             << plots frame at position c=1, z=0, t=100 (python type indexing), note: round brackets; always 2d array with 1 frame
            >> data = im[:,:,0,0,:25]
             << retrieves 5d numpy array containing first 25 frames at c=0, z=0, note: square brackets; always 5d array
            >> plt.imshow(im.maxz(0, 0))
             << plots max-z projection at c=0, t=0
            >> len(im)
             << total number of frames
            >> im.pxsize
             << 0.09708737864077668 image-plane pixel size in um
            >> im.laserwavelengths
             << [642, 488]
            >> im.laserpowers
             << [0.02, 0.0005] in %

            See __init__ and other functions for more ideas.

        wp@tl2019
    '''

    def __init__(self, path, series=0, transform=False, meta=None, beadfile=None):
        if isinstance(path, np.ndarray):
            self.path = path
            self.filetype = 'ndarray'
        else:
            if isinstance(path, (tuple, list)):
                try:
                    from packages.utilities import dfind
                    path = dfind(*path)
                except ModuleNotFoundError:
                    raise ModuleNotFoundError('Need dfind utility for this.')
            elif isinstance(path, type(self)):
                path = path.path
            self.path = os.path.abspath(path)
            self.filetype = os.path.splitext(path)[1]
            if path == '' and not meta is None:
                self.filetype = meta['filetype']
        self.beadfile = beadfile

        self.shape = (0, 0, 0, 0, 0)
        self.series = series
        self.pxsize = 1e-1
        self.settimeinterval = 0
        self.pxsizecam = 0
        self.magnification = 0
        if self.filetype == 'ndarray':
            self.title = 'numpy array'
            self.acquisitiondate = 'now'
        else:
            self.title = os.path.splitext(os.path.basename(self.path))[0]
            self.acquisitiondate = datetime.fromtimestamp(os.path.getmtime(self.path)).strftime('%y-%m-%dT%H:%M:%S')
        self.exposuretime = (0,)
        self.deltaz = 1
        self.pcf = (1, 1)
        self.timeseries = False
        self.zstack = False
        self.laserwavelengths = ()
        self.laserpowers = ()
        self.powermode = 'normal'
        self.optovar = (1,)
        self.binning = 1
        self.collimator = (1,)
        self.tirfangle = (0,)
        self.gain = (100, 100)
        self.objective = 'unknown'
        self.filter = 'unknown'
        self.NA = 1
        self.cyllens = ['None', 'A']
        self.duolink = '488/640'
        self.detector = [0, 1]
        self.metadata = {}

        self.cachesize = 16
        self.cache = []
        self.cacheidx = []
        self._frame_decorator = None

        # how far is the center of the frame removed from the center of the sensor
        self.frameoffset = (self.shape[0] / 2, self.shape[1] / 2)

        if self.filetype == '':
            self.seqread()
        elif self.filetype == 'ndarray':
            self.ndarray()
        elif self.filetype:
            self.bfread()

        if not meta is None:
            for key, item in meta.items():
                self.__dict__[key] = item

        if 'None' in self.cyllens:
            self.slavech = self.cyllens.index('None')
            self.masterch = 1 - self.slavech  # channel with cyllens
        else:
            self.masterch, self.slavech = 1, 0

        m = self.extrametadata
        if not m is None:
            try:
                self.cyllens = m['CylLens']
                self.duolink = m['DLFilterSet'].split(' & ')[m['DLFilterChannel']]
                self.masterch = m['FeedbackChannel']
                self.slavech = 1 - self.masterch
            except:
                pass

        self.zstack = self.shape[3] > 1

        parameter = np.zeros((3, 5))
        parameter[0,] = self.shape
        parameter[1,] = (2, 1, 3, 4, 5)
        parameter[2,] = (self.pxsize, self.pxsize, 0, 0, self.timeinterval)
        self.parameter = parameter

        # handle transforms
        if not transform is False:
            if __package__ is None or __package__=='':
                from transforms import frame_transform, init_transform
            else:
                from .transforms import frame_transform, init_transform
            self.dotransform = True
            self.__framet__ = lambda c, z, t: frame_transform(self, c, z, t)
            init_transform(self, transform)
        else:
            self.dotransform = False

        # self.xmeta = xmldata(self.omedata)

    def seqread(self):
        with open(os.path.join(self.path, 'metadata.txt'), 'r') as metadatafile:
            metadata = metadatafile.read()

        self.close = lambda: None
        self.metadata = xmldata(json.loads(metadata))

        filelist = os.listdir(self.path)
        cnamelist = self.metadata.search('ChNames')

        rm = []
        for file in filelist:
            if not re.search('^img_\d{3,}.*\d{3,}.*\.tif$', file):
                rm.append(file)

        for file in rm:
            filelist.remove(file)

        filedict = dict()
        maxc = 0
        maxz = 0
        maxt = 0
        for file in filelist:
            T = re.search('(?<=img_)\d{3,}', file)
            Z = re.search('\d{3,}(?=\.tif$)', file)
            C = file[T.end() + 1:Z.start() - 1]
            t = int(T.group(0))
            z = int(Z.group(0))
            if C in cnamelist:
                c = cnamelist.index(C)
            else:
                c = len(cnamelist)
                cnamelist.append(C)

            filedict[(c, z, t)] = file
            if c > maxc:
                maxc = c
            if z > maxz:
                maxz = z
            if t > maxt:
                maxt = t
        self.filedict = filedict
        self.__frame__ = lambda c=0, z=0, t=0: tiffread(os.path.join(self.path, self.filedict[(c, z, t)]))

        if py2:
            self.omedataa = untangle.parse(
                bioformats.get_omexml_metadata(os.path.join(self.path, self.filedict[(0, 0, 0)])).encode('utf-8'))
        else:
            self.omedataa = untangle.parse(
                bioformats.get_omexml_metadata(os.path.join(self.path, self.filedict[(0, 0, 0)])))
        self.metadata = xmldata(dict(self.metadata, **xmldata(self.omedataa)))

        X = self.metadata.search('SizeX')[0]
        Y = self.metadata.search('SizeY')[0]
        self.shape = (int(X), int(Y), maxc + 1, maxz + 1, maxt + 1)

        self.timeseries = self.shape[4] > 1
        self.zstack = self.shape[3] > 1

        self.pxsize = self.metadata.search('PixelSize_um')[0]
        if self.pxsize == 0:
            self.pxsize = 0.065
        if self.zstack:
            self.deltaz = self.metadata.re_search('z-step_um', 0)[0]
        if self.timeseries:
            self.settimeinterval = self.metadata.search('Interval_ms')[0] / 1000
        if re.search('Hamamatsu', self.metadata.search('Core-Camera')[0]):
            self.pxsizecam = 6.5
        self.magnification = self.pxsizecam / self.pxsize
        self.title = self.metadata.search('Prefix')[0]
        self.acquisitiondate = self.metadata.search('Time')[0]
        self.exposuretime = [i / 1000 for i in self.metadata.search('Exposure-ms')]
        self.objective = self.metadata.search('ZeissObjectiveTurret-Label')[0]
        optovar = self.metadata.search('ZeissOptovar-Label')
        self.optovar = []
        for o in optovar:
            a = re.search('\d?\d*[,\.]?\d+(?=x$)', o)
            if hasattr(a, 'group'):
                self.optovar.append(float(a.group(0).replace(',', '.')))

    def bfread(self):
        key = np.random.randint(1e9)
        self.reader = bioformats.get_image_reader(key, self.path)
        self.close = lambda: bioformats.release_image_reader(key)

        if py2:
            omexml = bioformats.get_omexml_metadata(self.path).encode('utf-8', 'ignore')  # Otherwise unicode errors in 2.7 :(
        else:
            omexml = bioformats.get_omexml_metadata(self.path)
        self.metadata = xmldata(untangle.parse(omexml))

        s = self.reader.rdr.getSeriesCount()
        if self.series >= s:
            print('Series {} does not exist.'.format(self.series))
        self.reader.rdr.setSeries(self.series)

        self.__frame__ = lambda *args: self.reader.read(*args, rescale=False).astype('float')

        X = self.reader.rdr.getSizeX()
        Y = self.reader.rdr.getSizeY()
        C = self.reader.rdr.getSizeC()
        Z = self.reader.rdr.getSizeZ()
        T = self.reader.rdr.getSizeT()
        self.shape = (X, Y, C, Z, T)

        self.timeseries = self.shape[4] > 1
        self.zstack = self.shape[3] > 1

        image = list(self.metadata.search_all('Image').values())
        if len(image) and self.series in image[0]:
            image = xmldata(image[0][self.series])
        else:
            image = self.metadata

        unit = lambda u: 10 ** {'nm': 9, 'µm': 6, 'um': 6, u'\xb5m': 6, 'mm': 3, 'm': 0}[u]

        pxsizeunit = image.search('PhysicalSizeXUnit')[0]
        pxsize = image.search('PhysicalSizeX')[0]
        if not pxsize is None:
            self.pxsize = pxsize / unit(pxsizeunit) * 1e6

        if self.zstack:
            deltazunit = image.search('PhysicalSizeZUnit')[0]
            deltaz = image.search('PhysicalSizeZ')[0]
            if not deltaz is None:
                self.deltaz = deltaz / unit(deltazunit) * 1e6

        if self.filetype == '.czi':
            self.title = self.metadata.re_search('Information\|Document\|Name', self.title)[0]
            self.acquisitiondate = self.metadata.re_search('Information\|Document\|CreationDate', self.acquisitiondate)[
                0]
            self.exposuretime = self.metadata.re_search('TrackSetup\|CameraIntegrationTime', self.exposuretime)
            if self.timeseries:
                self.settimeinterval = self.metadata.re_search('Interval\|TimeSpan\|Value', self.settimeinterval * 1e3)[
                                           0] / 1000
                if not self.settimeinterval:
                    self.settimeinterval = self.exposuretime[0]
            self.pxsizecam = self.metadata.re_search('AcquisitionModeSetup\|PixelPeriod', self.pxsizecam)
            self.magnification = self.metadata.re_search('NominalMagnification', self.magnification)[0]
            self.laserwavelengths = [1e9 * i for i in self.metadata.re_search('Attenuator\|Wavelength', 488)]
            self.laserpowers = self.metadata.re_search('Attenuator\|Transmission')
            self.collimator = self.metadata.re_search('Collimator\|Position')
            self.gain = self.metadata.re_search('Detector\|AmplifierGain')
            self.powermode = self.metadata.re_search('TrackSetup\|FWFOVPosition')[0]
            optovar = self.metadata.re_search('TrackSetup\|TubeLensPosition', '1x')
            self.optovar = []
            for o in optovar:
                a = re.search('\d?\d*[,\.]?\d+(?=x$)', o)
                if hasattr(a, 'group'):
                    self.optovar.append(float(a.group(0).replace(',', '.')))
            self.pcf = [2 ** self.metadata.re_search('Image\|ComponentBitCount', 14)[0] / i \
                        for i in self.metadata.re_search('Channel\|PhotonConversionFactor', 1)]
            self.binning = self.metadata.re_search('AcquisitionModeSetup\|CameraBinning', 1)[0]
            self.objective = self.metadata.re_search('AcquisitionModeSetup\|Objective')[0]
            self.NA = self.metadata.re_search('Instrument\|Objective\|LensNA')[0]
            self.filter = self.metadata.re_search('TrackSetup\|BeamSplitter\|Filter')[0]
            self.tirfangle = [50 * i for i in self.metadata.re_search('TrackSetup\|TirfAngle', 0)]
            self.frameoffset = [self.metadata.re_search('AcquisitionModeSetup\|CameraFrameOffsetX')[0],
                                self.metadata.re_search('AcquisitionModeSetup\|CameraFrameOffsetY')[0]]
            self.detector = [int(i[-1]) for i in self.metadata.re_search('Instrument\|Detector\|Id', 0)]

        elif self.filetype == '.lif':
            self.title = os.path.splitext(os.path.basename(self.path))[0]
            self.exposuretime = self.metadata.re_search('WideFieldChannelInfo\|ExposureTime', self.exposuretime)
            if self.timeseries:
                self.settimeinterval = \
                self.metadata.re_search('ATLCameraSettingDefinition\|CycleTime', self.settimeinterval * 1e3)[0] / 1000
                if not self.settimeinterval:
                    self.settimeinterval = self.exposuretime[0]
            self.pxsizecam = self.metadata.re_search('ATLCameraSettingDefinition\|TheoCamSensorPixelSizeX',
                                                     self.pxsizecam)
            self.objective = self.metadata.re_search('ATLCameraSettingDefinition\|ObjectiveName', 'none')[0]
            self.magnification = \
            self.metadata.re_search('ATLCameraSettingDefinition|Magnification', self.magnification)[0]

        elif self.filetype == '.ims':
            self.magnification = self.metadata.search('LensPower', 100)[0]
            self.NA = self.metadata.search('NumericalAperture', 1.47)[0]
            self.title = self.metadata.search('Name', self.title)
            self.binning = self.metadata.search('BinningX', 1)[0]

        else:
            self.title = self.metadata.search('Name')[0]

    def ndarray(self):
        assert isinstance(self.path, np.ndarray), 'Not a numpy array'
        self.close = lambda: None
        if np.ndim(self.path) == 5:
            self.shape = self.path.shape
            cache = np.reshape(self.path, (self.shape[0], self.shape[1], len(self)))
        elif np.ndim(self.path) == 3:
            self.shape = tuple(np.hstack((self.path.shape[:2], 1, 1, self.path.shape[-1])))
            cache = self.path
        elif np.ndim(self.path) == 2:
            self.shape = tuple(np.hstack((self.path.shape, 1, 1, 1)))
            cache = np.expand_dims(self.path, 2)
        for i in range(len(self)):
            self.cache.append(cache[:, :, i])
        self.cachesize = len(self)
        z, c, t = np.meshgrid(range(self.shape[3]), range(self.shape[2]), range(self.shape[4]))
        self.cacheidx = [(C, Z, T) for C, Z, T in zip(c.flatten(), z.flatten(), t.flatten())]
        self.path = ''

    @staticmethod
    def kill_vm():
        javabridge.kill_vm()

    @property
    def frame_decorator(self):
        return self._frame_decorator

    @frame_decorator.setter
    def frame_decorator(self, decorator):
        if self.filetype == 'ndarray':
            if not 'origcache' in self:
                self.origcache = self.cache
                self.origcacheidx = self.cache
            if decorator is None:
                self.cache = self.origcache
                self.cacheidx = self.origcacheidx
            else:
                for i in range(len(self.origcache)):
                    self.cache[i] = decorator(self, self.cache[i], *self.cacheidx[i])
        else:
            self._frame_decorator = decorator
            self.cache = []
            self.cacheidx = []

    def __iter__(self):
        self.index = 0
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.index >= len(self):
            raise StopIteration
        else:
            res = self(self.index)
            self.index += 1
            return res

    def __repr__(self):
        """ gives a helpfull summary of the recorded experiment
        """
        s = '##########################################################################################################\n'
        s += 'path/filename: {}\n'.format(self.path)
        s += 'shape (xyczt): {} x {} x {} x {} x {}\n'.format(*self.shape)
        s += 'pixelsize:     {:.2f} nm\n'.format(self.pxsize * 1000)
        if self.zstack:
            s += 'z-interval:    {:.2f} nm\n'.format(self.deltaz * 1000)
        s += 'Exposuretime:  ' + ('{:.2f} ' * len(self.exposuretime)).format(
            *(np.array(self.exposuretime) * 1000)) + 'ms\n'
        if self.timeseries:
            if np.diff(self.timeval[::self.shape[2] * self.shape[3]]).shape[0]:
                s += 't-interval:    {:.3f} ± {:.3f} s\n'.format(
                    np.diff(self.timeval[::self.shape[2] * self.shape[3]]).mean(),
                    np.diff(self.timeval[::self.shape[2] * self.shape[3]]).std())
            else:
                s += 't-interval:    {:.2f} s\n'.format(self.settimeinterval)
        s += 'binning:       {}x{}\n'.format(self.binning, self.binning)
        s += 'laser colors:  ' + ('{:.0f} ' * len(self.laserwavelengths)).format(*self.laserwavelengths) + 'nm\n'
        s += 'laser powers:  ' + ('{} ' * len(self.laserpowers)).format(*(np.array(self.laserpowers) * 100)) + '%\n'
        s += 'objective:     {}\n'.format(self.objective)
        s += 'magnification: {}x\n'.format(self.magnification)
        s += 'optovar:      ' + (' {}' * len(self.optovar)).format(*self.optovar) + 'x\n'
        s += 'filterset:     {}\n'.format(self.filter)
        s += 'powermode:     {}\n'.format(self.powermode)
        s += 'collimator:   ' + (' {}' * len(self.collimator)).format(*self.collimator) + '\n'
        s += 'TIRF angle:   ' + (' {:.2f}°' * len(self.tirfangle)).format(*self.tirfangle) + '\n'
        s += 'gain:         ' + (' {:.0f}' * len(self.gain)).format(*self.gain) + '\n'
        s += 'pcf:          ' + (' {:.2f}' * len(self.pcf)).format(*self.pcf)
        return s

    def __str__(self):
        return self.path

    def __getstate__(self):
        if self.filetype == 'ndarray':
            return {'path': self.path, 'series': self.series, 'dotransform': self.dotransform,
                    'filetype': self.filetype, '_frame_decorator': self._frame_decorator, 'cache': self.cache,
                    'cacheidx': self.cacheidx, 'cachesize': self.cachesize, 'shape': self.shape,
                    'frameoffset': self.frameoffset}
        else:
            return {'path': self.path, 'series': self.series, 'dotransform': self.dotransform,
                    'filetype': self.filetype, '_frame_decorator': self._frame_decorator}

    def __setstate__(self, state):
        self.__init__(state['path'], state['series'], state['dotransform'], meta=state)

    def __len__(self):
        return self.shape[2] * self.shape[3] * self.shape[4]

    def __call__(self, *n):
        """ returns single 2D frame
            im(n):     index linearly in czt order
            im(c,z):   return im(c,z,t=0)
            im(c,z,t): return im(c,z,t)
        """
        if len(n) == 1:
            n = n[0]
            c = n % self.shape[2]
            z = (n // self.shape[2]) % self.shape[3]
            t = (n // (self.shape[2] * self.shape[3])) % self.shape[4]
            return self.frame(c, z, t)
        return self.frame(*n)

    def __getitem__(self, n):
        """ returns sliced 5D block
            im[n]:     index linearly in czt order
            im[c,z]:   return im(c,z,t=0)
            im[c,z,t]: return im(c,z,t)
        """
        if not isinstance(n, tuple):
            c = n % self.shape[2]
            z = (n // self.shape[2]) % self.shape[3]
            t = (n // (self.shape[2] * self.shape[3])) % self.shape[4]
            return self.block(c, z, t)
        n = list(n)
        if (len(n) == 2) or (len(n) == 4):
            n.append(slice(0, -1, 1))
        if len(n) == 3:
            n = list(n)
            for i, e in enumerate(n):
                if isinstance(e, slice):
                    a = [e.start, e.stop, e.step]
                    if a[0] is None:
                        a[0] = 0
                    if a[1] is None:
                        a[1] = -1
                    if a[2] is None:
                        a[2] = 1
                    for j in range(2):
                        if a[j] < 0:
                            a[j] %= self.shape[2 + i]
                            a[j] += 1
                    n[i] = np.arange(*a)
            n = [np.array(i) for i in n]
            return self.block(*n)
        if len(n) == 5:
            return self[n[2], n[3], n[4]][n[0], n[1]]

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        if hasattr(self, 'close'):
            self.close()

    def __reduce__(self):
        return (self.__class__, (self.path,))

    @property
    def sigma(self):
        """ gives the sigma of the theoretical psf in in the two channels
            assume typical stokes-shift is 22 nm
            Do not blindly rely on this to give the correct answer.
        """
        if len(self.laserwavelengths) == 1:
            return [(self.laserwavelengths[0] + 22) / 2 / self.NA / self.pxsize / 1000] * self.shape[2]
        else:
            return [(self.laserwavelengths[self.detector[self.czt(n)[0]]] + 22) / 2 / self.NA / self.pxsize / 1000 for n
                    in range(self.shape[2])]

    def czt(self, n):
        """ returns indices c, z, t used when calling im(n)
        """
        if not isinstance(n, tuple):
            c = n % self.shape[2]
            z = (n // self.shape[2]) % self.shape[3]
            t = (n // (self.shape[2] * self.shape[3])) % self.shape[4]
            return (c, z, t)
        n = list(n)
        if len(n) == 2 or len(n) == 4:
            n.append(slice(0, -1, 1))
        if len(n) == 3:
            n = list(n)
            for i, e in enumerate(n):
                if isinstance(e, slice):
                    a = [e.start, e.stop, e.step]
                    if a[0] is None:
                        a[0] = 0
                    if a[1] is None:
                        a[1] = -1
                    if a[2] is None:
                        a[2] = 1
                    for j in range(2):
                        if a[j] < 0:
                            a[j] %= self.shape[2 + i]
                            a[j] += 1
                    n[i] = np.arange(*a)
            n = [np.array(i) for i in n]
            return tuple(n)
        if len(n) == 5:
            return tuple(n[2:5])

    def czt2n(self, c, z, t):
        return c + z * self.shape[2] + t * self.shape[2] * self.shape[3]

    def transform_frame(self, frame, c, *args):
        if __package__ is None or __package__=='':
            from transforms import tfilter_transform
        else:
            from .transforms import tfilter_transform
        if self.dotransform and self.detector[c] == self.masterch:
            return tfilter_transform(frame, self.tfilter)
        else:
            return frame

    def max(self, c):
        T = np.full(self.shape[:2], -np.inf)
        for z in range(self.shape[3]):
            for t in range(self.shape[4]):
                T = np.max(np.dstack((T, self.__frame__(c, z, t))), 2)
        return self.transform_frame(T, c)

    def min(self, c):
        T = np.full(self.shape[:2], np.inf)
        for z in range(self.shape[3]):
            for t in range(self.shape[4]):
                T = np.min(np.dstack((T, self.__frame__(c, z, t))), 2)
        return self.transform_frame(T, c)

    def maxz(self, c=0, t=0):
        """ returns max-z projection at color c and time t
        """
        T = np.full(self.shape[:2], -np.inf)
        for z in range(self.shape[3]):
            T = np.max(np.dstack((T, self.__frame__(c, z, t))), 2)
        return self.transform_frame(T, c)

    def meanz(self, c=0, t=0):
        """ returns mean-z projection at color c and time t
        """
        T = np.zeros(self.shape[:2])
        l = self.shape[3]
        for z in range(self.shape[3]):
            T += self.__frame__(c, z, t) / self.shape[3]
        return self.transform_frame(T, c)

    def sumz(self, c=0, t=0):
        """ returns sum-z projection at color c and time t
        """
        T = np.zeros(self.shape[:2])
        for z in range(self.shape[3]):
            T += self.__frame__(c, z, t)
        return self.transform_frame(T, c)

    def maxt(self, c=0, z=0):
        """ returns max-t projection at color c and slice z
        """
        T = np.full(self.shape[:2], -np.inf)
        for t in range(self.shape[4]):
            T = np.max(np.dstack((T, self.__frame__(c, z, t))), 2)
        return self.transform_frame(T, c)

    def meant(self, c=0, z=0):
        """ returns mean-t projection at color c and slice z
        """
        T = np.zeros(self.shape[:2])
        l = self.shape[3]
        for t in range(self.shape[4]):
            T += self.__frame__(c, z, t) / self.shape[4]
        return self.transform_frame(T, c)

    def sumt(self, c=0, z=0):
        """ returns sum-t projection at color c and slice z
        """
        T = np.zeros(self.shape[:2])
        for t in range(self.shape[4]):
            T += self.__frame__(c, z, t)
        return self.transform_frame(T, c)

    def frame(self, c=0, z=0, t=0):
        """ returns single 2D frame
        """
        c %= self.shape[2]
        z %= self.shape[3]
        t %= self.shape[4]

        # cache last n (default 16) frames in memory for speed (~250x faster)
        if (c, z, t) in self.cacheidx:
            n = self.cacheidx.index((c, z, t))
            if len(self) != self.cachesize:
                self.cacheidx.append(self.cacheidx.pop(n))
                self.cache.append(self.cache.pop(n))
                return self.cache[-1].copy()
            else:
                return self.cache[n].copy()
        else:
            self.cacheidx.append((c, z, t))
            if self.dotransform and self.detector[c] == self.masterch:
                fr = self.__framet__(c, z, t)
            else:
                fr = self.__frame__(c, z, t)
            if self.frame_decorator is None:
                self.cache.append(fr)
            else:
                self.cache.append(self.frame_decorator(self, fr, c, z, t))
            while len(self.cacheidx) > self.cachesize:
                self.cacheidx.pop(0)
                self.cache.pop(0)
        return self.cache[-1].copy()

    def data(self, c=0, z=0, t=0):
        """ returns 3D stack of frames
        """
        if np.any(c == None):
            c = range(self.shape[2])
        if np.any(z == None):
            z = range(self.shape[3])
        if np.any(t == None):
            t = range(self.shape[4])
        c = tolist(c)
        z = tolist(z)
        t = tolist(t)
        s = len(c) * len(z) * len(t)
        d = np.full((self.shape[0], self.shape[1], s), np.nan)
        t, z, c = np.meshgrid(t, z, c)
        c = c.flatten()
        z = z.flatten()
        t = t.flatten()
        for i in range(s):
            d[:, :, i] = self.frame(c[i], z[i], t[i])
        return d

    def block(self, c=None, z=None, t=None):
        """ returns 5D block of frames
        """
        if np.any(c == None):
            c = range(self.shape[2])
        if np.any(z == None):
            z = range(self.shape[3])
        if np.any(t == None):
            t = range(self.shape[4])
        c = tolist(c)
        z = tolist(z)
        t = tolist(t)
        s = len(c) * len(z) * len(t)
        d = np.full((self.shape[0], self.shape[1], len(c), len(z), len(t)), np.nan)
        t, z, c = np.meshgrid(t, z, c)
        c = c.flatten()
        z = z.flatten()
        t = t.flatten()
        C = c - min(c)
        Z = z - min(z)
        T = t - min(t)
        for i in range(s):
            d[:, :, C[i], Z[i], T[i]] = self.frame(c[i], z[i], t[i])
        return d

    @property
    def timeval(self):
        if hasattr(self, 'metadata'):
            image = self.metadata.search('Image')
            if (isinstance(image, dict) and self.series in image) or (isinstance(image, list) and len(image)):
                image = xmldata(image[0])
                return sorted(image.search_all('DeltaT').values())
        else:
            return np.arange(self.shape[4]) * self.timeinterval

    @property
    def timeinterval(self):
        try:
            if hasattr(self, 'timeval'):
                if len(self.timeval[::self.shape[2] * self.shape[3]]) > 1:
                    return np.diff(self.timeval[::self.shape[2] * self.shape[3]]).mean()
                else:
                    return self.settimeinterval
            else:
                return self.settimeinterval
        except:
            return self.settimeinterval

    @property
    def piezoval(self):
        """ gives the height of the piezo and focus motor, only available when CylLensGUI was used
        """

        def upack(idx):
            time = list()
            val = list()
            if len(idx) == 0:
                return time, val
            for i in idx:
                time.append(int(re.search('\d+', n[i]).group(0)))
                val.append(w[i])
            return zip(*sorted(zip(time, val)))

        # Maybe the values are stored in the metadata
        n = self.metadata.search('LsmTag|Name')
        w = self.metadata.search('LsmTag')
        if not n is None:
            # n = self.metadata['LsmTag|Name'][1:-1].split(', ')
            # w = str2float(self.metadata['LsmTag'][1:-1].split(', '))

            pidx = np.where([re.search('^Piezo\s\d+$', x) is not None for x in n])[0]
            sidx = np.where([re.search('^Zstage\s\d+$', x) is not None for x in n])[0]

            ptime, pval = upack(pidx)
            stime, sval = upack(sidx)

        # Or maybe in an extra '.pzl' file
        else:
            m = self.extrametadata
            if not m is None and 'p' in m:
                q = np.array(m['p'])
                if not len(q.shape):
                    q = np.zeros((1, 3))

                ptime = [int(i) for i in q[:, 0]]
                pval = [float(i) for i in q[:, 1]]
                sval = [float(i) for i in q[:, 2]]

            else:
                ptime = []
                pval = []
                sval = []

        df = pandas.DataFrame(columns=['frame', 'piezoZ', 'stageZ'])
        df['frame'] = ptime
        df['piezoZ'] = pval
        df['stageZ'] = np.array(sval) - np.array(pval) - self.metadata.re_search('AcquisitionModeSetup\|ReferenceZ', 0)[
            0] * 1e6

        # remove duplicates
        df = df[~df.duplicated('frame', 'last')]
        return df

    @property
    def extrametadata(self):
        if len(self.path) > 3:
            if os.path.isfile(self.path[:-3] + 'pzl2'):
                pname = self.path[:-3] + 'pzl2'
            elif os.path.isfile(self.path[:-3] + 'pzl'):
                pname = self.path[:-3] + 'pzl'
            else:
                return
            try:
                return getConfig(pname)
            except:
                return
        return

    def save_as_tiff(self, fname=None, c=None, z=None, t=None, split=False, bar=True, pixel_type='uint16', rgb=False):
        """ saves the image as a tiff-file
            split: split channels into different files
        """
        if __package__ is None or __package__=='':
            from tiffwrite import IJTiffWriter
        else:
            from .tiffwrite import IJTiffWriter
        if fname is None:
            fname = self.path[:-3] + 'tif'
        elif not fname[-3:] == 'tif':
            fname += '.tif'
        if split:
            for i in range(self.shape[2]):
                if self.timeseries:
                    self.save_as_tiff(fname[:-3] + 'C[i]{:01d}.tif'.format(i), i, 0, None, False, bar, pixel_type)
                else:
                    self.save_as_tiff(fname[:-3] + 'C[i]{:01d}.tif'.format(i), i, None, 0, False, bar, pixel_type)
        else:
            n = [c, z, t]
            s = ('C', 'Z', 'T')
            for i in range(len(n)):
                if n[i] is None:
                    n[i] = range(self.shape[i + 2])
                elif not isinstance(n[i], (tuple, list)):
                    n[i] = (n[i],)

            if rgb:
                mx = [self.max(c) for c in n[0]]
                mn = [self.min(c) for c in n[0]]

                with IJTiffWriter(fname, (len(n[0]), len(n[1]), len(n[2]))) as f:
                    with tqdm(total=np.prod((len(n[1]), len(n[2]))),
                              desc='Saving frames', leave=False, disable=not bar) as bar:
                        for j, z in enumerate(n[1]):
                            for k, t in enumerate(n[2]):
                                frame = []
                                for i, c in enumerate(n[0]):
                                    cframe = self(c, z, t)
                                    cframe -= mn[i]
                                    cframe /= (mx[i] - mn[i])
                                    cframe *= 255
                                    frame.append(cframe)
                                frame = np.dstack(frame).astype('uint8')
                                f.save(frame, j, k)
                                bar.update()
            else:
                with IJTiffWriter(fname, (len(n[0]), len(n[1]), len(n[2]))) as f:
                    with tqdm(total=np.prod((len(n[0]), len(n[1]), len(n[2]))),
                              desc='Saving frames', leave=False, disable=not bar) as bar:
                        for i, c in enumerate(n[0]):
                            for j, z in enumerate(n[1]):
                                for k, t in enumerate(n[2]):
                                    f.save(self(c, z, t).astype(pixel_type), i, j, k)
                                    bar.update()

    @property
    def summary(self):
        print(self.__repr__())

    def close_all_references(self):
        for f in [f.fd for f in psutil.Process().open_files() if f.path == os.path.realpath(self.path)]:
            os.close(f)