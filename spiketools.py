from scipy.signal import convolve2d
from bisect import bisect_right
import sys
sys.path.append("..")
import numpy as np
# import pyximport; pyximport.install(setup_args={"include_dirs":np.get_include()},
#                   reload_support=True)
# from  Cspiketools import time_resolved_cv_two as _time_resolved_cv_two
import numpy as np
def spiketimes_to_binary(spiketimes,tlim = None,dt = 1.):
    """ takes a n array of spiketimes and turns it into a binary
        array of spikes.
            spiketimes:  array where - spiketimes[0,:] -> spike times in [ms]
                                     - spiketimes[1,:] -> trial indices
            tlim:        [tmin,tmax] - time limits for the binary array
                         if None (default), extreme values from spiketimes 
                         are taken                             
            dt:          time resolution of the binary array in [ms]
        
        returns:
            binary:      binary array (trials,time)
            time:        time axis for the binary array (length = binary.shape[1])
        """
    if tlim is None:
        tlim = _get_tlim(spiketimes)
        
    time = np.arange(tlim[0],tlim[1]+dt,dt).astype(float)
    
    if dt<=1:
        time-=0.5*float(dt)

    trials = np.array([-1] + list(range(int(spiketimes[1, :].max() + 1)))) + 0.5
    
    tlim_spikes = cut_spiketimes(spiketimes, tlim)
    tlim_spikes = tlim_spikes[:,np.isnan(tlim_spikes[0,:])==False]
    
    
    
    if tlim_spikes.shape[1]>0:
        binary = np.histogram2d(tlim_spikes[0,:],tlim_spikes[1,:],[time,trials])[0].T
    else:
        binary = np.zeros((len(trials)-1,len(time)-1))
    return binary,time[:-1]



def binary_to_spiketimes(binary,time):
    """ takes a binary array of spikes and a corresponding time axis
        and converts it to an array of spike times.
            binary:  binary array of spikes shape = (ntrials,len(time))
            time:    time axis for binary
        returns:
            spiketimes:  array where - spiketimes[0,:] -> spike times
                                     - spiketimes[1,:] -> trial indices
        """
    time = np.array(time)
    spiketimes = [[],[]]
    max_count = binary.max()
    spikes = binary.copy()
    while max_count>0:
        if max_count == 1:
            trial,t_index = spikes.nonzero()

        else:
            trial,t_index = np.where(spikes==max_count)
        
        spiketimes[0] +=time[t_index].tolist()
        spiketimes[1] += trial.tolist()
        spikes[spikes==max_count]-=1
        max_count-=1
    spiketimes = np.array(spiketimes)
    #account for trials that have no spikes by adding a nan
    all_trials = set(range(binary.shape[0]))
    found_trials = set(spiketimes[1,:])
    missing_trials = list(all_trials.difference(found_trials))
    for mt in missing_trials:
        spiketimes = np.append(spiketimes,np.array([[np.nan],[mt]]),axis = 1)
    return spiketimes.astype(float)
def spiketimes_to_list(spiketimes):
    """converts spiketimes from array to list format.
           spiketimes:   array where - spiketimes[0,:] -> spike times in [ms]
                                     - spiketimes[1,:] -> trial indices
       returns:
           spikelist:    list, length = ntrials. list[n] -> spiketimes from trial n   
        """
    if spiketimes.shape[1] ==0:
        return []
    trials = range(int(spiketimes[1,:].max()+1))#list(set(spiketimes[1,:].tolist()))
    orderedspiketimes = spiketimes.copy()
    orderedspiketimes = orderedspiketimes[:,np.isnan(orderedspiketimes[0])==False]
    spike_order = np.argsort(orderedspiketimes[0],kind = 'mergesort')
    orderedspiketimes = orderedspiketimes[:,spike_order]
    trial_order = np.argsort(orderedspiketimes[1],kind = 'mergesort')
    orderedspiketimes = orderedspiketimes[:,trial_order]
    
    spikelist = [None] * len(trials)
    start = 0
    for trial in trials:
        end = bisect_right(orderedspiketimes[1], trial)
        
        #trialspikes = spiketimes[0,spiketimes[1,:]==trial]
        trialspikes = orderedspiketimes[0,start:end]
        start = end
        #trialspikes = trialspikes[np.isnan(trialspikes)==False]
        spikelist[trial] = trialspikes
    return spikelist
        
def ff(spiketimes,mintrials=None,tlim = None):
    """ computes the fano-factor for spiketimes.
            spiketimes:  array where - spiketimes[0,:] -> spike times
                                     - spiketimes[1,:] -> trial indices
        returns:
            fano-factor
        """
    if tlim is None:
        tlim = _get_tlim(spiketimes)
    # only counts per trial are important so binsize = tlim
    dt = tlim[1]-tlim[0]
    binary,time = spiketimes_to_binary(spiketimes,dt = dt,tlim = tlim)
    counts = binary.sum(axis = 1)
    if (counts == 0).all():
        return np.nan
    if mintrials is not None:
        if len(counts)< mintrials:
            return np.nan
    return counts.var()/counts.mean()
                 



def cv2(spiketimes,pool=True,return_all = False,bessel_correction = False,minvals = 0):
    """ computes the squared coefficient of variation for spiketimes.
            spiketimes:  array where - spiketimes[0,:] -> spike times
                                     - spiketimes[1,:] -> trial indices
            pool:        if True, ISIs for all trials are pooled
                         if False, CV^2 is calculated per trial and averaged
        returns:
            squared coefficient of variation
        """

    #trials = len(np.unique(spiketimes[1]))
    #print 'mean count: ',spiketimes.shape[1]/float(trials)
    if spiketimes.shape[1]<3:
        # need at least three spikes to calculate anything
        if return_all:
            return np.array([np.nan])
        return np.nan
    if bessel_correction:
        ddof = 1
    else:
        ddof = 0
    spikelist = spiketimes_to_list(spiketimes)
    maxlen = max([len(sl) for sl in spikelist])
    if maxlen <3:
        return np.nan
    #spikearray = np.array([sl.tolist()+[np.nan]*(maxlen-len(sl)) for sl in spikelist])
    spikearray = np.zeros((len(spikelist),maxlen))*np.nan
    spike_counts = []
    for i,sl in enumerate(spikelist):
        spikearray[i,:len(sl)] = sl
        spike_counts.append(len(sl))
    spike_counts = np.array(spike_counts)
    spikearray = spikearray[spike_counts>2]
    intervals = np.diff(spikearray,axis=1)
    

    if pool:

        var = np.nanvar(intervals,ddof=ddof)
        mean_squared = np.nanmean(intervals)**2
        return var/mean_squared
    else:
        
        intervals[np.isfinite(intervals).sum(axis = 1)<minvals,:] = np.nan
        var = np.nanvar(intervals,axis=1,ddof=ddof)
        mean_squared = np.nanmean(intervals,axis = 1)**2
        if return_all:
            return var/mean_squared
        else:
            return np.nanmean(var/mean_squared)
def time_resolved(spiketimes,window,func,kwargs = {},tlim = None,tstep = 1.):
    """ applies a function to spiketimes in a time resolved manner.
            spiketimes:  array where - spiketimes[0,:] -> spike times
                                     - spiketimes[1,:] -> trial indices
            window:      time window considered at each step
            func:        function to be applied at each step
                         must take spiketimes as its first argument
            kwargs:      optional arguments to func
            tlim:        [tmin,tmax] time limits of the calculation
            tstep:       increment by which the window is moved in each step
        returns:
            out:        output of func for each window
            time:       time axis of out (centered on the window)
        """     
    if tlim is None:
        tlim = _get_tlim(spiketimes)
    cut_spikes = cut_spiketimes(spiketimes,tlim)
    
    # initial time window
    tmin = tlim[0]
    tmax = tmin+window
    tcenter = tmin + 0.5 *(tmax-tstep-tmin)
    
    time=[]
    func_out = []
    while tmax <= tlim[1]:
        windowspikes = cut_spiketimes(cut_spikes,[tmin,tmax])
        time.append(tcenter) 
        func_out.append(func(windowspikes,**kwargs)) 
        
        tmin+=tstep
        tmax+=tstep
        tcenter+=tstep
    
    return func_out,np.array(time)


def time_resolved_new(spiketimes,window,func,kwargs = {},tlim = None,tstep = 1.):
    """ applies a function to spiketimes in a time resolved manner.
            spiketimes:  array where - spiketimes[0,:] -> spike times
                                     - spiketimes[1,:] -> trial indices
            window:      time window considered at each step
            func:        function to be applied at each step
                         must take spiketimes as its first argument
            kwargs:      optional arguments to func
            tlim:        [tmin,tmax] time limits of the calculation
            tstep:       increment by which the window is moved in each step
        returns:
            out:        output of func for each window
            time:       time axis of out (centered on the window)
        """     
    if tlim is None:
        tlim = _get_tlim(spiketimes)
    
    
    binary,btime = spiketimes_to_binary(spiketimes,tlim = tlim)
    # initial time window
    tmin = tlim[0]
    tmax = tmin+window
    tcenter = 0.5 *(tmax+tmin)
    
    time=[]
    func_out = []
    while tmax <= tlim[1]:
        #print tlim,tmin,tmax,binary[:,int(tmin):int(tmax)].shape
        windowspikes = binary_to_spiketimes(binary[:,int(tmin):int(tmax)], btime[int(tmin):int(tmax)])
        time.append(tcenter) 
        func_out.append(func(windowspikes,**kwargs)) 
        
        tmin+=tstep
        tmax+=tstep
        tcenter+=tstep
    return func_out,np.array(time)

def _consecutive_intervals_old(spiketimes):
    trial_spikes = spiketimes_to_list(spiketimes)
    trial_isis = [np.diff(spikes) for spikes in trial_spikes]
    consecutive_isis = np.zeros((0,2))
    for isis in trial_isis:

        inds = np.array([range(i,i+2) for i in range(len(isis)-1)])
        if len(inds)==0:
            continue
        consecutive_isis = np.append(consecutive_isis, isis[inds],axis = 0)
    return consecutive_isis

def _consecutive_intervals(spiketimes):
    
    order = np.argsort(spiketimes[0])
    sorted_spiketimes = spiketimes[:,order]
    order = np.argsort(sorted_spiketimes[1],kind = 'mergesort')
    sorted_spiketimes = sorted_spiketimes[:,order]
    isis = np.diff(sorted_spiketimes[0])
    trial = np.diff(sorted_spiketimes[1])
    isis[trial!=0] = np.nan
    inds = np.array([range(i,i+2) for i in range(len(isis)-1)]) 
    try:
        consecutive_isis = isis[inds]
    except:
        consecutive_isis = np.zeros((1,2))*np.nan
    consecutive_isis = consecutive_isis[np.isfinite(consecutive_isis.sum(axis=1))]
    
    return consecutive_isis

def cv_two(spiketimes,min_vals=20):
    """
    ms = []

    spikelist = spiketimes_to_list(spiketimes)
    for trialspikes in spikelist:
        
        trial_isis = np.diff(np.sort(trialspikes))
        for i in range(len(trial_isis)-1):
            ms.append(2.*np.absolute(trial_isis[i] -trial_isis[i+1])/(trial_isis[i]+trial_isis[i+1]))
    """
    consecutive_isis = _consecutive_intervals(spiketimes)
    ms = 2*np.absolute(consecutive_isis[:,0]-consecutive_isis[:,1])/(consecutive_isis[:,0]+consecutive_isis[:,1])
    if len(ms)>=min_vals:
        return np.mean(ms)
    else:
        return np.nan
    
def lv(spiketimes,min_vals=20):
    """
    ms = []
    spikelist = spiketimes_to_list(spiketimes)
    for trialspikes in spikelist:
        trial_isis = np.diff(np.sort(trialspikes))
        for i in range(len(trial_isis)-1):
            ms.append(3.*(trial_isis[i] -trial_isis[i+1])**2/(trial_isis[i]+trial_isis[i+1])**2)
    """
    consecutive_isis = _consecutive_intervals(spiketimes)
    ms = 3*(consecutive_isis[:,0]-consecutive_isis[:,1])**2 / (consecutive_isis[:,0]+consecutive_isis[:,1])**2
    if len(ms)>=min_vals:
        return np.mean(ms)
    else:
        return np.nan
    
def gaussian_kernel(sigma,dt = 1.,nstd=3.):
    """ returns a gaussian kernel with standard deviation sigma.
        sigma:  standard deviation of the kernel
        dt:     time resolution
        nstd:   overall width of the kernel specified in multiples
                of the standard deviation.
        """
    t = np.arange(-nstd*sigma,nstd*sigma+dt,dt)
    gauss = np.exp(-t**2 / sigma**2)
    gauss/= gauss.sum()*dt
    return gauss        

def triangular_kernel(sigma,dt=1):
    half_base =int(np.around(sigma * np.sqrt(6)))
    half_kernel = np.linspace(0.,1.,half_base+1)
    kernel = np.append(half_kernel,half_kernel[:-1][::-1])
    kernel /=dt*kernel.sum()
    
    return kernel

    
    
            
def kernel_rate(spiketimes,kernel,tlim = None,dt=1.,pool = True):
    """ computes a kernel rate-estimate for spiketimes.
            spiketimes:  array where - spiketimes[0,:] -> spike times [ms]
                                     - spiketimes[1,:] -> trial indices
            kernel:      1D centered kernel
            tlim:        [tmin,tmax] time-limits for the calculation
                         if None (default), extreme values from spiketimes
                         are taken
            dt:          time resolution of output [ms]
            pool:        if True, mean rate over all trials is returned
                         if False, trial wise rates are returned
        returns:
            rates:       the rate estimate s^-1 (if not pooled (ntrials,len(time)))
            time:        time axis for rate
        """
    if tlim is None:
        tlim = _get_tlim(spiketimes)
    
    binary,time = spiketimes_to_binary(spiketimes, tlim, dt)
    
    if pool:
        binary = binary.mean(axis = 0)[np.newaxis,:]
    
    
    
    rates = convolve2d(binary,kernel[np.newaxis,:],'same')
    kwidth = len(kernel)
    rates = rates[:,int(kwidth/2):-int(kwidth/2)]
    time = time[int(kwidth/2):-int(kwidth/2)]
    return rates*1000.,time
   
def sliding_counts(spiketimes,window,dt = 1.,tlim = None):
    if tlim is None:
        tlim = _get_tlim(spiketimes)
    binary,time = spiketimes_to_binary(spiketimes, dt=dt,tlim = tlim) 
    
    kernel = np.ones((1,int(window//dt)))
    counts = convolve2d(binary,kernel,'valid')
    
    dif = time.shape[0]-counts.shape[1]
    time =  time[int(np.ceil(dif/2)):int(-dif/2)]
    
    return counts,time

def kernel_fano(spiketimes,window,dt = 1.,tlim = None,components = False):
    if tlim is None:
        tlim = _get_tlim(spiketimes)
    if tlim[1]-tlim[0]==window:
        binary,time = spiketimes_to_binary(spiketimes,tlim = tlim)
        counts = binary.sum(axis=1)
        return np.array([counts.var()/counts.mean()]),np.array(time.mean())
    counts,time = sliding_counts(spiketimes,window,dt,tlim)

    vars = counts.var(axis = 0)
    means = counts.mean(axis = 0)
    fano = np.zeros_like(vars)*np.nan
    if components:
        return vars,means,time
    else:
        fano[means>0] = vars[means>0]/means[means>0]
           
        return fano,time
    
def time_resolved_cv2(spiketimes,window = None,ot = 5.,tlim = None,pool = False,tstep = 1.,bessel_correction=False,minvals = 0,return_all = False):
    
    if tlim is None:
        tlim = _get_tlim(spiketimes)

    if window is None:
        spikelist = spiketimes_to_list(spiketimes)
        isis = [] 
        for spikes in spikelist:
            isis.append(np.diff(spikes))
        meanisi = []
        for isi in isis:
            if np.isnan(isi.mean())==False:
                meanisi.append(isi.mean())
        if len(meanisi)>0:
            meanisi = sum(meanisi)/float(len(meanisi))
            window = meanisi*ot
        else:
            window = tlim[1]-tlim[0]
    
    time = []
    cvs = []
    tmin = tlim[0]
    tmax = tlim[0]+window
    order = np.argsort(spiketimes[0])
    ordered_spiketimes = spiketimes[:,order]
    while tmax<tlim[1]:
        start_ind = bisect_right(ordered_spiketimes[0],tmin)
        end_ind = bisect_right(ordered_spiketimes[0],tmax)
        window_spikes = ordered_spiketimes[:,start_ind:end_ind]

        #window_spikes = cut_spiketimes(spiketimes,[tmin,tmax])

        cvs.append(cv2(window_spikes, pool,bessel_correction=bessel_correction,minvals = minvals,return_all=return_all))
        time.append(0.5*(tmin+tmax))
        tmin+=tstep
        tmax+=tstep
    if return_all:
        if len(cvs)>0:
            # make sure the same number of values is there for each time step
            maxlen = max([len(cv) for cv in cvs])
            cvs = [cv.tolist() +[np.nan]*(maxlen-len(cv)) for cv in cvs]
            #print maxlen,[len(cv) for cv in cvs]
        else:
            cvs = [[]]
            
    return np.array(cvs),np.array(time)
    
   
def time_warped_cv2(spiketimes,window = None,ot=5.,tstep=1.,tlim = None,rate = None,kernel_width = 50.,nstd=3.,pool = True,dt = 1.,inspection_plots = False,kernel_type = 'gaussian',bessel_correction = False,interpolate = False,minvals = 0,return_all = False):
    
    if tlim is None:
        tlim = _get_tlim(spiketimes)   
      
    # estimate rate if not given  
    if rate is None:
        kernel = gaussian_kernel(kernel_width, dt)
        rate,trate = kernel_rate(spiketimes, kernel,tlim = tlim, dt=dt)
    else:
        
        trate = rate[1]
        rate = rate[0]
    
    # this is to avoid completely flat bits in the transformed time axis
    rate[rate==0]=1e-4
    
    rate = rate.flatten()
    notnaninds = np.isnan(rate)==False
    
    rate = rate[notnaninds]
    trate = trate[notnaninds]
    
    # integrate rate to obtain transformed time axis
    tdash = rate_integral(rate, dt)
    # rescale tdash to match original time
    
    tdash -= tdash.min()
    tdash/= tdash.max()
    tdash *= trate.max()-trate.min()
    tdash+=trate.min()
    
    
    transformed_spikes = spiketimes.copy().astype(float)
    
    transformed_spikes[0,:] = time_warp(transformed_spikes[0,:], trate,tdash)
    
    if inspection_plots:
        
        np.figure()
        np.subplot(1,2,1)
        stepsize =max(1, int(spiketimes.shape[1]/100))
        for i in range(0,spiketimes.shape[1],stepsize)+[-1]:
            if np.isnan(transformed_spikes[0,i]) or np.isnan(spiketimes[0,i]):
                continue
            np.plot([spiketimes[0,i]]*2,[0,tdash[np.find(((trate-spiketimes[0,i])**2)==((trate-spiketimes[0,i])**2).min())]],'--k',linewidth = 0.5)
            np.plot([0,trate[np.find(((tdash-transformed_spikes[0,i])**2)==((tdash-transformed_spikes[0,i])**2).min())]],[transformed_spikes[0,i]]*2,'--k',linewidth = 0.5)
        np.plot(trate,tdash,linewidth = 2.)
        np.xlabel('real time')
        np.ylabel('transformed time')
        np.title('transformation of spiketimes')
        
    
    
    cv2,tcv2 = time_resolved_cv2(transformed_spikes, window, ot, None, pool, tstep,bessel_correction=bessel_correction,minvals = minvals,return_all = return_all)
    if inspection_plots:
        np.subplot(1,2,2)
        stepsize =max(1, int(len(tcv2) /50))
        tcv22 = time_warp(tcv2,tdash,trate)
        for i in range(0,len(tcv2),stepsize)+[-1]:
            np.plot([tcv2[i]]*2,[0,trate[np.find(((tdash-tcv2[i])**2)==((tdash-tcv2[i])**2).min())]],'--k',linewidth = 0.5)
            np.plot([0,tdash[np.find(((trate-tcv22[i])**2)==((trate-tcv22[i])**2).min())]],[tcv22[i]]*2,'--k',linewidth = 0.5)
        
        np.plot(tdash,trate,linewidth = 2)
        np. title('re-transformation of cv2')
        np.xlabel('transformed time')
        np.ylabel('real time')
        
    tcv2 = time_warp(tcv2,tdash,trate)

    if interpolate:
        
        time = np.arange(tlim[0],tlim[1],dt)
        if len(cv2)==0 or (return_all and cv2.shape[1]==0):
            cv2 = np.zeros_like(time)*np.nan
        else:

            if return_all:
                cv2 = np.array([np.interp(time,tcv2,cv2[:,i],left = np.nan,right = np.nan) for i in range(cv2.shape[1])])

            else:
                cv2 = np.interp(time, tcv2, cv2,left = np.nan,right = np.nan)
        
            
        return cv2,time
    return cv2,tcv2
    
def rate_integral(rate,dt):
    """ integrates a rate in s^-1 to obtain average number of spikes. """
    return np.cumsum(rate/1000.)*dt
      

def time_warp(events,told,tnew):
    """transforms the events from the time axis in tnew to the one in told."""
    # get rid of events lying outside the defined range
    return np.interp(events,told,tnew,left = np.nan,right=np.nan)
    
def _get_tlim(spiketimes):
    try:
        tlim = [np.nanmin(spiketimes[0,:]),np.nanmax(spiketimes[0,:])+1]
    except:
        tlim = [0, 1]
    if np.isnan(tlim).any():
        tlim = [0,1]
    return tlim
    
def cut_spiketimes(spiketimes,tlim):

    alltrials = list(set(spiketimes[1,:]))
    cut_spikes = spiketimes[:,np.isfinite(spiketimes[0])]
    cut_spikes = cut_spikes[:,cut_spikes[0,:]>=tlim[0]]
    
    if cut_spikes.shape[1]>0:
        cut_spikes = cut_spikes[:,cut_spikes[0,:]<tlim[1]]
    for trial in alltrials:
        if not trial in cut_spikes[1,:]:
            cut_spikes = np.append(cut_spikes,np.array([[np.nan],[trial]]),axis = 1)
    return cut_spikes

def gamma_spikes(rates,order=[1],tlim = [0.,1000.],dt=0.1):
    """ produces a spike trains drawn from a homogeneous gamma process.
            rates:       list of spike-rates in s^-1. 
                         length determines number of trials
            order:      orders of the processes. can be length 1 or len(rates)
            tlim:        time window for the spikes
            dt:          time step in ms
        returns:
            spiketimes:  array where - spiketimes[0,:] -> spike times
                                     - spiketimes[1,:] -> trial indices
        """
        
    time = np.arange(tlim[0],tlim[1]+dt,dt)
    if len(rates)==1:
        rates = rates[0]*order[0]
    else:
        if len(order)==1:
            rates = [r*order[0] for r in rates]
        else:
            rates = [r*o for r,o in zip(rates,order)]
        rates = np.tile(np.array(rates)[:,np.newaxis],(1,len(time)))
    
    spikes = 1.* np.random.rand(rates.shape[0],rates.shape[1])< rates/1000.*dt
    
    if len(order)==1:
        order*=spikes.shape[0]
    for i,o in enumerate(order):
        if o ==1:
            continue
        # find trial spikes
        inds = np.find(spikes[i,:])
        selection = range(0,len(inds),int(o))
        spikes[i,:]=0. 
        spikes[i,inds[selection]]=1
    # convert spikes boolen array to array of 1 and 0
    spikes = spikes.astype(float)
    return binary_to_spiketimes(spikes, time)


def time_stretch(spiketimes,stretchstart,stretchend,endtime = None):
    if endtime is None:
        endtime = stretchend.mean()
    
    trials = np.unique(spiketimes[1,:])
    
    for i,trial in enumerate(trials):
        trialmask = spiketimes[1,:]==trial
        trialspikes = spiketimes[0,trialmask]
        trialspikes -= stretchstart[i]
        se = stretchend[i]-stretchstart[i]
        trialspikes/=se
        trialspikes*= endtime-stretchstart[i]
        trialspikes+= stretchstart[i]
        spiketimes[0,trialmask] = trialspikes
    
    return spiketimes
        

def rate_warped_analysis(spiketimes,window,step=1.,tlim=None,rate = None,func =lambda x:x.shape[1]/x[1].max(),kwargs = {},rate_kernel=gaussian_kernel(50.),dt=1.):
    if rate is None:   
        rate = kernel_rate(spiketimes, rate_kernel, tlim) 
        
    rate,trate = rate
    
    
    
    
    
    rate_tlim = [trate.min(),trate.max()+1]   
    
    spiketimes = cut_spiketimes(spiketimes, rate_tlim)
    
    ot = np.cumsum(rate)/1000.*dt
   
    w_spiketimes = spiketimes.copy()
    w_spiketimes[0,:] = time_warp(spiketimes[0,:], trate, ot)
    
    if window =='full':
        return func(w_spiketimes,**kwargs),ot.max()
    else:
        result,tresult = time_resolved(w_spiketimes, window, func, kwargs,  tstep=step)    
        
        tresult = time_warp(tresult,ot,trate)
        
        return result,tresult           


def synchrony(spikes,ignore_zero_rows= True):
    """ takes an array of spikes (or another characteristic quantity) and 
        calculates the synchrony measure according to golomb and hansel 2000
        accross the population.
        If the number of dimensions is more than 2, trials are assumed to be
        along the first axis and the average over trials is returned.
        """
    if len(spikes.shape)>2:
        return np.array([synchrony(s,ignore_zero_rows) for s in spikes]).mean()
    if ignore_zero_rows:
        sync= spikes[spikes.sum(axis=1)>0].mean(axis = 0).var() / spikes[spikes.sum(axis=1)>0].var(axis = 1).mean()
    else:
        sync= spikes.mean(axis = 0).var() / spikes.var(axis = 1).mean()
    return sync**0.5


def resample(vals,time,new_time):
    """ interpolate the result from time resolved analysis (vals,time) to 
        the new time axis new_time. """
    if len(vals)>0:
        return np.interp(new_time,time,vals,right = np.nan,left = np.nan)
    else:
        return np.ones(new_time.shape)*np.nan
    
def time_resolved_cv_two( spiketimes,window=400, tlim = None, min_vals = 10,tstep = 1):
    """ compute the time resolved local coefficient of variation. """
    return _time_resolved_cv_two(spiketimes,window,tlim,min_vals,tstep)
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    triangular_kernel(50,dt=1)

    trials = 50
    time = np.arange(1000.)
    rate = 0.05+0.15*np.exp(-((500-time[np.newaxis,:])/200.)**2)
    spikes = 1.*(np.random.rand(trials,1000)<np.repeat(rate,trials,axis = 0))
    print(spikes.mean())
    spiketimes = binary_to_spiketimes(spikes, time)

    print("CV_two: ",time_resolved_cv_two(spiketimes))
    
    warped,twarped = rate_warped_analysis(spiketimes,5,func = cv2,kwargs = {'pool':False},rate = (rate*1000,time))
    plt.figure()
    plt.plot(twarped,warped,'-o')
    #np.ylim(0,2)
    plt.show()
    """
    
    t1 = np.arange(0,1000)
    t2 = (500./(1+np.exp(-.02*(t1-0.5*t1[-1])))+t1)*0.1
    events =np.around(np.random.rand(1,10)*1000.)
   
    
    print events.shape,t1.shape,t2.shape
    
    
    
    
    np.plot(events,[0]*len(events),'ok')
    
    
    new_events = time_warp(events,t1,t2)
    
    
    np.plot([0]*len(new_events),new_events,'ok')
   
    

    
    
    for i in range(len(events[0,:])):
        np.plot([events[0,i]]*2,[0,t2[np.find(t1==events[0,i])]],'--k',linewidth = 0.5)
        np.plot([0,t1[np.find(t2==new_events[0,i])]],[new_events[0,i]]*2,'--k',linewidth = 0.5)
    
    np.plot(t1,t2,linewidth = 2)
    #np.axis('off')
    np.box(False)
    np.show()  
    """   
       

