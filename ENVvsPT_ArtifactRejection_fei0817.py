#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 18:20:20 2022

@author: shiyi
"""

from sys import platform
import numpy as np 
import numpy.matlib
import os  
from matplotlib import pyplot as plt
import sys
from numpy.lib.stride_tricks import as_strided
from scipy.signal import butter, filtfilt, iirnotch, resample_poly,resample,welch, csd, get_window
if platform == "darwin":
    libPath='/Users/fei/Documents/CI_projects/DataAnalysis'
    sys.path.append(libPath)
elif platform =="linux":
    libPath='/home/colliculus/behaviourBoxes/software/ratCageProgramsV2/'
    sys.path.append(libPath)
import RZ2ephys as ep

# Load the stimulus information
if platform == "darwin":
    stimPath = '/Users/fei/Documents/CI_projects/StimData_v1/'
elif platform == "linux":
    stimPath = '/home/colliculus/ephys/4/CIproject/0_ENVvsPTephys/Analysis/'
     
stimulusInfo = 'TemplateData'

if stimulusInfo == 'EachTrialData':
    stimdata=np.load(stimPath+'20220127_ENVvsPT_5_P10_1_stimWavedata.npy')
    stimclickidx=np.load(stimPath+'20220127_ENVvsPT_5_P10_1_stmClickIdx.npy',allow_pickle=True)
elif stimulusInfo == 'AllTrialData':
    StimulusData = np.load(stimPath+'Stim_ENVvsFS.npy')
elif stimulusInfo == 'TemplateData':
    StimulusData = np.load(stimPath+'Stim_ENVvsFS_template.npy')

# Load the recording data
if platform == "darwin":  
    RawDataPath = '/Users/fei/Documents/CI_projects/0_ENV_PT_ITD/Data/20220225/'
    fname = '20220225_ENVvsPT_5_P1_2'   
elif platform == "linux":
    RawDataPath = '/home/colliculus/ephys/4/CIproject/0_ENVvsPTephys/Data/2022_01_27/'
    fname = '20220127_ENVvsPT_5_P2_1'     
    
# Choose Results save path
if platform == "darwin":
    stimPath = '/Users/fei/Documents/CI_projects/StimData_v1/'
    results_path = '/Users/fei/Documents/CI_projects/0_ENV_PT_ITD/Data/20220225Results/'
elif platform == "linux":
    results_path = '/home/shiyi/Documents/CI_project/2022_01_27/Results_Data/'
    resultsfig_path = '/home/shiyi/Documents/CI_project/2022_01_27/Figure/'  

#%%
def _check_arg(x, xname):
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError('%s must be one-dimensional.' % xname)
    return x

def autocorrelation(x, maxlag):
    """
    Autocorrelation with a maximum number of lags.
    `x` must be a one-dimensional numpy array.
    This computes the same result as
        numpy.correlate(x, x, mode='full')[len(x)-1:len(x)+maxlag]
    The return value has length maxlag + 1.
    """
    x = _check_arg(x, 'x')
    p = np.pad(x.conj(), maxlag, mode='constant')
    T = as_strided(p[maxlag:], shape=(maxlag+1, len(x) + maxlag),
                    strides=(-p.strides[0], p.strides[0]))
    return T.dot(p[maxlag:].conj())
  



def crosscorrelation(x, y, maxlag):
    """
    Cross correlation with a maximum number of lags.
    `x` and `y` must be one-dimensional numpy arrays with the same length.
    This computes the same result as
        numpy.correlate(x, y, mode='full')[len(a)-maxlag-1:len(a)+maxlag]
    The return vaue has length 2*maxlag + 1.
    """
    x = _check_arg(x, 'x')
    y = _check_arg(y, 'y')
    py = np.pad(y.conj(), 2*maxlag, mode='constant')
    T = as_strided(py[2*maxlag:], shape=(2*maxlag+1, len(y) + 2*maxlag),
                    strides=(-py.strides[0], py.strides[0]))
    px = np.pad(x, maxlag, mode='constant')
    return T.dot(px)
    
    

def signal_extraction(swps_new,stimParam,stm,nsamples,Chan,Fs): 
        stm_select = stm[(stm['clickRate (Hz)'] == stimParam[0]) & (stm['duration (s)'] == stimParam[1]) & (stm['ITD (ms)'] == stimParam[2]) & (stm['env ITD (ms)'] == stimParam[3])]
        stmIdx = np.array(stm_select.index)
        ntrials = np.shape(stmIdx)[0]
        
        signal_alltrials = np.zeros((ntrials,nsamples),dtype='float32')
       
        sampleIdx1 = int(Fs*stimParam[1])
        
        # response signal aligh up 
        for ii in range(ntrials):
            signal_singleTri = swps_new[stmIdx[ii]].signal[:nsamples+100,Chan]
            signal_singleTri = signal_singleTri - np.mean(signal_singleTri)
            # signal_singleTri = signal_singleTri - signal_singleTri[0]

            if ii == 0:
                tempSigRef = signal_singleTri[:sampleIdx1]
                sigcorrRef=np.correlate(tempSigRef,tempSigRef,'full')
                templen=np.shape(tempSigRef)[0]
                sigcorrRef = sigcorrRef[(templen-5):(templen+5)]
                lagIdxRef= np.argmax(sigcorrRef)
                signal_alltrials[ii,:] = signal_singleTri[:nsamples]
                signalCat = signal_alltrials[ii,:]
                
            if ii >= 1:
                tempSig = signal_singleTri[:sampleIdx1]
                sigcorr=np.correlate((tempSigRef-np.mean(tempSigRef)),(tempSig-np.mean(tempSig)),'full')
                templen=np.shape(tempSig)[0]
                sigcorr2 = sigcorr[(templen-5):(templen+5)]
                lagIdx_temp = np.argmax(sigcorr2)
                peakIdx = lagIdxRef - lagIdx_temp
                # print(peakIdx)
                if peakIdx<0:
                    sig_padding = np.zeros((np.abs(peakIdx)),dtype = 'float32')
                    signal_padding = np.concatenate((sig_padding,signal_singleTri))
                    signal_alltrials[ii,:]=signal_padding[:nsamples]
                elif peakIdx>=0:
                    signal_alltrials[ii,:] = signal_singleTri[peakIdx:peakIdx+nsamples]
                signalCat = np.concatenate((signalCat,signal_alltrials[ii,:]))
        return signalCat,ntrials

def stim_extraction(stm,stimParam,stimdata,nsamples):
        if not len(stimdata[0,:,0])==nsamples:
            concatenate_sample = np.zeros((len(stimdata[:,0,0]), nsamples-len(stimdata[0,:,0]), len(stimdata[0,0,:])))
            stim_wave = np.concatenate((stimdata, concatenate_sample), axis = 1)
        else:
            stim_wave = stimdata
        stm_select = stm[(stm['clickRate (Hz)'] == stimParam[0]) & (stm['duration (s)'] == stimParam[1]) & (stm['ITD (ms)'] == stimParam[2]) & (stm['env ITD (ms)'] == stimParam[3])]
        stmIdx = np.array(stm_select.index)
        ntrials = np.shape(stmIdx)[0]      
        stimdata_alltrials = np.zeros((ntrials,np.shape(stim_wave)[1],np.shape(stim_wave)[2]),dtype = 'float32')
        for ii in range(ntrials):
#            print(stmIdx[ii])
            stimdata_alltrials[ii,:,:] = stim_wave[stmIdx[ii],:,:]
            temp0 = stim_wave[stmIdx[ii],:,:]
#            temp0[40:,:] = 0
            
            if ii ==0:
                stimLongData = temp0
            else:
                stimLongData = np.concatenate((stimLongData,temp0))
        return stimLongData,stimdata_alltrials,ntrials
    
    
def wienerfilt1(X,Y,N,Fs):
    Y1 = Y-np.mean(Y)
    X1 = X-np.mean(X)
    H1 = crosscorrelation(Y1,X1,N)
    H = (H1 / np.var(X))/Fs
    
    H = H - np.mean(H[:N])
    H = H[N:]
    return H


def AMUAFilterCoeffs(fs,lowpass=6000):
        nyq = 0.5*fs
        bBand,aBand = butter(4,(300/nyq, 6000/nyq),'bandpass')
        bLow,aLow = butter(4,(lowpass/nyq),'lowpass')
        bNotch, aNotch = iirnotch(50, 30, fs)
        return [[bBand, aBand], [bLow, aLow], [bNotch, aNotch]]
    
def calcAMUA(fs, ori_signal, Fs_downsample, padLen=300):
        coefs=AMUAFilterCoeffs(fs)
        bpCoefs=coefs[0]
        lpCoefs=coefs[1]
        NotchCoefs = coefs[2]
        insig = filtfilt(NotchCoefs[0], NotchCoefs[1], ori_signal, axis=0, padlen=padLen)
        insig = np.flip(insig)
        insig=filtfilt(bpCoefs[0],bpCoefs[1], insig, axis=0, padlen=padLen)
        insig=np.abs(insig)
        insig=filtfilt(lpCoefs[0],lpCoefs[1],insig,axis=0, padlen=padLen)
        insig = np.flip(insig)          
        # Fs_downsample
        # signal = resample_poly(insig, Fs_downsample, int(fs), axis=0)
        downsample_length = int((insig.shape[0]/fs)*Fs_downsample)
        signal=resample(insig,downsample_length)
        
        return signal


def ComparePredictArtifact_OriginalSigal(Predict_Artifact_temp,signal_org_temp0):
    
    PeakIdx = np.argmax(Predict_Artifact_temp)
    
    signal_org_temp0_maximumPeak = signal_org_temp0[PeakIdx-12:PeakIdx+13]
    signal_org_temp0_maximumPeak = signal_org_temp0_maximumPeak - signal_org_temp0_maximumPeak[0,:]   # subtract the first point of peak artifact of each trial
    
    SigOrg_temp_avg = np.mean(signal_org_temp0_maximumPeak,1)
    
   
    SigOrg_temp_avg_ptp = np.max(SigOrg_temp_avg) - np.min(SigOrg_temp_avg)
    Artifact_temp = Predict_Artifact_temp[PeakIdx-12:PeakIdx+13]
    
    
    
    Artifact_temp_ptp = np.max(Artifact_temp) - np.min(Artifact_temp)
    Artifact_temp = Artifact_temp/Artifact_temp_ptp*SigOrg_temp_avg_ptp
    scaleFactor = Artifact_temp_ptp/SigOrg_temp_avg_ptp
    
    PeakSigOrg_artifact = np.zeros(ntrials,dtype='float32')
    PeakCorr_sigClean_sigOrg = np.zeros(ntrials,dtype = 'float32')
    PeakCorr_sigClean_Artifact = np.zeros(ntrials,dtype = 'float32')

    tempSigCleanTrials = np.zeros((ntrials,WienerFilterOrder),dtype='float32')

    for tt in range(ntrials):
        
       
        temp_sigOrg = signal_org_temp0_maximumPeak[:,tt]
        temp_cleanSig = temp_sigOrg - Artifact_temp
        tempSigCleanTrials[tt,:] = temp_cleanSig
        corr_sigOrg_artifact = np.correlate(Artifact_temp,temp_sigOrg,'full')   
        PeakSigOrg_artifact[tt] = np.argmax(np.abs(corr_sigOrg_artifact))
        corr_sigClean_sigOrg = np.correlate(temp_sigOrg,temp_cleanSig,'full')
        corr_sigClean_artifact = np.correlate(Artifact_temp,temp_cleanSig,'full')
        PeakCorr_sigClean_sigOrg[tt] = np.argmax(np.abs(corr_sigClean_sigOrg))
        PeakCorr_sigClean_Artifact[tt] = np.argmax(np.abs(corr_sigClean_artifact))
        
        # plt.subplot(2,2,1)
        # plt.plot(temp_sigOrg,'-o')
        # plt.plot(Artifact_temp,'r-o')
        
        # plt.subplot(2,2,3)
        # plt.plot(corr_sigOrg_artifact)
        
        # plt.subplot(2,2,2)
        # plt.plot(corr_sigClean_sigOrg)
        
        # plt.subplot(2,2,4)
        # plt.plot(corr_sigClean_artifact)
        
        
    PeakSigOrg_artifact_value = np.unique(PeakSigOrg_artifact)
    PeakCorr_sigClean_sigOrg_value = np.unique(PeakCorr_sigClean_sigOrg)
    PeakCorr_sigClean_Artifact_value = np.unique(PeakCorr_sigClean_Artifact)
    
    
    return scaleFactor, PeakSigOrg_artifact_value, PeakCorr_sigClean_sigOrg_value, PeakCorr_sigClean_Artifact_value  
        
  

def SRNfft(sig_clean_temp0,stiFre):
    
    NoiseRatioTrials = []
  
    for tt in range(ntrials):
        
        
        sig_clean_temp = sig_clean_temp0[:Artifact_Length,tt]
        
        padding_temp = np.zeros((int(Fs/2) - Artifact_Length))
        sig_clean_temp = np.concatenate((sig_clean_temp,padding_temp),0)
        sigClean_temp_fft = np.fft.rfft(sig_clean_temp)
        sigClean_tempFFT_amp = np.abs(sigClean_temp_fft)
        fre = np.linspace(0,Fs/2,len(sigClean_tempFFT_amp))
        # plt.subplot(1,2,2)
        # plt.plot(fre,sigClean_tempFFT_amp,'-o')
        fre = np.around(fre)
    
        # stiFrequencies = [n for n in range(1,int(Fs0/2)) if n%stiFre ==0]
        # stiFrequencies = np.arange(stiFre,int(Fs0/2),stiFre)   
       
        stiFrequencies = np.arange(stiFre,int(2*stiFre),stiFre)
       
        Amp_stiFre = []
        Amp_baseline = []
        NeuralSig_amp_stiFre = []
        for ff in range(len(stiFrequencies)):
            idx_stifre = np.where((fre<stiFrequencies[ff]+10) & (fre>stiFrequencies[ff]-10))
            Amp_stiFre_temp = np.mean(sigClean_tempFFT_amp[idx_stifre[0]])
            Amp_stiFre.append(Amp_stiFre_temp)
            
            idx_baseline = np.where((fre<stiFrequencies[ff]+100) & (fre>stiFrequencies[ff]+20))
            Amp_baseline_temp = np.mean(sigClean_tempFFT_amp[idx_baseline[0]])
            Amp_baseline.append(Amp_baseline_temp)
            
           
        NoiseRatio= np.mean(np.array(Amp_stiFre)/np.array(Amp_baseline))
        NoiseRatioTrials.append(NoiseRatio)
        
    NoiseRatioTrials = np.array(NoiseRatioTrials) 
    
    return NoiseRatioTrials
    

def PSD_Diff_PredictArtifactvsSignalCSD(signal_org_temp0,Predict_Artifact_temp):
    
    signal_length_fft = int(Fs*0.4)
    sig_padding = np.zeros((signal_length_fft-Artifact_Length),dtype='float32')
    
    kaiser_window = get_window(('kaiser', 5), 512)
        
    power_spectrum_trials = []
    Diff_PSD_trials = []
    
    sig_org_avgTrial = np.mean(signal_org_temp0,1)
    sig_CSD_alltrials = []
    # spectrum of the Predict Artifact
    Artifact_temp = Predict_Artifact_temp  # all trials of the predict artifact are same
    Artifact_temp = np.concatenate((Artifact_temp,sig_padding),0)
    (fre,Artifact_power_spec) = welch(Artifact_temp,Fs,window=kaiser_window,nperseg=len(kaiser_window))
    Artifact_power_spectrum = 10*np.log10(Artifact_power_spec)
    
    for tt in range(ntrials):
        
        sig_temp = signal_org_temp0[:Artifact_Length,tt]
        sig_temp = np.concatenate((sig_temp,sig_padding),0)
        # cross-spectral density of two trials (it's supposed to be the artifact PSD)
        (fre,power_Pxy) = csd(sig_temp,sig_org_avgTrial,Fs,window=kaiser_window,nperseg=len(kaiser_window))
        Sig_CSD = 10*np.log10(np.abs(power_Pxy))
        Idx_fre = np.where((fre>800)& (fre<6000))
    
        Diff_PSD = Sig_CSD[Idx_fre[0]] - Artifact_power_spectrum[Idx_fre[0]]
        Diff_PSD = np.abs(Diff_PSD)
        Diff_PSD_trials.append(Diff_PSD)
      
    Diff_PSD_trials = np.array(Diff_PSD_trials)
    DiffPSD_temp = np.reshape(Diff_PSD_trials,(ntrials,len(Idx_fre[0]))).T
    DiffPSD_value = np.mean(DiffPSD_temp,0)
    
    
    
    return DiffPSD_value


#%% Import data
swps, stm = ep.readEphysFile(RawDataPath+fname)
print(swps.shape)
print(stm.shape)

if len(swps)-1 == len(stm):
    swps_new=np.delete(swps,0)
else:
    exit(0)
  
stiDur = np.sort(stm['duration (s)'].unique())
stiRate = np.sort(stm['clickRate (Hz)'].unique())   
stiITD = np.sort(stm['ITD (ms)'].unique())
stienvITD = np.sort(stm['env ITD (ms)'].unique())

Fs = swps[0].sampleRate
nsamples = int(Fs*0.5)
nchans = swps_new[0].signal.shape[1]
nchans = 32

print(nchans)
ntrials = np.shape(np.array(stm[(stm['clickRate (Hz)'] == stiRate[0]) & (stm['duration (s)'] == stiDur[0]) & (stm['ITD (ms)'] == stiITD[0]) & (stm['env ITD (ms)'] == stienvITD[0])].index))[0]    
#%% Artifact prediction and rejection
Fs_downsample = 2000
nsamples_ds = int(Fs_downsample*0.5)
amua_array = np.zeros((nchans,len(stiRate),len(stiDur),len(stiITD),len(stienvITD),nsamples,ntrials),dtype = 'float32')
Sig_clean = np.zeros((nchans,len(stiRate),len(stiDur),len(stiITD),len(stienvITD),nsamples,ntrials),dtype = 'float32')
Sig_org = np.zeros((nchans,len(stiRate),len(stiDur),len(stiITD),len(stienvITD),nsamples,ntrials),dtype = 'float32')
H0_data = np.zeros((nchans,len(stiRate),len(stiDur),len(stiITD),len(stienvITD),30),dtype = 'float32')
Sig_Predict =  np.zeros((nchans,len(stiRate),len(stiDur),len(stiITD),len(stienvITD),nsamples),dtype = 'float32')
scale_factor_array = np.zeros((nchans, len(stiRate), len(stiDur),len(stiITD),len(stienvITD)), dtype = 'float32')
Sig_AMUA = np.zeros((nchans,len(stiRate),len(stiDur),len(stiITD),len(stienvITD),nsamples_ds,ntrials),dtype='float32')
PeakCorr_sigClean_sigOrg_data =  np.zeros((nchans,len(stiRate),len(stiDur),len(stiITD),len(stienvITD)),dtype='float32')
PeakCorr_sigClean_Artifact_data = np.zeros((nchans,len(stiRate),len(stiDur),len(stiITD),len(stienvITD)),dtype='float32')
PeakCorr_sigOrg_artifact_data = np.zeros((nchans,len(stiRate),len(stiDur),len(stiITD),len(stienvITD)),dtype='float32')

Idx_goodTrials = np.zeros((nchans,len(stiRate),len(stiDur),len(stiITD),len(stienvITD),ntrials),dtype='float32')
SNRAtgoodTrials =  np.zeros((nchans,len(stiRate),len(stiDur),len(stiITD),len(stienvITD),ntrials),dtype='float32')

FilterRatioValues = np.zeros((nchans,len(stiRate),len(stiDur),len(stiITD),len(stienvITD)),dtype='float32')
CleanParameter =  np.zeros((nchans,len(stiRate),len(stiDur),len(stiITD),len(stienvITD)),dtype='float32')
ScaleFactor_data = np.zeros((nchans,len(stiRate),len(stiDur),len(stiITD),len(stienvITD)),dtype='float32')
SNRfft_data =  np.zeros((nchans,len(stiRate),len(stiDur),len(stiITD),len(stienvITD),ntrials),dtype='float32')
DiffPSD_data =  np.zeros((nchans,len(stiRate),len(stiDur),len(stiITD),len(stienvITD),ntrials),dtype='float32')

#%%
for cc in range(nchans): 
    # print('Chan'+str(cc+1))      
    for dd in range(len(stiDur)): 
    # for dd in range(1):
        dur = stiDur[dd]
        Artifact_Length = int(Fs*dur)
        for ii in range(len(stiITD)):
        # for ii in range(1):
            i = stiITD[ii]
            for jj in range(len(stienvITD)): 
            # for jj in range(1):
                e = stienvITD[jj]  
                
                # Use 900pps to predict artifact waveform
                stimParam = [900,dur,i,e]   # clickRate, duration, ITD, envITD
                # print(stimParam)
                # generate long recording signal                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
               
                #generate long stimulus signal and only keep the positive value part
                if stimulusInfo == 'EachTrialData':
                    stim_9 = stim_extraction(stm,stimParam,stimdata,nsamples)
                    stim_9 = stim_9[:,0]         
                elif stimulusInfo == 'TemplateData':
                    stim_temp_dat = StimulusData[0,dd,ii,jj,:,0]
                    stim_temp_long = np.matlib.repmat(stim_temp_dat,1,ntrials)
                    stim_9_temp = stim_temp_long.T                  
                stim_9 = stim_9_temp[:,0]
                
                stimSignal = StimulusData[0,dd,ii,jj,:,0]
                [signal_9,signal_ntrials] = signal_extraction(swps_new,stimParam,stm,nsamples,cc,Fs)
                stim_9[stim_9 < 0] = 0
                
                WienerFilterOrder = 25  
                
                H0 = wienerfilt1(stim_9, signal_9, WienerFilterOrder, Fs/WienerFilterOrder)
                H0 = H0-H0[0]
                H0_data[cc,0,dd,ii,jj,:len(H0)] = H0
                
                Predict_Artifact_Org = np.convolve(H0,stim_9)                                                
                sigLen = len(signal_9)
                #% Use the first stimuli of first trial to check the Shift between signal and predicted Artifact
                SigPredictArtifactCorr_9 = np.correlate(signal_9[:Artifact_Length],Predict_Artifact_Org[:Artifact_Length],'full')
                idxMaxCorr_9 = np.argmax(SigPredictArtifactCorr_9[Artifact_Length-5:Artifact_Length+5])
                SampleShift = idxMaxCorr_9 - (5-1)
                if SampleShift == 0:
                    Predict_Artifact = Predict_Artifact_Org[:sigLen]
                elif SampleShift < 0:
                    Predict_Artifact = Predict_Artifact_Org[np.abs(SampleShift):sigLen+np.abs(SampleShift)]
                elif SampleShift > 0:
                    temp_padding = np.zeros((SampleShift),dtype = 'float32')
                    Predict_Artifact = np.concatenate((temp_padding,Predict_Artifact_Org[:sigLen-SampleShift]))  
                
                # Predict_Artifact = Predict_Artifact_Org
                
                Predict_Artifact_temp = Predict_Artifact[:Artifact_Length]
                
                
                #get the original signal array
                signal_org_temp0 = np.reshape(signal_9,(signal_ntrials,nsamples)).T
                Sig_org[cc,0,dd,ii,jj,:,:] = signal_org_temp0
                
                [scaleFactor, PeakSigOrg_artifact_value, PeakCorr_sigClean_sigOrg_value, PeakCorr_sigClean_Artifact_value] = ComparePredictArtifact_OriginalSigal(Predict_Artifact_temp,signal_org_temp0)
                ScaleFactor_data[cc,0,dd,ii,jj] = scaleFactor
                
                Sig_Predict[cc,0,dd,ii,jj,:Artifact_Length] = Predict_Artifact_temp/scaleFactor
                
                PeakCorr_sigClean_sigOrg_value_len = len(PeakCorr_sigClean_sigOrg_value)
                
                PeakCorr_sigClean_Artifact_value_len = len(PeakCorr_sigClean_Artifact_value)
                PeakCorr_sigOrg_artifact_data[cc,0,dd,ii,jj] = len(PeakSigOrg_artifact_value)
                PeakCorr_sigClean_sigOrg_data[cc,0,dd,ii,jj] = PeakCorr_sigClean_sigOrg_value_len
                PeakCorr_sigClean_Artifact_data[cc,0,dd,ii,jj] = PeakCorr_sigClean_Artifact_value_len
                
                
                
                if len(PeakSigOrg_artifact_value) == 1:
                    if PeakSigOrg_artifact_value == WienerFilterOrder - 1:
                        if len(PeakCorr_sigClean_Artifact_value)>= 5:
                            print('CleanChannel'+str(cc))
                            
                            sig_orgSub_predictArtifact = signal_9-Predict_Artifact[:len(signal_9)]/scaleFactor
                                  
                            sig_clean_temp0 = np.reshape(sig_orgSub_predictArtifact, (signal_ntrials,nsamples)).T
                            
                            
                            Predict_Artifact_temp_scaled = Predict_Artifact_temp/scaleFactor
                            DiffPSD_value = PSD_Diff_PredictArtifactvsSignalCSD(signal_org_temp0,Predict_Artifact_temp_scaled)
                            DiffPSD_data[cc,0,dd,ii,jj,:] = DiffPSD_value
                            
                            NoiseRatioTrials = SRNfft(sig_clean_temp0,900)
                            SNRfft_data[cc,0,dd,ii,jj,:] = NoiseRatioTrials
                            
                            Sig_clean[cc,0,dd,ii,jj,:,:] = sig_clean_temp0
                    
                            AMUA_temp = calcAMUA(Fs, sig_clean_temp0[5:,:], Fs_downsample)
                            Sig_AMUA[cc,0,dd,ii,jj,:AMUA_temp.shape[0],:] = AMUA_temp
                            CleanParameter[cc,0,dd,ii,jj] = 1
                            
                            
                            stimParam_45 = [4500,dur,i,e]
                            print(stimParam_45)
                            [signal_45,signal_ntrials] = signal_extraction(swps_new,stimParam_45,stm,nsamples,cc,Fs)                
                            if stimulusInfo =='EachTrialData':
                                stim_45 = stim_extraction(stm,stimParam_45,stimdata,nsamples)
                                stim_45 = stim_45[:,0]    
                            elif stimulusInfo == 'TemplateData':
                                stimdata_temp = StimulusData[1,dd,ii,jj,:,0]
                                stim_temp = np.matlib.repmat(stimdata_temp,1,ntrials)
                                stim_45_temp = stim_temp.T
                            
                            # predict artifact on H0
                            stim_45 = stim_45_temp[:,0]    
                            stim_45[stim_45 < 0] = 0
                            PredictArt_45_Org = np.convolve(H0,stim_45)
                            signal_45_Len = len(signal_45)
                           
                            signal_45_org_temp0 = np.reshape(signal_45,(ntrials,nsamples)).T
                            Sig_org[cc,1,dd,ii,jj,:,:] = signal_45_org_temp0
                            
                            
                            # Align up the signal and the predicted Artifact
                            # compute the correlation between Predict Artifact and Responses
                            SigPredictArtifactCorr_45 = np.correlate(signal_45[:Artifact_Length],PredictArt_45_Org[:Artifact_Length],'full')
                            idxMaxCorr_45 =  np.argmax(SigPredictArtifactCorr_45[Artifact_Length-5:Artifact_Length+5])
                            print(idxMaxCorr_45-(5-1))  # We compute the lag from the 5 point before the signal length and 5 point after the signal length
                                                        # the index of the lag is the signal length -1  Here it should be 5-1
                        
                            SampleShift_45 = idxMaxCorr_45 - (5-1) 
                            if SampleShift_45 == 0:
                                PredictArt_45 = PredictArt_45_Org[:signal_45_Len]
                            elif SampleShift_45 < 0:
                                PredictArt_45 = PredictArt_45_Org[np.abs(SampleShift_45):signal_45_Len+np.abs(SampleShift_45)]
                            elif SampleShift_45 > 0:
                                temp_padding_45 = np.zeros((SampleShift_45),dtype = 'float32')
                                PredictArt_45 = np.concatenate((temp_padding_45,PredictArt_45_Org[:signal_45_Len-SampleShift_45]))
                                
                                
                            Sig_Predict[cc,1,dd,ii,jj,:Artifact_Length] = PredictArt_45[:Artifact_Length]/scaleFactor
                            
                            Predict_Artifact_temp_scaled_45 = PredictArt_45[:Artifact_Length]/scaleFactor
                            DiffPSD_value_45 = PSD_Diff_PredictArtifactvsSignalCSD(signal_45_org_temp0,Predict_Artifact_temp_scaled_45)
                            DiffPSD_data[cc,1,dd,ii,jj,:] = DiffPSD_value_45
                            
                            #get the artifact cleaned signal array
                            sig_orgSub_predictArtifact_45 = signal_45-PredictArt_45[:len(signal_45)]/scaleFactor
                            
                            sig_clean_temp45 = np.reshape(sig_orgSub_predictArtifact_45, (signal_ntrials,nsamples)).T
                            Sig_clean[cc,1,ii,jj,:,:] = sig_clean_temp45
                            
                            
                            
                            NoiseRatioTrials_45 = SRNfft(sig_clean_temp45,4500)
                            SNRfft_data[cc,1,dd,ii,jj,:] = NoiseRatioTrials_45
                            
                            # AMUA_temp = calcAMUA(Fs, sig_clean_temp0[5:,:], Fs_downsample)
                            AMUA_temp = calcAMUA(Fs, sig_clean_temp0, Fs_downsample)
                            Sig_AMUA[cc,1,dd,ii,jj,:AMUA_temp.shape[0],:] = AMUA_temp
                
                else:
                    print('NoiseChannel'+str(cc))
                    continue
               

#%%
np.save(results_path+fname+'SigOrg.npy',Sig_org)
np.save(results_path+fname+'SigClean.npy',Sig_clean)
np.save(results_path+fname+'Sig_AMUA.npy',Sig_AMUA)
np.save(results_path+fname+'PSD_diff.npy',DiffPSD_data)
np.save(results_path+fname+'ScaleFactor.npy',ScaleFactor_data)
np.save(results_path+fname+'SNRfft.npy',SNRfft_data)


#%% Need to find the noisy channel SNR larger than 10?

# Sig_org = np.load(results_path+fname+'SigClean.npy')
# Sig_clean = np.load(results_path+fname+'SigClean.npy')
# Sig_AMUA = np.load(results_path+fname+'Sig_AMUA.npy')
# DiffPSD_data = np.load(results_path+fname+'PSD_diff.npy')
# ScaleFactor_data = np.load(results_path+fname+'ScaleFactor.npy')
# SNRfft_data = np.load(results_path+fname+'SNRfft.npy')



for cc in np.arange(3,4):
    for ff in range(2):
        for dd in range(3):
            for ii in range(3):
                for jj in range(3):
                    print(cc,ff,dd,ii,jj)
                    SNR_chan_temp = SNRfft_data[cc,ff,dd,ii,jj,:]
                    idx_noise = np.where(SNR_chan_temp>10)
                    print((idx_noise[0]))
                    if len(idx_noise)>1:
                        break
                    # SNR_chan = SNR_chan_temp.flatten()
                    PSD_diff_temp = DiffPSD_data[cc,ff,dd,ii,jj,:]
                    
                    # PSD_diff_chan = PSD_diff_temp.flatten()


# plt.subplot(2,1,1)
# plt.plot(SNR_chan)
# plt.subplot(2,1,2)
# plt.plot(PSD_diff_chan)




#%% compare the original signal with the cleaned signal

cc = 3
ff = 0
dd = 2
ii = 2
jj = 2
Fs = 24414
stiDur = [0.01,0.05,0.2]
stiRate = [900,4500]
stiITD = [-0.1,0,0.1]
stienvITD = [-0.1,0,0.1]
Artifact_Length = int(Fs*stiDur[dd])
tt=9
ntrials = Sig_clean.shape[6]
plt.figure(figsize=(10,15))
plt.subplot(4,2,1)
sigOrg_temp = Sig_org[cc,ff,dd,ii,jj,:Artifact_Length,:]
plt.plot(sigOrg_temp)
plt.title('Original signal')
# plt.plot(sigOrg_temp[:,tt])

sigPredict_temp = Sig_Predict[cc,ff,dd,ii,jj,:Artifact_Length]
# sigPredict_temp = sigPredict_temp/ScaleFactor_data[cc,0,dd,ii,jj]
plt.plot(sigPredict_temp,'r')
plt.xlim([0,2000])  


plt.subplot(4,2,2)
sigClean_temp = Sig_clean[cc,ff,dd,ii,jj,:Artifact_Length,:]
plt.plot(sigClean_temp)
# plt.plot(sigClean_temp[:,tt])
plt.xlim([0,2000])      
plt.title('Clean signal')        
         
#
for tt in range(ntrials):
    dat_temp_fft = np.fft.rfft(sigOrg_temp[:,tt])
    dat_tempFFT_amp = np.abs(dat_temp_fft)
    fre = np.linspace(0,Fs/2,len(dat_tempFFT_amp))
    plt.subplot(4,2,3)
    plt.plot(fre,dat_tempFFT_amp)
plt.ylim(0,0.05)    
plt.title('Spectrum of original signal')
for tt in range(ntrials):
    dat_temp_fft = np.fft.rfft(sigClean_temp[:,tt])
    dat_tempFFT_amp = np.abs(dat_temp_fft)
    fre = np.linspace(0,Fs/2,len(dat_tempFFT_amp))
    plt.subplot(4,2,4)
    plt.plot(fre,dat_tempFFT_amp)
plt.ylim(0,0.05)      
plt.title('Spectrum of clean signal')
plt.suptitle('Channel'+str(cc)+'Stim Rate'+str(stiRate[ff])+'\nStiDur'+str(stiDur[dd])+' StiITD'+str(stiITD[ii])+' StiEnvITD'+str(stienvITD[jj]))


#%want to plot SNR across different channels 
# plt.figure()
plt.subplot(4,2,5)
plt.plot(SNRfft_data[cc,ff,dd,ii,jj,:])
plt.title('SNRfft')
plt.subplot(4,2,6)                
plt.plot(DiffPSD_data[cc,ff,dd,ii,jj,:])
plt.title('PSD difference')
 

dat_temp = np.mean(Sig_AMUA[cc,ff,dd,ii,jj,:,:],1)
plt.subplot(4,2,7)
plt.plot(dat_temp)
plt.title('AMUA')
Fs_downsample = 2000

dat_temp_fft = np.fft.rfft(dat_temp)
dat_tempFFT_amp = np.abs(dat_temp_fft)
fre = np.linspace(0,Fs_downsample/2,len(dat_tempFFT_amp))
plt.subplot(4,2,8)
plt.plot(fre,dat_tempFFT_amp)
plt.ylim(0,0.0001)
plt.title('Spectrum of AMUA')
plt.savefig(results_path+fname+'Channel'+str(cc)+'Stim'+str(ff)+str(dd)+str(ii)+str(jj))



#%% 
for ff in range(2):
    for dd in np.arange(0,1):
        for ii in range(1):
            for jj in range(1):
                plt.subplot(1,2,1)
                plt.plot(SNRfft_data[cc,ff,dd,ii,jj,:])
                plt.subplot(1,2,2)                
                plt.plot(DiffPSD_data[cc,ff,dd,ii,jj,:])

#%%
# plt.figure(figsize=(12,5))

# dat_temp = np.mean(Sig_AMUA[cc,ff,dd,ii,jj,:,:],1)
# plt.subplot(1,2,1)
# plt.plot(dat_temp)


# dat_temp_fft = np.fft.rfft(dat_temp)
# dat_tempFFT_amp = np.abs(dat_temp_fft)
# fre = np.linspace(0,Fs_downsample/2,len(dat_tempFFT_amp))
# plt.subplot(1,2,2)
# plt.plot(fre,dat_tempFFT_amp)
# plt.ylim(0,0.0001)

         
# #%% Check origin+clean signal 
# Sig_org_average = np.mean(Sig_org, 6)
# Sig_clean_average = np.mean(Sig_clean, 6)
# time_window = 0.3
# taxis = np.arange(0, time_window, 1/Fs)
# for cc in range(nchans):
#     for ff in range(len(stiRate)):
#         figs,ax=plt.subplots(3,9,figsize=(25,8))
#         figs.suptitle('channel_' + str(cc+1) + ' freq_' + str(stiRate[ff]))
#         for dd in range(len(stiDur)):
#             duration = stiDur[dd]
#             for ii in range(len(stiITD)):
#                 for ee in range(len(stienvITD)):
#                     idxITD = ii*3+ee
#                     ax[dd,idxITD].plot(taxis,Sig_org_average[cc,ff,dd,ii,ee,:len(taxis)])
#                     ax[dd,idxITD].plot(taxis,Sig_clean_average[cc,ff,dd,ii,ee,:len(taxis)])
#                     if dd == 0:
#                         ax[dd,idxITD].set_title('ptITD_' + str(stiITD[ii]) + ' envITD_' + str(stienvITD[ee]))
#                     if idxITD == 0:
#                         ax[dd,idxITD].set_ylabel('duration_' + str(duration))
#                     if idxITD > 0:
#                         ax[dd,idxITD].get_yaxis().set_visible(False)
# #        plt.savefig(resultsfig_path+fname+'Channel'+str(cc+1)+'_Frequency'+str(stiRate[ff])+'.png')
# #        plt.close()
# #%% test, plot P10-ch18-900-0.2-0.1-0.0, clean+origin and predict+origin
# clean_temp = np.mean(Sig_clean[17, 0, 2, 2, 1, :4913, :], 1)                         
# origin_temp = np.mean(Sig_org[17, 0, 2, 2, 1, :4913, :], 1)
# plt.figure()
# plt.plot(origin_temp)
# plt.plot(clean_temp)
# clean_temp1 = origin_temp-(Sig_Predict[17, 0, 2, 2, 1, :len(taxis)]/scale_factor_array[17, 0, 2, 2, 1])
# plt.figure()
# plt.plot(origin_temp)
# plt.plot(clean_temp1)
# temp = Sig_Predict[17, 0, 2, 2, 1, :len(taxis)]/scale_factor_array[17, 0, 2, 2, 1]
# plt.figure()
# plt.plot(temp)
# plt.plot(origin_temp)
#%% Forier transform
                # sigfft = np.fft.rfft(test_signal)
                # sigfftAmp=np.abs(sigfft)