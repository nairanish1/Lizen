import numpy as np 
import streamlit as st 
import scipy.signal as signal
import librosa
import soundfile as sf 
import sofa 
import os 
import resampy
from io import BytesIO
#--- Library Dependencies---#

#function to find the closest array member to target value (find nearest angle in .sofa file to target angle needed)
def find_nearest(array,value): 
    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    return array[idx], idx

def binaural_rendering(sofa_file, source_files, angle_positions, elevation_positions, target_fs = 48000):
    #Loading in HRTF dataset
    HRTF = sofa.Database.open(sofa_file)
    fs_H = HRTF.Data.SamplingRate.get_values()[0]
    positions = HRTF.Source.Position.get_values(system = "spherical")
    H = np.zeros([HRTF.Dimensions.N, 2])
    Stereo_3D = np.zeros([HRTF.Dimensions.N, 2])

    for n in range(len(source_files)):
        angle = angle_positions[n]
        elev = elevation_positions[n]
    
    #Adjust angle to match database convention 
        angle_label = angle
        angle = 360-angle
        if angle == 360:
            angle = 0
    
    # Retrieve HRTF data for the angle and the elevation 
        [az, _] = find_nearest(positions[:,0], angle)
        az_indices = np.where(positions[:,0] ==az)[0]
        [el, el_idx] = find_nearest(positions[az_indices][:,1], elev)
        M_idx = az_indices[el_idx]

        H[:, 0] = HRTF.Data.IR.get_values(indices ={"M": M_idx, "R":0, "E":0})
        H[:, 1] = HRTF.Data.IR.get_values(indices ={"M": M_idx, "R":1, "E":0})

        if fs_H != target_fs:
            H = librosa.core.resample(H.transpose(), fs_H, target_fs, res_type = "kaiser_best").transpose()
    
    #Read the audio source as a one-dimensional array 
        x, fs_x = sf.read(source_files[n] , always_2d= False)
        if x.ndim > 1:
            x = np.mean(x, axis=1)
        if fs_x != target_fs:
            x = librosa.resample(x, orig_sr=fs_x, target_sr=target_fs, res_type="kaiser_best")
    
    #Perform frequency based convolution with one-dimensional arrays and windowing
        window = signal.windows.hann(len(x))
        x_windowed = x* window
        H_left = H[:, 0] * window[:len(H[:,0])]
        H_right = H[:, 1] * window[:len(H[:,1])]
        rend_L = signal.fftconvolve(x, H_left, mode = 'full')
        rend_R = signal.fftconvolve(x, H_right, mode = 'full')

    #Normalization factor as a scalar 
        M = max(np.max(np.abs(rend_L)), np.max(np.abs(rend_R)))

    #Extend Stereo_3D if necessary 
        if len(Stereo_3D) < len(rend_L):
            diff = len(rend_L) - len(Stereo_3D)
            Stereo_3D = np.append(Stereo_3D, np.zeros([diff, 2]), axis = 0)

    #Add the rendered signals to the final output
        Stereo_3D[0:len(rend_L),0] += (rend_L/M)
        Stereo_3D[0:len(rend_R),1] += (rend_R/M)
    return Stereo_3D, target_fs












