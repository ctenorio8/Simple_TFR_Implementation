import numpy as np
import pathlib
import pyvisa as visa
import matplotlib.pyplot as plt
import time
import pandas as pd
from datetime import date
import PySimpleGUI as sg
import json
import numpy as np
from typing import Callable
import matplotlib.pyplot as plt
from scipy.optimize import line_search, minimize_scalar
from scipy import signal as sig


def config_filenames_dsp(
        location_array: np.ndarray,
        measurement_date: str,
        measurement_specs: pathlib.Path,
        dsp_specs: pathlib.Path
) -> list:
    """
    function: config_filename_dsp

    Author: Nate Tenorio

    Purpose: This function handles file name writing for calculated STFTs

    Arguments:
    - location_array: np.ndarray, an N x 2 array of x and y positions for your provided measurements, mm
    - measurement_date: str, the date of the measurement you are using in your analysis
    - measurement_specs: a JSON file referring to the configs of the measurement
    - file_type: str, the style of file you wish to save, including the decimal (default: .csv)
    """
    nSignals, nDimensions = np.shape(location_array)
    stft_filenames = list()
    measurement_properties_dict = read_JSON_data(measurement_specs)  # Reading the config JSON file
    dsp_properties_dict = read_JSON_data(dsp_specs)
    msmt_string = ""
    dsp_string = ""
    for keys in measurement_properties_dict:
        if keys == '-material-':
            msmt_string = '_' + measurement_properties_dict[keys] + msmt_string
        elif keys == '-excitation-':
            msmt_string = msmt_string + '_' + measurement_properties_dict[keys]
    for dsp_keys in dsp_properties_dict:
        if dsp_keys == '-method-':
            dsp_string = dsp_properties_dict[dsp_keys] + dsp_string
    for loc in range(nSignals):
        curr_position = location_array[loc, :]
        curr_name = (dsp_string + '_' + str(curr_position[0]) + 'mm_x_' + str(curr_position[1]) + 'mm_y_'
                     + measurement_date + msmt_string)
        stft_filenames = stft_filenames.append(curr_name)
    return stft_filenames


def saveDictAsJSON(dictIn: dict, path: pathlib.Path, fileName: str):
    """
    function: saveDictAsJSON

    Author: Nate Tenorio

    Purpose: This function is a simple dictionary to JSON mapping that allows for easy saving/interpretation of, in this
    case, your measurement inputs.

    Arguments:
     - dictIn: the dictionary you would like to save
     - path: the location of the file you would like to save
     - fileName: the name of the file you are looking to save
    """
    fileNameJSON = fileName + '.json'
    dumpPath = path / fileNameJSON
    with open(dumpPath, 'w') as writeFile:
        json.dump(dictIn, writeFile)
        print("Settings Saved Successfully")


def pathMaker(inPath=pathlib.Path(__file__).parent, msmtType='measurement') -> pathlib.Path:
    """
    function: pathMaker

    author: Nate Tenorio

    purpose: this function hunts for a folder within the directory passed by the inPath argument named 'data' and creates
    a child path with the day's date. It then creates a child path within the date of the format MSMT#, where the # represents
    the number of measurements made on that specific day.

    arguments:
    - inPath: (optional) the path you would like to begin hunting for 'data' from
    - msmtType: (optional) 'calibration' or 'measurement' - helps organize calibration vs. msmt data. It will use custom
    names if you so choose, though. Sometimes 'config' comes up.
    """
    if type(msmtType) != type('hello'):
        print('Invalid Measurement Type. Strings usable as directories are acceptable.')
        raise SystemExit
    dataPath = inPath / 'data'  # Creating path of inPath/data
    if not dataPath.exists():  # Checks if data already exists
        dataPath.mkdir()  # Makes directory if it does not exist
    todayDate = date.today()  # Pull today's date
    todayPath = dataPath / todayDate.strftime('20%y_%m_%d')  # Creatin path of inPath/data/YYYY_MM_DD
    if not todayPath.exists():
        todayPath.mkdir()
    pathMade = False
    checkMSMT = 1
    while not pathMade:
        if msmtType == 'measurement':
            msmtStrCheck = 'MSMT' + str(checkMSMT)
            finalPath = todayPath / msmtStrCheck
            if not finalPath.exists():
                finalPath.mkdir()
                pathMade = True
            checkMSMT = checkMSMT + 1
        if msmtType == 'calibration':
            calibStrCheck = 'CALIB' + str(checkMSMT)
            finalPath = todayPath / calibStrCheck
            if not finalPath.exists():
                finalPath.mkdir()
                pathMade = True
            checkMSMT = checkMSMT + 1
        if msmtType not in ('measurement', 'calibration'):
            strCheck = msmtType + str(checkMSMT)
            finalPath = todayPath / strCheck
            if not finalPath.exists():
                finalPath.mkdir()
                pathMade = True
            checkMSMT = checkMSMT + 1
    return finalPath


def dataPathHunter(basePath=pathlib.Path(__file__).parent,
                   mDate=date.today().strftime('20%y_%m_%d'),
                   mType='measurement',
                   mNumber=1) -> pathlib.Path:
    """
    Function: dataPathHunter

    Author: Nate Tenorio

    Purpose: This function hunts for the file path of data set that you want to process for the provided measurement.
    The function hunts in the basePath for a folder named 'data', indexes by 'mDate', then 'mType', then 'mNumber'.

    Arguments:
    - basePath (Optional): (type: path) the root of the path you want to search
    - mDate (Optional): (type: string) the date of the measurement you would like to observe, "YYYY_MM_DD"
    - mType (Optional): (type: string) the type of measurement you would like to load. Valid for 'measurement', 'calibration', 'configuration'
    - mNumber (Optional): (type: int) the number of the measurement you would like to take

    Returns:
    - mPath: the path of all data taken during a given measurement
    """
    dataPath = basePath / 'data'  # Checking for C://users/.../data | / operator on a path object expands it
    if not dataPath.exists():
        print('Data must be located within a folder named "data" at location specified in basePath.')
        raise SystemExit()
    else:
        todayPath = dataPath / mDate  # Checking for C://users/.../data/YYYY_MM_DD
        if not todayPath.exists():
            print('No data detected at given date. Input should be string "YYYY_MM_DD".')
            raise SystemExit()
        else:
            if mType not in ['measurement', 'calibration', 'configuration']:
                dataPath = todayPath / (mType + str(mNumber))
            elif mType == 'measurement':
                dataPath = todayPath / ('MSMT' + str(mNumber))
            elif mType == 'calibration':
                dataPath = todayPath / ('CALIB' + str(mNumber))
            elif mType == 'configuration':
                dataPath = todayPath / ('config' + str(mNumber))
            if not dataPath.exists():  # Checking for C://users/.../data/YYYY_MM_DD/mTypemNumber
                print(
                    'Data of the listed mType and mNumber does not exist. Check utils.py -> arguments for formatting.')
                raise SystemExit()
    return dataPath  # Returns C://users/.../data/YYYY_MM_DD/MSMT#

def read_JSON_data(filePath: pathlib.Path):
    """
    Function: read_JSON_data(filePath)

    Author: Nate Tenorio

    Purpose: Simple file reading from JSON

    Arguments: Path - a pathlib.Path object WITH FILE EXTENSION

    Returns: data - the data from your json file
    """
    with open(filePath) as json_file:
        data = json.load(json_file)
        print("Data Loaded Successfully")
    return data

########################################################################################################################
# Loss Functions                                                                                                       #
########################################################################################################################
"""
Function: toneBurstLoss()
Author: Nate Tenorio
Purpose: Calculates squared error between the provided signal and a tone burst that arrives at fitted parameter t0
Arguments:
- signal(N x 1), the current signal you are performing a line search for
- t0, the current arrival time you are investigating
- frequency, the frequency of the tone burst of note
- cycles, the number of cycles in the tone burst
- timeRes, the time resolution of the sample
Returns:
- L, the value of the total loss
"""

def toneBurstLoss_t0(t0, timeVec, signal, frequency, cycles):
    y = np.sin(2 * np.pi * frequency * timeVec) * (
            np.heaviside(timeVec - t0, 0.5) - np.heaviside(timeVec - t0 - cycles / frequency, 0.5))
    L = np.sum((y - signal) ** 2)
    return L


"""
Function: toneBurstGrad()
Author: Nate Tenorio
Purpose: Calculates the gradient of the LOSS FUNCTION of squared error between the given signal and an ideal tone burst
Arguments: See Above
Returns: The gradient of the loss function taken *with respect to the time offset*
"""


def toneBurstGrad_t0(t0, timeVec, signal, frequency, cycles):
    t0_loc = np.argmin(np.abs(timeVec - t0))
    tf_loc = np.argmin(np.abs(timeVec - t0 - cycles / frequency))
    L_prime_func = (2 * np.sin(2 * np.pi * frequency * timeVec)
                    * (np.heaviside(timeVec - t0, 0.5) - np.heaviside(timeVec - t0 - cycles / frequency, 0.5)) - signal) \
                   * np.sin(2 * np.pi * frequency * timeVec) \
                   * (sig.unit_impulse(len(timeVec), tf_loc)).reshape(-1, 1) - sig.unit_impulse(len(timeVec),
                                                                                                t0_loc).reshape(-1, 1)
    L_prime = np.sum(L_prime_func)
    return L_prime

def translate_excitation(t0: float,
                         shape: np.ndarray,
                         timeVec: np.ndarray,) -> np.ndarray:
    """
    Function: translate_excitation

    Purpose: This takes an arbitrary waveform shape, the time vector of your signal, and a time offset to
    reconstruct the excitation signal for loss minimization. This is designed to aid in the automatic
    windowing of signals with unusual excitation shapes.

    :param t0: float, the time offset of your waveform, given in seconds
    :param shape: np.ndarray, points representing signal locations for your waveform vector
    :param timeVec: np.ndarray, the time vector of your measurement
    :return: shifted_waveform: np.ndarray, output waveform padded with zeroes in accordance with given t0
    """
    dummyVec = np.zeros(np.shape(timeVec))  # Creating a dummy time vector to inject our shape into
    waveformLen = len(shape)  # Getting the length of our waveform for later use
    startIndex = np.argmin(timeVec-t0)
    try:
        dummyVec[startIndex:(waveformLen+startIndex-1)] = shape
    except IndexError:
        print('Given t0 creates out of range solution - creating highly lossy function.')
        dummyVec[startIndex:] = 999
    exShift = dummyVec
    return exShift

def arbitrary_loss_function(*modifiers,
                            shiftFunc: Callable,
                            timeVec: np.ndarray,
                            shape: np.ndarray,
                            signal: np.ndarray) -> np.ndarray:
    """
    Function: arbitrary_loss_function()

    Purpose: This function is designed to be a naive loss function creation method. It is designed such that
    the arguments in *modifiers are what any minimization/gradient descent algorithm is looking to minimize.

    The callable given in shiftFunc should take modifiers, the shape, and your time vector

    :param modifiers: The modifiers of the loss function that you are looking to solve for. Examples - t0, phase, wavespeed
    :param shiftFunc: The function you are using to shift around the waveform given via shape
    :param timeVec: The time vector of the recorded signal
    :param shape: The shape of your excitation
    :param signal: The signal you are looking to automatically window
    :return: loss: The value of MSE of your function
    """
    exShift = shiftFunc(*modifiers, shape, timeVec)
    loss = np.sum((exShift-signal)**2)
    return loss


########################################################################################################################
# Optimization Methods - Useful for any ML Applications                                                                #
########################################################################################################################
"""
Function: bruteForceMinimizer()
Author: Nate Tenorio
Purpose: This code implements a brute force method of searching for the location of 
the minimum value of a scalar input loss function. It is very computationally expensive, but
is guaranteed to converge. If needed, further work will be done with scipy optimization.

This is an exercise in breaking perfectionism :) (I hate this code)
Arguments:
-scalarValues -> An N x 1 array of values you want to check for loss minimization
-func() -> The loss function you want to minimize
-*searchArgs - other arguments you need to pass to your loss function
Returns:
minimizer - the value that minimizes the loss function
minimum - the minimum value of the loss function - useful for 'scoring' how monochromatic a msmt is
"""


def bruteForceMinimizer(variable, func, *searchArgs):
    minimum = func(variable[0])
    minimizer = variable[0]
    for val in variable[1:]:
        L = func(val, *searchArgs)
        if L < minimum:
            minimum = L
            minimizer = val
    return minimizer, minimum

