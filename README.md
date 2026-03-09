#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.2.4),
    on February 18, 2026, at 10:27
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.2.4'
expName = 'Study1_online'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': '',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1280, 800]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Study1_online\\Study1_online_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('exp')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # update experiment info
    expInfo['date'] = data.getDateStr()
    expInfo['expName'] = expName
    expInfo['expVersion'] = expVersion
    expInfo['psychopyVersion'] = psychopyVersion
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Begin_demo" ---
    # Run 'Begin Experiment' code from code_2
    # --- imports & globals ---
    from random import shuffle
    
    # round index (0-based)
    round_num = 0
    
    # cumulative tallies
    total_likes_you = 0
    total_likes_partner = 0
    
    # histories
    participant_choices = []  # 0 = Neutral, 1 = Negative
    partner_choices = []
    
    # side mapping history (optional, useful for bias analyses)
    side_history = []
    
    # --- decision routine exposed vars (so Text components won't error on first draw) ---
    left_label = ''
    right_label = ''
    # If you ever bind these in Text fields, initializing avoids UnboundLocalError:
    you_chose_text = ''
    you_likes_text = ''
    partner_chose_text = ''
    partner_likes_text = ''
    round_text = ''
    you_likes_thisround_text = ''
    partner_likes_thisround_text = ''
    
    
    # --- refresh routine exposed vars (used by Text set every frame) ---
    likes_display = ''
    timer_display = ''
    revealed_likes_you = 0
    revealed_likes_partner = 0
    refresh_clicks = 0
    _prevPressed = [0, 0, 0]
    
    # --- per-trial targets/texts shared across routines (set after Decision) ---
    final_likes_target_you = 0
    final_likes_target_partner = 0
    your_option_text_this_round = ''
    partner_option_text_this_round = ''
    likes_this_round_you = 0
    likes_this_round_partner = 0
    
    
    
    # helper: partner move rule
    def partner_move_for_round(r_idx, p_choices):
        """
        r_idx: 0-based round index
        Rules:
          r1 (idx 0): partner Negative (1)
          r2 (idx 1): partner Neutral (0)
          thereafter:
            if participant was Negative on round 2 (p_choices[1] == 1) -> Tit-for-Tat (copy p's previous move)
            else (participant Neutral on round 2) -> Axelrod alternator:
                 Negative on rounds 3,5,7,... (idx 2,4,6,...), Neutral on the others.
        """
        if r_idx == 0:
            return 1
        if r_idx == 1:
            return 0
        if len(p_choices) >= 2 and p_choices[1] == 1:
            # Tit-for-Tat: copy participant's previous move
            return p_choices[r_idx - 1]
        else:
            # Axelrod alternator after round 2: Negative on idx even (2,4,6,...) else Neutral
            return 1 if (r_idx % 2 == 0) else 0
    
    import random
    
    condFiles = [
        "allhigh_online.xlsx",
        "low_online.xlsx",
        "high_online.xlsx",
        "flat_online.xlsx",
    ]
    
    # pick ONE condition for the entire session
    condFile = random.choice(condFiles)
    
    # clean label
    condition = condFile.replace(".xlsx", "")
    
    # save assignment
    thisExp.addData("condition", condition)
    thisExp.addData("condFile", condFile)
    
    background = visual.Rect(
        win=win, name='background',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    begin_text_2 = visual.TextStim(win=win, name='begin_text_2',
        text='You must be on a computer to take this study. You will not be able to move forward if on a smartphone or tablet.\n\nPress SPACEBAR to begin the demo.',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_resp_3 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "demo_1" ---
    background_2 = visual.Rect(
        win=win, name='background_2',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    text_6 = visual.TextStim(win=win, name='text_6',
        text='At the beginning of each round, you will be given a social dilemma.\n\nThen you must choose, out of two social media posts, what you would be most likely to post on social media if this dilemma occurred to you.\n\n\nPress SPACEBAR to continue',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_4 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "demo1_c" ---
    background_22 = visual.Rect(
        win=win, name='background_22',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    text_24 = visual.TextStim(win=win, name='text_24',
        text='This task is not relative to a specific type of social media. It is not necessarily about what you would share if public or identifiable as yourself.\n\n\nPress SPACEBAR to continue',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_11 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "demo1_c_2" ---
    background_23 = visual.Rect(
        win=win, name='background_23',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    text_25 = visual.TextStim(win=win, name='text_25',
        text='Think about your regular social media habits. Maybe you have an anonymous account, or a private story, or a secondary account.\n\nPress SPACEBAR to continue',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_12 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "demo1_c_3" ---
    background_24 = visual.Rect(
        win=win, name='background_24',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    text_26 = visual.TextStim(win=win, name='text_26',
        text='When reading these social dilemmas, really think about what you would post in response to them, even if that would only be from an anonymous account or private story. \n\n\nPress SPACEBAR to continue',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_13 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "demo1_c_4" ---
    background_25 = visual.Rect(
        win=win, name='background_25',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    text_27 = visual.TextStim(win=win, name='text_27',
        text='This study is just about understanding how college age students make decisions about posting on social media - not specific to any type or context. \nYou are anonymous in this task, so choose what you would be most likely to choose in the real world.\n\nPress SPACEBAR to continue',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_14 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "demo_2" ---
    background_3 = visual.Rect(
        win=win, name='background_3',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    dilemma_txt_2 = visual.TextStim(win=win, name='dilemma_txt_2',
        text='',
        font='Open Sans',
        pos=(0, 0.35), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    poly_left_2 = visual.Rect(
        win=win, name='poly_left_2',
        width=(0.35, 0.35)[0], height=(0.35, 0.35)[1],
        ori=0.0, pos=(-0.4, -.15), draggable=False, anchor='center',
        lineWidth=8.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, 0.0902], fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    poly_right_2 = visual.Rect(
        win=win, name='poly_right_2',
        width=(0.35, 0.35)[0], height=(0.35, 0.35)[1],
        ori=0.0, pos=(0.4, -.15), draggable=False, anchor='center',
        lineWidth=8.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, 0.0902], fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    left_txt_2 = visual.TextStim(win=win, name='left_txt_2',
        text='',
        font='Open Sans',
        pos=(-0.4, -.15), draggable=False, height=0.03, wrapWidth=0.3, ori=0.0, 
        color=[-1.0000, -1.0000, 0.0902], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    right_txt_2 = visual.TextStim(win=win, name='right_txt_2',
        text='',
        font='Open Sans',
        pos=(0.4, -.15), draggable=False, height=0.03, wrapWidth=0.3, ori=0.0, 
        color=[-1.0000, -1.0000, 0.0902], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    happened_text_2 = visual.TextStim(win=win, name='happened_text_2',
        text='You will then read both options below, and pick the post in which you would be most likely to post if the dilemma happened to you.\n\nClick on one of the options to continue.',
        font='Open Sans',
        pos=(-0, .15), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    mouse_2 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_2.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "demo_3" ---
    background_4 = visual.Rect(
        win=win, name='background_4',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    text_7 = visual.TextStim(win=win, name='text_7',
        text='After you make your choice of what you would post, it takes our servers approximately 15 seconds to pull the likes from the server.\n\nThe likes will accumulate over time as they are pulled into the task.\nYou are able to refresh and view them coming in if you would like.\n\nPress SPACEBAR to continue.',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_5 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "demo_4" ---
    background_5 = visual.Rect(
        win=win, name='background_5',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    text_8 = visual.TextStim(win=win, name='text_8',
        text='As a reminder, each post was pre-rated by 1000 UCF students.\n\nEach post can receive anywhere from 0-1000 likes.\n\nPress SPACEBAR to continue.',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_6 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "demo_5" ---
    background_6 = visual.Rect(
        win=win, name='background_6',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    refresh_btn_2 = visual.Rect(
        win=win, name='refresh_btn_2',
        width=(0.25, 0.1)[0], height=(0.25, 0.1)[1],
        ori=0.0, pos=(0, -.05), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-2.0, interpolate=True)
    refresh_click_2 = visual.TextStim(win=win, name='refresh_click_2',
        text='click to refresh \nlike count',
        font='Open Sans',
        pos=(0, -0.05), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    likes_txt_2 = visual.TextStim(win=win, name='likes_txt_2',
        text='',
        font='Open Sans',
        pos=(0, 0.1), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, 0.0902], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    timer_txt_2 = visual.TextStim(win=win, name='timer_txt_2',
        text='',
        font='Open Sans',
        pos=(0, -.275), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    reminder_text_2 = visual.TextStim(win=win, name='reminder_text_2',
        text='Click on the refresh button to see how likes come in.',
        font='Open Sans',
        pos=(0, 0.25), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    demo_mouse = event.Mouse(win=win)
    x, y = [None, None]
    demo_mouse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "demo_6" ---
    background_7 = visual.Rect(
        win=win, name='background_7',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    text_9 = visual.TextStim(win=win, name='text_9',
        text='',
        font='Open Sans',
        pos=(0, 0.1), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, 0.0902], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_10 = visual.TextStim(win=win, name='text_10',
        text='After 15 seconds, you will see the total amount of likes received for your post.\n\n\n\n\n\nPress SPACEBAR to continue.',
        font='Open Sans',
        pos=(0, .1), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_resp_7 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "demo_7" ---
    background_8 = visual.Rect(
        win=win, name='background_8',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    text_11 = visual.TextStim(win=win, name='text_11',
        text='Finally, you will be able to see what the other participant posted on each round, and they will also see what you posted on each round.\n\n\nPress SPACEBAR to continue',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_8 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "demo_7b" ---
    background_21 = visual.Rect(
        win=win, name='background_21',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    text_28 = visual.TextStim(win=win, name='text_28',
        text='Keep in mind that you are playing with another participant in real time. The time in between decisions and task components may vary depending on how quickly they complete the study. This may feel very fast paced if they move through quickly.\n\nPress SPACEBAR to continue',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_10 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "demo_8" ---
    background_9 = visual.Rect(
        win=win, name='background_9',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    you_choice_box_2 = visual.Rect(
        win=win, name='you_choice_box_2',
        width=(0.4, 0.35)[0], height=(0.4, 0.35)[1],
        ori=0.0, pos=(-.4, 0), draggable=False, anchor='center',
        lineWidth=8.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, 0.0902], fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    you_chose_txt_2 = visual.TextStim(win=win, name='you_chose_txt_2',
        text='',
        font='Open Sans',
        pos=(-.4, 0), draggable=False, height=0.03, wrapWidth=0.35, ori=0.0, 
        color=[-1.0000, -1.0000, 0.0902], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    partner_choice_box_2 = visual.Rect(
        win=win, name='partner_choice_box_2',
        width=(0.4,0.35)[0], height=(0.4,0.35)[1],
        ori=0.0, pos=(0.4, 0), draggable=False, anchor='center',
        lineWidth=8.0,
        colorSpace='rgb', lineColor=[-1.0000, -0.2157, -1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-3.0, interpolate=True)
    partner_chose_txt_2 = visual.TextStim(win=win, name='partner_chose_txt_2',
        text='',
        font='Open Sans',
        pos=(0.4, 0), draggable=False, height=0.03, wrapWidth=0.35, ori=0.0, 
        color=[-1.0000, -0.2157, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    text_12 = visual.TextStim(win=win, name='text_12',
        text='Press SPACEBAR to continue.',
        font='Open Sans',
        pos=(0, -.35), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    key_resp_9 = keyboard.Keyboard(deviceName='defaultKeyboard')
    text_13 = visual.TextStim(win=win, name='text_13',
        text='Your choice will be on the left, in the blue box.\n',
        font='Open Sans',
        pos=(-.4, .2), draggable=False, height=0.02, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    text_23 = visual.TextStim(win=win, name='text_23',
        text='Their choice will be on the right, in the green box\n',
        font='Open Sans',
        pos=(.4, .2), draggable=False, height=0.02, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    remember_demo_text = visual.TextStim(win=win, name='remember_demo_text',
        text='At the end of each round, the other participant will also see what you posted.',
        font='Open Sans',
        pos=(0, 0.4), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    
    # --- Initialize components for Routine "Start" ---
    background_10 = visual.Rect(
        win=win, name='background_10',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    Welcome_text = visual.TextStim(win=win, name='Welcome_text',
        text='You will first answer a few questions, then begin the task when synced with the other participant.\n\nClick SPACEBAR to begin.',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "SP" ---
    # Run 'Begin Experiment' code from sp_code
    
    
    background_11 = visual.Rect(
        win=win, name='background_11',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    slider = visual.Slider(win=win, name='slider',
        startValue=50, size=(1.0, 0.1), pos=(0, -0.15), units=win.units,
        labels=None, ticks=(0, 50, 100), granularity=0.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    text_14 = visual.TextStim(win=win, name='text_14',
        text='I currently feel connected to the other participant as if we are sharing the same space.',
        font='Open Sans',
        pos=(0, .3), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    polygon = visual.Rect(
        win=win, name='polygon',
        width=(0.3, 0.1)[0], height=(0.3, 0.1)[1],
        ori=0.0, pos=(0, -0.35), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-4.0, interpolate=True)
    text_16 = visual.TextStim(win=win, name='text_16',
        text='Continue',
        font='Open Sans',
        pos=(0, -.35), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    mouse_3 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_3.mouseClock = core.Clock()
    q_rating_n = visual.TextStim(win=win, name='q_rating_n',
        text='Not at all',
        font='Open Sans',
        pos=(-.5, -0.05), draggable=False, height=0.02, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    q_rating_n_2 = visual.TextStim(win=win, name='q_rating_n_2',
        text='Very much so',
        font='Open Sans',
        pos=(0.5, -0.05), draggable=False, height=0.02, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    text_15 = visual.TextStim(win=win, name='text_15',
        text="Drag the slider, then click 'continue' to submit your answer.",
        font='Open Sans',
        pos=(0, 0.2), draggable=False, height=0.02, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    
    # --- Initialize components for Routine "anon" ---
    background_12 = visual.Rect(
        win=win, name='background_12',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    slider_2 = visual.Slider(win=win, name='slider_2',
        startValue=50, size=(1.0, 0.1), pos=(0, -0.15), units=win.units,
        labels=None, ticks=(0, 50, 100), granularity=0.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    text_17 = visual.TextStim(win=win, name='text_17',
        text='I currently feel that my identity is concealed from the other participant.',
        font='Open Sans',
        pos=(0, .3), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    polygon_2 = visual.Rect(
        win=win, name='polygon_2',
        width=(0.3, 0.1)[0], height=(0.3, 0.1)[1],
        ori=0.0, pos=(0, -0.35), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-4.0, interpolate=True)
    text_18 = visual.TextStim(win=win, name='text_18',
        text='Continue',
        font='Open Sans',
        pos=(0, -.35), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    mouse_4 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_4.mouseClock = core.Clock()
    q_rating_n_3 = visual.TextStim(win=win, name='q_rating_n_3',
        text='Not at all',
        font='Open Sans',
        pos=(-.5, -0.05), draggable=False, height=0.02, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    q_rating_n_4 = visual.TextStim(win=win, name='q_rating_n_4',
        text='Very much so',
        font='Open Sans',
        pos=(0.5, -0.05), draggable=False, height=0.02, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    text_19 = visual.TextStim(win=win, name='text_19',
        text="Drag the slider, then click 'continue' to submit your answer.",
        font='Open Sans',
        pos=(0, 0.2), draggable=False, height=0.02, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    
    # --- Initialize components for Routine "invis" ---
    background_13 = visual.Rect(
        win=win, name='background_13',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    slider_3 = visual.Slider(win=win, name='slider_3',
        startValue=50, size=(1.0, 0.1), pos=(0, -0.15), units=win.units,
        labels=None, ticks=(0, 50, 100), granularity=0.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    text_20 = visual.TextStim(win=win, name='text_20',
        text='I currently feel unseen by other participant.',
        font='Open Sans',
        pos=(0, .3), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    polygon_3 = visual.Rect(
        win=win, name='polygon_3',
        width=(0.3, 0.1)[0], height=(0.3, 0.1)[1],
        ori=0.0, pos=(0, -0.35), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-4.0, interpolate=True)
    text_21 = visual.TextStim(win=win, name='text_21',
        text='Continue',
        font='Open Sans',
        pos=(0, -.35), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    mouse_5 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_5.mouseClock = core.Clock()
    q_rating_n_5 = visual.TextStim(win=win, name='q_rating_n_5',
        text='Not at all',
        font='Open Sans',
        pos=(-.5, -0.05), draggable=False, height=0.02, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    q_rating_n_6 = visual.TextStim(win=win, name='q_rating_n_6',
        text='Very much so',
        font='Open Sans',
        pos=(0.5, -0.05), draggable=False, height=0.02, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    text_22 = visual.TextStim(win=win, name='text_22',
        text="Drag the slider, then click 'continue' to submit your answer.",
        font='Open Sans',
        pos=(0, 0.2), draggable=False, height=0.02, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    
    # --- Initialize components for Routine "Begin" ---
    background_14 = visual.Rect(
        win=win, name='background_14',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    begin_text = visual.TextStim(win=win, name='begin_text',
        text='Press SPACEBAR to begin the task.',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_2 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "connect" ---
    text_4 = visual.TextStim(win=win, name='text_4',
        text='Waiting for other participant to connect...\n\nIf this takes more than 2 minutes, please try again later.\n',
        font='Open Sans',
        pos=(0, .15), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_29 = visual.TextStim(win=win, name='text_29',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    text_30 = visual.TextStim(win=win, name='text_30',
        text='Current wait time: ...',
        font='Arial',
        pos=(0, -.1), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    text_31 = visual.TextStim(win=win, name='text_31',
        text='Current wait time: Less than one minute',
        font='Arial',
        pos=(0, -.1), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "connected" ---
    text_5 = visual.TextStim(win=win, name='text_5',
        text='Connection successful.',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "Decision" ---
    background_15 = visual.Rect(
        win=win, name='background_15',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    dilemma_txt = visual.TextStim(win=win, name='dilemma_txt',
        text='',
        font='Open Sans',
        pos=(0, 0.35), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    poly_left = visual.Rect(
        win=win, name='poly_left',
        width=(0.4, 0.4)[0], height=(0.4, 0.4)[1],
        ori=0.0, pos=(-0.4, -.15), draggable=False, anchor='center',
        lineWidth=5.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, 0.0902], fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    poly_right = visual.Rect(
        win=win, name='poly_right',
        width=(0.4, 0.4)[0], height=(0.4, 0.4)[1],
        ori=0.0, pos=(0.4, -.15), draggable=False, anchor='center',
        lineWidth=5.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, 0.0902], fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    left_txt = visual.TextStim(win=win, name='left_txt',
        text='',
        font='Open Sans',
        pos=(-0.4, -.15), draggable=False, height=0.03, wrapWidth=0.35, ori=0.0, 
        color=[-1.0000, -1.0000, 0.0902], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    right_txt = visual.TextStim(win=win, name='right_txt',
        text='',
        font='Open Sans',
        pos=(0.4, -.15), draggable=False, height=0.03, wrapWidth=0.35, ori=0.0, 
        color=[-1.0000, -1.0000, 0.0902], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    mouse_decision = event.Mouse(win=win)
    x, y = [None, None]
    mouse_decision.mouseClock = core.Clock()
    happened_text = visual.TextStim(win=win, name='happened_text',
        text='If this happened to you, what would you be most likely to post on social media?',
        font='Open Sans',
        pos=(0, .2), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    text_2 = visual.TextStim(win=win, name='text_2',
        text='Click your choice.',
        font='Open Sans',
        pos=(0, 0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    
    # --- Initialize components for Routine "Wait_Choice" ---
    background_16 = visual.Rect(
        win=win, name='background_16',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    wait_txt_2 = visual.TextStim(win=win, name='wait_txt_2',
        text='Waiting for other participant to post...',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "Refresh" ---
    background_17 = visual.Rect(
        win=win, name='background_17',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    likes_txt = visual.TextStim(win=win, name='likes_txt',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, 0.0902], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    refresh_btn = visual.Rect(
        win=win, name='refresh_btn',
        width=(0.25, 0.1)[0], height=(0.25, 0.1)[1],
        ori=0.0, pos=(0, -.15), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-3.0, interpolate=True)
    refresh_click = visual.TextStim(win=win, name='refresh_click',
        text='click to refresh \nlike count',
        font='Open Sans',
        pos=(0, -0.15), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    mouse_refresh = event.Mouse(win=win)
    x, y = [None, None]
    mouse_refresh.mouseClock = core.Clock()
    timer_txt = visual.TextStim(win=win, name='timer_txt',
        text='',
        font='Open Sans',
        pos=(0, -.275), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    choice_box = visual.Rect(
        win=win, name='choice_box',
        width=(1, 0.25)[0], height=(1, 0.25)[1],
        ori=0.0, pos=(0, 0.25), draggable=False, anchor='center',
        lineWidth=5.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, 0.0902], fillColor='white',
        opacity=None, depth=-7.0, interpolate=True)
    choice_text = visual.TextStim(win=win, name='choice_text',
        text='',
        font='Open Sans',
        pos=(0, 0.25), draggable=False, height=0.03, wrapWidth=0.85, ori=0.0, 
        color=[-1.0000, -1.0000, 0.0902], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    reminder_text = visual.TextStim(win=win, name='reminder_text',
        text='It takes approximately 15 seconds for likes to come in from our servers.\nAs a reminder, each post can receive between 0-1000 likes.',
        font='Open Sans',
        pos=(0, -.35), draggable=False, height=0.02, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    your_choice_txt = visual.TextStim(win=win, name='your_choice_txt',
        text='You posted:',
        font='Open Sans',
        pos=(0, 0.4), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, 0.0902], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-10.0);
    
    # --- Initialize components for Routine "likes_revealed" ---
    background_18 = visual.Rect(
        win=win, name='background_18',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    text_3 = visual.TextStim(win=win, name='text_3',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, 0.0902], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    choice_box_2 = visual.Rect(
        win=win, name='choice_box_2',
        width=(1, 0.25)[0], height=(1, 0.25)[1],
        ori=0.0, pos=(0, 0.25), draggable=False, anchor='center',
        lineWidth=5.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, 0.0902], fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    choice_text_2 = visual.TextStim(win=win, name='choice_text_2',
        text='',
        font='Open Sans',
        pos=(0, 0.25), draggable=False, height=0.03, wrapWidth=0.85, ori=0.0, 
        color=[-1.0000, -1.0000, 0.0902], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    your_choice_txt_2 = visual.TextStim(win=win, name='your_choice_txt_2',
        text='You posted:',
        font='Open Sans',
        pos=(0, 0.4), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, 0.0902], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    
    # --- Initialize components for Routine "Outcome" ---
    background_19 = visual.Rect(
        win=win, name='background_19',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    round_txt = visual.TextStim(win=win, name='round_txt',
        text='',
        font='Open Sans',
        pos=(0, 0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    you_choice_box = visual.Rect(
        win=win, name='you_choice_box',
        width=(0.4, 0.4)[0], height=(0.4, 0.4)[1],
        ori=0.0, pos=(-.4, 0), draggable=False, anchor='center',
        lineWidth=5.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, 0.0902], fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    you_chose_txt = visual.TextStim(win=win, name='you_chose_txt',
        text='',
        font='Open Sans',
        pos=(-.4, 0), draggable=False, height=0.03, wrapWidth=0.35, ori=0.0, 
        color=[-1.0000, -1.0000, 0.0902], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    partner_choice_box = visual.Rect(
        win=win, name='partner_choice_box',
        width=(0.4,0.4)[0], height=(0.4,0.4)[1],
        ori=0.0, pos=(0.4, 0), draggable=False, anchor='center',
        lineWidth=5.0,
        colorSpace='rgb', lineColor=[-1.0000, -0.2157, -1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-5.0, interpolate=True)
    partner_chose_txt = visual.TextStim(win=win, name='partner_chose_txt',
        text='',
        font='Open Sans',
        pos=(0.4, 0), draggable=False, height=0.03, wrapWidth=0.35, ori=0.0, 
        color=[-1.0000, -0.2157, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    click_box = visual.Rect(
        win=win, name='click_box',
        width=(0.35, 0.15)[0], height=(0.35, 0.15)[1],
        ori=0.0, pos=(0, -0.35), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-7.0, interpolate=True)
    continue_click = visual.TextStim(win=win, name='continue_click',
        text='Click to continue',
        font='Open Sans',
        pos=(0, -0.35), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    mouse = event.Mouse(win=win)
    x, y = [None, None]
    mouse.mouseClock = core.Clock()
    you_posted = visual.TextStim(win=win, name='you_posted',
        text='You posted: ',
        font='Open Sans',
        pos=(-.4, 0.25), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, 0.0902], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-10.0);
    they_posted = visual.TextStim(win=win, name='they_posted',
        text='They posted: ',
        font='Open Sans',
        pos=(.4, 0.25), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -0.2157, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-11.0);
    remember_view = visual.TextStim(win=win, name='remember_view',
        text='Remember: the other participant can also see what you posted.',
        font='Open Sans',
        pos=(0, .325), draggable=False, height=0.02, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-12.0);
    
    # --- Initialize components for Routine "WaitScreen" ---
    background_20 = visual.Rect(
        win=win, name='background_20',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    wait_txt = visual.TextStim(win=win, name='wait_txt',
        text='Waiting for other participant to continue...',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "end" ---
    text = visual.TextStim(win=win, name='text',
        text='DO NOT EXIT OR CLICK ESC. \n\nThank you! You wll now be redirected to complete two brief surveys.\n\nNote: you must complete the surveys in order to receive credit for this experiment.\n\nClick SPACEBAR to safely exit the experiment. click ok when complete.',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_15 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    if eyetracker is not None:
        eyetracker.enableEventReporting()
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "Begin_demo" ---
    # create an object to store info about Routine Begin_demo
    Begin_demo = data.Routine(
        name='Begin_demo',
        components=[background, begin_text_2, key_resp_3],
    )
    Begin_demo.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_3
    key_resp_3.keys = []
    key_resp_3.rt = []
    _key_resp_3_allKeys = []
    # store start times for Begin_demo
    Begin_demo.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Begin_demo.tStart = globalClock.getTime(format='float')
    Begin_demo.status = STARTED
    thisExp.addData('Begin_demo.started', Begin_demo.tStart)
    Begin_demo.maxDuration = None
    # keep track of which components have finished
    Begin_demoComponents = Begin_demo.components
    for thisComponent in Begin_demo.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Begin_demo" ---
    thisExp.currentRoutine = Begin_demo
    Begin_demo.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *background* updates
        
        # if background is starting this frame...
        if background.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            background.frameNStart = frameN  # exact frame index
            background.tStart = t  # local t and not account for scr refresh
            background.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(background, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'background.started')
            # update status
            background.status = STARTED
            background.setAutoDraw(True)
        
        # if background is active this frame...
        if background.status == STARTED:
            # update params
            pass
        
        # *begin_text_2* updates
        
        # if begin_text_2 is starting this frame...
        if begin_text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            begin_text_2.frameNStart = frameN  # exact frame index
            begin_text_2.tStart = t  # local t and not account for scr refresh
            begin_text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(begin_text_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'begin_text_2.started')
            # update status
            begin_text_2.status = STARTED
            begin_text_2.setAutoDraw(True)
        
        # if begin_text_2 is active this frame...
        if begin_text_2.status == STARTED:
            # update params
            pass
        
        # *key_resp_3* updates
        waitOnFlip = False
        
        # if key_resp_3 is starting this frame...
        if key_resp_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_3.frameNStart = frameN  # exact frame index
            key_resp_3.tStart = t  # local t and not account for scr refresh
            key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_3.started')
            # update status
            key_resp_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_3.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_3.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_3_allKeys.extend(theseKeys)
            if len(_key_resp_3_allKeys):
                key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
                key_resp_3.rt = _key_resp_3_allKeys[-1].rt
                key_resp_3.duration = _key_resp_3_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Begin_demo,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Begin_demo.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Begin_demo.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Begin_demo.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Begin_demo" ---
    for thisComponent in Begin_demo.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Begin_demo
    Begin_demo.tStop = globalClock.getTime(format='float')
    Begin_demo.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Begin_demo.stopped', Begin_demo.tStop)
    # check responses
    if key_resp_3.keys in ['', [], None]:  # No response was made
        key_resp_3.keys = None
    thisExp.addData('key_resp_3.keys',key_resp_3.keys)
    if key_resp_3.keys != None:  # we had a response
        thisExp.addData('key_resp_3.rt', key_resp_3.rt)
        thisExp.addData('key_resp_3.duration', key_resp_3.duration)
    thisExp.nextEntry()
    # the Routine "Begin_demo" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "demo_1" ---
    # create an object to store info about Routine demo_1
    demo_1 = data.Routine(
        name='demo_1',
        components=[background_2, text_6, key_resp_4],
    )
    demo_1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_4
    key_resp_4.keys = []
    key_resp_4.rt = []
    _key_resp_4_allKeys = []
    # store start times for demo_1
    demo_1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    demo_1.tStart = globalClock.getTime(format='float')
    demo_1.status = STARTED
    thisExp.addData('demo_1.started', demo_1.tStart)
    demo_1.maxDuration = None
    # keep track of which components have finished
    demo_1Components = demo_1.components
    for thisComponent in demo_1.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "demo_1" ---
    thisExp.currentRoutine = demo_1
    demo_1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *background_2* updates
        
        # if background_2 is starting this frame...
        if background_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            background_2.frameNStart = frameN  # exact frame index
            background_2.tStart = t  # local t and not account for scr refresh
            background_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(background_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'background_2.started')
            # update status
            background_2.status = STARTED
            background_2.setAutoDraw(True)
        
        # if background_2 is active this frame...
        if background_2.status == STARTED:
            # update params
            pass
        
        # *text_6* updates
        
        # if text_6 is starting this frame...
        if text_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_6.frameNStart = frameN  # exact frame index
            text_6.tStart = t  # local t and not account for scr refresh
            text_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_6.started')
            # update status
            text_6.status = STARTED
            text_6.setAutoDraw(True)
        
        # if text_6 is active this frame...
        if text_6.status == STARTED:
            # update params
            pass
        
        # *key_resp_4* updates
        waitOnFlip = False
        
        # if key_resp_4 is starting this frame...
        if key_resp_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_4.frameNStart = frameN  # exact frame index
            key_resp_4.tStart = t  # local t and not account for scr refresh
            key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_4.started')
            # update status
            key_resp_4.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_4.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_4.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_4.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_4_allKeys.extend(theseKeys)
            if len(_key_resp_4_allKeys):
                key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
                key_resp_4.rt = _key_resp_4_allKeys[-1].rt
                key_resp_4.duration = _key_resp_4_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=demo_1,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            demo_1.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if demo_1.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in demo_1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "demo_1" ---
    for thisComponent in demo_1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for demo_1
    demo_1.tStop = globalClock.getTime(format='float')
    demo_1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('demo_1.stopped', demo_1.tStop)
    # check responses
    if key_resp_4.keys in ['', [], None]:  # No response was made
        key_resp_4.keys = None
    thisExp.addData('key_resp_4.keys',key_resp_4.keys)
    if key_resp_4.keys != None:  # we had a response
        thisExp.addData('key_resp_4.rt', key_resp_4.rt)
        thisExp.addData('key_resp_4.duration', key_resp_4.duration)
    thisExp.nextEntry()
    # the Routine "demo_1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "demo1_c" ---
    # create an object to store info about Routine demo1_c
    demo1_c = data.Routine(
        name='demo1_c',
        components=[background_22, text_24, key_resp_11],
    )
    demo1_c.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_11
    key_resp_11.keys = []
    key_resp_11.rt = []
    _key_resp_11_allKeys = []
    # store start times for demo1_c
    demo1_c.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    demo1_c.tStart = globalClock.getTime(format='float')
    demo1_c.status = STARTED
    thisExp.addData('demo1_c.started', demo1_c.tStart)
    demo1_c.maxDuration = None
    # keep track of which components have finished
    demo1_cComponents = demo1_c.components
    for thisComponent in demo1_c.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "demo1_c" ---
    thisExp.currentRoutine = demo1_c
    demo1_c.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *background_22* updates
        
        # if background_22 is starting this frame...
        if background_22.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            background_22.frameNStart = frameN  # exact frame index
            background_22.tStart = t  # local t and not account for scr refresh
            background_22.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(background_22, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'background_22.started')
            # update status
            background_22.status = STARTED
            background_22.setAutoDraw(True)
        
        # if background_22 is active this frame...
        if background_22.status == STARTED:
            # update params
            pass
        
        # *text_24* updates
        
        # if text_24 is starting this frame...
        if text_24.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_24.frameNStart = frameN  # exact frame index
            text_24.tStart = t  # local t and not account for scr refresh
            text_24.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_24, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_24.started')
            # update status
            text_24.status = STARTED
            text_24.setAutoDraw(True)
        
        # if text_24 is active this frame...
        if text_24.status == STARTED:
            # update params
            pass
        
        # *key_resp_11* updates
        waitOnFlip = False
        
        # if key_resp_11 is starting this frame...
        if key_resp_11.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_11.frameNStart = frameN  # exact frame index
            key_resp_11.tStart = t  # local t and not account for scr refresh
            key_resp_11.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_11, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_11.started')
            # update status
            key_resp_11.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_11.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_11.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_11.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_11.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_11_allKeys.extend(theseKeys)
            if len(_key_resp_11_allKeys):
                key_resp_11.keys = _key_resp_11_allKeys[-1].name  # just the last key pressed
                key_resp_11.rt = _key_resp_11_allKeys[-1].rt
                key_resp_11.duration = _key_resp_11_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=demo1_c,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            demo1_c.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if demo1_c.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in demo1_c.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "demo1_c" ---
    for thisComponent in demo1_c.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for demo1_c
    demo1_c.tStop = globalClock.getTime(format='float')
    demo1_c.tStopRefresh = tThisFlipGlobal
    thisExp.addData('demo1_c.stopped', demo1_c.tStop)
    # check responses
    if key_resp_11.keys in ['', [], None]:  # No response was made
        key_resp_11.keys = None
    thisExp.addData('key_resp_11.keys',key_resp_11.keys)
    if key_resp_11.keys != None:  # we had a response
        thisExp.addData('key_resp_11.rt', key_resp_11.rt)
        thisExp.addData('key_resp_11.duration', key_resp_11.duration)
    thisExp.nextEntry()
    # the Routine "demo1_c" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "demo1_c_2" ---
    # create an object to store info about Routine demo1_c_2
    demo1_c_2 = data.Routine(
        name='demo1_c_2',
        components=[background_23, text_25, key_resp_12],
    )
    demo1_c_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_12
    key_resp_12.keys = []
    key_resp_12.rt = []
    _key_resp_12_allKeys = []
    # store start times for demo1_c_2
    demo1_c_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    demo1_c_2.tStart = globalClock.getTime(format='float')
    demo1_c_2.status = STARTED
    thisExp.addData('demo1_c_2.started', demo1_c_2.tStart)
    demo1_c_2.maxDuration = None
    # keep track of which components have finished
    demo1_c_2Components = demo1_c_2.components
    for thisComponent in demo1_c_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "demo1_c_2" ---
    thisExp.currentRoutine = demo1_c_2
    demo1_c_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *background_23* updates
        
        # if background_23 is starting this frame...
        if background_23.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            background_23.frameNStart = frameN  # exact frame index
            background_23.tStart = t  # local t and not account for scr refresh
            background_23.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(background_23, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'background_23.started')
            # update status
            background_23.status = STARTED
            background_23.setAutoDraw(True)
        
        # if background_23 is active this frame...
        if background_23.status == STARTED:
            # update params
            pass
        
        # *text_25* updates
        
        # if text_25 is starting this frame...
        if text_25.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_25.frameNStart = frameN  # exact frame index
            text_25.tStart = t  # local t and not account for scr refresh
            text_25.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_25, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_25.started')
            # update status
            text_25.status = STARTED
            text_25.setAutoDraw(True)
        
        # if text_25 is active this frame...
        if text_25.status == STARTED:
            # update params
            pass
        
        # *key_resp_12* updates
        waitOnFlip = False
        
        # if key_resp_12 is starting this frame...
        if key_resp_12.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_12.frameNStart = frameN  # exact frame index
            key_resp_12.tStart = t  # local t and not account for scr refresh
            key_resp_12.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_12, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_12.started')
            # update status
            key_resp_12.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_12.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_12.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_12.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_12.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_12_allKeys.extend(theseKeys)
            if len(_key_resp_12_allKeys):
                key_resp_12.keys = _key_resp_12_allKeys[-1].name  # just the last key pressed
                key_resp_12.rt = _key_resp_12_allKeys[-1].rt
                key_resp_12.duration = _key_resp_12_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=demo1_c_2,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            demo1_c_2.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if demo1_c_2.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in demo1_c_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "demo1_c_2" ---
    for thisComponent in demo1_c_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for demo1_c_2
    demo1_c_2.tStop = globalClock.getTime(format='float')
    demo1_c_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('demo1_c_2.stopped', demo1_c_2.tStop)
    # check responses
    if key_resp_12.keys in ['', [], None]:  # No response was made
        key_resp_12.keys = None
    thisExp.addData('key_resp_12.keys',key_resp_12.keys)
    if key_resp_12.keys != None:  # we had a response
        thisExp.addData('key_resp_12.rt', key_resp_12.rt)
        thisExp.addData('key_resp_12.duration', key_resp_12.duration)
    thisExp.nextEntry()
    # the Routine "demo1_c_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "demo1_c_3" ---
    # create an object to store info about Routine demo1_c_3
    demo1_c_3 = data.Routine(
        name='demo1_c_3',
        components=[background_24, text_26, key_resp_13],
    )
    demo1_c_3.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_13
    key_resp_13.keys = []
    key_resp_13.rt = []
    _key_resp_13_allKeys = []
    # store start times for demo1_c_3
    demo1_c_3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    demo1_c_3.tStart = globalClock.getTime(format='float')
    demo1_c_3.status = STARTED
    thisExp.addData('demo1_c_3.started', demo1_c_3.tStart)
    demo1_c_3.maxDuration = None
    # keep track of which components have finished
    demo1_c_3Components = demo1_c_3.components
    for thisComponent in demo1_c_3.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "demo1_c_3" ---
    thisExp.currentRoutine = demo1_c_3
    demo1_c_3.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *background_24* updates
        
        # if background_24 is starting this frame...
        if background_24.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            background_24.frameNStart = frameN  # exact frame index
            background_24.tStart = t  # local t and not account for scr refresh
            background_24.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(background_24, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'background_24.started')
            # update status
            background_24.status = STARTED
            background_24.setAutoDraw(True)
        
        # if background_24 is active this frame...
        if background_24.status == STARTED:
            # update params
            pass
        
        # *text_26* updates
        
        # if text_26 is starting this frame...
        if text_26.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_26.frameNStart = frameN  # exact frame index
            text_26.tStart = t  # local t and not account for scr refresh
            text_26.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_26, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_26.started')
            # update status
            text_26.status = STARTED
            text_26.setAutoDraw(True)
        
        # if text_26 is active this frame...
        if text_26.status == STARTED:
            # update params
            pass
        
        # *key_resp_13* updates
        waitOnFlip = False
        
        # if key_resp_13 is starting this frame...
        if key_resp_13.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_13.frameNStart = frameN  # exact frame index
            key_resp_13.tStart = t  # local t and not account for scr refresh
            key_resp_13.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_13, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_13.started')
            # update status
            key_resp_13.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_13.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_13.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_13.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_13.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_13_allKeys.extend(theseKeys)
            if len(_key_resp_13_allKeys):
                key_resp_13.keys = _key_resp_13_allKeys[-1].name  # just the last key pressed
                key_resp_13.rt = _key_resp_13_allKeys[-1].rt
                key_resp_13.duration = _key_resp_13_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=demo1_c_3,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            demo1_c_3.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if demo1_c_3.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in demo1_c_3.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "demo1_c_3" ---
    for thisComponent in demo1_c_3.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for demo1_c_3
    demo1_c_3.tStop = globalClock.getTime(format='float')
    demo1_c_3.tStopRefresh = tThisFlipGlobal
    thisExp.addData('demo1_c_3.stopped', demo1_c_3.tStop)
    # check responses
    if key_resp_13.keys in ['', [], None]:  # No response was made
        key_resp_13.keys = None
    thisExp.addData('key_resp_13.keys',key_resp_13.keys)
    if key_resp_13.keys != None:  # we had a response
        thisExp.addData('key_resp_13.rt', key_resp_13.rt)
        thisExp.addData('key_resp_13.duration', key_resp_13.duration)
    thisExp.nextEntry()
    # the Routine "demo1_c_3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "demo1_c_4" ---
    # create an object to store info about Routine demo1_c_4
    demo1_c_4 = data.Routine(
        name='demo1_c_4',
        components=[background_25, text_27, key_resp_14],
    )
    demo1_c_4.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_14
    key_resp_14.keys = []
    key_resp_14.rt = []
    _key_resp_14_allKeys = []
    # store start times for demo1_c_4
    demo1_c_4.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    demo1_c_4.tStart = globalClock.getTime(format='float')
    demo1_c_4.status = STARTED
    thisExp.addData('demo1_c_4.started', demo1_c_4.tStart)
    demo1_c_4.maxDuration = None
    # keep track of which components have finished
    demo1_c_4Components = demo1_c_4.components
    for thisComponent in demo1_c_4.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "demo1_c_4" ---
    thisExp.currentRoutine = demo1_c_4
    demo1_c_4.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *background_25* updates
        
        # if background_25 is starting this frame...
        if background_25.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            background_25.frameNStart = frameN  # exact frame index
            background_25.tStart = t  # local t and not account for scr refresh
            background_25.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(background_25, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'background_25.started')
            # update status
            background_25.status = STARTED
            background_25.setAutoDraw(True)
        
        # if background_25 is active this frame...
        if background_25.status == STARTED:
            # update params
            pass
        
        # *text_27* updates
        
        # if text_27 is starting this frame...
        if text_27.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_27.frameNStart = frameN  # exact frame index
            text_27.tStart = t  # local t and not account for scr refresh
            text_27.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_27, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_27.started')
            # update status
            text_27.status = STARTED
            text_27.setAutoDraw(True)
        
        # if text_27 is active this frame...
        if text_27.status == STARTED:
            # update params
            pass
        
        # *key_resp_14* updates
        waitOnFlip = False
        
        # if key_resp_14 is starting this frame...
        if key_resp_14.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_14.frameNStart = frameN  # exact frame index
            key_resp_14.tStart = t  # local t and not account for scr refresh
            key_resp_14.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_14, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_14.started')
            # update status
            key_resp_14.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_14.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_14.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_14.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_14.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_14_allKeys.extend(theseKeys)
            if len(_key_resp_14_allKeys):
                key_resp_14.keys = _key_resp_14_allKeys[-1].name  # just the last key pressed
                key_resp_14.rt = _key_resp_14_allKeys[-1].rt
                key_resp_14.duration = _key_resp_14_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=demo1_c_4,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            demo1_c_4.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if demo1_c_4.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in demo1_c_4.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "demo1_c_4" ---
    for thisComponent in demo1_c_4.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for demo1_c_4
    demo1_c_4.tStop = globalClock.getTime(format='float')
    demo1_c_4.tStopRefresh = tThisFlipGlobal
    thisExp.addData('demo1_c_4.stopped', demo1_c_4.tStop)
    # check responses
    if key_resp_14.keys in ['', [], None]:  # No response was made
        key_resp_14.keys = None
    thisExp.addData('key_resp_14.keys',key_resp_14.keys)
    if key_resp_14.keys != None:  # we had a response
        thisExp.addData('key_resp_14.rt', key_resp_14.rt)
        thisExp.addData('key_resp_14.duration', key_resp_14.duration)
    thisExp.nextEntry()
    # the Routine "demo1_c_4" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "demo_2" ---
    # create an object to store info about Routine demo_2
    demo_2 = data.Routine(
        name='demo_2',
        components=[background_3, dilemma_txt_2, poly_left_2, poly_right_2, left_txt_2, right_txt_2, happened_text_2, mouse_2],
    )
    demo_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    dilemma_txt_2.setText('The Social Dilemma will be at the top of the page, in black text.')
    left_txt_2.setText('Option 1')
    right_txt_2.setText('Option 2')
    # setup some python lists for storing info about the mouse_2
    mouse_2.x = []
    mouse_2.y = []
    mouse_2.leftButton = []
    mouse_2.midButton = []
    mouse_2.rightButton = []
    mouse_2.time = []
    mouse_2.clicked_name = []
    gotValidClick = False  # until a click is received
    # store start times for demo_2
    demo_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    demo_2.tStart = globalClock.getTime(format='float')
    demo_2.status = STARTED
    thisExp.addData('demo_2.started', demo_2.tStart)
    demo_2.maxDuration = None
    # keep track of which components have finished
    demo_2Components = demo_2.components
    for thisComponent in demo_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "demo_2" ---
    thisExp.currentRoutine = demo_2
    demo_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *background_3* updates
        
        # if background_3 is starting this frame...
        if background_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            background_3.frameNStart = frameN  # exact frame index
            background_3.tStart = t  # local t and not account for scr refresh
            background_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(background_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'background_3.started')
            # update status
            background_3.status = STARTED
            background_3.setAutoDraw(True)
        
        # if background_3 is active this frame...
        if background_3.status == STARTED:
            # update params
            pass
        
        # *dilemma_txt_2* updates
        
        # if dilemma_txt_2 is starting this frame...
        if dilemma_txt_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            dilemma_txt_2.frameNStart = frameN  # exact frame index
            dilemma_txt_2.tStart = t  # local t and not account for scr refresh
            dilemma_txt_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(dilemma_txt_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'dilemma_txt_2.started')
            # update status
            dilemma_txt_2.status = STARTED
            dilemma_txt_2.setAutoDraw(True)
        
        # if dilemma_txt_2 is active this frame...
        if dilemma_txt_2.status == STARTED:
            # update params
            pass
        
        # *poly_left_2* updates
        
        # if poly_left_2 is starting this frame...
        if poly_left_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            poly_left_2.frameNStart = frameN  # exact frame index
            poly_left_2.tStart = t  # local t and not account for scr refresh
            poly_left_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(poly_left_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'poly_left_2.started')
            # update status
            poly_left_2.status = STARTED
            poly_left_2.setAutoDraw(True)
        
        # if poly_left_2 is active this frame...
        if poly_left_2.status == STARTED:
            # update params
            pass
        
        # *poly_right_2* updates
        
        # if poly_right_2 is starting this frame...
        if poly_right_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            poly_right_2.frameNStart = frameN  # exact frame index
            poly_right_2.tStart = t  # local t and not account for scr refresh
            poly_right_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(poly_right_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'poly_right_2.started')
            # update status
            poly_right_2.status = STARTED
            poly_right_2.setAutoDraw(True)
        
        # if poly_right_2 is active this frame...
        if poly_right_2.status == STARTED:
            # update params
            pass
        
        # *left_txt_2* updates
        
        # if left_txt_2 is starting this frame...
        if left_txt_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            left_txt_2.frameNStart = frameN  # exact frame index
            left_txt_2.tStart = t  # local t and not account for scr refresh
            left_txt_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(left_txt_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'left_txt_2.started')
            # update status
            left_txt_2.status = STARTED
            left_txt_2.setAutoDraw(True)
        
        # if left_txt_2 is active this frame...
        if left_txt_2.status == STARTED:
            # update params
            pass
        
        # *right_txt_2* updates
        
        # if right_txt_2 is starting this frame...
        if right_txt_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            right_txt_2.frameNStart = frameN  # exact frame index
            right_txt_2.tStart = t  # local t and not account for scr refresh
            right_txt_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(right_txt_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'right_txt_2.started')
            # update status
            right_txt_2.status = STARTED
            right_txt_2.setAutoDraw(True)
        
        # if right_txt_2 is active this frame...
        if right_txt_2.status == STARTED:
            # update params
            pass
        
        # *happened_text_2* updates
        
        # if happened_text_2 is starting this frame...
        if happened_text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            happened_text_2.frameNStart = frameN  # exact frame index
            happened_text_2.tStart = t  # local t and not account for scr refresh
            happened_text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(happened_text_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'happened_text_2.started')
            # update status
            happened_text_2.status = STARTED
            happened_text_2.setAutoDraw(True)
        
        # if happened_text_2 is active this frame...
        if happened_text_2.status == STARTED:
            # update params
            pass
        # *mouse_2* updates
        
        # if mouse_2 is starting this frame...
        if mouse_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_2.frameNStart = frameN  # exact frame index
            mouse_2.tStart = t  # local t and not account for scr refresh
            mouse_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('mouse_2.started', t)
            # update status
            mouse_2.status = STARTED
            mouse_2.mouseClock.reset()
            prevButtonState = mouse_2.getPressed()  # if button is down already this ISN'T a new click
        if mouse_2.status == STARTED:  # only update if started and not finished!
            buttons = mouse_2.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames([poly_left_2, poly_right_2], namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse_2):
                            gotValidClick = True
                            mouse_2.clicked_name.append(obj.name)
                    if not gotValidClick:
                        mouse_2.clicked_name.append(None)
                    x, y = mouse_2.getPos()
                    mouse_2.x.append(float(x))
                    mouse_2.y.append(float(y))
                    buttons = mouse_2.getPressed()
                    mouse_2.leftButton.append(buttons[0])
                    mouse_2.midButton.append(buttons[1])
                    mouse_2.rightButton.append(buttons[2])
                    mouse_2.time.append(mouse_2.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=demo_2,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            demo_2.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if demo_2.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in demo_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "demo_2" ---
    for thisComponent in demo_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for demo_2
    demo_2.tStop = globalClock.getTime(format='float')
    demo_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('demo_2.stopped', demo_2.tStop)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_2.x', mouse_2.x)
    thisExp.addData('mouse_2.y', mouse_2.y)
    thisExp.addData('mouse_2.leftButton', mouse_2.leftButton)
    thisExp.addData('mouse_2.midButton', mouse_2.midButton)
    thisExp.addData('mouse_2.rightButton', mouse_2.rightButton)
    thisExp.addData('mouse_2.time', mouse_2.time)
    thisExp.addData('mouse_2.clicked_name', mouse_2.clicked_name)
    thisExp.nextEntry()
    # the Routine "demo_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "demo_3" ---
    # create an object to store info about Routine demo_3
    demo_3 = data.Routine(
        name='demo_3',
        components=[background_4, text_7, key_resp_5],
    )
    demo_3.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_5
    key_resp_5.keys = []
    key_resp_5.rt = []
    _key_resp_5_allKeys = []
    # store start times for demo_3
    demo_3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    demo_3.tStart = globalClock.getTime(format='float')
    demo_3.status = STARTED
    thisExp.addData('demo_3.started', demo_3.tStart)
    demo_3.maxDuration = None
    # keep track of which components have finished
    demo_3Components = demo_3.components
    for thisComponent in demo_3.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "demo_3" ---
    thisExp.currentRoutine = demo_3
    demo_3.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *background_4* updates
        
        # if background_4 is starting this frame...
        if background_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            background_4.frameNStart = frameN  # exact frame index
            background_4.tStart = t  # local t and not account for scr refresh
            background_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(background_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'background_4.started')
            # update status
            background_4.status = STARTED
            background_4.setAutoDraw(True)
        
        # if background_4 is active this frame...
        if background_4.status == STARTED:
            # update params
            pass
        
        # *text_7* updates
        
        # if text_7 is starting this frame...
        if text_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_7.frameNStart = frameN  # exact frame index
            text_7.tStart = t  # local t and not account for scr refresh
            text_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_7.started')
            # update status
            text_7.status = STARTED
            text_7.setAutoDraw(True)
        
        # if text_7 is active this frame...
        if text_7.status == STARTED:
            # update params
            pass
        
        # *key_resp_5* updates
        waitOnFlip = False
        
        # if key_resp_5 is starting this frame...
        if key_resp_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_5.frameNStart = frameN  # exact frame index
            key_resp_5.tStart = t  # local t and not account for scr refresh
            key_resp_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_5.started')
            # update status
            key_resp_5.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_5.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_5.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_5.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_5.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_5_allKeys.extend(theseKeys)
            if len(_key_resp_5_allKeys):
                key_resp_5.keys = _key_resp_5_allKeys[-1].name  # just the last key pressed
                key_resp_5.rt = _key_resp_5_allKeys[-1].rt
                key_resp_5.duration = _key_resp_5_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=demo_3,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            demo_3.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if demo_3.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in demo_3.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "demo_3" ---
    for thisComponent in demo_3.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for demo_3
    demo_3.tStop = globalClock.getTime(format='float')
    demo_3.tStopRefresh = tThisFlipGlobal
    thisExp.addData('demo_3.stopped', demo_3.tStop)
    # check responses
    if key_resp_5.keys in ['', [], None]:  # No response was made
        key_resp_5.keys = None
    thisExp.addData('key_resp_5.keys',key_resp_5.keys)
    if key_resp_5.keys != None:  # we had a response
        thisExp.addData('key_resp_5.rt', key_resp_5.rt)
        thisExp.addData('key_resp_5.duration', key_resp_5.duration)
    thisExp.nextEntry()
    # the Routine "demo_3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "demo_4" ---
    # create an object to store info about Routine demo_4
    demo_4 = data.Routine(
        name='demo_4',
        components=[background_5, text_8, key_resp_6],
    )
    demo_4.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_6
    key_resp_6.keys = []
    key_resp_6.rt = []
    _key_resp_6_allKeys = []
    # store start times for demo_4
    demo_4.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    demo_4.tStart = globalClock.getTime(format='float')
    demo_4.status = STARTED
    thisExp.addData('demo_4.started', demo_4.tStart)
    demo_4.maxDuration = None
    # keep track of which components have finished
    demo_4Components = demo_4.components
    for thisComponent in demo_4.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "demo_4" ---
    thisExp.currentRoutine = demo_4
    demo_4.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *background_5* updates
        
        # if background_5 is starting this frame...
        if background_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            background_5.frameNStart = frameN  # exact frame index
            background_5.tStart = t  # local t and not account for scr refresh
            background_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(background_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'background_5.started')
            # update status
            background_5.status = STARTED
            background_5.setAutoDraw(True)
        
        # if background_5 is active this frame...
        if background_5.status == STARTED:
            # update params
            pass
        
        # *text_8* updates
        
        # if text_8 is starting this frame...
        if text_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_8.frameNStart = frameN  # exact frame index
            text_8.tStart = t  # local t and not account for scr refresh
            text_8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_8, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_8.started')
            # update status
            text_8.status = STARTED
            text_8.setAutoDraw(True)
        
        # if text_8 is active this frame...
        if text_8.status == STARTED:
            # update params
            pass
        
        # *key_resp_6* updates
        waitOnFlip = False
        
        # if key_resp_6 is starting this frame...
        if key_resp_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_6.frameNStart = frameN  # exact frame index
            key_resp_6.tStart = t  # local t and not account for scr refresh
            key_resp_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_6.started')
            # update status
            key_resp_6.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_6.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_6.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_6.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_6.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_6_allKeys.extend(theseKeys)
            if len(_key_resp_6_allKeys):
                key_resp_6.keys = _key_resp_6_allKeys[-1].name  # just the last key pressed
                key_resp_6.rt = _key_resp_6_allKeys[-1].rt
                key_resp_6.duration = _key_resp_6_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=demo_4,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            demo_4.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if demo_4.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in demo_4.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "demo_4" ---
    for thisComponent in demo_4.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for demo_4
    demo_4.tStop = globalClock.getTime(format='float')
    demo_4.tStopRefresh = tThisFlipGlobal
    thisExp.addData('demo_4.stopped', demo_4.tStop)
    # check responses
    if key_resp_6.keys in ['', [], None]:  # No response was made
        key_resp_6.keys = None
    thisExp.addData('key_resp_6.keys',key_resp_6.keys)
    if key_resp_6.keys != None:  # we had a response
        thisExp.addData('key_resp_6.rt', key_resp_6.rt)
        thisExp.addData('key_resp_6.duration', key_resp_6.duration)
    thisExp.nextEntry()
    # the Routine "demo_4" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "demo_5" ---
    # create an object to store info about Routine demo_5
    demo_5 = data.Routine(
        name='demo_5',
        components=[background_6, refresh_btn_2, refresh_click_2, likes_txt_2, timer_txt_2, reminder_text_2, demo_mouse],
    )
    demo_5.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code
    # ----------------- DEMO Begin Routine -----------------
    demo_refresh_duration = 15.0      # seconds
    demo_likes_cap = 500              # set None to remove cap
    demo_likes_per_frame = 1          # change to 2,3,... if you want faster growth per frame
    
    demo_current_likes = 0
    demo_likes_display = f"Likes: {demo_current_likes}"
    demo_timer_display = f"{int(demo_refresh_duration)}"
    
    demo_start_time = globalClock.getTime()
    
    # ADDED for edge detection + invisible accumulation target
    demo_prev_buttons = [0, 0, 0]
    # use the cap as the accumulation target (fallback to 500 if None)
    demo_accum_target = demo_likes_cap if demo_likes_cap is not None else 500
    
    # setup some python lists for storing info about the demo_mouse
    demo_mouse.x = []
    demo_mouse.y = []
    demo_mouse.leftButton = []
    demo_mouse.midButton = []
    demo_mouse.rightButton = []
    demo_mouse.time = []
    gotValidClick = False  # until a click is received
    # store start times for demo_5
    demo_5.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    demo_5.tStart = globalClock.getTime(format='float')
    demo_5.status = STARTED
    thisExp.addData('demo_5.started', demo_5.tStart)
    demo_5.maxDuration = None
    # keep track of which components have finished
    demo_5Components = demo_5.components
    for thisComponent in demo_5.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "demo_5" ---
    thisExp.currentRoutine = demo_5
    demo_5.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from code
        # ----------------- DEMO Begin Routine -----------------
        demo_refresh_duration = 15.0      # seconds
        demo_likes_cap = 500              # set None to remove cap
        demo_likes_per_frame = 1          # change to 2,3,... if you want faster growth per frame
        
        demo_current_likes = 0
        demo_likes_display = f"Likes: {demo_current_likes}"
        demo_timer_display = f"{int(demo_refresh_duration)}"
        
        demo_start_time = t
        
        # ADDED for edge detection + invisible accumulation target
        demo_prev_buttons = [0, 0, 0]
        # use the cap as the accumulation target (fallback to 500 if None)
        demo_accum_target = demo_likes_cap if demo_likes_cap is not None else 500
        
        
        # ----------------- DEMO Each Frame -----------------
        elapsed_demo = t - demo_start_time
        
        # ADDED: compute invisible accumulation (linear ramp toward target over the routine)
        demo_accum_you = int(demo_accum_target * min(1.0, elapsed_demo / demo_refresh_duration))
        
        # REPLACED: edge-detected click (down event) on refresh_btn_2 reveals accumulated total
        _buttons = demo_mouse.getPressed()
        if (_buttons[0] == 1) and (demo_prev_buttons[0] == 0) and demo_mouse.isPressedIn(refresh_btn_2):
            if demo_accum_you > demo_current_likes:
                demo_current_likes = demo_accum_you
        demo_prev_buttons = _buttons
        
        # cap
        if (demo_likes_cap is not None) and (demo_current_likes > demo_likes_cap):
            demo_current_likes = demo_likes_cap
        
        # update displays
        demo_likes_display = f"Likes: {demo_current_likes}"
        remaining_demo = max(0, demo_refresh_duration - elapsed_demo)
        demo_timer_display = f"{int(remaining_demo)}"
        
        # end after 15s
        if elapsed_demo >= demo_refresh_duration:
            continueRoutine = False
        
        
        # *background_6* updates
        
        # if background_6 is starting this frame...
        if background_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            background_6.frameNStart = frameN  # exact frame index
            background_6.tStart = t  # local t and not account for scr refresh
            background_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(background_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'background_6.started')
            # update status
            background_6.status = STARTED
            background_6.setAutoDraw(True)
        
        # if background_6 is active this frame...
        if background_6.status == STARTED:
            # update params
            pass
        
        # *refresh_btn_2* updates
        
        # if refresh_btn_2 is starting this frame...
        if refresh_btn_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            refresh_btn_2.frameNStart = frameN  # exact frame index
            refresh_btn_2.tStart = t  # local t and not account for scr refresh
            refresh_btn_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(refresh_btn_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'refresh_btn_2.started')
            # update status
            refresh_btn_2.status = STARTED
            refresh_btn_2.setAutoDraw(True)
        
        # if refresh_btn_2 is active this frame...
        if refresh_btn_2.status == STARTED:
            # update params
            pass
        
        # *refresh_click_2* updates
        
        # if refresh_click_2 is starting this frame...
        if refresh_click_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            refresh_click_2.frameNStart = frameN  # exact frame index
            refresh_click_2.tStart = t  # local t and not account for scr refresh
            refresh_click_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(refresh_click_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'refresh_click_2.started')
            # update status
            refresh_click_2.status = STARTED
            refresh_click_2.setAutoDraw(True)
        
        # if refresh_click_2 is active this frame...
        if refresh_click_2.status == STARTED:
            # update params
            pass
        
        # *likes_txt_2* updates
        
        # if likes_txt_2 is starting this frame...
        if likes_txt_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            likes_txt_2.frameNStart = frameN  # exact frame index
            likes_txt_2.tStart = t  # local t and not account for scr refresh
            likes_txt_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(likes_txt_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'likes_txt_2.started')
            # update status
            likes_txt_2.status = STARTED
            likes_txt_2.setAutoDraw(True)
        
        # if likes_txt_2 is active this frame...
        if likes_txt_2.status == STARTED:
            # update params
            likes_txt_2.setText(demo_likes_display, log=False)
        
        # *timer_txt_2* updates
        
        # if timer_txt_2 is starting this frame...
        if timer_txt_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            timer_txt_2.frameNStart = frameN  # exact frame index
            timer_txt_2.tStart = t  # local t and not account for scr refresh
            timer_txt_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(timer_txt_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'timer_txt_2.started')
            # update status
            timer_txt_2.status = STARTED
            timer_txt_2.setAutoDraw(True)
        
        # if timer_txt_2 is active this frame...
        if timer_txt_2.status == STARTED:
            # update params
            timer_txt_2.setText(demo_timer_display, log=False)
        
        # *reminder_text_2* updates
        
        # if reminder_text_2 is starting this frame...
        if reminder_text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            reminder_text_2.frameNStart = frameN  # exact frame index
            reminder_text_2.tStart = t  # local t and not account for scr refresh
            reminder_text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(reminder_text_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'reminder_text_2.started')
            # update status
            reminder_text_2.status = STARTED
            reminder_text_2.setAutoDraw(True)
        
        # if reminder_text_2 is active this frame...
        if reminder_text_2.status == STARTED:
            # update params
            pass
        # *demo_mouse* updates
        
        # if demo_mouse is starting this frame...
        if demo_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            demo_mouse.frameNStart = frameN  # exact frame index
            demo_mouse.tStart = t  # local t and not account for scr refresh
            demo_mouse.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(demo_mouse, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('demo_mouse.started', t)
            # update status
            demo_mouse.status = STARTED
            demo_mouse.mouseClock.reset()
            prevButtonState = [0, 0, 0]  # if now button is down we will treat as 'new' click
        if demo_mouse.status == STARTED:  # only update if started and not finished!
            buttons = demo_mouse.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    pass
                    x, y = demo_mouse.getPos()
                    demo_mouse.x.append(float(x))
                    demo_mouse.y.append(float(y))
                    buttons = demo_mouse.getPressed()
                    demo_mouse.leftButton.append(buttons[0])
                    demo_mouse.midButton.append(buttons[1])
                    demo_mouse.rightButton.append(buttons[2])
                    demo_mouse.time.append(demo_mouse.mouseClock.getTime())
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=demo_5,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            demo_5.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if demo_5.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in demo_5.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "demo_5" ---
    for thisComponent in demo_5.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for demo_5
    demo_5.tStop = globalClock.getTime(format='float')
    demo_5.tStopRefresh = tThisFlipGlobal
    thisExp.addData('demo_5.stopped', demo_5.tStop)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('demo_mouse.x', demo_mouse.x)
    thisExp.addData('demo_mouse.y', demo_mouse.y)
    thisExp.addData('demo_mouse.leftButton', demo_mouse.leftButton)
    thisExp.addData('demo_mouse.midButton', demo_mouse.midButton)
    thisExp.addData('demo_mouse.rightButton', demo_mouse.rightButton)
    thisExp.addData('demo_mouse.time', demo_mouse.time)
    thisExp.nextEntry()
    # the Routine "demo_5" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "demo_6" ---
    # create an object to store info about Routine demo_6
    demo_6 = data.Routine(
        name='demo_6',
        components=[background_7, text_9, text_10, key_resp_7],
    )
    demo_6.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    text_9.setText('Likes: 500')
    # create starting attributes for key_resp_7
    key_resp_7.keys = []
    key_resp_7.rt = []
    _key_resp_7_allKeys = []
    # store start times for demo_6
    demo_6.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    demo_6.tStart = globalClock.getTime(format='float')
    demo_6.status = STARTED
    thisExp.addData('demo_6.started', demo_6.tStart)
    demo_6.maxDuration = None
    # keep track of which components have finished
    demo_6Components = demo_6.components
    for thisComponent in demo_6.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "demo_6" ---
    thisExp.currentRoutine = demo_6
    demo_6.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *background_7* updates
        
        # if background_7 is starting this frame...
        if background_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            background_7.frameNStart = frameN  # exact frame index
            background_7.tStart = t  # local t and not account for scr refresh
            background_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(background_7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'background_7.started')
            # update status
            background_7.status = STARTED
            background_7.setAutoDraw(True)
        
        # if background_7 is active this frame...
        if background_7.status == STARTED:
            # update params
            pass
        
        # *text_9* updates
        
        # if text_9 is starting this frame...
        if text_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_9.frameNStart = frameN  # exact frame index
            text_9.tStart = t  # local t and not account for scr refresh
            text_9.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_9, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_9.started')
            # update status
            text_9.status = STARTED
            text_9.setAutoDraw(True)
        
        # if text_9 is active this frame...
        if text_9.status == STARTED:
            # update params
            pass
        
        # *text_10* updates
        
        # if text_10 is starting this frame...
        if text_10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_10.frameNStart = frameN  # exact frame index
            text_10.tStart = t  # local t and not account for scr refresh
            text_10.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_10, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_10.started')
            # update status
            text_10.status = STARTED
            text_10.setAutoDraw(True)
        
        # if text_10 is active this frame...
        if text_10.status == STARTED:
            # update params
            pass
        
        # *key_resp_7* updates
        waitOnFlip = False
        
        # if key_resp_7 is starting this frame...
        if key_resp_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_7.frameNStart = frameN  # exact frame index
            key_resp_7.tStart = t  # local t and not account for scr refresh
            key_resp_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_7.started')
            # update status
            key_resp_7.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_7.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_7.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_7.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_7.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_7_allKeys.extend(theseKeys)
            if len(_key_resp_7_allKeys):
                key_resp_7.keys = _key_resp_7_allKeys[-1].name  # just the last key pressed
                key_resp_7.rt = _key_resp_7_allKeys[-1].rt
                key_resp_7.duration = _key_resp_7_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=demo_6,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            demo_6.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if demo_6.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in demo_6.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "demo_6" ---
    for thisComponent in demo_6.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for demo_6
    demo_6.tStop = globalClock.getTime(format='float')
    demo_6.tStopRefresh = tThisFlipGlobal
    thisExp.addData('demo_6.stopped', demo_6.tStop)
    # check responses
    if key_resp_7.keys in ['', [], None]:  # No response was made
        key_resp_7.keys = None
    thisExp.addData('key_resp_7.keys',key_resp_7.keys)
    if key_resp_7.keys != None:  # we had a response
        thisExp.addData('key_resp_7.rt', key_resp_7.rt)
        thisExp.addData('key_resp_7.duration', key_resp_7.duration)
    thisExp.nextEntry()
    # the Routine "demo_6" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "demo_7" ---
    # create an object to store info about Routine demo_7
    demo_7 = data.Routine(
        name='demo_7',
        components=[background_8, text_11, key_resp_8],
    )
    demo_7.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_8
    key_resp_8.keys = []
    key_resp_8.rt = []
    _key_resp_8_allKeys = []
    # store start times for demo_7
    demo_7.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    demo_7.tStart = globalClock.getTime(format='float')
    demo_7.status = STARTED
    thisExp.addData('demo_7.started', demo_7.tStart)
    demo_7.maxDuration = None
    # keep track of which components have finished
    demo_7Components = demo_7.components
    for thisComponent in demo_7.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "demo_7" ---
    thisExp.currentRoutine = demo_7
    demo_7.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *background_8* updates
        
        # if background_8 is starting this frame...
        if background_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            background_8.frameNStart = frameN  # exact frame index
            background_8.tStart = t  # local t and not account for scr refresh
            background_8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(background_8, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'background_8.started')
            # update status
            background_8.status = STARTED
            background_8.setAutoDraw(True)
        
        # if background_8 is active this frame...
        if background_8.status == STARTED:
            # update params
            pass
        
        # *text_11* updates
        
        # if text_11 is starting this frame...
        if text_11.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_11.frameNStart = frameN  # exact frame index
            text_11.tStart = t  # local t and not account for scr refresh
            text_11.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_11, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_11.started')
            # update status
            text_11.status = STARTED
            text_11.setAutoDraw(True)
        
        # if text_11 is active this frame...
        if text_11.status == STARTED:
            # update params
            pass
        
        # *key_resp_8* updates
        waitOnFlip = False
        
        # if key_resp_8 is starting this frame...
        if key_resp_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_8.frameNStart = frameN  # exact frame index
            key_resp_8.tStart = t  # local t and not account for scr refresh
            key_resp_8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_8, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_8.started')
            # update status
            key_resp_8.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_8.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_8.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_8.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_8.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_8_allKeys.extend(theseKeys)
            if len(_key_resp_8_allKeys):
                key_resp_8.keys = _key_resp_8_allKeys[-1].name  # just the last key pressed
                key_resp_8.rt = _key_resp_8_allKeys[-1].rt
                key_resp_8.duration = _key_resp_8_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=demo_7,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            demo_7.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if demo_7.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in demo_7.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "demo_7" ---
    for thisComponent in demo_7.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for demo_7
    demo_7.tStop = globalClock.getTime(format='float')
    demo_7.tStopRefresh = tThisFlipGlobal
    thisExp.addData('demo_7.stopped', demo_7.tStop)
    # check responses
    if key_resp_8.keys in ['', [], None]:  # No response was made
        key_resp_8.keys = None
    thisExp.addData('key_resp_8.keys',key_resp_8.keys)
    if key_resp_8.keys != None:  # we had a response
        thisExp.addData('key_resp_8.rt', key_resp_8.rt)
        thisExp.addData('key_resp_8.duration', key_resp_8.duration)
    thisExp.nextEntry()
    # the Routine "demo_7" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "demo_7b" ---
    # create an object to store info about Routine demo_7b
    demo_7b = data.Routine(
        name='demo_7b',
        components=[background_21, text_28, key_resp_10],
    )
    demo_7b.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_10
    key_resp_10.keys = []
    key_resp_10.rt = []
    _key_resp_10_allKeys = []
    # store start times for demo_7b
    demo_7b.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    demo_7b.tStart = globalClock.getTime(format='float')
    demo_7b.status = STARTED
    thisExp.addData('demo_7b.started', demo_7b.tStart)
    demo_7b.maxDuration = None
    # keep track of which components have finished
    demo_7bComponents = demo_7b.components
    for thisComponent in demo_7b.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "demo_7b" ---
    thisExp.currentRoutine = demo_7b
    demo_7b.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *background_21* updates
        
        # if background_21 is starting this frame...
        if background_21.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            background_21.frameNStart = frameN  # exact frame index
            background_21.tStart = t  # local t and not account for scr refresh
            background_21.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(background_21, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'background_21.started')
            # update status
            background_21.status = STARTED
            background_21.setAutoDraw(True)
        
        # if background_21 is active this frame...
        if background_21.status == STARTED:
            # update params
            pass
        
        # *text_28* updates
        
        # if text_28 is starting this frame...
        if text_28.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_28.frameNStart = frameN  # exact frame index
            text_28.tStart = t  # local t and not account for scr refresh
            text_28.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_28, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_28.started')
            # update status
            text_28.status = STARTED
            text_28.setAutoDraw(True)
        
        # if text_28 is active this frame...
        if text_28.status == STARTED:
            # update params
            pass
        
        # *key_resp_10* updates
        waitOnFlip = False
        
        # if key_resp_10 is starting this frame...
        if key_resp_10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_10.frameNStart = frameN  # exact frame index
            key_resp_10.tStart = t  # local t and not account for scr refresh
            key_resp_10.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_10, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_10.started')
            # update status
            key_resp_10.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_10.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_10.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_10.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_10.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_10_allKeys.extend(theseKeys)
            if len(_key_resp_10_allKeys):
                key_resp_10.keys = _key_resp_10_allKeys[-1].name  # just the last key pressed
                key_resp_10.rt = _key_resp_10_allKeys[-1].rt
                key_resp_10.duration = _key_resp_10_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=demo_7b,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            demo_7b.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if demo_7b.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in demo_7b.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "demo_7b" ---
    for thisComponent in demo_7b.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for demo_7b
    demo_7b.tStop = globalClock.getTime(format='float')
    demo_7b.tStopRefresh = tThisFlipGlobal
    thisExp.addData('demo_7b.stopped', demo_7b.tStop)
    # check responses
    if key_resp_10.keys in ['', [], None]:  # No response was made
        key_resp_10.keys = None
    thisExp.addData('key_resp_10.keys',key_resp_10.keys)
    if key_resp_10.keys != None:  # we had a response
        thisExp.addData('key_resp_10.rt', key_resp_10.rt)
        thisExp.addData('key_resp_10.duration', key_resp_10.duration)
    thisExp.nextEntry()
    # the Routine "demo_7b" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "demo_8" ---
    # create an object to store info about Routine demo_8
    demo_8 = data.Routine(
        name='demo_8',
        components=[background_9, you_choice_box_2, you_chose_txt_2, partner_choice_box_2, partner_chose_txt_2, text_12, key_resp_9, text_13, text_23, remember_demo_text],
    )
    demo_8.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    you_chose_txt_2.setText('Here is what you posted')
    partner_chose_txt_2.setText('Here is what the other participant posted.')
    # create starting attributes for key_resp_9
    key_resp_9.keys = []
    key_resp_9.rt = []
    _key_resp_9_allKeys = []
    # store start times for demo_8
    demo_8.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    demo_8.tStart = globalClock.getTime(format='float')
    demo_8.status = STARTED
    thisExp.addData('demo_8.started', demo_8.tStart)
    demo_8.maxDuration = None
    # keep track of which components have finished
    demo_8Components = demo_8.components
    for thisComponent in demo_8.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "demo_8" ---
    thisExp.currentRoutine = demo_8
    demo_8.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *background_9* updates
        
        # if background_9 is starting this frame...
        if background_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            background_9.frameNStart = frameN  # exact frame index
            background_9.tStart = t  # local t and not account for scr refresh
            background_9.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(background_9, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'background_9.started')
            # update status
            background_9.status = STARTED
            background_9.setAutoDraw(True)
        
        # if background_9 is active this frame...
        if background_9.status == STARTED:
            # update params
            pass
        
        # *you_choice_box_2* updates
        
        # if you_choice_box_2 is starting this frame...
        if you_choice_box_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            you_choice_box_2.frameNStart = frameN  # exact frame index
            you_choice_box_2.tStart = t  # local t and not account for scr refresh
            you_choice_box_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(you_choice_box_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'you_choice_box_2.started')
            # update status
            you_choice_box_2.status = STARTED
            you_choice_box_2.setAutoDraw(True)
        
        # if you_choice_box_2 is active this frame...
        if you_choice_box_2.status == STARTED:
            # update params
            pass
        
        # *you_chose_txt_2* updates
        
        # if you_chose_txt_2 is starting this frame...
        if you_chose_txt_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            you_chose_txt_2.frameNStart = frameN  # exact frame index
            you_chose_txt_2.tStart = t  # local t and not account for scr refresh
            you_chose_txt_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(you_chose_txt_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'you_chose_txt_2.started')
            # update status
            you_chose_txt_2.status = STARTED
            you_chose_txt_2.setAutoDraw(True)
        
        # if you_chose_txt_2 is active this frame...
        if you_chose_txt_2.status == STARTED:
            # update params
            pass
        
        # *partner_choice_box_2* updates
        
        # if partner_choice_box_2 is starting this frame...
        if partner_choice_box_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            partner_choice_box_2.frameNStart = frameN  # exact frame index
            partner_choice_box_2.tStart = t  # local t and not account for scr refresh
            partner_choice_box_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(partner_choice_box_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'partner_choice_box_2.started')
            # update status
            partner_choice_box_2.status = STARTED
            partner_choice_box_2.setAutoDraw(True)
        
        # if partner_choice_box_2 is active this frame...
        if partner_choice_box_2.status == STARTED:
            # update params
            pass
        
        # *partner_chose_txt_2* updates
        
        # if partner_chose_txt_2 is starting this frame...
        if partner_chose_txt_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            partner_chose_txt_2.frameNStart = frameN  # exact frame index
            partner_chose_txt_2.tStart = t  # local t and not account for scr refresh
            partner_chose_txt_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(partner_chose_txt_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'partner_chose_txt_2.started')
            # update status
            partner_chose_txt_2.status = STARTED
            partner_chose_txt_2.setAutoDraw(True)
        
        # if partner_chose_txt_2 is active this frame...
        if partner_chose_txt_2.status == STARTED:
            # update params
            pass
        
        # *text_12* updates
        
        # if text_12 is starting this frame...
        if text_12.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_12.frameNStart = frameN  # exact frame index
            text_12.tStart = t  # local t and not account for scr refresh
            text_12.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_12, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_12.started')
            # update status
            text_12.status = STARTED
            text_12.setAutoDraw(True)
        
        # if text_12 is active this frame...
        if text_12.status == STARTED:
            # update params
            pass
        
        # *key_resp_9* updates
        waitOnFlip = False
        
        # if key_resp_9 is starting this frame...
        if key_resp_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_9.frameNStart = frameN  # exact frame index
            key_resp_9.tStart = t  # local t and not account for scr refresh
            key_resp_9.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_9, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_9.started')
            # update status
            key_resp_9.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_9.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_9.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_9.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_9.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_9_allKeys.extend(theseKeys)
            if len(_key_resp_9_allKeys):
                key_resp_9.keys = _key_resp_9_allKeys[-1].name  # just the last key pressed
                key_resp_9.rt = _key_resp_9_allKeys[-1].rt
                key_resp_9.duration = _key_resp_9_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *text_13* updates
        
        # if text_13 is starting this frame...
        if text_13.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_13.frameNStart = frameN  # exact frame index
            text_13.tStart = t  # local t and not account for scr refresh
            text_13.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_13, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_13.started')
            # update status
            text_13.status = STARTED
            text_13.setAutoDraw(True)
        
        # if text_13 is active this frame...
        if text_13.status == STARTED:
            # update params
            pass
        
        # *text_23* updates
        
        # if text_23 is starting this frame...
        if text_23.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_23.frameNStart = frameN  # exact frame index
            text_23.tStart = t  # local t and not account for scr refresh
            text_23.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_23, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_23.started')
            # update status
            text_23.status = STARTED
            text_23.setAutoDraw(True)
        
        # if text_23 is active this frame...
        if text_23.status == STARTED:
            # update params
            pass
        
        # *remember_demo_text* updates
        
        # if remember_demo_text is starting this frame...
        if remember_demo_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            remember_demo_text.frameNStart = frameN  # exact frame index
            remember_demo_text.tStart = t  # local t and not account for scr refresh
            remember_demo_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(remember_demo_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'remember_demo_text.started')
            # update status
            remember_demo_text.status = STARTED
            remember_demo_text.setAutoDraw(True)
        
        # if remember_demo_text is active this frame...
        if remember_demo_text.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=demo_8,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            demo_8.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if demo_8.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in demo_8.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "demo_8" ---
    for thisComponent in demo_8.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for demo_8
    demo_8.tStop = globalClock.getTime(format='float')
    demo_8.tStopRefresh = tThisFlipGlobal
    thisExp.addData('demo_8.stopped', demo_8.tStop)
    # check responses
    if key_resp_9.keys in ['', [], None]:  # No response was made
        key_resp_9.keys = None
    thisExp.addData('key_resp_9.keys',key_resp_9.keys)
    if key_resp_9.keys != None:  # we had a response
        thisExp.addData('key_resp_9.rt', key_resp_9.rt)
        thisExp.addData('key_resp_9.duration', key_resp_9.duration)
    thisExp.nextEntry()
    # the Routine "demo_8" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Start" ---
    # create an object to store info about Routine Start
    Start = data.Routine(
        name='Start',
        components=[background_10, Welcome_text, key_resp],
    )
    Start.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # store start times for Start
    Start.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Start.tStart = globalClock.getTime(format='float')
    Start.status = STARTED
    thisExp.addData('Start.started', Start.tStart)
    Start.maxDuration = None
    # keep track of which components have finished
    StartComponents = Start.components
    for thisComponent in Start.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Start" ---
    thisExp.currentRoutine = Start
    Start.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *background_10* updates
        
        # if background_10 is starting this frame...
        if background_10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            background_10.frameNStart = frameN  # exact frame index
            background_10.tStart = t  # local t and not account for scr refresh
            background_10.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(background_10, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'background_10.started')
            # update status
            background_10.status = STARTED
            background_10.setAutoDraw(True)
        
        # if background_10 is active this frame...
        if background_10.status == STARTED:
            # update params
            pass
        
        # *Welcome_text* updates
        
        # if Welcome_text is starting this frame...
        if Welcome_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Welcome_text.frameNStart = frameN  # exact frame index
            Welcome_text.tStart = t  # local t and not account for scr refresh
            Welcome_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Welcome_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Welcome_text.started')
            # update status
            Welcome_text.status = STARTED
            Welcome_text.setAutoDraw(True)
        
        # if Welcome_text is active this frame...
        if Welcome_text.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Start,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Start.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Start.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Start.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Start" ---
    for thisComponent in Start.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Start
    Start.tStop = globalClock.getTime(format='float')
    Start.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Start.stopped', Start.tStop)
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "Start" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "SP" ---
    # create an object to store info about Routine SP
    SP = data.Routine(
        name='SP',
        components=[background_11, slider, text_14, polygon, text_16, mouse_3, q_rating_n, q_rating_n_2, text_15],
    )
    SP.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from sp_code
    slider.reset()
    slider.reset()
    # setup some python lists for storing info about the mouse_3
    mouse_3.x = []
    mouse_3.y = []
    mouse_3.leftButton = []
    mouse_3.midButton = []
    mouse_3.rightButton = []
    mouse_3.time = []
    mouse_3.clicked_name = []
    gotValidClick = False  # until a click is received
    # store start times for SP
    SP.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    SP.tStart = globalClock.getTime(format='float')
    SP.status = STARTED
    thisExp.addData('SP.started', SP.tStart)
    SP.maxDuration = None
    # keep track of which components have finished
    SPComponents = SP.components
    for thisComponent in SP.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "SP" ---
    thisExp.currentRoutine = SP
    SP.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *background_11* updates
        
        # if background_11 is starting this frame...
        if background_11.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            background_11.frameNStart = frameN  # exact frame index
            background_11.tStart = t  # local t and not account for scr refresh
            background_11.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(background_11, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'background_11.started')
            # update status
            background_11.status = STARTED
            background_11.setAutoDraw(True)
        
        # if background_11 is active this frame...
        if background_11.status == STARTED:
            # update params
            pass
        
        # *slider* updates
        
        # if slider is starting this frame...
        if slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider.frameNStart = frameN  # exact frame index
            slider.tStart = t  # local t and not account for scr refresh
            slider.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'slider.started')
            # update status
            slider.status = STARTED
            slider.setAutoDraw(True)
        
        # if slider is active this frame...
        if slider.status == STARTED:
            # update params
            pass
        
        # *text_14* updates
        
        # if text_14 is starting this frame...
        if text_14.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_14.frameNStart = frameN  # exact frame index
            text_14.tStart = t  # local t and not account for scr refresh
            text_14.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_14, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_14.started')
            # update status
            text_14.status = STARTED
            text_14.setAutoDraw(True)
        
        # if text_14 is active this frame...
        if text_14.status == STARTED:
            # update params
            pass
        
        # *polygon* updates
        
        # if polygon is starting this frame...
        if polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            polygon.frameNStart = frameN  # exact frame index
            polygon.tStart = t  # local t and not account for scr refresh
            polygon.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'polygon.started')
            # update status
            polygon.status = STARTED
            polygon.setAutoDraw(True)
        
        # if polygon is active this frame...
        if polygon.status == STARTED:
            # update params
            pass
        
        # *text_16* updates
        
        # if text_16 is starting this frame...
        if text_16.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_16.frameNStart = frameN  # exact frame index
            text_16.tStart = t  # local t and not account for scr refresh
            text_16.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_16, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_16.started')
            # update status
            text_16.status = STARTED
            text_16.setAutoDraw(True)
        
        # if text_16 is active this frame...
        if text_16.status == STARTED:
            # update params
            pass
        # *mouse_3* updates
        
        # if mouse_3 is starting this frame...
        if mouse_3.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_3.frameNStart = frameN  # exact frame index
            mouse_3.tStart = t  # local t and not account for scr refresh
            mouse_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('mouse_3.started', t)
            # update status
            mouse_3.status = STARTED
            mouse_3.mouseClock.reset()
            prevButtonState = mouse_3.getPressed()  # if button is down already this ISN'T a new click
        if mouse_3.status == STARTED:  # only update if started and not finished!
            buttons = mouse_3.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(polygon, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse_3):
                            gotValidClick = True
                            mouse_3.clicked_name.append(obj.name)
                    if not gotValidClick:
                        mouse_3.clicked_name.append(None)
                    x, y = mouse_3.getPos()
                    mouse_3.x.append(float(x))
                    mouse_3.y.append(float(y))
                    buttons = mouse_3.getPressed()
                    mouse_3.leftButton.append(buttons[0])
                    mouse_3.midButton.append(buttons[1])
                    mouse_3.rightButton.append(buttons[2])
                    mouse_3.time.append(mouse_3.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # *q_rating_n* updates
        
        # if q_rating_n is starting this frame...
        if q_rating_n.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            q_rating_n.frameNStart = frameN  # exact frame index
            q_rating_n.tStart = t  # local t and not account for scr refresh
            q_rating_n.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(q_rating_n, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'q_rating_n.started')
            # update status
            q_rating_n.status = STARTED
            q_rating_n.setAutoDraw(True)
        
        # if q_rating_n is active this frame...
        if q_rating_n.status == STARTED:
            # update params
            pass
        
        # *q_rating_n_2* updates
        
        # if q_rating_n_2 is starting this frame...
        if q_rating_n_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            q_rating_n_2.frameNStart = frameN  # exact frame index
            q_rating_n_2.tStart = t  # local t and not account for scr refresh
            q_rating_n_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(q_rating_n_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'q_rating_n_2.started')
            # update status
            q_rating_n_2.status = STARTED
            q_rating_n_2.setAutoDraw(True)
        
        # if q_rating_n_2 is active this frame...
        if q_rating_n_2.status == STARTED:
            # update params
            pass
        
        # *text_15* updates
        
        # if text_15 is starting this frame...
        if text_15.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_15.frameNStart = frameN  # exact frame index
            text_15.tStart = t  # local t and not account for scr refresh
            text_15.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_15, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_15.started')
            # update status
            text_15.status = STARTED
            text_15.setAutoDraw(True)
        
        # if text_15 is active this frame...
        if text_15.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=SP,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            SP.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if SP.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in SP.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "SP" ---
    for thisComponent in SP.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for SP
    SP.tStop = globalClock.getTime(format='float')
    SP.tStopRefresh = tThisFlipGlobal
    thisExp.addData('SP.stopped', SP.tStop)
    # Run 'End Routine' code from sp_code
    thisExp.addData('sp_rating', slider.getRating())
    thisExp.addData('sp_RT',     slider.getRT())
    
    thisExp.addData('slider.response', slider.getRating())
    thisExp.addData('slider.rt', slider.getRT())
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_3.x', mouse_3.x)
    thisExp.addData('mouse_3.y', mouse_3.y)
    thisExp.addData('mouse_3.leftButton', mouse_3.leftButton)
    thisExp.addData('mouse_3.midButton', mouse_3.midButton)
    thisExp.addData('mouse_3.rightButton', mouse_3.rightButton)
    thisExp.addData('mouse_3.time', mouse_3.time)
    thisExp.addData('mouse_3.clicked_name', mouse_3.clicked_name)
    thisExp.nextEntry()
    # the Routine "SP" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "anon" ---
    # create an object to store info about Routine anon
    anon = data.Routine(
        name='anon',
        components=[background_12, slider_2, text_17, polygon_2, text_18, mouse_4, q_rating_n_3, q_rating_n_4, text_19],
    )
    anon.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from sp_code_2
    slider_2.reset()
    slider_2.reset()
    # setup some python lists for storing info about the mouse_4
    mouse_4.x = []
    mouse_4.y = []
    mouse_4.leftButton = []
    mouse_4.midButton = []
    mouse_4.rightButton = []
    mouse_4.time = []
    mouse_4.clicked_name = []
    gotValidClick = False  # until a click is received
    # store start times for anon
    anon.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    anon.tStart = globalClock.getTime(format='float')
    anon.status = STARTED
    thisExp.addData('anon.started', anon.tStart)
    anon.maxDuration = None
    # keep track of which components have finished
    anonComponents = anon.components
    for thisComponent in anon.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "anon" ---
    thisExp.currentRoutine = anon
    anon.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *background_12* updates
        
        # if background_12 is starting this frame...
        if background_12.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            background_12.frameNStart = frameN  # exact frame index
            background_12.tStart = t  # local t and not account for scr refresh
            background_12.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(background_12, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'background_12.started')
            # update status
            background_12.status = STARTED
            background_12.setAutoDraw(True)
        
        # if background_12 is active this frame...
        if background_12.status == STARTED:
            # update params
            pass
        
        # *slider_2* updates
        
        # if slider_2 is starting this frame...
        if slider_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider_2.frameNStart = frameN  # exact frame index
            slider_2.tStart = t  # local t and not account for scr refresh
            slider_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'slider_2.started')
            # update status
            slider_2.status = STARTED
            slider_2.setAutoDraw(True)
        
        # if slider_2 is active this frame...
        if slider_2.status == STARTED:
            # update params
            pass
        
        # *text_17* updates
        
        # if text_17 is starting this frame...
        if text_17.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_17.frameNStart = frameN  # exact frame index
            text_17.tStart = t  # local t and not account for scr refresh
            text_17.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_17, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_17.started')
            # update status
            text_17.status = STARTED
            text_17.setAutoDraw(True)
        
        # if text_17 is active this frame...
        if text_17.status == STARTED:
            # update params
            pass
        
        # *polygon_2* updates
        
        # if polygon_2 is starting this frame...
        if polygon_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            polygon_2.frameNStart = frameN  # exact frame index
            polygon_2.tStart = t  # local t and not account for scr refresh
            polygon_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'polygon_2.started')
            # update status
            polygon_2.status = STARTED
            polygon_2.setAutoDraw(True)
        
        # if polygon_2 is active this frame...
        if polygon_2.status == STARTED:
            # update params
            pass
        
        # *text_18* updates
        
        # if text_18 is starting this frame...
        if text_18.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_18.frameNStart = frameN  # exact frame index
            text_18.tStart = t  # local t and not account for scr refresh
            text_18.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_18, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_18.started')
            # update status
            text_18.status = STARTED
            text_18.setAutoDraw(True)
        
        # if text_18 is active this frame...
        if text_18.status == STARTED:
            # update params
            pass
        # *mouse_4* updates
        
        # if mouse_4 is starting this frame...
        if mouse_4.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_4.frameNStart = frameN  # exact frame index
            mouse_4.tStart = t  # local t and not account for scr refresh
            mouse_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('mouse_4.started', t)
            # update status
            mouse_4.status = STARTED
            mouse_4.mouseClock.reset()
            prevButtonState = mouse_4.getPressed()  # if button is down already this ISN'T a new click
        if mouse_4.status == STARTED:  # only update if started and not finished!
            buttons = mouse_4.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(polygon_2, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse_4):
                            gotValidClick = True
                            mouse_4.clicked_name.append(obj.name)
                    if not gotValidClick:
                        mouse_4.clicked_name.append(None)
                    x, y = mouse_4.getPos()
                    mouse_4.x.append(float(x))
                    mouse_4.y.append(float(y))
                    buttons = mouse_4.getPressed()
                    mouse_4.leftButton.append(buttons[0])
                    mouse_4.midButton.append(buttons[1])
                    mouse_4.rightButton.append(buttons[2])
                    mouse_4.time.append(mouse_4.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # *q_rating_n_3* updates
        
        # if q_rating_n_3 is starting this frame...
        if q_rating_n_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            q_rating_n_3.frameNStart = frameN  # exact frame index
            q_rating_n_3.tStart = t  # local t and not account for scr refresh
            q_rating_n_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(q_rating_n_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'q_rating_n_3.started')
            # update status
            q_rating_n_3.status = STARTED
            q_rating_n_3.setAutoDraw(True)
        
        # if q_rating_n_3 is active this frame...
        if q_rating_n_3.status == STARTED:
            # update params
            pass
        
        # *q_rating_n_4* updates
        
        # if q_rating_n_4 is starting this frame...
        if q_rating_n_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            q_rating_n_4.frameNStart = frameN  # exact frame index
            q_rating_n_4.tStart = t  # local t and not account for scr refresh
            q_rating_n_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(q_rating_n_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'q_rating_n_4.started')
            # update status
            q_rating_n_4.status = STARTED
            q_rating_n_4.setAutoDraw(True)
        
        # if q_rating_n_4 is active this frame...
        if q_rating_n_4.status == STARTED:
            # update params
            pass
        
        # *text_19* updates
        
        # if text_19 is starting this frame...
        if text_19.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_19.frameNStart = frameN  # exact frame index
            text_19.tStart = t  # local t and not account for scr refresh
            text_19.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_19, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_19.started')
            # update status
            text_19.status = STARTED
            text_19.setAutoDraw(True)
        
        # if text_19 is active this frame...
        if text_19.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=anon,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            anon.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if anon.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in anon.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "anon" ---
    for thisComponent in anon.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for anon
    anon.tStop = globalClock.getTime(format='float')
    anon.tStopRefresh = tThisFlipGlobal
    thisExp.addData('anon.stopped', anon.tStop)
    # Run 'End Routine' code from sp_code_2
    thisExp.addData('an_rating', slider_2.getRating())
    thisExp.addData('an_RT',     slider_2.getRT())
    
    thisExp.addData('slider_2.response', slider_2.getRating())
    thisExp.addData('slider_2.rt', slider_2.getRT())
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_4.x', mouse_4.x)
    thisExp.addData('mouse_4.y', mouse_4.y)
    thisExp.addData('mouse_4.leftButton', mouse_4.leftButton)
    thisExp.addData('mouse_4.midButton', mouse_4.midButton)
    thisExp.addData('mouse_4.rightButton', mouse_4.rightButton)
    thisExp.addData('mouse_4.time', mouse_4.time)
    thisExp.addData('mouse_4.clicked_name', mouse_4.clicked_name)
    thisExp.nextEntry()
    # the Routine "anon" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "invis" ---
    # create an object to store info about Routine invis
    invis = data.Routine(
        name='invis',
        components=[background_13, slider_3, text_20, polygon_3, text_21, mouse_5, q_rating_n_5, q_rating_n_6, text_22],
    )
    invis.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from sp_code_3
    slider_3.reset()
    slider_3.reset()
    # setup some python lists for storing info about the mouse_5
    mouse_5.x = []
    mouse_5.y = []
    mouse_5.leftButton = []
    mouse_5.midButton = []
    mouse_5.rightButton = []
    mouse_5.time = []
    mouse_5.clicked_name = []
    gotValidClick = False  # until a click is received
    # store start times for invis
    invis.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    invis.tStart = globalClock.getTime(format='float')
    invis.status = STARTED
    thisExp.addData('invis.started', invis.tStart)
    invis.maxDuration = None
    # keep track of which components have finished
    invisComponents = invis.components
    for thisComponent in invis.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "invis" ---
    thisExp.currentRoutine = invis
    invis.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *background_13* updates
        
        # if background_13 is starting this frame...
        if background_13.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            background_13.frameNStart = frameN  # exact frame index
            background_13.tStart = t  # local t and not account for scr refresh
            background_13.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(background_13, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'background_13.started')
            # update status
            background_13.status = STARTED
            background_13.setAutoDraw(True)
        
        # if background_13 is active this frame...
        if background_13.status == STARTED:
            # update params
            pass
        
        # *slider_3* updates
        
        # if slider_3 is starting this frame...
        if slider_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider_3.frameNStart = frameN  # exact frame index
            slider_3.tStart = t  # local t and not account for scr refresh
            slider_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'slider_3.started')
            # update status
            slider_3.status = STARTED
            slider_3.setAutoDraw(True)
        
        # if slider_3 is active this frame...
        if slider_3.status == STARTED:
            # update params
            pass
        
        # *text_20* updates
        
        # if text_20 is starting this frame...
        if text_20.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_20.frameNStart = frameN  # exact frame index
            text_20.tStart = t  # local t and not account for scr refresh
            text_20.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_20, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_20.started')
            # update status
            text_20.status = STARTED
            text_20.setAutoDraw(True)
        
        # if text_20 is active this frame...
        if text_20.status == STARTED:
            # update params
            pass
        
        # *polygon_3* updates
        
        # if polygon_3 is starting this frame...
        if polygon_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            polygon_3.frameNStart = frameN  # exact frame index
            polygon_3.tStart = t  # local t and not account for scr refresh
            polygon_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'polygon_3.started')
            # update status
            polygon_3.status = STARTED
            polygon_3.setAutoDraw(True)
        
        # if polygon_3 is active this frame...
        if polygon_3.status == STARTED:
            # update params
            pass
        
        # *text_21* updates
        
        # if text_21 is starting this frame...
        if text_21.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_21.frameNStart = frameN  # exact frame index
            text_21.tStart = t  # local t and not account for scr refresh
            text_21.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_21, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_21.started')
            # update status
            text_21.status = STARTED
            text_21.setAutoDraw(True)
        
        # if text_21 is active this frame...
        if text_21.status == STARTED:
            # update params
            pass
        # *mouse_5* updates
        
        # if mouse_5 is starting this frame...
        if mouse_5.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_5.frameNStart = frameN  # exact frame index
            mouse_5.tStart = t  # local t and not account for scr refresh
            mouse_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('mouse_5.started', t)
            # update status
            mouse_5.status = STARTED
            mouse_5.mouseClock.reset()
            prevButtonState = mouse_5.getPressed()  # if button is down already this ISN'T a new click
        if mouse_5.status == STARTED:  # only update if started and not finished!
            buttons = mouse_5.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(polygon_3, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse_5):
                            gotValidClick = True
                            mouse_5.clicked_name.append(obj.name)
                    if not gotValidClick:
                        mouse_5.clicked_name.append(None)
                    x, y = mouse_5.getPos()
                    mouse_5.x.append(float(x))
                    mouse_5.y.append(float(y))
                    buttons = mouse_5.getPressed()
                    mouse_5.leftButton.append(buttons[0])
                    mouse_5.midButton.append(buttons[1])
                    mouse_5.rightButton.append(buttons[2])
                    mouse_5.time.append(mouse_5.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # *q_rating_n_5* updates
        
        # if q_rating_n_5 is starting this frame...
        if q_rating_n_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            q_rating_n_5.frameNStart = frameN  # exact frame index
            q_rating_n_5.tStart = t  # local t and not account for scr refresh
            q_rating_n_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(q_rating_n_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'q_rating_n_5.started')
            # update status
            q_rating_n_5.status = STARTED
            q_rating_n_5.setAutoDraw(True)
        
        # if q_rating_n_5 is active this frame...
        if q_rating_n_5.status == STARTED:
            # update params
            pass
        
        # *q_rating_n_6* updates
        
        # if q_rating_n_6 is starting this frame...
        if q_rating_n_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            q_rating_n_6.frameNStart = frameN  # exact frame index
            q_rating_n_6.tStart = t  # local t and not account for scr refresh
            q_rating_n_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(q_rating_n_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'q_rating_n_6.started')
            # update status
            q_rating_n_6.status = STARTED
            q_rating_n_6.setAutoDraw(True)
        
        # if q_rating_n_6 is active this frame...
        if q_rating_n_6.status == STARTED:
            # update params
            pass
        
        # *text_22* updates
        
        # if text_22 is starting this frame...
        if text_22.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_22.frameNStart = frameN  # exact frame index
            text_22.tStart = t  # local t and not account for scr refresh
            text_22.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_22, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_22.started')
            # update status
            text_22.status = STARTED
            text_22.setAutoDraw(True)
        
        # if text_22 is active this frame...
        if text_22.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=invis,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            invis.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if invis.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in invis.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "invis" ---
    for thisComponent in invis.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for invis
    invis.tStop = globalClock.getTime(format='float')
    invis.tStopRefresh = tThisFlipGlobal
    thisExp.addData('invis.stopped', invis.tStop)
    # Run 'End Routine' code from sp_code_3
    thisExp.addData('in_rating', slider_3.getRating())
    thisExp.addData('in_RT',     slider_3.getRT())
    
    thisExp.addData('slider_3.response', slider_3.getRating())
    thisExp.addData('slider_3.rt', slider_3.getRT())
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_5.x', mouse_5.x)
    thisExp.addData('mouse_5.y', mouse_5.y)
    thisExp.addData('mouse_5.leftButton', mouse_5.leftButton)
    thisExp.addData('mouse_5.midButton', mouse_5.midButton)
    thisExp.addData('mouse_5.rightButton', mouse_5.rightButton)
    thisExp.addData('mouse_5.time', mouse_5.time)
    thisExp.addData('mouse_5.clicked_name', mouse_5.clicked_name)
    thisExp.nextEntry()
    # the Routine "invis" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Begin" ---
    # create an object to store info about Routine Begin
    Begin = data.Routine(
        name='Begin',
        components=[background_14, begin_text, key_resp_2],
    )
    Begin.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_2
    key_resp_2.keys = []
    key_resp_2.rt = []
    _key_resp_2_allKeys = []
    # store start times for Begin
    Begin.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Begin.tStart = globalClock.getTime(format='float')
    Begin.status = STARTED
    thisExp.addData('Begin.started', Begin.tStart)
    Begin.maxDuration = None
    # keep track of which components have finished
    BeginComponents = Begin.components
    for thisComponent in Begin.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Begin" ---
    thisExp.currentRoutine = Begin
    Begin.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *background_14* updates
        
        # if background_14 is starting this frame...
        if background_14.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            background_14.frameNStart = frameN  # exact frame index
            background_14.tStart = t  # local t and not account for scr refresh
            background_14.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(background_14, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'background_14.started')
            # update status
            background_14.status = STARTED
            background_14.setAutoDraw(True)
        
        # if background_14 is active this frame...
        if background_14.status == STARTED:
            # update params
            pass
        
        # *begin_text* updates
        
        # if begin_text is starting this frame...
        if begin_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            begin_text.frameNStart = frameN  # exact frame index
            begin_text.tStart = t  # local t and not account for scr refresh
            begin_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(begin_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'begin_text.started')
            # update status
            begin_text.status = STARTED
            begin_text.setAutoDraw(True)
        
        # if begin_text is active this frame...
        if begin_text.status == STARTED:
            # update params
            pass
        
        # *key_resp_2* updates
        waitOnFlip = False
        
        # if key_resp_2 is starting this frame...
        if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_2.frameNStart = frameN  # exact frame index
            key_resp_2.tStart = t  # local t and not account for scr refresh
            key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_2.started')
            # update status
            key_resp_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_2_allKeys.extend(theseKeys)
            if len(_key_resp_2_allKeys):
                key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Begin,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Begin.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Begin.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Begin.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Begin" ---
    for thisComponent in Begin.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Begin
    Begin.tStop = globalClock.getTime(format='float')
    Begin.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Begin.stopped', Begin.tStop)
    # check responses
    if key_resp_2.keys in ['', [], None]:  # No response was made
        key_resp_2.keys = None
    thisExp.addData('key_resp_2.keys',key_resp_2.keys)
    if key_resp_2.keys != None:  # we had a response
        thisExp.addData('key_resp_2.rt', key_resp_2.rt)
        thisExp.addData('key_resp_2.duration', key_resp_2.duration)
    thisExp.nextEntry()
    # the Routine "Begin" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "connect" ---
    # create an object to store info about Routine connect
    connect = data.Routine(
        name='connect',
        components=[text_4, text_29, text_30, text_31],
    )
    connect.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_4
    import random
    
    sona_count = random.randint(70, 150)
    
    text_29.setText('f\\"There are currently {sona_count} SONA users online\\"\n')
    # store start times for connect
    connect.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    connect.tStart = globalClock.getTime(format='float')
    connect.status = STARTED
    thisExp.addData('connect.started', connect.tStart)
    connect.maxDuration = None
    # keep track of which components have finished
    connectComponents = connect.components
    for thisComponent in connect.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "connect" ---
    thisExp.currentRoutine = connect
    connect.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_4* updates
        
        # if text_4 is starting this frame...
        if text_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_4.frameNStart = frameN  # exact frame index
            text_4.tStart = t  # local t and not account for scr refresh
            text_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_4.started')
            # update status
            text_4.status = STARTED
            text_4.setAutoDraw(True)
        
        # if text_4 is active this frame...
        if text_4.status == STARTED:
            # update params
            pass
        
        # if text_4 is stopping this frame...
        if text_4.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_4.tStartRefresh + 14-frameTolerance:
                # keep track of stop time/frame for later
                text_4.tStop = t  # not accounting for scr refresh
                text_4.tStopRefresh = tThisFlipGlobal  # on global time
                text_4.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_4.stopped')
                # update status
                text_4.status = FINISHED
                text_4.setAutoDraw(False)
        
        # *text_29* updates
        
        # if text_29 is starting this frame...
        if text_29.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_29.frameNStart = frameN  # exact frame index
            text_29.tStart = t  # local t and not account for scr refresh
            text_29.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_29, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_29.started')
            # update status
            text_29.status = STARTED
            text_29.setAutoDraw(True)
        
        # if text_29 is active this frame...
        if text_29.status == STARTED:
            # update params
            pass
        
        # *text_30* updates
        
        # if text_30 is starting this frame...
        if text_30.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_30.frameNStart = frameN  # exact frame index
            text_30.tStart = t  # local t and not account for scr refresh
            text_30.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_30, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_30.started')
            # update status
            text_30.status = STARTED
            text_30.setAutoDraw(True)
        
        # if text_30 is active this frame...
        if text_30.status == STARTED:
            # update params
            pass
        
        # if text_30 is stopping this frame...
        if text_30.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_30.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                text_30.tStop = t  # not accounting for scr refresh
                text_30.tStopRefresh = tThisFlipGlobal  # on global time
                text_30.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_30.stopped')
                # update status
                text_30.status = FINISHED
                text_30.setAutoDraw(False)
        
        # *text_31* updates
        
        # if text_31 is starting this frame...
        if text_31.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
            # keep track of start time/frame for later
            text_31.frameNStart = frameN  # exact frame index
            text_31.tStart = t  # local t and not account for scr refresh
            text_31.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_31, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_31.started')
            # update status
            text_31.status = STARTED
            text_31.setAutoDraw(True)
        
        # if text_31 is active this frame...
        if text_31.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=connect,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            connect.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if connect.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in connect.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "connect" ---
    for thisComponent in connect.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for connect
    connect.tStop = globalClock.getTime(format='float')
    connect.tStopRefresh = tThisFlipGlobal
    thisExp.addData('connect.stopped', connect.tStop)
    thisExp.nextEntry()
    # the Routine "connect" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "connected" ---
    # create an object to store info about Routine connected
    connected = data.Routine(
        name='connected',
        components=[text_5],
    )
    connected.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for connected
    connected.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    connected.tStart = globalClock.getTime(format='float')
    connected.status = STARTED
    thisExp.addData('connected.started', connected.tStart)
    connected.maxDuration = None
    # keep track of which components have finished
    connectedComponents = connected.components
    for thisComponent in connected.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "connected" ---
    thisExp.currentRoutine = connected
    connected.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 2.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_5* updates
        
        # if text_5 is starting this frame...
        if text_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_5.frameNStart = frameN  # exact frame index
            text_5.tStart = t  # local t and not account for scr refresh
            text_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_5.started')
            # update status
            text_5.status = STARTED
            text_5.setAutoDraw(True)
        
        # if text_5 is active this frame...
        if text_5.status == STARTED:
            # update params
            pass
        
        # if text_5 is stopping this frame...
        if text_5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_5.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                text_5.tStop = t  # not accounting for scr refresh
                text_5.tStopRefresh = tThisFlipGlobal  # on global time
                text_5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_5.stopped')
                # update status
                text_5.status = FINISHED
                text_5.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=connected,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            connected.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if connected.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in connected.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "connected" ---
    for thisComponent in connected.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for connected
    connected.tStop = globalClock.getTime(format='float')
    connected.tStopRefresh = tThisFlipGlobal
    thisExp.addData('connected.stopped', connected.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if connected.maxDurationReached:
        routineTimer.addTime(-connected.maxDuration)
    elif connected.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-2.000000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(condFile), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        trials.status = STARTED
        if hasattr(thisTrial, 'status'):
            thisTrial.status = STARTED
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "Decision" ---
        # create an object to store info about Routine Decision
        Decision = data.Routine(
            name='Decision',
            components=[background_15, dilemma_txt, poly_left, poly_right, left_txt, right_txt, mouse_decision, happened_text, text_2],
        )
        Decision.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_decision
        # Build option tuples for this trial (coming from your conditions file):
        # (choice_code, label_for_you, likes_for_you, partner_label_for_that_valence, partner_likes_for_that_valence)
        # choice_code: 0=Neutral, 1=Negative
        options = [
            (0, option_neutral, likes_neutral, partner_option_neutral, partner_likes_neutral),
            (1, option_negative, likes_negative, partner_option_negative, partner_likes_negative)
        ]
        
        # Randomize which option appears on left/right
        shuffle(options)
        
        # Unpack to left/right; these names are read by your Text components ($left_label / $right_label)
        left_choice_code, left_label, left_likes_final, left_partner_label, left_partner_likes = options[0]
        right_choice_code, right_label, right_likes_final, right_partner_label, right_partner_likes = options[1]
        
        # Reset per-trial selection vars
        selected_choice_code = None
        selected_rt = None  # seconds since routine start
        
        # Record the mapping (nice to have for analysis)
        side_map = {
            'round': round_num + 1,
            'left_choice_code': left_choice_code,
            'left_label': left_label,
            'right_choice_code': right_choice_code,
            'right_label': right_label
        }
        
        
        # 1) Dilemma text (ONLY the text, no round number)
        try:
            dilemma_txt.text = str(dilemma_text)
        except NameError:
            print("ERROR: Column 'dilemma_text' not found in conditions file. Check header spelling.")
            dilemma_txt.text = ''
        
        # 2) Option labels on the polygons
        left_txt.text = str(left_label)
        right_txt.text = str(right_label)
        
        # Optional debug: confirm what this trial loaded
        print(f"[Decision] Round {round_num+1} | Dilemma: {dilemma_text}")
        print(f"           Left({left_choice_code}): {left_label}")
        print(f"           Right({right_choice_code}): {right_label}")
        
        # Python
        left_txt.alignHoriz = 'left'
        right_txt.alignHoriz = 'left'
        
        dilemma_txt.setText(dilemma_text)
        left_txt.setText(left_label)
        right_txt.setText(right_label)
        # setup some python lists for storing info about the mouse_decision
        mouse_decision.x = []
        mouse_decision.y = []
        mouse_decision.leftButton = []
        mouse_decision.midButton = []
        mouse_decision.rightButton = []
        mouse_decision.time = []
        gotValidClick = False  # until a click is received
        # store start times for Decision
        Decision.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Decision.tStart = globalClock.getTime(format='float')
        Decision.status = STARTED
        thisExp.addData('Decision.started', Decision.tStart)
        Decision.maxDuration = None
        # keep track of which components have finished
        DecisionComponents = Decision.components
        for thisComponent in Decision.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Decision" ---
        thisExp.currentRoutine = Decision
        Decision.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from code_decision
            # Detect first click and capture RT (Builder provides t = time since routine began)
            if selected_choice_code is None:
                if mouse_decision.isPressedIn(poly_left):
                    selected_choice_code = left_choice_code
                    selected_rt = t
                    continueRoutine = False
                elif mouse_decision.isPressedIn(poly_right):
                    selected_choice_code = right_choice_code
                    selected_rt = t
                    continueRoutine = False
            
            
            # *background_15* updates
            
            # if background_15 is starting this frame...
            if background_15.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                background_15.frameNStart = frameN  # exact frame index
                background_15.tStart = t  # local t and not account for scr refresh
                background_15.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(background_15, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'background_15.started')
                # update status
                background_15.status = STARTED
                background_15.setAutoDraw(True)
            
            # if background_15 is active this frame...
            if background_15.status == STARTED:
                # update params
                pass
            
            # *dilemma_txt* updates
            
            # if dilemma_txt is starting this frame...
            if dilemma_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dilemma_txt.frameNStart = frameN  # exact frame index
                dilemma_txt.tStart = t  # local t and not account for scr refresh
                dilemma_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dilemma_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dilemma_txt.started')
                # update status
                dilemma_txt.status = STARTED
                dilemma_txt.setAutoDraw(True)
            
            # if dilemma_txt is active this frame...
            if dilemma_txt.status == STARTED:
                # update params
                pass
            
            # *poly_left* updates
            
            # if poly_left is starting this frame...
            if poly_left.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                poly_left.frameNStart = frameN  # exact frame index
                poly_left.tStart = t  # local t and not account for scr refresh
                poly_left.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(poly_left, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'poly_left.started')
                # update status
                poly_left.status = STARTED
                poly_left.setAutoDraw(True)
            
            # if poly_left is active this frame...
            if poly_left.status == STARTED:
                # update params
                pass
            
            # *poly_right* updates
            
            # if poly_right is starting this frame...
            if poly_right.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                poly_right.frameNStart = frameN  # exact frame index
                poly_right.tStart = t  # local t and not account for scr refresh
                poly_right.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(poly_right, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'poly_right.started')
                # update status
                poly_right.status = STARTED
                poly_right.setAutoDraw(True)
            
            # if poly_right is active this frame...
            if poly_right.status == STARTED:
                # update params
                pass
            
            # *left_txt* updates
            
            # if left_txt is starting this frame...
            if left_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                left_txt.frameNStart = frameN  # exact frame index
                left_txt.tStart = t  # local t and not account for scr refresh
                left_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(left_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left_txt.started')
                # update status
                left_txt.status = STARTED
                left_txt.setAutoDraw(True)
            
            # if left_txt is active this frame...
            if left_txt.status == STARTED:
                # update params
                pass
            
            # *right_txt* updates
            
            # if right_txt is starting this frame...
            if right_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                right_txt.frameNStart = frameN  # exact frame index
                right_txt.tStart = t  # local t and not account for scr refresh
                right_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(right_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right_txt.started')
                # update status
                right_txt.status = STARTED
                right_txt.setAutoDraw(True)
            
            # if right_txt is active this frame...
            if right_txt.status == STARTED:
                # update params
                pass
            # *mouse_decision* updates
            
            # if mouse_decision is starting this frame...
            if mouse_decision.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouse_decision.frameNStart = frameN  # exact frame index
                mouse_decision.tStart = t  # local t and not account for scr refresh
                mouse_decision.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse_decision, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('mouse_decision.started', t)
                # update status
                mouse_decision.status = STARTED
                mouse_decision.mouseClock.reset()
                prevButtonState = mouse_decision.getPressed()  # if button is down already this ISN'T a new click
            if mouse_decision.status == STARTED:  # only update if started and not finished!
                buttons = mouse_decision.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        pass
                        x, y = mouse_decision.getPos()
                        mouse_decision.x.append(float(x))
                        mouse_decision.y.append(float(y))
                        buttons = mouse_decision.getPressed()
                        mouse_decision.leftButton.append(buttons[0])
                        mouse_decision.midButton.append(buttons[1])
                        mouse_decision.rightButton.append(buttons[2])
                        mouse_decision.time.append(mouse_decision.mouseClock.getTime())
            
            # *happened_text* updates
            
            # if happened_text is starting this frame...
            if happened_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                happened_text.frameNStart = frameN  # exact frame index
                happened_text.tStart = t  # local t and not account for scr refresh
                happened_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(happened_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'happened_text.started')
                # update status
                happened_text.status = STARTED
                happened_text.setAutoDraw(True)
            
            # if happened_text is active this frame...
            if happened_text.status == STARTED:
                # update params
                pass
            
            # *text_2* updates
            
            # if text_2 is starting this frame...
            if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_2.frameNStart = frameN  # exact frame index
                text_2.tStart = t  # local t and not account for scr refresh
                text_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_2.started')
                # update status
                text_2.status = STARTED
                text_2.setAutoDraw(True)
            
            # if text_2 is active this frame...
            if text_2.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Decision,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Decision.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Decision.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Decision.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Decision" ---
        for thisComponent in Decision.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Decision
        Decision.tStop = globalClock.getTime(format='float')
        Decision.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Decision.stopped', Decision.tStop)
        # Run 'End Routine' code from code_decision
        # Safety: if somehow no click occurred, you can decide to repeat or assign a default; here we just guard
        if selected_choice_code is None:
            selected_choice_code = 0  # default to Neutral to avoid crashes
            selected_rt = None
        
        # Determine YOUR like target and the exact option text you chose (from this row)
        if selected_choice_code == 0:
            your_final_likes_target = int(likes_neutral)
            your_option_text = str(option_neutral)
        else:
            your_final_likes_target = int(likes_negative)
            your_option_text = str(option_negative)
        
        # Store your choice
        participant_choices.append(selected_choice_code)
        
        # Compute partner move for THIS round (depends on your Round 2 behavior)
        partner_choice_code = partner_move_for_round(round_num, participant_choices)
        partner_choices.append(partner_choice_code)
        
        # Get the confederate's post text and like target for *their* chosen valence
        if partner_choice_code == 0:
            partner_option_text = str(partner_option_neutral)
            partner_final_likes_target = int(partner_likes_neutral)
        else:
            partner_option_text = str(partner_option_negative)
            partner_final_likes_target = int(partner_likes_negative)
        
        # Persist side mapping (optional, helpful later)
        side_history.append(side_map)
        
        # --- Log decision-level data now ---
        thisExp.addData('round', round_num + 1)
        thisExp.addData('dilemma_text', dilemma_text)
        
        # sides (for side-bias analysis)
        thisExp.addData('left_label', side_map['left_label'])
        thisExp.addData('left_choice_code', side_map['left_choice_code'])
        thisExp.addData('right_label', side_map['right_label'])
        thisExp.addData('right_choice_code', side_map['right_choice_code'])
        
        # your decision
        thisExp.addData('participant_choice_code', selected_choice_code)  # 0=Neutral, 1=Negative
        thisExp.addData('choice_rt', selected_rt)
        
        # partner decision (for this round)
        thisExp.addData('partner_choice_code', partner_choice_code)
        
        # --- Expose variables for the Refresh & Outcome routines ---
        # Targets (the final totals we will accumulate toward)
        final_likes_target_you = your_final_likes_target
        final_likes_target_partner = partner_final_likes_target
        
        # Texts for outcome display
        your_option_text_this_round = your_option_text
        partner_option_text_this_round = partner_option_text
        
        # Expose the chosen option for display on Refresh
        choice_string = your_option_text_this_round
        
        # Choice code (0=Neutral, 1=Negative)
        thisExp.addData('participant_choice_code', selected_choice_code)
        
        # Human-readable choice label
        choice_label = 'Neutral' if selected_choice_code == 0 else 'Negative'
        thisExp.addData('participant_choice_label', choice_label)
        
        # store data for trials (TrialHandler)
        trials.addData('mouse_decision.x', mouse_decision.x)
        trials.addData('mouse_decision.y', mouse_decision.y)
        trials.addData('mouse_decision.leftButton', mouse_decision.leftButton)
        trials.addData('mouse_decision.midButton', mouse_decision.midButton)
        trials.addData('mouse_decision.rightButton', mouse_decision.rightButton)
        trials.addData('mouse_decision.time', mouse_decision.time)
        # the Routine "Decision" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Wait_Choice" ---
        # create an object to store info about Routine Wait_Choice
        Wait_Choice = data.Routine(
            name='Wait_Choice',
            components=[background_16, wait_txt_2],
        )
        Wait_Choice.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_wait_2
        # Pick a random wait duration between 1 and 8 seconds
        import random
        wait_dur = random.randint(0, 9)
        
        # store start times for Wait_Choice
        Wait_Choice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Wait_Choice.tStart = globalClock.getTime(format='float')
        Wait_Choice.status = STARTED
        thisExp.addData('Wait_Choice.started', Wait_Choice.tStart)
        Wait_Choice.maxDuration = None
        # keep track of which components have finished
        Wait_ChoiceComponents = Wait_Choice.components
        for thisComponent in Wait_Choice.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Wait_Choice" ---
        thisExp.currentRoutine = Wait_Choice
        Wait_Choice.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from code_wait_2
            # End routine automatically after wait_dur seconds
            if t >= wait_dur:
                continueRoutine = False
            
            
            # *background_16* updates
            
            # if background_16 is starting this frame...
            if background_16.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                background_16.frameNStart = frameN  # exact frame index
                background_16.tStart = t  # local t and not account for scr refresh
                background_16.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(background_16, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'background_16.started')
                # update status
                background_16.status = STARTED
                background_16.setAutoDraw(True)
            
            # if background_16 is active this frame...
            if background_16.status == STARTED:
                # update params
                pass
            
            # *wait_txt_2* updates
            
            # if wait_txt_2 is starting this frame...
            if wait_txt_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                wait_txt_2.frameNStart = frameN  # exact frame index
                wait_txt_2.tStart = t  # local t and not account for scr refresh
                wait_txt_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(wait_txt_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'wait_txt_2.started')
                # update status
                wait_txt_2.status = STARTED
                wait_txt_2.setAutoDraw(True)
            
            # if wait_txt_2 is active this frame...
            if wait_txt_2.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Wait_Choice,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Wait_Choice.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Wait_Choice.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Wait_Choice.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Wait_Choice" ---
        for thisComponent in Wait_Choice.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Wait_Choice
        Wait_Choice.tStop = globalClock.getTime(format='float')
        Wait_Choice.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Wait_Choice.stopped', Wait_Choice.tStop)
        # the Routine "Wait_Choice" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Refresh" ---
        # create an object to store info about Routine Refresh
        Refresh = data.Routine(
            name='Refresh',
            components=[background_17, likes_txt, refresh_btn, refresh_click, mouse_refresh, timer_txt, choice_box, choice_text, reminder_text, your_choice_txt],
        )
        Refresh.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_refresh
        # ------------- Refresh: Begin Routine -------------
        
        # hard cap for this routine (seconds)
        refresh_duration = 15.0
        
        # Reset “revealed” likes (what the participant sees)
        revealed_likes_you = 0
        revealed_likes_partner = 0
        
        # Running display strings (bound to Text components)
        likes_display = f"Likes: {revealed_likes_you}"
        timer_display = f"{int(refresh_duration)}"
        
        # Count refresh clicks with edge detection (avoid counting holds)
        refresh_clicks = 0
        _prevPressed = [0, 0, 0]
        
        # (Optional) if you want to store exact times of each refresh:
        refresh_click_times = []
        
        # setup some python lists for storing info about the mouse_refresh
        mouse_refresh.x = []
        mouse_refresh.y = []
        mouse_refresh.leftButton = []
        mouse_refresh.midButton = []
        mouse_refresh.rightButton = []
        mouse_refresh.time = []
        gotValidClick = False  # until a click is received
        choice_text.setText(choice_string)
        # store start times for Refresh
        Refresh.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Refresh.tStart = globalClock.getTime(format='float')
        Refresh.status = STARTED
        thisExp.addData('Refresh.started', Refresh.tStart)
        Refresh.maxDuration = None
        # keep track of which components have finished
        RefreshComponents = Refresh.components
        for thisComponent in Refresh.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Refresh" ---
        thisExp.currentRoutine = Refresh
        Refresh.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from code_refresh
            # ------------- Refresh: Each Frame -------------
            
            # End this routine automatically after 15s
            if t >= refresh_duration:
                continueRoutine = False
            
            # 1) Compute the "true" accumulated likes (linear ramp over 15s)
            accum_you = int(final_likes_target_you * min(1.0, t / refresh_duration))
            accum_partner = int(final_likes_target_partner * min(1.0, t / refresh_duration))
            
            # 2) Edge-detect clicks on the refresh button (count only press-down events)
            pressed = mouse_refresh.getPressed()
            if pressed[0] == 1 and _prevPressed[0] == 0 and mouse_refresh.isPressedIn(refresh_btn):
                refresh_clicks += 1
                # Optionally store the timestamp of this refresh
                # refresh_click_times.append(t)
            
                # On each refresh, REVEAL whatever has accumulated so far
                if accum_you > revealed_likes_you:
                    revealed_likes_you = accum_you
                if accum_partner > revealed_likes_partner:
                    revealed_likes_partner = accum_partner
            
            _prevPressed = pressed
            
            # 3) Update on-screen text every frame
            likes_display = f"Likes: {revealed_likes_you}"
            timer_display = f"{int(max(0, refresh_duration - t))}s"
            
            
            # *background_17* updates
            
            # if background_17 is starting this frame...
            if background_17.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                background_17.frameNStart = frameN  # exact frame index
                background_17.tStart = t  # local t and not account for scr refresh
                background_17.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(background_17, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'background_17.started')
                # update status
                background_17.status = STARTED
                background_17.setAutoDraw(True)
            
            # if background_17 is active this frame...
            if background_17.status == STARTED:
                # update params
                pass
            
            # *likes_txt* updates
            
            # if likes_txt is starting this frame...
            if likes_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                likes_txt.frameNStart = frameN  # exact frame index
                likes_txt.tStart = t  # local t and not account for scr refresh
                likes_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(likes_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'likes_txt.started')
                # update status
                likes_txt.status = STARTED
                likes_txt.setAutoDraw(True)
            
            # if likes_txt is active this frame...
            if likes_txt.status == STARTED:
                # update params
                likes_txt.setText(likes_display, log=False)
            
            # *refresh_btn* updates
            
            # if refresh_btn is starting this frame...
            if refresh_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                refresh_btn.frameNStart = frameN  # exact frame index
                refresh_btn.tStart = t  # local t and not account for scr refresh
                refresh_btn.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(refresh_btn, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'refresh_btn.started')
                # update status
                refresh_btn.status = STARTED
                refresh_btn.setAutoDraw(True)
            
            # if refresh_btn is active this frame...
            if refresh_btn.status == STARTED:
                # update params
                pass
            
            # *refresh_click* updates
            
            # if refresh_click is starting this frame...
            if refresh_click.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                refresh_click.frameNStart = frameN  # exact frame index
                refresh_click.tStart = t  # local t and not account for scr refresh
                refresh_click.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(refresh_click, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'refresh_click.started')
                # update status
                refresh_click.status = STARTED
                refresh_click.setAutoDraw(True)
            
            # if refresh_click is active this frame...
            if refresh_click.status == STARTED:
                # update params
                pass
            # *mouse_refresh* updates
            
            # if mouse_refresh is starting this frame...
            if mouse_refresh.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouse_refresh.frameNStart = frameN  # exact frame index
                mouse_refresh.tStart = t  # local t and not account for scr refresh
                mouse_refresh.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse_refresh, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('mouse_refresh.started', t)
                # update status
                mouse_refresh.status = STARTED
                mouse_refresh.mouseClock.reset()
                prevButtonState = mouse_refresh.getPressed()  # if button is down already this ISN'T a new click
            if mouse_refresh.status == STARTED:  # only update if started and not finished!
                buttons = mouse_refresh.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        pass
                        x, y = mouse_refresh.getPos()
                        mouse_refresh.x.append(float(x))
                        mouse_refresh.y.append(float(y))
                        buttons = mouse_refresh.getPressed()
                        mouse_refresh.leftButton.append(buttons[0])
                        mouse_refresh.midButton.append(buttons[1])
                        mouse_refresh.rightButton.append(buttons[2])
                        mouse_refresh.time.append(mouse_refresh.mouseClock.getTime())
            
            # *timer_txt* updates
            
            # if timer_txt is starting this frame...
            if timer_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                timer_txt.frameNStart = frameN  # exact frame index
                timer_txt.tStart = t  # local t and not account for scr refresh
                timer_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(timer_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'timer_txt.started')
                # update status
                timer_txt.status = STARTED
                timer_txt.setAutoDraw(True)
            
            # if timer_txt is active this frame...
            if timer_txt.status == STARTED:
                # update params
                timer_txt.setText(timer_display, log=False)
            
            # *choice_box* updates
            
            # if choice_box is starting this frame...
            if choice_box.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                choice_box.frameNStart = frameN  # exact frame index
                choice_box.tStart = t  # local t and not account for scr refresh
                choice_box.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(choice_box, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'choice_box.started')
                # update status
                choice_box.status = STARTED
                choice_box.setAutoDraw(True)
            
            # if choice_box is active this frame...
            if choice_box.status == STARTED:
                # update params
                pass
            
            # *choice_text* updates
            
            # if choice_text is starting this frame...
            if choice_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                choice_text.frameNStart = frameN  # exact frame index
                choice_text.tStart = t  # local t and not account for scr refresh
                choice_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(choice_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'choice_text.started')
                # update status
                choice_text.status = STARTED
                choice_text.setAutoDraw(True)
            
            # if choice_text is active this frame...
            if choice_text.status == STARTED:
                # update params
                pass
            
            # *reminder_text* updates
            
            # if reminder_text is starting this frame...
            if reminder_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                reminder_text.frameNStart = frameN  # exact frame index
                reminder_text.tStart = t  # local t and not account for scr refresh
                reminder_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(reminder_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'reminder_text.started')
                # update status
                reminder_text.status = STARTED
                reminder_text.setAutoDraw(True)
            
            # if reminder_text is active this frame...
            if reminder_text.status == STARTED:
                # update params
                pass
            
            # *your_choice_txt* updates
            
            # if your_choice_txt is starting this frame...
            if your_choice_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                your_choice_txt.frameNStart = frameN  # exact frame index
                your_choice_txt.tStart = t  # local t and not account for scr refresh
                your_choice_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(your_choice_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'your_choice_txt.started')
                # update status
                your_choice_txt.status = STARTED
                your_choice_txt.setAutoDraw(True)
            
            # if your_choice_txt is active this frame...
            if your_choice_txt.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Refresh,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Refresh.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Refresh.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Refresh.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Refresh" ---
        for thisComponent in Refresh.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Refresh
        Refresh.tStop = globalClock.getTime(format='float')
        Refresh.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Refresh.stopped', Refresh.tStop)
        # Run 'End Routine' code from code_refresh
        # ------------- Refresh: End Routine -------------
        
        
        # Update cumulative tallies (globals set in Step 3)
        total_likes_you += final_likes_target_you
        total_likes_partner += final_likes_target_partner
        # Expose per-round outcomes for Outcome routine
        likes_this_round_you = final_likes_target_you
        likes_this_round_partner = final_likes_target_partner
        
        # Save refresh metrics now
        thisExp.addData('refresh_count', refresh_clicks)
        thisExp.addData('revealed_likes_you', revealed_likes_you)
        thisExp.addData('revealed_likes_partner', revealed_likes_partner)
        
        # If you decided to log exact click times, you can store as a string:
        # thisExp.addData('refresh_click_times', ','.join([f'{x:.3f}' for x in refresh_click_times]))
        
        # store data for trials (TrialHandler)
        trials.addData('mouse_refresh.x', mouse_refresh.x)
        trials.addData('mouse_refresh.y', mouse_refresh.y)
        trials.addData('mouse_refresh.leftButton', mouse_refresh.leftButton)
        trials.addData('mouse_refresh.midButton', mouse_refresh.midButton)
        trials.addData('mouse_refresh.rightButton', mouse_refresh.rightButton)
        trials.addData('mouse_refresh.time', mouse_refresh.time)
        # the Routine "Refresh" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "likes_revealed" ---
        # create an object to store info about Routine likes_revealed
        likes_revealed = data.Routine(
            name='likes_revealed',
            components=[background_18, text_3, choice_box_2, choice_text_2, your_choice_txt_2],
        )
        likes_revealed.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_likesrevealed
        you_likes_thisround_text = f"Likes: {likes_this_round_you}"
        
        text_3.setText(you_likes_thisround_text)
        choice_text_2.setText(choice_string)
        # store start times for likes_revealed
        likes_revealed.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        likes_revealed.tStart = globalClock.getTime(format='float')
        likes_revealed.status = STARTED
        thisExp.addData('likes_revealed.started', likes_revealed.tStart)
        likes_revealed.maxDuration = None
        # keep track of which components have finished
        likes_revealedComponents = likes_revealed.components
        for thisComponent in likes_revealed.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "likes_revealed" ---
        thisExp.currentRoutine = likes_revealed
        likes_revealed.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 5.0:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *background_18* updates
            
            # if background_18 is starting this frame...
            if background_18.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                background_18.frameNStart = frameN  # exact frame index
                background_18.tStart = t  # local t and not account for scr refresh
                background_18.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(background_18, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'background_18.started')
                # update status
                background_18.status = STARTED
                background_18.setAutoDraw(True)
            
            # if background_18 is active this frame...
            if background_18.status == STARTED:
                # update params
                pass
            
            # if background_18 is stopping this frame...
            if background_18.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > background_18.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    background_18.tStop = t  # not accounting for scr refresh
                    background_18.tStopRefresh = tThisFlipGlobal  # on global time
                    background_18.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'background_18.stopped')
                    # update status
                    background_18.status = FINISHED
                    background_18.setAutoDraw(False)
            
            # *text_3* updates
            
            # if text_3 is starting this frame...
            if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_3.frameNStart = frameN  # exact frame index
                text_3.tStart = t  # local t and not account for scr refresh
                text_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_3.started')
                # update status
                text_3.status = STARTED
                text_3.setAutoDraw(True)
            
            # if text_3 is active this frame...
            if text_3.status == STARTED:
                # update params
                pass
            
            # if text_3 is stopping this frame...
            if text_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_3.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    text_3.tStop = t  # not accounting for scr refresh
                    text_3.tStopRefresh = tThisFlipGlobal  # on global time
                    text_3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_3.stopped')
                    # update status
                    text_3.status = FINISHED
                    text_3.setAutoDraw(False)
            
            # *choice_box_2* updates
            
            # if choice_box_2 is starting this frame...
            if choice_box_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                choice_box_2.frameNStart = frameN  # exact frame index
                choice_box_2.tStart = t  # local t and not account for scr refresh
                choice_box_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(choice_box_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'choice_box_2.started')
                # update status
                choice_box_2.status = STARTED
                choice_box_2.setAutoDraw(True)
            
            # if choice_box_2 is active this frame...
            if choice_box_2.status == STARTED:
                # update params
                pass
            
            # if choice_box_2 is stopping this frame...
            if choice_box_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > choice_box_2.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    choice_box_2.tStop = t  # not accounting for scr refresh
                    choice_box_2.tStopRefresh = tThisFlipGlobal  # on global time
                    choice_box_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'choice_box_2.stopped')
                    # update status
                    choice_box_2.status = FINISHED
                    choice_box_2.setAutoDraw(False)
            
            # *choice_text_2* updates
            
            # if choice_text_2 is starting this frame...
            if choice_text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                choice_text_2.frameNStart = frameN  # exact frame index
                choice_text_2.tStart = t  # local t and not account for scr refresh
                choice_text_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(choice_text_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'choice_text_2.started')
                # update status
                choice_text_2.status = STARTED
                choice_text_2.setAutoDraw(True)
            
            # if choice_text_2 is active this frame...
            if choice_text_2.status == STARTED:
                # update params
                pass
            
            # if choice_text_2 is stopping this frame...
            if choice_text_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > choice_text_2.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    choice_text_2.tStop = t  # not accounting for scr refresh
                    choice_text_2.tStopRefresh = tThisFlipGlobal  # on global time
                    choice_text_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'choice_text_2.stopped')
                    # update status
                    choice_text_2.status = FINISHED
                    choice_text_2.setAutoDraw(False)
            
            # *your_choice_txt_2* updates
            
            # if your_choice_txt_2 is starting this frame...
            if your_choice_txt_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                your_choice_txt_2.frameNStart = frameN  # exact frame index
                your_choice_txt_2.tStart = t  # local t and not account for scr refresh
                your_choice_txt_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(your_choice_txt_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'your_choice_txt_2.started')
                # update status
                your_choice_txt_2.status = STARTED
                your_choice_txt_2.setAutoDraw(True)
            
            # if your_choice_txt_2 is active this frame...
            if your_choice_txt_2.status == STARTED:
                # update params
                pass
            
            # if your_choice_txt_2 is stopping this frame...
            if your_choice_txt_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > your_choice_txt_2.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    your_choice_txt_2.tStop = t  # not accounting for scr refresh
                    your_choice_txt_2.tStopRefresh = tThisFlipGlobal  # on global time
                    your_choice_txt_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'your_choice_txt_2.stopped')
                    # update status
                    your_choice_txt_2.status = FINISHED
                    your_choice_txt_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=likes_revealed,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                likes_revealed.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if likes_revealed.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in likes_revealed.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "likes_revealed" ---
        for thisComponent in likes_revealed.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for likes_revealed
        likes_revealed.tStop = globalClock.getTime(format='float')
        likes_revealed.tStopRefresh = tThisFlipGlobal
        thisExp.addData('likes_revealed.stopped', likes_revealed.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if likes_revealed.maxDurationReached:
            routineTimer.addTime(-likes_revealed.maxDuration)
        elif likes_revealed.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-5.000000)
        
        # --- Prepare to start Routine "Outcome" ---
        # create an object to store info about Routine Outcome
        Outcome = data.Routine(
            name='Outcome',
            components=[background_19, round_txt, you_choice_box, you_chose_txt, partner_choice_box, partner_chose_txt, click_box, continue_click, mouse, you_posted, they_posted, remember_view],
        )
        Outcome.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_outcome
        # Round header
        round_text = f"Round {round_num + 1} Complete."
        
        # What YOU chose
        you_chose_text = your_option_text_this_round
        
        # Partner’s choice
        partner_chose_text = partner_option_text_this_round
        
        # Python
        you_chose_txt.alignHoriz = 'left'
        partner_chose_txt.alignHoriz = 'left'
        
        round_txt.setText(round_text
        )
        you_chose_txt.setText(you_chose_text)
        partner_chose_txt.setText(partner_chose_text)
        # setup some python lists for storing info about the mouse
        mouse.x = []
        mouse.y = []
        mouse.leftButton = []
        mouse.midButton = []
        mouse.rightButton = []
        mouse.time = []
        mouse.clicked_name = []
        gotValidClick = False  # until a click is received
        # store start times for Outcome
        Outcome.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Outcome.tStart = globalClock.getTime(format='float')
        Outcome.status = STARTED
        thisExp.addData('Outcome.started', Outcome.tStart)
        Outcome.maxDuration = None
        # keep track of which components have finished
        OutcomeComponents = Outcome.components
        for thisComponent in Outcome.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Outcome" ---
        thisExp.currentRoutine = Outcome
        Outcome.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *background_19* updates
            
            # if background_19 is starting this frame...
            if background_19.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                background_19.frameNStart = frameN  # exact frame index
                background_19.tStart = t  # local t and not account for scr refresh
                background_19.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(background_19, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'background_19.started')
                # update status
                background_19.status = STARTED
                background_19.setAutoDraw(True)
            
            # if background_19 is active this frame...
            if background_19.status == STARTED:
                # update params
                pass
            
            # *round_txt* updates
            
            # if round_txt is starting this frame...
            if round_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                round_txt.frameNStart = frameN  # exact frame index
                round_txt.tStart = t  # local t and not account for scr refresh
                round_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(round_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'round_txt.started')
                # update status
                round_txt.status = STARTED
                round_txt.setAutoDraw(True)
            
            # if round_txt is active this frame...
            if round_txt.status == STARTED:
                # update params
                pass
            
            # *you_choice_box* updates
            
            # if you_choice_box is starting this frame...
            if you_choice_box.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                you_choice_box.frameNStart = frameN  # exact frame index
                you_choice_box.tStart = t  # local t and not account for scr refresh
                you_choice_box.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(you_choice_box, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'you_choice_box.started')
                # update status
                you_choice_box.status = STARTED
                you_choice_box.setAutoDraw(True)
            
            # if you_choice_box is active this frame...
            if you_choice_box.status == STARTED:
                # update params
                pass
            
            # *you_chose_txt* updates
            
            # if you_chose_txt is starting this frame...
            if you_chose_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                you_chose_txt.frameNStart = frameN  # exact frame index
                you_chose_txt.tStart = t  # local t and not account for scr refresh
                you_chose_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(you_chose_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'you_chose_txt.started')
                # update status
                you_chose_txt.status = STARTED
                you_chose_txt.setAutoDraw(True)
            
            # if you_chose_txt is active this frame...
            if you_chose_txt.status == STARTED:
                # update params
                pass
            
            # *partner_choice_box* updates
            
            # if partner_choice_box is starting this frame...
            if partner_choice_box.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                partner_choice_box.frameNStart = frameN  # exact frame index
                partner_choice_box.tStart = t  # local t and not account for scr refresh
                partner_choice_box.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(partner_choice_box, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'partner_choice_box.started')
                # update status
                partner_choice_box.status = STARTED
                partner_choice_box.setAutoDraw(True)
            
            # if partner_choice_box is active this frame...
            if partner_choice_box.status == STARTED:
                # update params
                pass
            
            # *partner_chose_txt* updates
            
            # if partner_chose_txt is starting this frame...
            if partner_chose_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                partner_chose_txt.frameNStart = frameN  # exact frame index
                partner_chose_txt.tStart = t  # local t and not account for scr refresh
                partner_chose_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(partner_chose_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'partner_chose_txt.started')
                # update status
                partner_chose_txt.status = STARTED
                partner_chose_txt.setAutoDraw(True)
            
            # if partner_chose_txt is active this frame...
            if partner_chose_txt.status == STARTED:
                # update params
                pass
            
            # *click_box* updates
            
            # if click_box is starting this frame...
            if click_box.status == NOT_STARTED and tThisFlip >= 10-frameTolerance:
                # keep track of start time/frame for later
                click_box.frameNStart = frameN  # exact frame index
                click_box.tStart = t  # local t and not account for scr refresh
                click_box.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(click_box, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'click_box.started')
                # update status
                click_box.status = STARTED
                click_box.setAutoDraw(True)
            
            # if click_box is active this frame...
            if click_box.status == STARTED:
                # update params
                pass
            
            # *continue_click* updates
            
            # if continue_click is starting this frame...
            if continue_click.status == NOT_STARTED and tThisFlip >= 10-frameTolerance:
                # keep track of start time/frame for later
                continue_click.frameNStart = frameN  # exact frame index
                continue_click.tStart = t  # local t and not account for scr refresh
                continue_click.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(continue_click, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'continue_click.started')
                # update status
                continue_click.status = STARTED
                continue_click.setAutoDraw(True)
            
            # if continue_click is active this frame...
            if continue_click.status == STARTED:
                # update params
                pass
            # *mouse* updates
            
            # if mouse is starting this frame...
            if mouse.status == NOT_STARTED and t >= 10-frameTolerance:
                # keep track of start time/frame for later
                mouse.frameNStart = frameN  # exact frame index
                mouse.tStart = t  # local t and not account for scr refresh
                mouse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('mouse.started', t)
                # update status
                mouse.status = STARTED
                mouse.mouseClock.reset()
                prevButtonState = mouse.getPressed()  # if button is down already this ISN'T a new click
            if mouse.status == STARTED:  # only update if started and not finished!
                buttons = mouse.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames(click_box, namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(mouse):
                                gotValidClick = True
                                mouse.clicked_name.append(obj.name)
                        if not gotValidClick:
                            mouse.clicked_name.append(None)
                        x, y = mouse.getPos()
                        mouse.x.append(float(x))
                        mouse.y.append(float(y))
                        buttons = mouse.getPressed()
                        mouse.leftButton.append(buttons[0])
                        mouse.midButton.append(buttons[1])
                        mouse.rightButton.append(buttons[2])
                        mouse.time.append(mouse.mouseClock.getTime())
                        if gotValidClick:
                            continueRoutine = False  # end routine on response
            
            # *you_posted* updates
            
            # if you_posted is starting this frame...
            if you_posted.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                you_posted.frameNStart = frameN  # exact frame index
                you_posted.tStart = t  # local t and not account for scr refresh
                you_posted.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(you_posted, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'you_posted.started')
                # update status
                you_posted.status = STARTED
                you_posted.setAutoDraw(True)
            
            # if you_posted is active this frame...
            if you_posted.status == STARTED:
                # update params
                pass
            
            # *they_posted* updates
            
            # if they_posted is starting this frame...
            if they_posted.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                they_posted.frameNStart = frameN  # exact frame index
                they_posted.tStart = t  # local t and not account for scr refresh
                they_posted.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(they_posted, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'they_posted.started')
                # update status
                they_posted.status = STARTED
                they_posted.setAutoDraw(True)
            
            # if they_posted is active this frame...
            if they_posted.status == STARTED:
                # update params
                pass
            
            # *remember_view* updates
            
            # if remember_view is starting this frame...
            if remember_view.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                remember_view.frameNStart = frameN  # exact frame index
                remember_view.tStart = t  # local t and not account for scr refresh
                remember_view.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(remember_view, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'remember_view.started')
                # update status
                remember_view.status = STARTED
                remember_view.setAutoDraw(True)
            
            # if remember_view is active this frame...
            if remember_view.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Outcome,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Outcome.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Outcome.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Outcome.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Outcome" ---
        for thisComponent in Outcome.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Outcome
        Outcome.tStop = globalClock.getTime(format='float')
        Outcome.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Outcome.stopped', Outcome.tStop)
        # Run 'End Routine' code from code_outcome
        # Log like outcomes and totals
        thisExp.addData('likes_this_round_you', likes_this_round_you)
        thisExp.addData('likes_this_round_partner', likes_this_round_partner)
        thisExp.addData('total_likes_you', total_likes_you)
        thisExp.addData('total_likes_partner', total_likes_partner)
        
        # Advance global round index
        round_num += 1
        
        # store data for trials (TrialHandler)
        trials.addData('mouse.x', mouse.x)
        trials.addData('mouse.y', mouse.y)
        trials.addData('mouse.leftButton', mouse.leftButton)
        trials.addData('mouse.midButton', mouse.midButton)
        trials.addData('mouse.rightButton', mouse.rightButton)
        trials.addData('mouse.time', mouse.time)
        trials.addData('mouse.clicked_name', mouse.clicked_name)
        # the Routine "Outcome" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "WaitScreen" ---
        # create an object to store info about Routine WaitScreen
        WaitScreen = data.Routine(
            name='WaitScreen',
            components=[background_20, wait_txt],
        )
        WaitScreen.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_wait
        # Pick a random wait duration between 1 and 8 seconds
        import random
        wait_dur = random.randint(1, 8)
        
        # store start times for WaitScreen
        WaitScreen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        WaitScreen.tStart = globalClock.getTime(format='float')
        WaitScreen.status = STARTED
        thisExp.addData('WaitScreen.started', WaitScreen.tStart)
        WaitScreen.maxDuration = None
        # skip Routine WaitScreen if its 'Skip if' condition is True
        WaitScreen.skipped = continueRoutine and not (round == 20)
        continueRoutine = WaitScreen.skipped
        # keep track of which components have finished
        WaitScreenComponents = WaitScreen.components
        for thisComponent in WaitScreen.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "WaitScreen" ---
        thisExp.currentRoutine = WaitScreen
        WaitScreen.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from code_wait
            # End routine automatically after wait_dur seconds
            if t >= wait_dur:
                continueRoutine = False
            
            
            # *background_20* updates
            
            # if background_20 is starting this frame...
            if background_20.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                background_20.frameNStart = frameN  # exact frame index
                background_20.tStart = t  # local t and not account for scr refresh
                background_20.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(background_20, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'background_20.started')
                # update status
                background_20.status = STARTED
                background_20.setAutoDraw(True)
            
            # if background_20 is active this frame...
            if background_20.status == STARTED:
                # update params
                pass
            
            # *wait_txt* updates
            
            # if wait_txt is starting this frame...
            if wait_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                wait_txt.frameNStart = frameN  # exact frame index
                wait_txt.tStart = t  # local t and not account for scr refresh
                wait_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(wait_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'wait_txt.started')
                # update status
                wait_txt.status = STARTED
                wait_txt.setAutoDraw(True)
            
            # if wait_txt is active this frame...
            if wait_txt.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=WaitScreen,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                WaitScreen.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if WaitScreen.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in WaitScreen.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "WaitScreen" ---
        for thisComponent in WaitScreen.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for WaitScreen
        WaitScreen.tStop = globalClock.getTime(format='float')
        WaitScreen.tStopRefresh = tThisFlipGlobal
        thisExp.addData('WaitScreen.stopped', WaitScreen.tStop)
        # the Routine "WaitScreen" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisTrial as finished
        if hasattr(thisTrial, 'status'):
            thisTrial.status = FINISHED
        # if awaiting a pause, pause now
        if trials.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trials.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials'
    trials.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "end" ---
    # create an object to store info about Routine end
    end = data.Routine(
        name='end',
        components=[text, key_resp_15],
    )
    end.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_15
    key_resp_15.keys = []
    key_resp_15.rt = []
    _key_resp_15_allKeys = []
    # store start times for end
    end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    end.tStart = globalClock.getTime(format='float')
    end.status = STARTED
    thisExp.addData('end.started', end.tStart)
    end.maxDuration = None
    # keep track of which components have finished
    endComponents = end.components
    for thisComponent in end.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end" ---
    thisExp.currentRoutine = end
    end.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # *key_resp_15* updates
        waitOnFlip = False
        
        # if key_resp_15 is starting this frame...
        if key_resp_15.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_15.frameNStart = frameN  # exact frame index
            key_resp_15.tStart = t  # local t and not account for scr refresh
            key_resp_15.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_15, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_15.started')
            # update status
            key_resp_15.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_15.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_15.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_15.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_15.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_15_allKeys.extend(theseKeys)
            if len(_key_resp_15_allKeys):
                key_resp_15.keys = _key_resp_15_allKeys[-1].name  # just the last key pressed
                key_resp_15.rt = _key_resp_15_allKeys[-1].rt
                key_resp_15.duration = _key_resp_15_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=end,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            end.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if end.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in end.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end" ---
    for thisComponent in end.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for end
    end.tStop = globalClock.getTime(format='float')
    end.tStopRefresh = tThisFlipGlobal
    thisExp.addData('end.stopped', end.tStop)
    # check responses
    if key_resp_15.keys in ['', [], None]:  # No response was made
        key_resp_15.keys = None
    thisExp.addData('key_resp_15.keys',key_resp_15.keys)
    if key_resp_15.keys != None:  # we had a response
        thisExp.addData('key_resp_15.rt', key_resp_15.rt)
        thisExp.addData('key_resp_15.duration', key_resp_15.duration)
    thisExp.nextEntry()
    # the Routine "end" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
