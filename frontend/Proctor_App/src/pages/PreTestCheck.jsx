import React, { useState, useEffect, useRef } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { Camera, Mic, Maximize, AlertCircle, CheckCircle2, Loader2, Volume2 } from 'lucide-react';
import { studentTestsAPI, proctoringAPI } from '../services/api';

const PreTestCheck = () => {
  const { testId } = useParams();
  const navigate = useNavigate();
  
  // Stage management
  const [currentStage, setCurrentStage] = useState(1); // 1: Voice, 2: Camera, 3: Fullscreen
  const [stageStatus, setStageStatus] = useState({
    voice: 'pending', // pending, checking, passed, failed
    camera: 'pending',
    fullscreen: 'pending'
  });

  // Stage 1: Voice checks
  const [voiceChecks, setVoiceChecks] = useState({
    permission: { status: 'pending', label: 'Microphone Permission' },
    builtIn: { status: 'pending', label: 'Built-in Microphone' },
    audioLevel: { status: 'pending', label: 'Audio Level Detection' }
  });
  const [audioLevel, setAudioLevel] = useState(0);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const animationRef = useRef(null);

  // Stage 2: Camera preview
  const [cameraFrame, setCameraFrame] = useState(null);
  const [frameTimestamp, setFrameTimestamp] = useState(null);
  const [aiProcessSpawned, setAiProcessSpawned] = useState(false);
  const [frameCount, setFrameCount] = useState(0);
  const REQUIRED_FRAMES = 10;

  // Stage 3: Fullscreen
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [fullscreenLocked, setFullscreenLocked] = useState(false);

  // Error states
  const [errorMessage, setErrorMessage] = useState('');

  const builtInKeywords = ['built-in', 'internal', 'onboard', 'integrated', 'array', 'default', 'apple', 'macbook'];

  // =========================
  // STAGE 1: VOICE CHECKS
  // =========================
  const runVoiceChecks = async () => {
    try {
      setStageStatus(prev => ({ ...prev, voice: 'checking' }));
      setErrorMessage('');

      // 1. Request microphone permission
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setVoiceChecks(prev => ({
        ...prev,
        permission: { ...prev.permission, status: 'pass' }
      }));

      // 2. Check for built-in microphone
      const devices = await navigator.mediaDevices.enumerateDevices();
      const mics = devices.filter(d => d.kind === 'audioinput');
      
      const hasBuiltIn = mics.some(m => 
        builtInKeywords.some(keyword => m.label.toLowerCase().includes(keyword))
      );

      if (!hasBuiltIn) {
        throw new Error("Built-in laptop microphone not detected. Please use your system's integrated audio.");
      }
      setVoiceChecks(prev => ({
        ...prev,
        builtIn: { ...prev.builtIn, status: 'pass' }
      }));

      // 3. Start audio level monitoring
      await startAudioLevelMonitoring(stream);
      setVoiceChecks(prev => ({
        ...prev,
        audioLevel: { ...prev.audioLevel, status: 'pass' }
      }));

      setStageStatus(prev => ({ ...prev, voice: 'passed' }));

    } catch (err) {
      const errorText = err.message || "Voice verification failed.";
      setErrorMessage(errorText);
      setStageStatus(prev => ({ ...prev, voice: 'failed' }));
      
      // Log failure
      try {
        await proctoringAPI.reportViolation(testId, {
          type: 'hardware-precheck-voice-fail',
          severity: 'high',
          message: errorText,
          timestamp: new Date().toISOString()
        });
      } catch (logErr) {
        console.error("Failed to log violation:", logErr);
      }
    }
  };

  const startAudioLevelMonitoring = async (stream) => {
    try {
      // Create audio context and analyser
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      analyserRef.current = audioContextRef.current.createAnalyser();
      
      const source = audioContextRef.current.createMediaStreamSource(stream);
      source.connect(analyserRef.current);
      
      analyserRef.current.fftSize = 256;
      const bufferLength = analyserRef.current.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      // Animation loop to update audio level
      const updateLevel = () => {
        analyserRef.current.getByteFrequencyData(dataArray);
        const average = dataArray.reduce((a, b) => a + b) / bufferLength;
        const normalizedLevel = Math.min(100, (average / 255) * 100);
        setAudioLevel(normalizedLevel);
        animationRef.current = requestAnimationFrame(updateLevel);
      };
      
      updateLevel();
    } catch (error) {
      console.error('Audio monitoring error:', error);
      throw new Error('Failed to start audio level monitoring');
    }
  };

  const stopAudioMonitoring = () => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }
  };

  // =========================
  // STAGE 2: CAMERA CHECK
  // =========================
  const startCameraCheck = async () => {
    try {
      setStageStatus(prev => ({ ...prev, camera: 'checking' }));
      setErrorMessage('');
      setFrameCount(0);

      // Set up frame listener BEFORE starting stream to catch first frame
      const handleFrame = (frame) => {
        if (frame.format === 'jpeg' || frame.format === 'raw-bgr') {
          setCameraFrame(`data:image/jpeg;base64,${frame.data}`);
          setFrameTimestamp(frame.timestamp);
          
          // Increment frame count
          setFrameCount(prev => {
            const newCount = prev + 1;
            
            // Mark as passed after receiving required number of frames
            if (newCount >= REQUIRED_FRAMES) {
              setStageStatus(prevStatus => ({ ...prevStatus, camera: 'passed' }));
              console.log(`[PreTest] ${REQUIRED_FRAMES} frames received, camera feed stable`);
            }
            
            return newCount;
          });
        }
      };
      
      // Register listener
      window.electron.receive('proctoring:frame', handleFrame);

      // Spawn AI proctoring process
      console.log('[PreTest] Spawning AI process...');
      const spawnResult = await window.electron.invoke('proctoring:start', {
        testId,
        sessionId: `pretest_${testId}_${Date.now()}`
      });

      if (!spawnResult.success) {
        throw new Error('Failed to start AI proctoring process');
      }

      setAiProcessSpawned(true);

      // Wait for shared memory file to be created
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Start frame streaming
      console.log('[PreTest] Starting frame stream...');
      const streamResult = await window.electron.invoke('proctoring:startFrameStream', {
        fps: 15,
        quality: 70
      });

      if (!streamResult.success) {
        throw new Error('Failed to start frame streaming');
      }
      
      console.log('[PreTest] Frame stream started, waiting for first frame...');
      // Note: camera will be marked as 'passed' when first frame arrives

    } catch (err) {
      const errorText = err.message || "Camera check failed.";
      setErrorMessage(errorText);
      setStageStatus(prev => ({ ...prev, camera: 'failed' }));
    }
  };

  const stopCameraCheck = async () => {
    await window.electron.invoke('proctoring:stopFrameStream');
  };

  // =========================
  // STAGE 3: FULLSCREEN
  // =========================
  const enterAndLockFullscreen = async () => {
    try {
      setStageStatus(prev => ({ ...prev, fullscreen: 'checking' }));
      
      const element = document.documentElement;
      
      // Request fullscreen
      if (element.requestFullscreen) {
        await element.requestFullscreen();
      } else if (element.webkitRequestFullscreen) {
        await element.webkitRequestFullscreen();
      } else if (element.mozRequestFullScreen) {
        await element.mozRequestFullScreen();
      } else if (element.msRequestFullscreen) {
        await element.msRequestFullscreen();
      } else {
        throw new Error('Fullscreen not supported');
      }

      setIsFullscreen(true);
      setFullscreenLocked(true);
      setStageStatus(prev => ({ ...prev, fullscreen: 'passed' }));
      
      // Disable preview mode (stop Python from writing frames)
      console.log('[PreTest] Disabling preview mode...');
      await window.electron.invoke('proctoring:disablePreview');
      
      // Navigate to test after entering fullscreen
      console.log('[PreTest] Fullscreen locked, navigating to test...');
      stopCameraCheck();
      navigate(`/student/tests/${testId}`);

    } catch (err) {
      setErrorMessage('Failed to enter fullscreen mode');
      setStageStatus(prev => ({ ...prev, fullscreen: 'failed' }));
    }
  };

  // Monitor fullscreen exit attempts
  useEffect(() => {
    const handleFullscreenChange = () => {
      const isCurrentlyFullscreen = !!(
        document.fullscreenElement ||
        document.webkitFullscreenElement ||
        document.mozFullScreenElement ||
        document.msFullscreenElement
      );

      setIsFullscreen(isCurrentlyFullscreen);

      // If fullscreen was locked and user exits, re-enter
      if (fullscreenLocked && !isCurrentlyFullscreen) {
        console.warn('[PreTest] Fullscreen exit detected, re-entering...');
        enterAndLockFullscreen();
      }
    };

    // Block F11 and Escape keys
    const handleKeyDown = (e) => {
      if (fullscreenLocked && (e.key === 'F11' || e.key === 'Escape')) {
        e.preventDefault();
        e.stopPropagation();
      }
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
    document.addEventListener('mozfullscreenchange', handleFullscreenChange);
    document.addEventListener('keydown', handleKeyDown, true);

    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
      document.removeEventListener('webkitfullscreenchange', handleFullscreenChange);
      document.removeEventListener('mozfullscreenchange', handleFullscreenChange);
      document.removeEventListener('keydown', handleKeyDown, true);
    };
  }, [fullscreenLocked]);

  // =========================
  // STAGE PROGRESSION
  // =========================
  const proceedToNextStage = () => {
    if (currentStage === 1 && stageStatus.voice === 'passed') {
      stopAudioMonitoring();
      setCurrentStage(2);
      startCameraCheck();
    } else if (currentStage === 2 && stageStatus.camera === 'passed') {
      stopCameraCheck();
      setCurrentStage(3);
    } else if (currentStage === 3 && stageStatus.fullscreen === 'passed') {
      // All stages complete, navigate to test
      navigate(`/student/tests/${testId}`);
    }
  };

  const retryCurrentStage = () => {
    setErrorMessage('');
    if (currentStage === 1) {
      runVoiceChecks();
    } else if (currentStage === 2) {
      startCameraCheck();
    } else if (currentStage === 3) {
      enterAndLockFullscreen();
    }
  };

  // Start first stage on mount
  useEffect(() => {
    if (currentStage === 1) {
      runVoiceChecks();
    }
  }, []);

  // =========================
  // RENDER
  // =========================
  const getStageTitle = () => {
    if (currentStage === 1) return 'Stage 1: Voice Verification';
    if (currentStage === 2) return 'Stage 2: Camera Preview';
    return 'Stage 3: Fullscreen Mode';
  };

  const getStageDescription = () => {
    if (currentStage === 1) return 'Verifying microphone access and audio levels';
    if (currentStage === 2) return 'Checking camera feed from AI proctoring system';
    return 'Entering secure fullscreen mode';
  };

  return (
    <div className="min-h-screen bg-gray-900 flex items-center justify-center p-4">
      <div className="max-w-2xl w-full bg-white rounded-2xl shadow-2xl p-8">
        {/* Stage Progress Indicator */}
        <div className="flex justify-between mb-8">
          {[1, 2, 3].map((stage) => (
            <div key={stage} className="flex-1 flex items-center">
              <div className={`w-10 h-10 rounded-full flex items-center justify-center font-bold ${
                currentStage > stage ? 'bg-green-500 text-white' :
                currentStage === stage ? 'bg-black text-white' :
                'bg-gray-200 text-gray-500'
              }`}>
                {stage}
              </div>
              {stage < 3 && (
                <div className={`flex-1 h-1 mx-2 ${
                  currentStage > stage ? 'bg-green-500' : 'bg-gray-200'
                }`} />
              )}
            </div>
          ))}
        </div>

        {/* Stage Header */}
        <div className="text-center mb-8">
          <h1 className="text-2xl font-bold text-gray-900">{getStageTitle()}</h1>
          <p className="text-gray-600 mt-2">{getStageDescription()}</p>
        </div>

        {/* Stage 1: Voice Checks */}
        {currentStage === 1 && (
          <div>
            <div className="space-y-4 mb-6">
              {Object.entries(voiceChecks).map(([key, check]) => (
                <div key={key} className="flex items-center justify-between p-4 bg-gray-50 border border-gray-200 rounded-xl">
                  <div className="flex items-center gap-3">
                    <Mic className="w-5 h-5 text-gray-600" />
                    <span className="font-medium text-gray-800">{check.label}</span>
                  </div>
                  {check.status === 'pass' ? (
                    <CheckCircle2 className="w-6 h-6 text-green-500" />
                  ) : check.status === 'pending' && stageStatus.voice === 'checking' ? (
                    <Loader2 className="w-6 h-6 text-gray-300 animate-spin" />
                  ) : (
                    <AlertCircle className="w-6 h-6 text-red-500" />
                  )}
                </div>
              ))}
            </div>

            {/* Audio Level Meter */}
            {stageStatus.voice === 'checking' && (
              <div className="mb-6">
                <div className="flex items-center gap-3 mb-2">
                  <Volume2 className="w-5 h-5 text-gray-600" />
                  <span className="text-sm font-medium text-gray-700">Audio Level</span>
                </div>
                <div className="w-full h-8 bg-gray-200 rounded-lg overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-green-400 to-green-600 transition-all duration-100"
                    style={{ width: `${audioLevel}%` }}
                  />
                </div>
                <p className="text-xs text-gray-500 mt-1">Speak to test your microphone</p>
              </div>
            )}
          </div>
        )}

        {/* Stage 2: Camera Preview */}
        {currentStage === 2 && (
          <div>
            <div className="bg-black rounded-lg overflow-hidden mb-4" style={{ aspectRatio: '16/9' }}>
              {cameraFrame ? (
                <img src={cameraFrame} alt="Camera Preview" className="w-full h-full object-cover" />
              ) : (
                <div className="w-full h-full flex items-center justify-center">
                  <Loader2 className="w-12 h-12 text-white animate-spin" />
                  <span className="ml-3 text-white">Loading camera feed...</span>
                </div>
              )}
            </div>
            
            {/* Camera feed status with progress */}
            <div className="mb-4">
              {stageStatus.camera === 'checking' && (
                <div className="flex items-center gap-3 p-3 bg-blue-50 rounded-lg">
                  <Loader2 className="w-5 h-5 text-blue-600 animate-spin" />
                  <div className="flex-1">
                    <p className="text-sm font-medium text-blue-900">
                      {frameCount === 0 ? 'Initializing camera feed...' : `Stabilizing feed... (${frameCount}/${REQUIRED_FRAMES})`}
                    </p>
                    {frameCount > 0 && (
                      <div className="w-full h-2 bg-blue-200 rounded-full mt-2 overflow-hidden">
                        <div 
                          className="h-full bg-blue-600 transition-all duration-300"
                          style={{ width: `${(frameCount / REQUIRED_FRAMES) * 100}%` }}
                        />
                      </div>
                    )}
                  </div>
                </div>
              )}
              {stageStatus.camera === 'passed' && (
                <div className="flex items-center gap-3 p-3 bg-green-50 rounded-lg">
                  <CheckCircle2 className="w-5 h-5 text-green-600" />
                  <p className="text-sm font-medium text-green-900">Camera feed stable and ready</p>
                </div>
              )}
            </div>
            
            {frameTimestamp && (
              <p className="text-xs text-gray-500 text-center">
                Last update: {new Date(frameTimestamp * 1000).toLocaleTimeString()}
              </p>
            )}
          </div>
        )}

        {/* Stage 3: Fullscreen Instructions */}
        {currentStage === 3 && (
          <div className="text-center">
            <Maximize className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-700 mb-6">
              The test will run in fullscreen mode. Press the button below to enter fullscreen.
              You will not be able to exit until the test is complete.
            </p>
          </div>
        )}

        {/* Error Message */}
        {errorMessage && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
            <p className="text-red-800 text-sm font-medium flex items-center gap-2">
              <AlertCircle className="w-4 h-4" />
              {errorMessage}
            </p>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex gap-4">
          {stageStatus[currentStage === 1 ? 'voice' : currentStage === 2 ? 'camera' : 'fullscreen'] === 'failed' && (
            <button onClick={retryCurrentStage} className="flex-1 py-3 bg-gray-200 text-gray-800 rounded-lg font-bold hover:bg-gray-300">
              Retry
            </button>
          )}
          
          <button
            onClick={currentStage === 3 ? enterAndLockFullscreen : proceedToNextStage}
            disabled={
              currentStage === 1 ? stageStatus.voice !== 'passed' :
              currentStage === 2 ? stageStatus.camera !== 'passed' :
              false  // Stage 3: Always enabled (need to click to enter fullscreen)
            }
            className={`flex-1 py-3 font-bold rounded-lg transition-all ${
              (currentStage === 1 && stageStatus.voice === 'passed') ||
              (currentStage === 2 && stageStatus.camera === 'passed') ||
              (currentStage === 3)
                ? 'bg-black text-white hover:bg-gray-800 shadow-lg' 
                : 'bg-gray-100 text-gray-400 cursor-not-allowed'
            }`}
          >
            {currentStage === 3 ? 'Enter Fullscreen & Start Test' : 'Next Stage'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default PreTestCheck;
