package turndetector

import (
	"fmt"
	"math"
	"time"
)

type TurnEventType int

const (
	EventNone TurnEventType = iota
	EventSpeechStart
	EventSpeechStop
	EventNoiseDiscarded
)

func (t TurnEventType) String() string {
	switch t {
	case EventSpeechStart:
		return "SpeechStart"
	case EventSpeechStop:
		return "SpeechStop (User Stopped Talking)"
	case EventNoiseDiscarded:
		return "NoiseDiscarded (Transient Rejected)"
	default:
		return "None"
	}
}

type TurnEvent struct {
	Type         TurnEventType
	StartSec     float64
	EndSec       float64
	DurationSec  float64
	Probability  float32
	AudioPCM     []float32 // Full audio slice for this speech turn (when EventSpeechStop)
}

type Config struct {
	ModelPath          string
	SampleRate         int           // 16000 or 8000 (default: 16000)
	StartThreshold     float32       // Speech onset threshold (default: 0.55)
	StopThreshold      float32       // Speech offset threshold (default: 0.30)
	MinSpeechDuration  time.Duration // Minimum duration to qualify as speech (default: 250ms)
	SilenceTimeout     time.Duration // Silence duration after speech to trigger Turn Complete (default: 800ms)
	PreRollDuration    time.Duration // Audio prepended before speech onset to avoid clipping (default: 200ms)
	MinRMS             float32       // Minimum RMS energy to consider (default: 0.002, ~ -54dBFS)
}

func DefaultConfig(modelPath string) Config {
	return Config{
		ModelPath:         modelPath,
		SampleRate:        16000,
		StartThreshold:    0.55,
		StopThreshold:     0.30,
		MinSpeechDuration: 250 * time.Millisecond,
		SilenceTimeout:    800 * time.Millisecond,
		PreRollDuration:   200 * time.Millisecond,
		MinRMS:            0.002,
	}
}

type detectorState int

const (
	stateSilence detectorState = iota
	stateCandidateSpeech
	stateSpeaking
	stateTrailingSilence
)

type TurnDetector struct {
	cfg        Config
	engine     *vadEngine
	chunkSize  int

	state            detectorState
	currSample       int64
	speechStartSample int64
	silenceStartSample int64

	// Audio buffers
	preRollBuffer  [][]float32
	maxPreRollChunks int
	utteranceAudio []float32
}

func New(cfg Config) (*TurnDetector, error) {
	if cfg.SampleRate == 0 {
		cfg.SampleRate = 16000
	}
	if cfg.StartThreshold == 0 {
		cfg.StartThreshold = 0.55
	}
	if cfg.StopThreshold == 0 {
		cfg.StopThreshold = 0.30
	}
	if cfg.MinSpeechDuration == 0 {
		cfg.MinSpeechDuration = 250 * time.Millisecond
	}
	if cfg.SilenceTimeout == 0 {
		cfg.SilenceTimeout = 800 * time.Millisecond
	}
	if cfg.PreRollDuration == 0 {
		cfg.PreRollDuration = 200 * time.Millisecond
	}

	engine, err := newVADEngine(cfg.ModelPath, cfg.SampleRate)
	if err != nil {
		return nil, fmt.Errorf("failed to init vad engine: %w", err)
	}

	chunkSize := 512
	if cfg.SampleRate == 8000 {
		chunkSize = 256
	}

	preRollSamples := int(float64(cfg.SampleRate) * cfg.PreRollDuration.Seconds())
	maxPreRollChunks := (preRollSamples + chunkSize - 1) / chunkSize
	if maxPreRollChunks < 1 {
		maxPreRollChunks = 1
	}

	td := &TurnDetector{
		cfg:              cfg,
		engine:           engine,
		chunkSize:        chunkSize,
		state:            stateSilence,
		maxPreRollChunks: maxPreRollChunks,
		preRollBuffer:    make([][]float32, 0, maxPreRollChunks),
	}

	return td, nil
}

func (td *TurnDetector) ChunkSize() int {
	return td.chunkSize
}

func (td *TurnDetector) Reset() {
	td.engine.Reset()
	td.state = stateSilence
	td.currSample = 0
	td.speechStartSample = 0
	td.silenceStartSample = 0
	td.preRollBuffer = td.preRollBuffer[:0]
	td.utteranceAudio = td.utteranceAudio[:0]
}

func (td *TurnDetector) Close() error {
	return td.engine.Close()
}

// ProcessChunk processes a single frame of audio (512 samples for 16kHz).
func (td *TurnDetector) ProcessChunk(chunk []float32) (*TurnEvent, error) {
	if len(chunk) != td.chunkSize {
		return nil, fmt.Errorf("expected chunk of %d samples, got %d", td.chunkSize, len(chunk))
	}

	// Compute RMS energy
	var sumSquares float64
	for _, s := range chunk {
		sumSquares += float64(s * s)
	}
	rms := float32(math.Sqrt(sumSquares / float64(len(chunk))))

	// Run VAD model
	prob, err := td.engine.Infer(chunk)
	if err != nil {
		return nil, err
	}

	// If RMS is below ambient noise floor gate, suppress low-level artifacts
	if rms < td.cfg.MinRMS {
		prob = 0.0
	}

	chunkTimeSec := float64(td.currSample) / float64(td.cfg.SampleRate)
	td.currSample += int64(len(chunk))

	var event *TurnEvent

	switch td.state {
	case stateSilence:
		// Maintain pre-roll circular buffer
		chunkCopy := make([]float32, len(chunk))
		copy(chunkCopy, chunk)
		if len(td.preRollBuffer) >= td.maxPreRollChunks {
			td.preRollBuffer = td.preRollBuffer[1:]
		}
		td.preRollBuffer = append(td.preRollBuffer, chunkCopy)

		if prob >= td.cfg.StartThreshold {
			td.state = stateCandidateSpeech
			td.speechStartSample = td.currSample - int64(len(chunk))
			td.utteranceAudio = td.utteranceAudio[:0]
			for _, pr := range td.preRollBuffer {
				td.utteranceAudio = append(td.utteranceAudio, pr...)
			}
		}

	case stateCandidateSpeech:
		td.utteranceAudio = append(td.utteranceAudio, chunk...)
		speechDuration := time.Duration(float64(td.currSample-td.speechStartSample)/float64(td.cfg.SampleRate)*1000) * time.Millisecond

		if prob < td.cfg.StopThreshold {
			// Energy dropped before meeting minimum speech duration: Discard as transient noise!
			event = &TurnEvent{
				Type:        EventNoiseDiscarded,
				StartSec:    float64(td.speechStartSample) / float64(td.cfg.SampleRate),
				EndSec:      chunkTimeSec,
				DurationSec: speechDuration.Seconds(),
				Probability: prob,
			}
			td.state = stateSilence
			td.utteranceAudio = td.utteranceAudio[:0]
			td.preRollBuffer = td.preRollBuffer[:0]
		} else if speechDuration >= td.cfg.MinSpeechDuration {
			// Speech sustained past min duration: Confirmed speech!
			td.state = stateSpeaking
			event = &TurnEvent{
				Type:        EventSpeechStart,
				StartSec:    float64(td.speechStartSample) / float64(td.cfg.SampleRate),
				Probability: prob,
			}
		}

	case stateSpeaking:
		td.utteranceAudio = append(td.utteranceAudio, chunk...)

		if prob < td.cfg.StopThreshold {
			// Speech energy dropped: enter trailing silence hangover
			td.state = stateTrailingSilence
			td.silenceStartSample = td.currSample - int64(len(chunk))
		}

	case stateTrailingSilence:
		td.utteranceAudio = append(td.utteranceAudio, chunk...)
		silenceDuration := time.Duration(float64(td.currSample-td.silenceStartSample)/float64(td.cfg.SampleRate)*1000) * time.Millisecond

		if prob >= td.cfg.StartThreshold {
			// Speaker resumed talking within the pause (intra-sentence pause)!
			td.state = stateSpeaking
			td.silenceStartSample = 0
		} else if silenceDuration >= td.cfg.SilenceTimeout {
			// Silence exceeded timeout: Speaker has stopped talking!
			speechEndSec := float64(td.silenceStartSample) / float64(td.cfg.SampleRate)
			speechStartSec := float64(td.speechStartSample) / float64(td.cfg.SampleRate)

			collectedAudio := make([]float32, len(td.utteranceAudio))
			copy(collectedAudio, td.utteranceAudio)

			event = &TurnEvent{
				Type:        EventSpeechStop,
				StartSec:    speechStartSec,
				EndSec:      speechEndSec,
				DurationSec: speechEndSec - speechStartSec,
				AudioPCM:    collectedAudio,
				Probability: prob,
			}

			td.state = stateSilence
			td.utteranceAudio = td.utteranceAudio[:0]
			td.preRollBuffer = td.preRollBuffer[:0]
			td.silenceStartSample = 0
		}
	}

	return event, nil
}

// Flush checks if there is an unfinished utterance and completes it.
func (td *TurnDetector) Flush() *TurnEvent {
	if td.state == stateSpeaking || td.state == stateTrailingSilence {
		endSec := float64(td.currSample) / float64(td.cfg.SampleRate)
		if td.silenceStartSample > 0 {
			endSec = float64(td.silenceStartSample) / float64(td.cfg.SampleRate)
		}
		startSec := float64(td.speechStartSample) / float64(td.cfg.SampleRate)
		collectedAudio := make([]float32, len(td.utteranceAudio))
		copy(collectedAudio, td.utteranceAudio)

		event := &TurnEvent{
			Type:        EventSpeechStop,
			StartSec:    startSec,
			EndSec:      endSec,
			DurationSec: endSec - startSec,
			AudioPCM:    collectedAudio,
		}
		td.Reset()
		return event
	}
	td.Reset()
	return nil
}

// ProcessBuffer processes a slice of audio samples, chunk by chunk, returning all emitted events.
func (td *TurnDetector) ProcessBuffer(pcm []float32) ([]TurnEvent, error) {
	var events []TurnEvent
	for i := 0; i+td.chunkSize <= len(pcm); i += td.chunkSize {
		ev, err := td.ProcessChunk(pcm[i : i+td.chunkSize])
		if err != nil {
			return events, err
		}
		if ev != nil && ev.Type != EventNone {
			events = append(events, *ev)
		}
	}
	if flushEv := td.Flush(); flushEv != nil {
		events = append(events, *flushEv)
	}
	return events, nil
}
