package main

import (
	"fmt"
	"log"
	"os"
	"time"

	"silero/turndetector"

	"github.com/go-audio/wav"
)

func main() {
	audioPath := "testdata/test-16k.wav"
	if len(os.Args) > 1 {
		audioPath = os.Args[1]
	}

	modelPath := "models/silero_vad.onnx"



	fmt.Printf("=== TurnDetector Go Demonstration ===\n")
	fmt.Printf("Audio File: %s\n", audioPath)
	fmt.Printf("Model Path: %s\n\n", modelPath)

	// 1. Initialize TurnDetector with noise-resistant configuration
	cfg := turndetector.Config{
		ModelPath:         modelPath,
		SampleRate:        16000,
		StartThreshold:    0.55,              // High threshold prevents false-starts from ambient noise
		StopThreshold:     0.30,              // Low threshold prevents premature cut-off on trailing consonants
		MinSpeechDuration: 250 * time.Millisecond, // Discard transient acoustic clicks/coughs < 250ms
		SilenceTimeout:    800 * time.Millisecond, // Detect stopped talking when silence >= 800ms
		PreRollDuration:   200 * time.Millisecond, // Prepend audio before speech trigger so consonants aren't lost
		MinRMS:            0.002,             // Ambient noise floor gate
	}

	td, err := turndetector.New(cfg)
	if err != nil {
		log.Fatalf("Failed to initialize TurnDetector: %v", err)
	}
	defer td.Close()

	// 2. Open and decode WAV file
	f, err := os.Open(audioPath)
	if err != nil {
		log.Fatalf("Failed to open audio file: %v", err)
	}
	defer f.Close()

	dec := wav.NewDecoder(f)
	if !dec.IsValidFile() {
		log.Fatalf("Invalid WAV file: %s", audioPath)
	}

	buf, err := dec.FullPCMBuffer()
	if err != nil {
		log.Fatalf("Failed to decode PCM buffer: %v", err)
	}

	pcmData := buf.AsFloat32Buffer().Data
	totalDuration := float64(len(pcmData)) / float64(cfg.SampleRate)
	fmt.Printf("Loaded %.2fs of audio (%d samples at %d Hz)\n", totalDuration, len(pcmData), cfg.SampleRate)
	fmt.Println("Simulating real-time chunk streaming (chunk size: 512 samples = 32ms)...")
	fmt.Println("----------------------------------------------------------------------")

	chunkSize := td.ChunkSize()
	var completedTurns []turndetector.TurnEvent

	// 3. Stream chunk-by-chunk through TurnDetector
	for i := 0; i+chunkSize <= len(pcmData); i += chunkSize {
		chunk := pcmData[i : i+chunkSize]
		event, err := td.ProcessChunk(chunk)
		if err != nil {
			log.Fatalf("ProcessChunk error at sample %d: %v", i, err)
		}

		if event != nil {
			switch event.Type {
			case turndetector.EventSpeechStart:
				fmt.Printf("[%05.2fs] 🎙️  Speech STARTED (user began speaking)\n", event.StartSec)
			case turndetector.EventSpeechStop:
				fmt.Printf("[%05.2fs] 🛑 User STOPPED talking! (Speech ended at %.2fs, Duration: %.2fs, Buffer: %d samples)\n",
					event.EndSec+cfg.SilenceTimeout.Seconds(),
					event.EndSec,
					event.DurationSec,
					len(event.AudioPCM),
				)
				completedTurns = append(completedTurns, *event)
			case turndetector.EventNoiseDiscarded:
				fmt.Printf("[%05.2fs] ⚡ Transient noise discarded (duration: %.3fs < 250ms)\n", event.EndSec, event.DurationSec)
			}
		}
	}

	// 4. Flush any trailing speech turn at end of stream
	if lastEvent := td.Flush(); lastEvent != nil && lastEvent.Type == turndetector.EventSpeechStop {
		fmt.Printf("[%05.2fs] 🛑 Stream ended: User STOPPED talking! (Duration: %.2fs)\n", lastEvent.EndSec, lastEvent.DurationSec)
		completedTurns = append(completedTurns, *lastEvent)
	}

	fmt.Println("----------------------------------------------------------------------")
	fmt.Printf("\nDetection Summary: %d speech turn(s) detected\n", len(completedTurns))
	for idx, turn := range completedTurns {
		fmt.Printf("  Turn #%d: [%.2fs -> %.2fs] (duration: %.2fs, %d PCM samples)\n",
			idx+1, turn.StartSec, turn.EndSec, turn.DurationSec, len(turn.AudioPCM))
	}
}
