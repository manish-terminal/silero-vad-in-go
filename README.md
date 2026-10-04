# Silero VAD Turn Detector (Go)

A high-performance, real-time speech turn detector and voice activity detection (VAD) library for Go, powered by [Silero VAD](https://github.com/snakers4/silero-vad) and ONNX Runtime.

It reliably detects when a speaker has **started speaking** and **stopped talking**, without false-triggering on background noise or cutting off natural intra-sentence pauses.

---

## Features

- **End-of-Speech / Turn Completion Detection**: Accurately detects when the user finishes speaking, triggering downstream AI/ASR pipelines.
- **Dual-Threshold Hysteresis**:
  - `StartThreshold` (e.g. `0.55`): Avoids false starts from room noise, breathing, or background chatter.
  - `StopThreshold` (e.g. `0.30`): Low threshold ensures soft word endings (trailing consonants, whispers) are not clipped.
- **Noise Spike Rejection**: Acoustic bursts shorter than `MinSpeechDuration` (e.g. `250ms`) like keyboard clicks, desk taps, or coughs are automatically discarded.
- **Hangover Silence Timeout**: Handles natural conversational pauses (100–300ms) within sentences without breaking the utterance.
- **Pre-Roll Ring Buffer**: Retains a 200ms audio buffer before speech trigger to ensure leading consonants are preserved.
- **Pure Go Streaming API**: Process chunk-by-chunk in real time (e.g. 512 samples = 32ms at 16kHz) from microphone, WebRTC, or WebSocket audio streams.

---

## Requirements

- **Go**: >= 1.21
- **ONNX Runtime**: Installed on your system
  - **macOS** (Homebrew): `brew install onnxruntime`
  - **Linux** (Debian/Ubuntu): `apt-get install libonnxruntime-dev` or download the prebuilt binary from Microsoft.

---

## Project Structure

```
.
├── cmd/
│   └── main.go                 # Turn detection demo entry point
├── turndetector/               # Core TurnDetector package
│   ├── detector.go             # State machine (hysteresis, silence timeout, buffers)
│   ├── engine.go               # ONNX Runtime CGO wrapper
│   ├── ort_bridge.c            # C bridge for ONNX Runtime C API
│   └── ort_bridge.h            # C header
├── models/
│   └── silero_vad.onnx         # Bundled Silero VAD ONNX model
├── testdata/
│   └── test-16k.wav            # Sample 16kHz mono audio test file
├── go.mod                      # Go module definition
├── go.sum                      # Dependency checksums
└── README.md
```

---

## Quick Start

### 1. Run the Demo

```bash
go run ./cmd/main.go testdata/test-16k.wav
```

### 2. Library Usage

```go
package main

import (
	"fmt"
	"log"
	"time"

	"silero/turndetector"
)

func main() {
	cfg := turndetector.Config{
		ModelPath:         "models/silero_vad.onnx",
		SampleRate:        16000,
		StartThreshold:    0.55,              // Onset threshold
		StopThreshold:     0.30,              // Offset threshold
		MinSpeechDuration: 250 * time.Millisecond, // Discard clicks/coughs < 250ms
		SilenceTimeout:    800 * time.Millisecond, // Time to declare turn complete
		PreRollDuration:   200 * time.Millisecond, // Audio preserved before speech
		MinRMS:            0.002,             // Ambient noise gate
	}

	td, err := turndetector.New(cfg)
	if err != nil {
		log.Fatalf("Failed to create detector: %v", err)
	}
	defer td.Close()

	// In your real-time audio loop (e.g., 512 samples = 32ms chunks at 16kHz):
	var chunk []float32 // 512 float32 samples in range [-1.0, 1.0]

	event, err := td.ProcessChunk(chunk)
	if err != nil {
		log.Fatal(err)
	}

	if event != nil {
		switch event.Type {
		case turndetector.EventSpeechStart:
			fmt.Printf("🎙️ User started talking at %.2fs\n", event.StartSec)
		case turndetector.EventSpeechStop:
			fmt.Printf("🛑 User STOPPED talking! Utterance duration: %.2fs (%d samples)\n",
				event.DurationSec, len(event.AudioPCM))
			// Pass event.AudioPCM directly to your STT / Whisper engine
		case turndetector.EventNoiseDiscarded:
			fmt.Printf("⚡ Transient noise spike ignored (< 250ms)\n")
		}
	}
}
```

---

## License

This project incorporates the Silero VAD model under the [MIT License](LICENSE).
