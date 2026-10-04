package turndetector

// #cgo CFLAGS: -Wall -std=c99
// #cgo darwin CFLAGS: -I/opt/homebrew/include/onnxruntime -I/opt/homebrew/include -I/usr/local/include/onnxruntime -I/usr/local/include
// #cgo darwin LDFLAGS: -L/opt/homebrew/lib -L/usr/local/lib -lonnxruntime
// #cgo linux CFLAGS: -I/usr/include/onnxruntime -I/usr/local/include/onnxruntime
// #cgo linux LDFLAGS: -lonnxruntime
// #include "ort_bridge.h"
import "C"

import (
	"fmt"
	"unsafe"
)

const (
	stateLen   = 2 * 1 * 128
	contextLen = 64
)

// vadEngine runs Silero VAD v5 ONNX model inference.
type vadEngine struct {
	api        *C.OrtApi
	env        *C.OrtEnv
	session    *C.OrtSession
	memoryInfo *C.OrtMemoryInfo

	sampleRate int
	state      [stateLen]float32
	ctx        [contextLen]float32
	currSample int64

	cInputName   *C.char
	cStateName   *C.char
	cSrName      *C.char
	cOutputName  *C.char
	cStateNName  *C.char
}

func newVADEngine(modelPath string, sampleRate int) (*vadEngine, error) {
	if sampleRate != 16000 && sampleRate != 8000 {
		return nil, fmt.Errorf("sample rate must be 8000 or 16000, got %d", sampleRate)
	}

	api := C.OrtGetApi()
	if api == nil {
		return nil, fmt.Errorf("failed to get ONNX Runtime API")
	}

	var env *C.OrtEnv
	cLogId := C.CString("turndetector")
	defer C.free(unsafe.Pointer(cLogId))

	status := C.OrtApiCreateEnv(api, C.ORT_LOGGING_LEVEL_WARNING, cLogId, &env)
	if status != nil {
		defer C.OrtApiReleaseStatus(api, status)
		return nil, fmt.Errorf("create ORT env failed: %s", C.GoString(C.OrtApiGetErrorMessage(api, status)))
	}

	var opts *C.OrtSessionOptions
	status = C.OrtApiCreateSessionOptions(api, &opts)
	if status != nil {
		C.OrtApiReleaseEnv(api, env)
		defer C.OrtApiReleaseStatus(api, status)
		return nil, fmt.Errorf("create session options failed: %s", C.GoString(C.OrtApiGetErrorMessage(api, status)))
	}
	defer C.OrtApiReleaseSessionOptions(api, opts)

	C.OrtApiSetIntraOpNumThreads(api, opts, 1)
	C.OrtApiSetInterOpNumThreads(api, opts, 1)

	var session *C.OrtSession
	cModelPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cModelPath))

	status = C.OrtApiCreateSession(api, env, cModelPath, opts, &session)
	if status != nil {
		C.OrtApiReleaseEnv(api, env)
		defer C.OrtApiReleaseStatus(api, status)
		return nil, fmt.Errorf("create session failed for model %q: %s", modelPath, C.GoString(C.OrtApiGetErrorMessage(api, status)))
	}

	var memInfo *C.OrtMemoryInfo
	status = C.OrtApiCreateCpuMemoryInfo(api, C.OrtArenaAllocator, C.OrtMemTypeDefault, &memInfo)
	if status != nil {
		C.OrtApiReleaseSession(api, session)
		C.OrtApiReleaseEnv(api, env)
		defer C.OrtApiReleaseStatus(api, status)
		return nil, fmt.Errorf("create memory info failed: %s", C.GoString(C.OrtApiGetErrorMessage(api, status)))
	}

	e := &vadEngine{
		api:         api,
		env:         env,
		session:     session,
		memoryInfo:  memInfo,
		sampleRate:  sampleRate,
		cInputName:  C.CString("input"),
		cStateName:  C.CString("state"),
		cSrName:     C.CString("sr"),
		cOutputName: C.CString("output"),
		cStateNName: C.CString("stateN"),
	}

	e.Reset()
	return e, nil
}

func (e *vadEngine) Reset() {
	for i := range e.state {
		e.state[i] = 0
	}
	for i := range e.ctx {
		e.ctx[i] = 0
	}
	e.currSample = 0
}

func (e *vadEngine) Infer(chunk []float32) (float32, error) {
	required := 512
	ctxLen := contextLen
	if e.sampleRate == 8000 {
		required = 256
		ctxLen = 32
	}
	if len(chunk) != required {
		return 0, fmt.Errorf("chunk size must be %d samples, got %d", required, len(chunk))
	}

	// Silero VAD v5 expects context prepended: [context + chunk]
	inputBuffer := make([]float32, ctxLen+len(chunk))
	copy(inputBuffer[:ctxLen], e.ctx[:ctxLen])
	copy(inputBuffer[ctxLen:], chunk)

	// Save the tail of this chunk as context for next chunk
	copy(e.ctx[:ctxLen], chunk[len(chunk)-ctxLen:])
	e.currSample += int64(len(chunk))

	// 1. Audio input tensor: shape [1, ctxLen + chunkSize]
	var pcmValue *C.OrtValue
	pcmDims := []C.int64_t{1, C.int64_t(len(inputBuffer))}
	status := C.OrtApiCreateTensorWithDataAsOrtValue(
		e.api, e.memoryInfo,
		unsafe.Pointer(&inputBuffer[0]),
		C.size_t(len(inputBuffer)*4),
		&pcmDims[0], C.size_t(len(pcmDims)),
		C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
		&pcmValue,
	)
	if status != nil {
		defer C.OrtApiReleaseStatus(e.api, status)
		return 0, fmt.Errorf("create pcm tensor failed: %s", C.GoString(C.OrtApiGetErrorMessage(e.api, status)))
	}
	defer C.OrtApiReleaseValue(e.api, pcmValue)

	// 2. State tensor: shape [2, 1, 128]
	var stateValue *C.OrtValue
	stateDims := []C.int64_t{2, 1, 128}
	status = C.OrtApiCreateTensorWithDataAsOrtValue(
		e.api, e.memoryInfo,
		unsafe.Pointer(&e.state[0]),
		C.size_t(stateLen*4),
		&stateDims[0], C.size_t(len(stateDims)),
		C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
		&stateValue,
	)
	if status != nil {
		defer C.OrtApiReleaseStatus(e.api, status)
		return 0, fmt.Errorf("create state tensor failed: %s", C.GoString(C.OrtApiGetErrorMessage(e.api, status)))
	}
	defer C.OrtApiReleaseValue(e.api, stateValue)

	// 3. Sample rate tensor: shape [1]
	var srValue *C.OrtValue
	srDims := []C.int64_t{1}
	srVal := []C.int64_t{C.int64_t(e.sampleRate)}
	status = C.OrtApiCreateTensorWithDataAsOrtValue(
		e.api, e.memoryInfo,
		unsafe.Pointer(&srVal[0]),
		C.size_t(8),
		&srDims[0], C.size_t(len(srDims)),
		C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
		&srValue,
	)
	if status != nil {
		defer C.OrtApiReleaseStatus(e.api, status)
		return 0, fmt.Errorf("create sr tensor failed: %s", C.GoString(C.OrtApiGetErrorMessage(e.api, status)))
	}
	defer C.OrtApiReleaseValue(e.api, srValue)

	// Run inference
	inputs := []*C.OrtValue{pcmValue, stateValue, srValue}
	outputs := []*C.OrtValue{nil, nil}

	inputNames := []*C.char{e.cInputName, e.cStateName, e.cSrName}
	outputNames := []*C.char{e.cOutputName, e.cStateNName}

	status = C.OrtApiRun(
		e.api, e.session, nil,
		&inputNames[0], &inputs[0], C.size_t(len(inputNames)),
		&outputNames[0], C.size_t(len(outputNames)),
		&outputs[0],
	)
	if status != nil {
		defer C.OrtApiReleaseStatus(e.api, status)
		return 0, fmt.Errorf("ort run failed: %s", C.GoString(C.OrtApiGetErrorMessage(e.api, status)))
	}

	var probData unsafe.Pointer
	status = C.OrtApiGetTensorMutableData(e.api, outputs[0], &probData)
	if status != nil {
		defer C.OrtApiReleaseStatus(e.api, status)
		C.OrtApiReleaseValue(e.api, outputs[0])
		C.OrtApiReleaseValue(e.api, outputs[1])
		return 0, fmt.Errorf("get prob tensor failed: %s", C.GoString(C.OrtApiGetErrorMessage(e.api, status)))
	}
	prob := *(*float32)(probData)

	var stateNData unsafe.Pointer
	status = C.OrtApiGetTensorMutableData(e.api, outputs[1], &stateNData)
	if status != nil {
		defer C.OrtApiReleaseStatus(e.api, status)
		C.OrtApiReleaseValue(e.api, outputs[0])
		C.OrtApiReleaseValue(e.api, outputs[1])
		return 0, fmt.Errorf("get stateN tensor failed: %s", C.GoString(C.OrtApiGetErrorMessage(e.api, status)))
	}

	// Update recurrent state for next chunk
	C.memcpy(unsafe.Pointer(&e.state[0]), stateNData, stateLen*4)

	C.OrtApiReleaseValue(e.api, outputs[0])
	C.OrtApiReleaseValue(e.api, outputs[1])

	return prob, nil
}

func (e *vadEngine) Close() error {
	if e.session != nil {
		C.OrtApiReleaseSession(e.api, e.session)
		e.session = nil
	}
	if e.memoryInfo != nil {
		C.OrtApiReleaseMemoryInfo(e.api, e.memoryInfo)
		e.memoryInfo = nil
	}
	if e.env != nil {
		C.OrtApiReleaseEnv(e.api, e.env)
		e.env = nil
	}

	C.free(unsafe.Pointer(e.cInputName))
	C.free(unsafe.Pointer(e.cStateName))
	C.free(unsafe.Pointer(e.cSrName))
	C.free(unsafe.Pointer(e.cOutputName))
	C.free(unsafe.Pointer(e.cStateNName))

	return nil
}
