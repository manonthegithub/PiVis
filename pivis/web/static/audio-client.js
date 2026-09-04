/**
 * Audio capture and streaming client for browser.
 * Captures audio via Web Audio API and streams to backend via WebSocket.
 */

class AudioClient {
  constructor(options = {}) {
    this.ws = null;
    this.audioContext = null;
    this.mediaStream = null;
    this.processor = null;
    this.isRecording = false;
    this.streamId = options.streamId || this.generateStreamId();
    // wss:// when the page itself is https:// — a plain ws:// call from an
    // https:// page is blocked by browsers as mixed content before it ever
    // reaches the network (this is what was breaking it behind the nginx
    // TLS terminator in front of pivis-aud.apps.arpa).
    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    this.serverUrl = options.serverUrl || `${wsProtocol}//${window.location.host}/ws/audio/${this.streamId}`;
    this.sampleRate = options.sampleRate || 16000;
    this.chunkDurationMs = options.chunkDurationMs || 100;
    this.callbacks = {
      onConnected: options.onConnected || (() => {}),
      onError: options.onError || ((e) => console.error(e)),
      onTranscription: options.onTranscription || (() => {}),
      onLLMResponse: options.onLLMResponse || (() => {}),
      onProcessing: options.onProcessing || (() => {}),
    };
  }

  generateStreamId() {
    return `browser-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  async initialize() {
    try {
      // Request microphone access
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: { ideal: this.sampleRate },
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: false,
        },
      });

      // Setup Web Audio
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: this.sampleRate,
      });

      const source = this.audioContext.createMediaStreamSource(this.mediaStream);

      // Create ScriptProcessor for raw audio capture
      this.processor = this.audioContext.createScriptProcessor(
        4096, // buffer size
        1,    // input channels
        1     // output channels
      );

      this.processor.onaudioprocess = (event) => this.handleAudioFrame(event);
      source.connect(this.processor);
      this.processor.connect(this.audioContext.destination);

      // Connect WebSocket
      await this.connectWebSocket();
      this.callbacks.onConnected();
    } catch (error) {
      this.callbacks.onError(`Initialization failed: ${error.message}`);
      throw error;
    }
  }

  connectWebSocket() {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.serverUrl);
        this.ws.binaryType = "arraybuffer";

        this.ws.onopen = () => {
          console.log("WebSocket connected");
          resolve();
        };

        this.ws.onmessage = (event) => this.handleServerMessage(event);
        this.ws.onerror = (error) => {
          this.callbacks.onError(`WebSocket error: ${error}`);
          reject(error);
        };
        this.ws.onclose = () => {
          console.log("WebSocket closed");
          this.isRecording = false;
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  handleAudioFrame(event) {
    if (!this.isRecording || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }

    const inputData = event.inputBuffer.getChannelData(0);
    const pcmData = this.floatTo16BitPCM(inputData);

    // Send frame as JSON with base64 encoding
    const frame = {
      type: "audio_chunk",
      timestamp: new Date().toISOString(),
      audio_base64: this.arrayBufferToBase64(pcmData),
      sample_rate: this.sampleRate,
      frame_id: `${this.streamId}-${Date.now()}`,
    };

    try {
      this.ws.send(JSON.stringify(frame));
    } catch (error) {
      this.callbacks.onError(`Failed to send frame: ${error.message}`);
    }
  }

  handleServerMessage(event) {
    try {
      const message = JSON.parse(event.data);

      if (message.type === "error") {
        this.callbacks.onError(message.message);
      } else if (message.type === "transcription") {
        this.callbacks.onTranscription(message);
      } else if (message.type === "llm_response") {
        this.callbacks.onLLMResponse(message);
      } else if (message.type === "processing") {
        // Server accepted a phrase and started transcribing -- this can
        // take tens of seconds on constrained hardware, so surface it
        // instead of leaving the UI looking frozen with no feedback.
        this.callbacks.onProcessing();
      }
    } catch (error) {
      console.error("Failed to parse server message:", error);
    }
  }

  floatTo16BitPCM(floatArray) {
    const pcmData = new Int16Array(floatArray.length);
    for (let i = 0; i < floatArray.length; i++) {
      // Clamp and convert to 16-bit
      const s = Math.max(-1, Math.min(1, floatArray[i]));
      pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return pcmData.buffer;
  }

  arrayBufferToBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    let binary = "";
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  }

  start() {
    if (!this.isRecording && this.audioContext) {
      this.isRecording = true;
      if (this.audioContext.state === "suspended") {
        this.audioContext.resume();
      }
      console.log("Recording started");
    }
  }

  stop() {
    this.isRecording = false;
    console.log("Recording stopped");
  }

  disconnect() {
    this.stop();

    if (this.processor) {
      this.processor.disconnect();
    }

    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => track.stop());
    }

    if (this.audioContext) {
      this.audioContext.close();
    }

    if (this.ws) {
      this.ws.close();
    }
  }
}

// Export for use in HTML
if (typeof module !== "undefined" && module.exports) {
  module.exports = AudioClient;
}
