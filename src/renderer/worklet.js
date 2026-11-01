// Mic capture worklet: brings whatever rate the context runs at down to 16 kHz and posts ~100 ms frames.
// Input 0 is the mic. Input 1, when connected, is the assistant's voice as it goes to the speaker: both are taken in the
// same render quantum, so the echo canceller (main/aec.js) gets a reference lined up with the mic sample for sample,
// whatever the playback queue does. Each output sample averages the input samples it covers (a little low-pass, so
// the step down to 16 kHz aliases less than picking every third sample).
class MicProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.acc = 0;
    this.sumMic = 0; this.sumRef = 0; this.count = 0;
    this.mic = new Float32Array(1600);
    this.ref = new Float32Array(1600);
    this.n = 0;
  }
  process(inputs) {
    const mic = inputs[0] && inputs[0][0];
    if (!mic) return true;
    const ref = inputs[1] && inputs[1][0]; // nothing playing (or not connected): silence
    for (let i = 0; i < mic.length; i++) {
      this.sumMic += mic[i]; this.sumRef += ref ? ref[i] : 0; this.count += 1;
      this.acc += 1;
      if (this.acc >= this.ratio) {
        this.acc -= this.ratio;
        this.mic[this.n] = this.sumMic / this.count;
        this.ref[this.n] = this.sumRef / this.count;
        this.sumMic = 0; this.sumRef = 0; this.count = 0;
        if (++this.n === this.mic.length) {
          this.port.postMessage({ mic: this.mic.slice(), ref: this.ref.slice() });
          this.n = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor('mic-processor', MicProcessor);
