// AudioWorklet: copies incoming PCM to the main thread and passes audio through unchanged.
class PcmCapture extends AudioWorkletProcessor {
  process(inputs, outputs) {
    const input = inputs[0];
    const output = outputs[0];
    if (input.length === 0) return true;
    const left = input[0];
    const right = input[1] || input[0];
    this.port.postMessage([Float32Array.from(left), Float32Array.from(right)]);
    if (output[0]) output[0].set(left);
    if (output[1]) output[1].set(right);
    return true;
  }
}
registerProcessor("pcm-capture", PcmCapture);
