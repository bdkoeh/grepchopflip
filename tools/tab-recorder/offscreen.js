// Offscreen document: holds the captured audio in memory as raw PCM, keeps the tab audible,
// and encodes a 16-bit stereo WAV on request.

let audioContext = null;
let mediaStream = null;
let workletNode = null;
let chunks = []; // arrays of [left Float32Array, right Float32Array]
let sampleRate = 48000;
let startedAt = 0;
let lastUrl = null;

async function start(streamId) {
  await discard();
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      mandatory: { chromeMediaSource: "tab", chromeMediaSourceId: streamId },
    },
    video: false,
  });
  audioContext = new AudioContext();
  sampleRate = audioContext.sampleRate;
  await audioContext.audioWorklet.addModule("worklet.js");
  const source = audioContext.createMediaStreamSource(mediaStream);
  workletNode = new AudioWorkletNode(audioContext, "pcm-capture", {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [2],
  });
  workletNode.port.onmessage = (event) => {
    chunks.push(event.data);
  };
  source.connect(workletNode);
  workletNode.connect(audioContext.destination); // keep the tab audible while recording
  startedAt = Date.now();
}

async function stop() {
  if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop());
  if (workletNode) workletNode.disconnect();
  if (audioContext) await audioContext.close();
  mediaStream = null;
  workletNode = null;
  audioContext = null;
  const frames = chunks.reduce((n, c) => n + c[0].length, 0);
  return { seconds: frames / sampleRate };
}

function encodeWav() {
  const frames = chunks.reduce((n, c) => n + c[0].length, 0);
  const channels = 2;
  const bytesPerSample = 2;
  const dataBytes = frames * channels * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buffer);
  const writeStr = (offset, s) => { for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i)); };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channels * bytesPerSample, true);
  view.setUint16(32, channels * bytesPerSample, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, dataBytes, true);
  let offset = 44;
  for (const [left, right] of chunks) {
    for (let i = 0; i < left.length; i++) {
      const l = Math.max(-1, Math.min(1, left[i]));
      const r = Math.max(-1, Math.min(1, right[i]));
      view.setInt16(offset, l < 0 ? l * 0x8000 : l * 0x7fff, true);
      view.setInt16(offset + 2, r < 0 ? r * 0x8000 : r * 0x7fff, true);
      offset += 4;
    }
  }
  if (lastUrl) URL.revokeObjectURL(lastUrl);
  lastUrl = URL.createObjectURL(new Blob([buffer], { type: "audio/wav" }));
  return { url: lastUrl, seconds: frames / sampleRate, sampleRate };
}

async function discard() {
  if (mediaStream || audioContext) await stop();
  chunks = [];
  // Keep lastUrl alive briefly so an in-flight download can read it; revoke on next encode.
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.target !== "offscreen") return false;
  (async () => {
    try {
      switch (message.type) {
        case "start":
          await start(message.streamId);
          sendResponse({ ok: true });
          break;
        case "stop":
          sendResponse({ ok: true, ...(await stop()) });
          break;
        case "encode":
          sendResponse({ ok: true, ...encodeWav() });
          break;
        case "discard":
          await discard();
          sendResponse({ ok: true });
          break;
        default:
          sendResponse({ ok: false, error: `Unknown message ${message.type}` });
      }
    } catch (err) {
      sendResponse({ ok: false, error: err?.message || String(err) });
    }
  })();
  return true;
});
