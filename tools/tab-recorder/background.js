// Service worker: owns recorder state, starts tab capture, hands the stream id to the
// offscreen document (which does the actual audio work), and saves the final WAV.

const FOLDER = "gcf_test_tracks";
const OFFSCREEN_URL = "offscreen.html";

async function getState() {
  const { state } = await chrome.storage.session.get("state");
  return state || { status: "idle", startedAt: null, tabTitle: "" };
}

async function setState(state) {
  await chrome.storage.session.set({ state });
  await chrome.action.setBadgeText({ text: state.status === "recording" ? "REC" : "" });
  if (state.status === "recording") {
    await chrome.action.setBadgeBackgroundColor({ color: "#d33" });
  }
}

async function ensureOffscreen() {
  const contexts = await chrome.runtime.getContexts({
    contextTypes: ["OFFSCREEN_DOCUMENT"],
  });
  if (contexts.length > 0) return;
  await chrome.offscreen.createDocument({
    url: OFFSCREEN_URL,
    reasons: ["USER_MEDIA"],
    justification: "Capture tab audio and encode it to WAV",
  });
}

function sendToOffscreen(message) {
  return chrome.runtime.sendMessage({ target: "offscreen", ...message });
}

async function startRecording() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) throw new Error("No active tab");
  const streamId = await chrome.tabCapture.getMediaStreamId({ targetTabId: tab.id });
  await ensureOffscreen();
  await sendToOffscreen({ type: "start", streamId });
  await setState({ status: "recording", startedAt: Date.now(), tabTitle: tab.title || "" });
}

async function stopRecording() {
  const info = await sendToOffscreen({ type: "stop" });
  const prev = await getState();
  await setState({ ...prev, status: "stopped", seconds: info?.seconds ?? 0 });
}

function safeName(name) {
  const cleaned = name.replace(/[<>:"/\\|?*\x00-\x1f]/g, "_").replace(/\s+/g, "_").trim();
  return cleaned || "recording";
}

async function saveRecording(name) {
  const { url } = await sendToOffscreen({ type: "encode" });
  const filename = `${FOLDER}/${safeName(name)}.wav`;
  await chrome.downloads.download({ url, filename, conflictAction: "uniquify", saveAs: false });
  await sendToOffscreen({ type: "discard" });
  await setState({ status: "idle", startedAt: null, tabTitle: "" });
  return filename;
}

async function deleteRecording() {
  await sendToOffscreen({ type: "discard" });
  await setState({ status: "idle", startedAt: null, tabTitle: "" });
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.target !== "background") return false;
  (async () => {
    try {
      switch (message.type) {
        case "getState":
          sendResponse({ ok: true, state: await getState() });
          break;
        case "start":
          await startRecording();
          sendResponse({ ok: true, state: await getState() });
          break;
        case "stop":
          await stopRecording();
          sendResponse({ ok: true, state: await getState() });
          break;
        case "save":
          sendResponse({ ok: true, filename: await saveRecording(message.name), state: await getState() });
          break;
        case "delete":
          await deleteRecording();
          sendResponse({ ok: true, state: await getState() });
          break;
        default:
          sendResponse({ ok: false, error: `Unknown message ${message.type}` });
      }
    } catch (err) {
      await setState({ status: "idle", startedAt: null, tabTitle: "" }).catch(() => {});
      sendResponse({ ok: false, error: err?.message || String(err) });
    }
  })();
  return true;
});
