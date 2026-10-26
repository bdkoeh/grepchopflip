const $ = (id) => document.getElementById(id);
let timerHandle = null;

function send(message) {
  return chrome.runtime.sendMessage({ target: "background", ...message });
}

function fmt(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function render(state) {
  $("idle").classList.toggle("hidden", state.status !== "idle");
  $("recording").classList.toggle("hidden", state.status !== "recording");
  $("stopped").classList.toggle("hidden", state.status !== "stopped");
  $("err").textContent = "";
  clearInterval(timerHandle);
  if (state.status === "recording") {
    const tick = () => { $("timer").textContent = fmt((Date.now() - state.startedAt) / 1000); };
    tick();
    timerHandle = setInterval(tick, 500);
    $("meta").textContent = state.tabTitle;
  } else if (state.status === "stopped") {
    $("meta").textContent = `${fmt(state.seconds || 0)} recorded · saves to Downloads/gcf_test_tracks`;
    if (!$("name").value) $("name").value = suggestName(state.tabTitle);
    $("name").focus();
    $("name").select();
  } else {
    $("meta").textContent = "Saves WAV to Downloads/gcf_test_tracks";
  }
}

function suggestName(title) {
  return (title || "")
    .replace(/\s*[-|–]\s*YouTube.*$/i, "")
    .replace(/[<>:"/\\|?*]/g, "")
    .trim()
    .slice(0, 60);
}

async function act(message) {
  const res = await send(message);
  if (!res?.ok) {
    $("err").textContent = res?.error || "Something went wrong";
    const { state } = await send({ type: "getState" });
    render(state);
    return null;
  }
  render(res.state);
  return res;
}

$("record").onclick = () => act({ type: "start" });
$("stop").onclick = () => act({ type: "stop" });
$("delete").onclick = () => { $("name").value = ""; act({ type: "delete" }); };
$("save").onclick = async () => {
  const name = $("name").value.trim() || "recording";
  const res = await act({ type: "save", name });
  if (res) { $("meta").textContent = `Saved ${res.filename}`; $("name").value = ""; }
};
$("name").addEventListener("keydown", (e) => { if (e.key === "Enter") $("save").click(); });

send({ type: "getState" }).then((res) => render(res?.state || { status: "idle" }));
