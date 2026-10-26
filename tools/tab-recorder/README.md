# GCF Tab Recorder (test helper, not part of the product)

Records the audio of the current Chrome tab to a 16-bit stereo WAV at the tab's sample rate
(usually 48 kHz) and saves it to `Downloads\gcf_test_tracks\`. The project's `test_tracks` folder
is a junction to that same folder, so `gcf` can be pointed at `test_tracks\<name>.wav`.

## Install (once)
1. Open `chrome://extensions`, turn on **Developer mode** (top right).
2. Click **Load unpacked** and pick this folder: `C:\Users\bob\projects\beat_making\tools\tab-recorder`.
3. Pin the extension from the puzzle-piece menu so the icon is visible.

## Use
1. Open the tab that plays the music and start playback (or pause it right before the part you want).
2. Click the extension icon, then **Record this tab**. The icon shows a red REC badge. The tab stays audible.
3. Click the icon again and **Stop** when done.
4. Type a name and press **Save** (or Enter), or press **Delete** to throw the take away.

Recording continues while the popup is closed; the badge tells you it's running. Switching tabs is fine;
it keeps recording the tab you started on. The file is written by Chrome's download system, so it may
flash in the download bar.

## Notes
- Recordings are for private testing only. Recording from a streaming site may breach that site's terms;
  never redistribute the files or ship them with the tool.
- Chrome extensions cannot write to arbitrary folders; the Downloads subfolder is the only option without
  a native helper. Change `FOLDER` in `background.js` to use another subfolder name.
- If recording fails with a permission error, click the icon once on the target tab first (the tab-capture
  API needs the extension to have been invoked on that tab).
