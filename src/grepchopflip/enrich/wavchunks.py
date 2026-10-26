"""In-house RIFF/WAVE `smpl` and `acid` chunk writer (D-006).

No maintained Python library writes these chunks, so this is a small byte-level editor: it parses the RIFF
chunk list, keeps every other chunk byte for byte (fmt, data, LIST/INFO, ...), drops any earlier `smpl` /
`acid`, appends the new ones at the end and fixes the RIFF size. The audio is never decoded or touched.

Layouts (little endian):
  smpl  36-byte header: manufacturer, product, sample_period_ns, midi_unity_note, midi_pitch_fraction,
        smpte_format, smpte_offset, num_sample_loops, sampler_data; then 24 bytes per loop: cue_point_id,
        type (0 = forward), start, end (inclusive, in sample frames), fraction, play_count (0 = infinite).
  acid  24 bytes: flags (0x01 one-shot, 0x02 root note set, 0x04 stretch, 0x08 disk based, 0x10 acidizer),
        root_note uint16, uint16 0x8000, float32 0, num_beats uint32, meter denominator uint16,
        meter numerator uint16, tempo float32.
"""

from __future__ import annotations

import hashlib
import os
import struct
from dataclasses import dataclass
from pathlib import Path

SMPL = b"smpl"
ACID = b"acid"

ACID_ONE_SHOT = 0x01
ACID_ROOT_SET = 0x02
ACID_STRETCH = 0x04


class WavChunkError(ValueError):
    """Not a RIFF/WAVE file, or its chunk list is damaged."""


@dataclass(frozen=True)
class Chunk:
    cid: bytes
    offset: int      # of the 8-byte chunk header
    size: int        # payload size as declared (without the pad byte)

    @property
    def end(self) -> int:  # offset after the payload and its pad byte
        return self.offset + 8 + self.size + (self.size & 1)


def parse(blob: bytes) -> list[Chunk]:
    if len(blob) < 12 or blob[:4] != b"RIFF" or blob[8:12] != b"WAVE":
        raise WavChunkError("not a RIFF/WAVE file")
    chunks: list[Chunk] = []
    pos = 12
    while pos + 8 <= len(blob):
        cid = blob[pos:pos + 4]
        (size,) = struct.unpack_from("<I", blob, pos + 4)
        if pos + 8 + size > len(blob):
            raise WavChunkError(f"chunk {cid!r} at {pos} runs past the end of the file")
        chunks.append(Chunk(cid, pos, size))
        pos += 8 + size + (size & 1)
    if not any(c.cid == b"data" for c in chunks) or not any(c.cid == b"fmt " for c in chunks):
        raise WavChunkError("missing fmt or data chunk")
    return chunks


def read_chunks(path: Path) -> dict[bytes, bytes]:
    """Payloads of every chunk (the last one wins for a repeated id)."""
    blob = Path(path).read_bytes()
    return {c.cid: blob[c.offset + 8:c.offset + 8 + c.size] for c in parse(blob)}


def audio_digest(path: Path) -> str:
    """sha256 of the fmt and data payloads: identifies the audio whatever metadata chunks the file carries."""
    blob = Path(path).read_bytes()
    h = hashlib.sha256()
    for c in parse(blob):
        if c.cid in (b"fmt ", b"data"):
            h.update(c.cid)
            h.update(blob[c.offset + 8:c.offset + 8 + c.size])
    return h.hexdigest()


def build_smpl(sample_rate: int, unity_note: int, loop: tuple[int, int] | None = None,
               pitch_fraction: int = 0) -> bytes:
    period_ns = int(round(1e9 / sample_rate)) if sample_rate else 0
    loops = [loop] if loop else []
    out = struct.pack("<9I", 0, 0, period_ns, int(unity_note) & 0x7F, pitch_fraction, 0, 0, len(loops), 0)
    for i, (start, end) in enumerate(loops):
        out += struct.pack("<6I", i, 0, int(start), int(end), 0, 0)
    return out


def parse_smpl(data: bytes) -> dict:
    f = struct.unpack_from("<9I", data, 0)
    loops = []
    for i in range(f[7]):
        if 36 + 24 * (i + 1) > len(data):
            break
        cue, typ, start, end, frac, count = struct.unpack_from("<6I", data, 36 + 24 * i)
        loops.append({"type": typ, "start": start, "end": end, "play_count": count})
    return {"sample_period_ns": f[2], "unity_note": f[3], "pitch_fraction": f[4], "loops": loops}


def build_acid(*, one_shot: bool, root_note: int | None, num_beats: int, tempo: float,
               meter: tuple[int, int] = (4, 4)) -> bytes:
    flags = (ACID_ONE_SHOT if one_shot else 0) | (ACID_ROOT_SET if root_note is not None else 0)
    if not one_shot and tempo > 0:
        flags |= ACID_STRETCH
    num, den = meter
    return struct.pack("<IHHfIHHf", flags, int(root_note if root_note is not None else 60), 0x8000, 0.0,
                       int(num_beats), int(den), int(num), float(tempo))


def parse_acid(data: bytes) -> dict:
    flags, root, _u1, _u2, beats, den, num, tempo = struct.unpack_from("<IHHfIHHf", data, 0)
    return {"one_shot": bool(flags & ACID_ONE_SHOT), "root_note": root if flags & ACID_ROOT_SET else None,
            "stretch": bool(flags & ACID_STRETCH), "num_beats": beats, "meter": [num, den],
            "tempo": round(tempo, 4)}


def write_chunks(path: Path, new: dict[bytes, bytes]) -> bool:
    """Replace or append the given chunks; every other byte stays as it was. False when the file already
    carries exactly these chunks at the end (nothing written). The rewrite is atomic (tmp file + rename)."""
    path = Path(path)
    blob = path.read_bytes()
    chunks = parse(blob)
    body = bytearray()
    for c in chunks:
        if c.cid not in new:
            body += blob[c.offset:c.end]
    # a last chunk without its pad byte is legal in the wild; keep what is there, pad what we add
    if len(body) % 2:
        body += b"\x00"
    for cid, payload in new.items():
        body += cid + struct.pack("<I", len(payload)) + payload
        if len(payload) % 2:
            body += b"\x00"
    out = b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + bytes(body)
    if out == blob:
        return False
    tmp = path.with_name(path.name + ".gcf-chunks")
    with open(tmp, "wb") as fh:
        fh.write(out)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return True
