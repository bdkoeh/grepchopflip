"""Separation depth (D-054 r2): which of the four separated stems stay apart and which are combined.

Pure: no audio, no files. Every other module asks this one for group names, class codes and the folder token.

Grammar (case-insensitive, spaces ignored):

    spec   := "all" | "none" | group ("," group)*
    group  := "rest" | member ("+" member)*
    member := vocals | drums | bass | other

- `all` (default): the four stems apart, as before separation-depth support. `none`: one group of the whole record (`mix`).
- `rest` = every member not named elsewhere, as one group. Members named nowhere form that group implicitly, so
  `vocals` means `vocals,rest`.
- Canonical form: members inside a group in the order vocals, drums, bass, other; groups ordered by their
  first member; `+` inside a group, `,` between groups. Four singletons are `all`; one group of all four is
  `none`. There are exactly 15 groupings.
- Group names: `inst` = drums+bass+other, `mix` = all four, a singleton keeps its member name, anything else
  is its members joined with `+` (`vocals+bass+other`).
- Class codes: Vox, Drum, Bass, Mel per member, concatenated in canonical order; `Inst` and `Mix` are the two
  special names. Loops add `Loop` (`InstLoop`, `VoxBassLoop`); phrases use the bare code (`Inst`, `VoxMel`).
- Folder token: `<name>-<hash8>` for `all`, else `<name>-<hash8>-split-<token>` with the member initials of
  each group joined by `-` (`-split-v-dbo`), or `-split-none`.

Errors raise SplitError (exit 2 at the CLI): unknown or repeated member, empty group, `rest` twice, `rest` that
matches nothing, `all`/`none` mixed with groups.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

MEMBERS = ("vocals", "drums", "bass", "other")       # the separator's four outputs, canonical order
INITIAL = {"vocals": "v", "drums": "d", "bass": "b", "other": "o"}
CODE = {"vocals": "Vox", "drums": "Drum", "bass": "Bass", "other": "Mel"}
SPECIAL_NAMES = {("drums", "bass", "other"): "inst", MEMBERS: "mix"}
SPECIAL_CODES = {("drums", "bass", "other"): "Inst", MEMBERS: "Mix"}

HELP = ("What gets separated: all (default: vocals, drums, bass, other apart), none (the record as one `mix` "
        "stem), or groups of vocals/drums/bass/other joined with + and separated by commas; `rest` (or anything "
        "left out) is one more group. Examples: vocals (= vocals,rest: vocals + inst), drums,rest, "
        "vocals+other,drums+bass. --split decides what exists; chop --stems picks among what exists. "
        "One split per song folder: other splits get their own folder (<name>-<hash8>-split-<initials>).")


class SplitError(ValueError):
    """A --split value that does not parse, or one that conflicts with what a song folder records (exit 2)."""


def _order(members) -> tuple[str, ...]:
    return tuple(sorted(members, key=MEMBERS.index))


def group_name(members) -> str:
    m = _order(members)
    if m in SPECIAL_NAMES:
        return SPECIAL_NAMES[m]
    return "+".join(m)


def members_of(name: str) -> tuple[str, ...]:
    """Inverse of group_name: `inst` -> (drums, bass, other); `vocals+bass` -> (vocals, bass). Raises SplitError."""
    for m, n in SPECIAL_NAMES.items():
        if name == n:
            return m
    parts = tuple(name.split("+"))
    if not parts or any(p not in MEMBERS for p in parts) or len(set(parts)) != len(parts) or _order(parts) != parts:
        raise SplitError(f"not a group name: {name!r}")
    if parts in SPECIAL_NAMES:      # e.g. "drums+bass+other" must be spelled "inst"
        raise SplitError(f"not a group name: {name!r} (use {SPECIAL_NAMES[parts]!r})")
    return parts


def is_group_name(name: str) -> bool:
    try:
        members_of(name)
        return True
    except SplitError:
        return False


def class_code(members, kind: str | None = None) -> str:
    """Vox/Drum/Bass/Mel joined (Inst, Mix special); `Loop` appended for loops."""
    m = _order(members)
    code = SPECIAL_CODES.get(m) or "".join(CODE[x] for x in m)
    return code + ("Loop" if kind == "loop" else "")


#: every combined group (2+ members) by name, in canonical order; 11 of them
COMBINED_GROUPS: tuple[str, ...] = tuple(
    group_name(c) for n in (2, 3, 4) for c in itertools.combinations(MEMBERS, n))
#: class codes of combined groups: loops and phrases (22)
COMBINED_CLASSES: tuple[str, ...] = tuple(
    class_code(members_of(g), k) for g in COMBINED_GROUPS for k in ("loop", "phrase"))
#: every possible stem name (4 singletons + 11 combined groups)
ALL_GROUP_NAMES: tuple[str, ...] = (*MEMBERS, *COMBINED_GROUPS)


@dataclass(frozen=True)
class Split:
    groups: tuple[tuple[str, ...], ...]       # canonical: members ordered, groups ordered by first member

    # ------------------------------------------------------------------ text forms
    @property
    def spec(self) -> str:
        if self.is_all:
            return "all"
        if self.is_none:
            return "none"
        return ",".join("+".join(g) for g in self.groups)

    def __str__(self) -> str:
        return self.spec

    @property
    def token(self) -> str:
        """Folder token: '' for all, 'none', else member initials per group (`v-dbo`)."""
        if self.is_all:
            return ""
        if self.is_none:
            return "none"
        return "-".join("".join(INITIAL[m] for m in g) for g in self.groups)

    @property
    def slug_suffix(self) -> str:
        return f"-split-{self.token}" if self.token else ""

    # ------------------------------------------------------------------ facts
    @property
    def is_all(self) -> bool:
        return len(self.groups) == 4

    @property
    def is_none(self) -> bool:
        return len(self.groups) == 1

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(group_name(g) for g in self.groups)

    def members(self, name: str) -> tuple[str, ...]:
        for g in self.groups:
            if group_name(g) == name:
                return g
        raise KeyError(name)

    def isolated(self, member: str) -> bool:
        """True when `member` is a group of its own (it gets its full pipeline: drum parts, bass notes, words...)."""
        return (member,) in self.groups

    @property
    def combined(self) -> tuple[str, ...]:
        """Names of groups with two or more members (they get loops and phrases only)."""
        return tuple(group_name(g) for g in self.groups if len(g) > 1)

    @property
    def drum_parts_hidden(self) -> bool:
        """Drums sit inside a combined group (or `none`): their parts are grid inputs under stems/.grid/ only."""
        return not self.isolated("drums")

    @property
    def method(self) -> str:
        """How the group stems were made: `source` (none: the original file) or `sum` (separated stems, a group
        of one being the stem itself)."""
        return "source" if self.is_none else "sum"

    def record(self) -> dict:
        """What source.json / stems/.done.json store."""
        return {"split": self.spec, "groups": {group_name(g): list(g) for g in self.groups}}

    def describe(self) -> dict:
        """The `separation` object of index.json."""
        return {
            "split": self.spec,
            "groups": [{"name": group_name(g), "members": list(g), "isolated": len(g) == 1} for g in self.groups],
            "method": self.method,
            "drum_grid": "hidden-parts" if self.drum_parts_hidden else "parts",
        }


def from_groups(groups) -> Split:
    gs = [_order(g) for g in groups if g]
    gs.sort(key=lambda g: MEMBERS.index(g[0]))
    return Split(tuple(gs))


ALL = from_groups([(m,) for m in MEMBERS])
NONE = from_groups([MEMBERS])


def parse_split(text: str | Split | None) -> Split:
    """`--split` text -> canonical Split. None or '' -> all. Raises SplitError."""
    if isinstance(text, Split):
        return text
    if text is None:
        return ALL
    if not isinstance(text, str):
        raise SplitError(f"expected text, got {text!r}")
    s = "".join(text.split()).lower()
    if s in ("", "all"):
        return ALL
    if s == "none":
        return NONE
    raw = s.split(",")
    if "all" in raw or "none" in raw:
        raise SplitError(f"{text!r}: `all` and `none` stand alone; they cannot be mixed with groups")
    groups: list[tuple[str, ...]] = []
    seen: set[str] = set()
    rest = 0
    for g in raw:
        if g == "":
            raise SplitError(f"{text!r}: empty group (two commas in a row, or a comma at an end)")
        if g == "rest":
            rest += 1
            if rest > 1:
                raise SplitError(f"{text!r}: `rest` given twice")
            continue
        members = g.split("+")
        if "" in members:
            raise SplitError(f"{text!r}: empty member in {g!r} (a `+` with nothing on one side)")
        for m in members:
            if m == "rest":
                raise SplitError(f"{text!r}: `rest` is a group of its own; it cannot be joined with +")
            if m not in MEMBERS:
                raise SplitError(f"{text!r}: unknown member {m!r} (use {', '.join(MEMBERS)}, rest, all or none)")
            if m in seen:
                raise SplitError(f"{text!r}: {m!r} listed twice")
            seen.add(m)
        groups.append(tuple(members))
    left = [m for m in MEMBERS if m not in seen]
    if rest and not left:
        raise SplitError(f"{text!r}: `rest` matches nothing (all four stems are already named)")
    if left:
        groups.append(tuple(left))
    return from_groups(groups)


def from_token(token: str) -> Split:
    """Inverse of Split.token: '' -> all, 'none', 'v-dbo' -> vocals,drums+bass+other. Raises SplitError."""
    if token == "":
        return ALL
    if token == "none":
        return NONE
    by_initial = {v: k for k, v in INITIAL.items()}
    groups = []
    for part in token.split("-"):
        if not part or any(c not in by_initial for c in part):
            raise SplitError(f"not a split token: {token!r}")
        groups.append(tuple(by_initial[c] for c in part))
    s = parse_split(",".join("+".join(g) for g in groups))
    if s.token != token:
        raise SplitError(f"not a canonical split token: {token!r} (canonical: {s.token or 'all'})")
    return s


def all_splits() -> list[Split]:
    """The 15 groupings (Bell number B4), `all` first, then `none`, then the rest by spec."""
    def partitions(items):
        if not items:
            yield []
            return
        first, rest = items[0], items[1:]
        for p in partitions(rest):
            for i in range(len(p)):
                yield p[:i] + [[first, *p[i]]] + p[i + 1:]
            yield [[first], *p]

    out = {from_groups(p) for p in partitions(list(MEMBERS))}
    return sorted(out, key=lambda s: (not s.is_all, not s.is_none, s.spec))


def from_record(rec: object) -> Split | None:
    """Split from a recorded `split` value (source.json / .done.json); None when absent or unreadable."""
    if not isinstance(rec, str):
        return None
    try:
        return parse_split(rec)
    except SplitError:
        return None
