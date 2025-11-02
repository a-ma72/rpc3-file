"""RPC3 file class."""

# ruff: noqa: ANN401 B028 C901 E501 N806 PLR0911 PLR0912 PLR0913 PLR0915 PLR2004 S101

from __future__ import annotations

import datetime
import mmap
import os
import re
import warnings
from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Self, TypeAlias

from numpy import (
    append,
    arange,
    asarray,
    char,
    dtype,
    empty,
    finfo,
    frombuffer,
    fromfile,
    isclose,
    nan,
    ndarray,
    ones,
)
from tqdm import tqdm

if TYPE_CHECKING:
    from typing import BinaryIO

    from numpy.typing import NDArray

__all__ = [
    "Channel",
    "ChannelDict",
    "ChannelList",
    "FileFormatError",
    "OrderedDict",
    "__author__",
    "__version__",
    "find_channel",
    "read",
    "to_dict",
    "write",
]


_INT_FULL_SCALE: int = 2**15 - 16  # Specific scale (don't change!)
_INT16: dtype = dtype("<i2")
_FLOAT32: dtype = dtype("<f4")
progressbar: bool = True

__version__ = "1.0.0rc6"
__author__ = "Andreas Martin"


class FileFormatError(OSError):
    """Exception class indicating file format errors."""


@dataclass
class Channel:
    """Channel dataclass.

    Describing one channel.

    Parameters
    ----------
    name : str
        Channel name.
    unit : str
        Channel unit.
    dt : float
        Unique timestep between samples in `data` in [s].
    data : Sequence, optional
        The data vector (1d). If ``None``, an empty float32 array is used.
    minval : float, optional
        Smallest value in `data`.
        (Set on read and ignored on write.)
    maxval : float, optional
        Largest value in `data`.
        (Set on read and ignored on write.)
    resolution : float, optional
        Resolution of `data`.
        (Set on read and ignored on write.)
        For INTEGER RPC3 files, `resolution` is an absolute value (bit resolution),
        for FLOATING POINT RPC3 files, `resolution` is a relative value.
    x0 : float, optional
        Start time in [s], default 0.
    unitx : str, optional
        Unit of time axis, always ``'s'`` (set internally).

    """

    name: str
    unit: str
    dt: float
    data: Sequence | None
    minval: float = nan
    maxval: float = nan
    resolution: float = nan
    x0: float = 0.0
    unitx: str = field(default="s", init=False)

    def __post_init__(self) -> None:
        """Ensure `data` is a flat NumPy array (or empty array if None)."""
        if self.data is None:
            # Use an empty array; dtype can be adjusted to your needs.
            # (Kept generic here to avoid referencing project-specific constants.)
            self.data = empty(0)
        else:
            self.data = asarray(self.data).flatten()

    def time(self) -> NDArray:
        """Create the time vector for this channel."""
        return arange(len(self.data)) * self.dt + self.x0


# ---- Type aliases ------------------------------------------------------------

ChannelList: TypeAlias = list[Channel]
ChannelDict: TypeAlias = Mapping[str, Channel | ChannelList]
ParameterList: TypeAlias = Mapping[str, str | float | int]


def to_dict(
    channels: ChannelList | ChannelDict | None,
) -> ChannelDict | None:
    """Transform a channel collection into a dictionary keyed by channel names.

    Groups `Channel` objects by their ``name`` attribute while preserving the
    encounter order of the first occurrence of each name. If a name appears
    once, the value is the single `Channel`. If a name appears multiple times,
    the value is a `list[Channel]` in encounter order. A single warning is
    issued per ambiguous name.

    Parameters
    ----------
    channels : list[Channel] or OrderedDict[str, Channel | list[Channel]] or None
        The channels to transform.
        - If a mapping is provided, it is returned as an ``OrderedDict`` copy.
        - If ``None``, ``None`` is returned.

    Returns
    -------
    OrderedDict[str, Channel | list[Channel]] or None
        Channels grouped by name (or ``None`` if input was ``None``).

    Notes
    -----
    - In Python >=3.7, ``dict`` preserves insertion order, but this function
      returns an ``OrderedDict`` to match the declared `ChannelDict` alias.
    - Exactly one warning is emitted per ambiguous name (name with >1 channel).

    Examples
    --------
    >>> a = Channel(name="left",  unit="V", dt=0.1, data=[1,2,3])
    >>> b = Channel(name="right", unit="V", dt=0.1, data=[4,5])
    >>> c = Channel(name="left",  unit="V", dt=0.1, data=[6])
    >>> out = to_dict([a, b, c])
    >>> isinstance(out["left"], list) and out["left"][0] is a and out["left"][1] is c
    True
    >>> out["right"] is b
    True

    """
    # Fast path: nothing to do if input is None.
    if channels is None:
        return None

    # If a mapping is provided, assume caller already shaped it as desired.
    # Normalize to OrderedDict for a consistent return type.
    if isinstance(channels, Mapping):
        return OrderedDict(channels)

    # First pass: group by channel name; dict preserves first-seen order of names.
    groups: dict[str, list[Channel]] = defaultdict(list)
    for ch in channels:
        # Assumes each item is a `Channel` instance with `.name`
        groups[ch.name].append(ch)

    # Second pass: collapse singletons and warn once for ambiguous names.
    out: ChannelDict = OrderedDict()
    for name, items in groups.items():
        if len(items) == 1:
            out[name] = items[0]
        else:
            msg = f"Channel name is ambiguous: `{name}`"
            warnings.warn(msg)
            out[name] = items

    return out


def find_channel(
    channels: ChannelList | ChannelDict | None,
    *,
    name: str | None = None,
    unit: str | None = None,
    regex: bool = False,
    match: Literal["any", "all"] = "any",
    assert_once: bool = False,
    as_dict: bool = False,
) -> Channel | ChannelDict | ChannelList | None:
    r"""Find channels by name and/or unit with optional regex and selection semantics.

    Parameters
    ----------
    channels : Sequence[Channel] | Mapping[str, Channel] | None
        Container of channels. For mappings, values are channels and keys are ignored
        in the result. For sequences, elements are channels.
    name : str | None, optional
        Name pattern to match against `ch.name`. If `None`, the name criterion is
        ignored. When `regex=True`, interpreted as a regular expression and matched
        with ``fullmatch``.
    unit : str | None, optional
        Unit pattern to match against `ch.unit`. If `None`, the unit criterion is
        ignored. When `regex=True`, interpreted as a regular expression and matched
        with ``fullmatch``.
    regex : bool, default False
        If True, compile `name`/`unit` as regular expressions and apply full-match.
        If False, use exact string equality.
    match : "all" or "any", default "any"
        If "all", require both criteria to match (AND). If "any", at least one must
        match (OR).
    assert_once : bool, default False
        If True, enforce that there is at most one match:
        - Return ``None`` if there are zero matches.
        - Return the single match if there is exactly one match.
        - Raise ``AssertionError`` if more than one match is found.
        This mode short-circuits as soon as a second match is detected.
    as_dict : bool, default False
        If True, return matches as a ``Channel`` or
        ``dict[str, Channel | list[Channel]]`` keyed by ``ch.name``.
        If False, return a ``Channel`` or ``list[Channel]``.

    Returns
    -------
    dict[str, Channel | list[Channel]] | list[Channel] | Channel | None

    Raises
    ------
    AssertionError
        If `assert_once=True` and more than one match is found.
    TypeError
        If `channels` is not a mapping or a (non-string) sequence.

    Notes
    -----
    - Regex patterns, when enabled, use ``re.fullmatch`` semantics.
    - Dict outputs are always keyed by ``ch.name`` regardless of the input container.

    Examples
    --------
    >>> # Sequence input, exact match on name
    >>> find_channel(seq, name="voltage")
    [Channel(name='voltage', unit='V'), ...]

    >>> # Sequence input, AND condition on regex name and unit, single match
    >>> find_channel(seq, name=r"temp_\\d+", unit=r"K|°C", regex=True, match="all", assert_once=True)
    Channel(name='temp_1', unit='K')

    >>> # Mapping input, OR condition, dict result keyed by channel name
    >>> find_channel(mapping, name="current", unit="A")
    {'current': Channel(name='current', unit='A')}

    """

    name_is_set = name is not None
    unit_is_set = unit is not None

    def make_matcher(pattern: str | None) -> Callable[[str], bool]:
        if pattern is None:
            # We'll gate on *_is_set in pred(), so this value won't be used.
            return lambda _v: False
        if regex:
            rx = re.compile(pattern)
            return lambda v: rx.fullmatch(v) is not None
        return lambda v: v == pattern

    name_match = make_matcher(name)
    unit_match = make_matcher(unit)

    def pred(ch: Channel) -> bool:
        # Evaluate only for criteria that are actually set
        a = name_match(ch.name) if name_is_set else False
        b = unit_match(ch.unit) if unit_is_set else False

        if not name_is_set and not unit_is_set:
            # No filters given -> return everything (matches docstring intuition)
            return True

        if match == "all":
            # AND across only the set criteria
            if name_is_set and unit_is_set:
                return a and b
            return a if name_is_set else b
        # OR across only the set criteria
        return a or b

    # Mapping branch: preserve only values; result keys are channel names.
    if isinstance(channels, Mapping):
        def iter_channels() -> Channel:
            for v in channels.values():
                if isinstance(v, Channel):
                    yield v
                else:
                    # assume iterable of Channel
                    yield from v
    # Sequence branch (exclude str/bytes to avoid accidental iteration over characters).
    elif isinstance(channels, Sequence) and not isinstance(channels, (str, bytes)):
        def iter_channels() -> Channel:
            yield from channels
    else:
        # Unsupported container type.
        msg = "channels must be Mapping[str, Channel] or Sequence[Channel]."
        raise TypeError(msg)

    if assert_once:
        # Generator to short-circuit on second match.
        it = (ch for ch in iter_channels() if pred(ch))
        try:
            ch1 = next(it)
        except StopIteration:
            return None
        try:
            next(it)  # If this succeeds, there is a second match.
        except StopIteration:
            return ch1
        msg = "More than one match."
        raise AssertionError(msg)

    # Gather all matches.
    hits = [ch for ch in iter_channels() if pred(ch)]
    return to_dict(hits) if as_dict else hits


class BufferedFileReader:
    _file: Path
    _file_mapping: bool
    _f: BinaryIO | None = None
    _m: mmap.mmap | None = None
    _data: ndarray | None = None

    def __init__(self, filepath: str, *, file_mapping: bool = False) -> None:
        self._file = Path(filepath)
        self._file_mapping = file_mapping

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()

    def open(self) -> None:
        if self._f:
            return
        self._f = self._file.open("rb")
        if self._file_mapping:
            self._m = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
            self._data = frombuffer(self._m, dtype="u1")
            if self._data is None:
                msg = "Memory map not available for file."
                raise FileFormatError(msg)


    @property
    def file_mapping(self) -> bool:
        return self._file_mapping

    @property
    def file_size(self) -> int:
        return self._file.stat().st_size

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._f.seek(offset, whence)

    def read(self, count: int) -> bytes:
        return self._f.read(count)

    def fromfile(self, count: int, dt: dtype) -> NDArray:
        return fromfile(self._f, dt, count)

    def data_view(self, dt: dtype, offset: int = 0) -> NDArray | None:
        return self._data[offset:].view(dt) if self._data is not None else None

    def close(self) -> None:
        self._data = None
        if self._m is not None:
            try:
                self._m.close()
            finally:
                self._m = None
        if self._f is not None:
            try:
                self._f.close()
            finally:
                self._f = None


class _RPC3Reader:
    _file: Path
    _buffered_io: BufferedFileReader | None
    _params: ParameterList
    _header_size: int
    # header parameters
    _num_header_blocks: int
    _num_params: int
    _num_channels: int | None
    _num_samples: int | None
    _pts_per_group: int | None
    _pts_per_frame: int | None
    _dtype: dtype | None
    _dt: float
    # calculated
    _num_groups: int | None
    _num_frames: int | None
    _frames_per_group: int | None
    _pts_per_channels_group: int | None
    _sample_pts: int | None
    _channels: ChannelList | None
    _frame_size_bytes: int

    def __init__(self, filepath: str, *, file_mapping: bool = False) -> None:
        self._file = Path(filepath)
        self._buffered_io = BufferedFileReader(filepath, file_mapping=file_mapping)
        self.reset_params()

    def reset_params(self) -> None:
        self._params = {"NUM_HEADER_BLOCKS": 1, "NUM_PARAMS": 3}
        self._num_header_blocks = 1
        self._num_params = 3
        self._header_size = 4 * 512
        self._num_channels = None
        self._num_samples = None
        self._num_frames = None
        self._num_groups = None
        self._pts_per_group = None
        self._pts_per_frame = None
        self._pts_per_channels_group = None
        self._sample_pts = None
        self._channels = None

    @property
    def exists(self) -> bool:
        return self._file.exists()

    def open(self) -> None:
        self._buffered_io.open()

    def close(self) -> None:
        self._buffered_io.close()
        self.reset_params()

    @staticmethod
    def param_to_number(
        key: str,
        value: str | bytes,
        *,
        ignore_error: bool = False,
    ) -> int | float | str:
        key, value = _RPC3Reader.decode_key_value(key, value)
        if key.replace("_", ".").split(".")[0] not in (
            "OPERATION",
            "PARENT",
            "DESC",
            "UNIT",
        ):
            try:
                return float(value) if "." in value else int(value)
            except Exception as _ex:
                if not ignore_error:
                    raise
        return value

    @staticmethod
    def decode_key_value(key: str | bytes, value: Any) -> tuple[str, Any]:
        if isinstance(key, bytes):
            key = key.decode("latin-1")
        if isinstance(value, bytes):
            value = value.decode("latin-1")
        return key, value

    def check_format(self, key: str, value: Any) -> None:
        key, value = self.decode_key_value(key, value)
        if key != "FORMAT":
            msg = "First header key must be `FORMAT`."
            raise FileFormatError(msg)
        # only Intel little endian format is supported
        if value not in ("BINARY_IEEE_LITTLE_END", "BINARY"):
            msg = "Only `BINARY_IEEE_LITTLE_END` and `BINARY` number formats supported."
            raise FileFormatError(msg)

    def check_num_header_blocks(self, key: str, value: Any) -> None:
        key, value = self.decode_key_value(key, value)
        # support large header (MTS(R) documentation "RPC3 file formats" specifies only max. 256 blocks)
        if key != "NUM_HEADER_BLOCKS":
            msg = "Second header key must be `NUM_HEADER_BLOCKS`."
            raise FileFormatError(msg)
        try:
            value = self.param_to_number(key, value)
        except Exception as ex:
            msg = f"Format error for `NUM_HEADER_BLOCKS`={value}."
            raise FileFormatError(msg) from ex
        if not (0 < value <= 256 * 4):
            msg = f"Too many header blocks (`NUM_HEADER_BLOCKS`={int(value)})."
            raise FileFormatError(msg)
        self._num_header_blocks = int(value)
        self._header_size = int(value) * 512

    def check_num_params(self, key: str, value: Any) -> None:
        key, value = self.decode_key_value(key, value)
        # support large header
        # (MTS(R) documentation "RPC3 file formats" specifies only max. 256 blocks)
        if key != "NUM_PARAMS":
            msg = "Third header key must be `NUM_PARAMS`."
            raise FileFormatError(msg)
        try:
            value = self.param_to_number(key, value)
        except Exception as ex:
            msg = f"Format error for `NUM_PARAMS`={value}."
            raise FileFormatError(msg) from ex
        if not (3 < value <= self._num_header_blocks * 4):
            msg = f"Wrong number of parameters (`NUM_PARAMS`={int(value)})."
            raise FileFormatError(msg)
        self._num_params = int(value)

    def check_params(self) -> None:
        if len(self._params) != self._num_params:
            msg = "Wrong number of parameters in file header."
            raise FileFormatError(msg)

        if "FILE_TYPE" in self._params and self._params["FILE_TYPE"] != "TIME_HISTORY":
            msg = "Only data file type `TIME_HISTORY` is supported."
            raise FileFormatError(msg)

        if "TIME_TYPE" in self._params:
            if isinstance(self._params["TIME_TYPE"], int):
                if self._params["TIME_TYPE"] not in [1, 2, 3, 4]:
                    msg = (
                        "Only time types 1..4 supported "
                        f"(`TIME_TYPE`={int(self._params['TIME_TYPE'])})."
                    )
                    raise FileFormatError(msg)
            elif self._params["TIME_TYPE"] not in [
                "DRIVE",
                "RESPONSE",
                "MULT_DRIVE",
                "MULT_RESP",
            ]:
                msg = (
                    "Only `DRIVE`, `RESPONSE`, `MULT_DRIVE` and `MULT_RESP` supported "
                    f"(`TIME_TYPE`={self._params['TIME_TYPE']!s})"
                )
                raise FileFormatError(msg)

        self._dt = self._params.get("DELTA_T")
        if not self._dt or self._dt < 0:
            msg = "Missing valid samplerate parameter `DELTA_T`."
            raise FileFormatError(msg)

        self._num_channels = int(self._params.get("CHANNELS", -1))
        if self._num_channels < 0:
            msg = "Missing or wrong number of channels."
            raise FileFormatError(msg)

        self._num_frames = int(self._params["FRAMES"])
        self._num_samples = self._params.get("SAMPLES")
        self._pts_per_group = int(self._params["PTS_PER_GROUP"])
        self._pts_per_frame = int(self._params["PTS_PER_FRAME"])

        if self._params.get("DATA_TYPE", "SHORT_INTEGER") == "SHORT_INTEGER":
            self._dtype = _INT16
        elif self._params["DATA_TYPE"] == "FLOATING_POINT":
            self._dtype = _FLOAT32
        else:
            msg = (
                "Only `SHORT_INTEGER` and `FLOATING_POINT` supported "
                f"({self._params['DATA_TYPE']})"
            )
            raise FileFormatError(msg)

        if self._pts_per_frame <= 0 or self._pts_per_group % self._pts_per_frame != 0:
            msg = "PTS_PER_GROUP must be a multiple of PTS_PER_FRAME and > 0."
            raise FileFormatError(msg)

        self._frames_per_group = self._pts_per_group // self._pts_per_frame
        if self._frames_per_group <= 0:
            msg = "Computed FRAMES_PER_GROUP <= 0."
            raise FileFormatError(msg)

        if self._num_frames % self._frames_per_group != 0:
            msg = "Partially filled last group."
            warnings.warn(msg)

        self._num_groups = (
            self._num_frames + self._frames_per_group - 1
        ) // self._frames_per_group

        self._sample_pts = self._num_frames * self._pts_per_frame
        if self._num_samples is not None and not (
            0 <= self._num_samples <= self._sample_pts
        ):
            msg = "SAMPLES must be in [0, FRAMES*PTS_PER_FRAME]."
            raise FileFormatError(msg)

        self._frame_size_bytes = (
            self._pts_per_frame * self._num_channels * self._dtype.itemsize
        )
        self._pts_per_channels_group = self._num_channels * self._pts_per_group

        expected = (
            self._header_size
            + self._num_groups
            * self._pts_per_group
            * self._num_channels
            * self._dtype.itemsize
        )
        actual = self._buffered_io.file_size
        if actual < expected:
            msg = "File truncated: header expects more data than present."
            raise FileFormatError(msg)
        if actual > expected:
            msg = f"File contains {actual - expected} extra bytes after expected data region."
            warnings.warn(msg)

    def prepare_channels(self, *, header_only: bool = False) -> None:
        channels = []
        for i in range(1, self._num_channels + 1):
            ch = Channel(
                name=self._params[f"DESC.CHAN_{i}"],
                unit=self._params[f"UNITS.CHAN_{i}"],
                resolution=self._params[f"SCALE.CHAN_{i}"],
                minval=self._params[f"LOWER_LIMIT.CHAN_{i}"],
                maxval=self._params[f"UPPER_LIMIT.CHAN_{i}"],
                dt=self._dt,
                data=empty(0 if header_only else self._sample_pts, self._dtype),
            )
            channels.append(ch)
        self._channels = channels

    def read_next_param(self) -> tuple[str, str]:
        """Return next header parameter at read position.

        Value is type casted to float or int if possible.

        Returns
        -------
        tuple[str, Any]
            Parameter as key/value pair.

        """
        key = self._buffered_io.read(32).decode("latin-1").strip(" \0")
        value = self._buffered_io.read(96).decode("latin-1").strip(" \0")
        return key, value

    def read_header(self) -> None:
        # First 3 entries are specified:
        # Pos. 1: "FORMAT" is one of "BINARY_IEEE_LITTLE_END", "BINARY_IEEE_BIG_END",
        #         "BINARY" or "ASCII"
        # Pos. 2: "NUM_HEADER_BLOCKS", number of 512-byte blocks (max. 256 blocks)
        # Pos. 3: "NUM_PARAMS", number of parameters in header (max. 1024 parameters)
        self._buffered_io.seek(0, os.SEEK_SET)
        i = 0
        while i < self._num_params:
            key, value = self.read_next_param()
            if i == 0:
                self.check_format(key, value)
            elif i == 1:
                self.check_num_header_blocks(key, value)
            elif i == 2:
                self.check_num_params(key, value)
            if len(key) > 0:
                self._params[key] = self.param_to_number(key, value, ignore_error=True)
            i += 1

    def read_header_mapped(self) -> None:
        dt = [("key", "S32"), ("value", "S96")]
        decode = {"encoding": "latin-1", "errors": "strict"}

        head = self._buffered_io.data_view(dt=dt)
        if head is None:
            msg = "Memory map not available for mapped header read."
            raise FileFormatError(msg)

        self.check_format(head["key"][0], head["value"][0])
        self.check_num_header_blocks(head["key"][1], head["value"][1])
        self.check_num_params(head["key"][2], head["value"][2])

        keys = head["key"][: self._num_params]
        keys = char.rstrip(keys, b"\x00 ")
        keys = char.decode(keys, **decode)
        vals = head["value"][: self._num_params]
        vals = char.rstrip(vals, b"\x00 ")
        vals = char.decode(vals, **decode)

        params = dict(zip(keys, vals))

        for key, value in params.items():
            self._params[key] = self.param_to_number(key, value, ignore_error=True)

    def read_data(self, batch_size: int) -> None:
        # Read multiplexed channels
        self._buffered_io.seek(self._header_size, os.SEEK_SET)
        start = 0
        frames_left = self._num_frames
        with tqdm(
            total=self._num_frames * self._frame_size_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            disable=not progressbar,
        ) as pbar:
            while frames_left > 0:
                frames_to_read = min(frames_left, self._frames_per_group * batch_size)
                groups_to_read = (
                    frames_to_read + self._frames_per_group - 1
                ) // self._frames_per_group
                pts = frames_to_read * self._pts_per_frame
                buffer = (
                    self._buffered_io.fromfile(
                        groups_to_read * self._pts_per_channels_group,
                        self._dtype,
                    )
                    .reshape(-1, self._num_channels, self._pts_per_group)
                    .swapaxes(0, 1)
                    .reshape(self._num_channels, -1)
                )
                for i in range(self._num_channels):
                    self._channels[i].data[start : start + pts] = buffer[i, :pts]
                start += pts
                frames_left -= frames_to_read
                pbar.update(frames_to_read * self._frame_size_bytes)

    def read_data_mapped(self, batch_size: int) -> None:
        # Read multiplexed channels
        dv = self._buffered_io.data_view(self._dtype, offset=self._header_size)
        if dv is None:
            msg = "Memory map not available for mapped data read."
            raise FileFormatError(msg)
        data = dv.reshape(-1, self._num_channels, self._pts_per_group)

        start = 0
        frames_left = self._num_frames
        group_first = 0
        with tqdm(
            total=self._num_frames * self._frame_size_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            disable=not progressbar,
        ) as pbar:
            while frames_left > 0:
                frames_to_read = min(frames_left, self._frames_per_group * batch_size)
                groups_to_read = (
                    frames_to_read + self._frames_per_group - 1
                ) // self._frames_per_group
                group_next = group_first + groups_to_read
                pts = frames_to_read * self._pts_per_frame
                buffer = (
                    data[group_first:group_next]  # .copy()
                    .swapaxes(0, 1)
                    .reshape(self._num_channels, -1)
                )
                for i in range(self._num_channels):
                    self._channels[i].data[start : start + pts] = buffer[i, :pts]
                start += pts
                group_first = group_next
                frames_left -= frames_to_read
                pbar.update(frames_to_read * self._frame_size_bytes)

    def post_process(self, *, strip: bool) -> None:
        # Truncate
        num_samples = self._sample_pts
        if self._num_samples is not None:
            num_samples = self._num_samples
        elif strip:
            # Remove gap from last group
            max_tail = max(0, self._pts_per_group - 1)
            tail = max_tail
            for ch in self._channels:
                a = ch.data
                if a.size == 0 or max_tail == 0:
                    tail = 0
                    break
                # Number of identical tailing values
                run = (a[-max_tail:-1] == a[-1]).astype(int)[::-1].cumprod().sum()
                tail = min(tail, int(run))
            num_samples -= max(0, tail)
        if num_samples < self._sample_pts:
            for ch in self._channels:
                if ch.data.size != num_samples:
                    ch.data = ch.data[:num_samples].copy()

        # Scale and type cast
        for ch in self._channels:
            if self._dtype == _INT16:
                # Rescale short int datatype
                ch.data = (ch.data * ch.resolution).astype(_FLOAT32)
                ch.minval = float(ch.data.min())
                ch.maxval = float(ch.data.max())
            else:
                ch.resolution = float(finfo(_FLOAT32).resolution)

    def read(
        self,
        *,
        strip: bool = True,
        as_dict: bool = False,
        header_only: bool = False,
        batch_size: int | None = None,
    ) -> tuple[ChannelList | ChannelDict | None, ParameterList]:
        if not self.exists:
            msg = f"File {self._file!s} not found."
            raise FileNotFoundError(msg)
        if not batch_size or batch_size < 0:
            batch_size = 1
        with self._buffered_io as bio:
            if bio.file_mapping:
                self.read_header_mapped()
            else:
                self.read_header()
            self.check_params()
            self.prepare_channels(header_only=header_only)

            if not header_only:
                if bio.file_mapping:
                    self.read_data_mapped(batch_size)
                else:
                    self.read_data(batch_size)
                self.post_process(strip=strip)

        channels = to_dict(self._channels) if as_dict else self._channels
        return channels, self._params


def read(
    filepath: str,
    *,
    strip: bool = True,
    as_dict: bool = False,
    header_only: bool = False,
    batch_size: int | None = None,
    file_mapping: bool = False,
) -> tuple[ChannelList | ChannelDict | None, ParameterList]:
    """Read RPC file content.

    Parameters
    ----------
    filepath : str
        Path to file, may be relative or absolute.
    strip : bool
        Removing gap from last frame. Default is True.
    as_dict : bool
        Return a dictionary of channels with names used as keys, instead of a list.
        Default is False.
    header_only : bool
        Don't read channel data (only inspect header).
    batch_size : int | None
        Batch size, default is None.
    file_mapping : bool
        Use file to memory mapping. Default is `False`.

    Raises
    ------
    FileNotFoundError
        If file doesn't exist.
    FileFormatError
        If an internal format error occurs.

    Returns
    -------
    Union[ChannelList, ChannelDict]
        A list or dictionary of `Channel` objects.

    """
    return _RPC3Reader(filepath, file_mapping=file_mapping).read(
        strip=strip,
        batch_size=batch_size,
        as_dict=as_dict,
        header_only=header_only,
    )


def write(
    filepath: str,
    channels: ChannelList,
    *,
    datatype: type | None = int,
    pts_per_group: int = 2048,
    overwrite: bool | None = False,
    extra_params: dict | None = None,
    omit_samples_param: bool | None = False,
) -> None:
    """Write list of channels into RPC3 file.

    Parameters
    ----------
    filepath : str
        The file name of the RPC3 file.
    channels : ChannelList
        The list of channels.
    datatype : Union[type, None], optional
        The data type used to write channel data, may be `int` or `float`.
        `int` : Data will be quantized to 16-bit resolution.
        `float' : Data will be written as 32-bit floating point.
        The default is `int`.
    pts_per_group : int, optional
        The number of sample points per data group. The default is 2048.
    overwrite : bool, optional
        Overwrite existing file. The default is `False`.
    extra_params : dict, optional
        Additional parameters that can be stored in the file header.
        Caution: Only use if you know what you are doing, parameters can
        overwrite reserved key names.
    omit_samples_param : bool, optional
        Don't write `SAMPLES` parameter into file header.
        Default is `False`.

    Raises
    ------
    FileExistsError
        If the file already exists.

    Returns
    -------
    None

    """
    # Pre checks
    if datatype not in (int, float, None):
        msg = "Invalid `datatype`."
        raise ValueError(msg)
    datatype = _INT16 if datatype in (int, None) else _FLOAT32
    if (pts_per_group & (pts_per_group - 1) != 0) or pts_per_group == 0:
        msg = "`pts_per_group` must be a power of 2 and greater than zero."
        raise ValueError(msg)
    dt = None
    SAMPLES = 0
    for ch in channels:
        if dt is None:
            dt = ch.dt
        elif not isclose(dt, ch.dt):
            msg = "RPC3 format only supports one general samplerate."
            raise ValueError(msg)
        SAMPLES = max(SAMPLES, ch.data.size)
    if dt is None:
        dt = 1.0

    def make_params() -> OrderedDict:
        """Make a parameter dictionary from given channel list and given extra parameters.

        Returns
        -------
        params : OrderedDict
            The dictionary with RPC3 header parameters.

        """
        # RPC3 header parameters as ordered dictionary
        params = OrderedDict()
        if datatype == _INT16:
            params["DATA_TYPE"] = "SHORT_INTEGER"
        else:
            # floating point
            params["DATA_TYPE"] = "FLOATING_POINT"
        params["FORMAT"] = "BINARY"
        params["FILE_TYPE"] = "TIME_HISTORY"
        params["DATE"] = datetime.datetime.now(tz=datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S",
        )  # ISO 8601
        params["OPERATION"] = f"Python rpc3 {__version__}"
        params["CHANNELS"] = len(channels)
        params["TIME_TYPE"] = "RESPONSE"
        params["DELTA_T"] = dt
        params["PTS_PER_FRAME"] = pts_per_group
        params["PTS_PER_GROUP"] = pts_per_group
        params["FRAMES"] = (SAMPLES + pts_per_group - 1) // pts_per_group
        params["SAMPLES"] = int(SAMPLES)
        params["BYPASS_FILTER"] = 0
        params["HALF_FRAMES"] = 0
        params["INT_FULL_SCALE"] = _INT_FULL_SCALE
        params["REPEATS"] = 1
        params["PART.CHAN_1"] = 1
        params["PART.NCHAN_1"] = len(channels)
        params["PARTITIONS"] = 1

        for i, ch in enumerate(channels, 1):
            maxval = ch.data.max()
            minval = ch.data.min()
            if datatype == _INT16:
                scale = max(abs(minval), abs(maxval)) / _INT_FULL_SCALE
                params[f"SCALE.CHAN_{i}"] = scale if scale != 0 else 1.0
            else:
                params[f"SCALE.CHAN_{i}"] = 1.0
            params[f"DESC.CHAN_{i}"] = ch.name
            params[f"LOWER_LIMIT.CHAN_{i}"] = minval
            params[f"UPPER_LIMIT.CHAN_{i}"] = maxval
            params[f"MAP.CHAN_{i}"] = i
            params[f"UNITS.CHAN_{i}"] = channels[i - 1].unit

        if extra_params is not None:
            params.update(extra_params)

        # Preset to get valid parameter count
        params["NUM_PARAMS"] = None
        params["NUM_HEADER_BLOCKS"] = None
        # Set with valid values
        params["NUM_PARAMS"] = len(params)
        params["NUM_HEADER_BLOCKS"] = (params["NUM_PARAMS"] + 3) // 4

        return params

    if Path(filepath).suffix.lower() not in [".rpc", ".rpc3", ".rsp"]:
        filepath += ".rpc"
    if not overwrite and Path(filepath).exists():
        msg = f"File {filepath} already exists."
        raise FileExistsError(msg)

    params = make_params()
    if omit_samples_param:
        del params["SAMPLES"]
        params["NUM_PARAMS"] -= 1
        params["NUM_HEADER_BLOCKS"] = (params["NUM_PARAMS"] + 3) // 4
    NUM_PARAMS = params["NUM_PARAMS"]
    with Path(filepath).open("wb") as f:
        # Write header
        for i in range(params["NUM_HEADER_BLOCKS"] * 4):
            if i < NUM_PARAMS:
                if i == 0:
                    key, value = "FORMAT", params.pop("FORMAT")
                elif i == 1:
                    key, value = "NUM_HEADER_BLOCKS", params.pop("NUM_HEADER_BLOCKS")
                elif i == 2:
                    key, value = "NUM_PARAMS", params.pop("NUM_PARAMS")
                    param = iter(params.items())
                else:
                    key, value = next(param)
            else:
                key, value = "", ""
            f.write(
                key.encode("latin-1", "strict").ljust(32, b"\0")
                + str(value).encode("latin-1", "strict").ljust(96, b"\0"),
            )
        # Write data
        CHANNELS = params["CHANNELS"]
        PTS_PER_FRAME = params["PTS_PER_FRAME"]
        PTS_PER_GROUP = params["PTS_PER_GROUP"]
        FRAMES_PER_GROUP = PTS_PER_GROUP // PTS_PER_FRAME
        GROUPS = (params["FRAMES"] + FRAMES_PER_GROUP - 1) // FRAMES_PER_GROUP
        scale = tuple(float(params[f"SCALE.CHAN_{i}"]) for i in range(1, CHANNELS + 1))
        total_bytes = GROUPS * PTS_PER_GROUP * CHANNELS * datatype.itemsize
        start = 0
        with tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            disable=not progressbar,
        ) as pbar:
            for _ in range(GROUPS):
                stop = start + PTS_PER_GROUP
                for i, ch in enumerate(channels):
                    buffer = ch.data[start:stop]  # view
                    if buffer.size < PTS_PER_GROUP:
                        buffer = append(
                            buffer,
                            ones(PTS_PER_GROUP - buffer.size) * ch.data[-1],
                        )

                    if datatype == _INT16:
                        q = (buffer / scale[i]).round()  # copy
                        q = q.clip(-_INT_FULL_SCALE, _INT_FULL_SCALE).astype(datatype)
                        f.write(q.tobytes())
                    else:
                        f.write(buffer.astype(datatype).tobytes())
                start = stop
                pbar.update(PTS_PER_GROUP * CHANNELS * datatype.itemsize)
