"""RPC3 file class."""

# ruff: noqa: E501 S101 B028 PLR2004 N806

from __future__ import annotations

import contextlib
import datetime
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Union
from warnings import warn

from numpy import (
    append,
    arange,
    argmax,
    asarray,
    dtype,
    empty,
    finfo,
    fromfile,
    isclose,
    nan,
    ones,
)
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import BinaryIO, TypeAlias

    from numpy.typing import NDArray

__all__ = [
    "Channel",
    "ChannelDict",
    "ChannelList",
    "FileFormatError",
    "OrderedDict",
    "__author__",
    "__version__",
    "read",
    "to_dict",
    "write",
]


_INT_FULL_SCALE: int = 2**15 - 16
_INT16: dtype = dtype("<i2")
_FLOAT32: dtype = dtype("<f4")
progressbar: bool = True

__version__ = "1.0.0rc4"
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
    x0 : float
        Start time in [s], always 0.
    dt : float
        Unique timestep between samples in `data` in [s].
    unitx : str
        Unit of time axis, always 's'.
    data : ndarray, optional
        The data vector (1d).
    minval : float
        Smallest value in `data`.
        (Set on read and ignored on write.)
    maxval : float
        Largest value in `data`.
        (Set on read and ignored on write.)
    resolution : float
        Resolution of `data`.
        (Set on read and ignored on write.)
        For INTEGER RPC3 files, `resolution` is an absolute value (bit resolution),
        for FLOATING POINT RPC3 files, `resolution` is a relative value.

    """

    name: str
    unit: str
    dt: float
    data: Sequence | None
    minval: float = nan
    maxval: float = nan
    resolution: float = nan

    def __post_init__(self) -> None:
        """Make flat numpy NDArray."""
        self.data = asarray(self.data).flatten()
        self.x0 = 0
        self.unitx = "s"

    def time(self) -> NDArray:
        """Make the time vector for this channel."""
        return arange(self.data.size) * self.dt + self.x0


ChannelList: TypeAlias = list[Channel]
ChannelDict: TypeAlias = OrderedDict[str, Union[Channel, list[Channel]]]


def _file_size(filepath: str) -> int | None:
    """Return the file size in bytes.

    Parameters
    ----------
    filepath : str
        Path to file, may be relative or absolute.

    Returns
    -------
    Optional[int]
        Size in bytes.

    """
    if Path(filepath).exists():
        return Path(filepath).stat().st_size

    return None


def to_dict(channels: ChannelList) -> ChannelDict:
    """Transform channel list into name mapped dictionary.

    Parameters
    ----------
    channels : ChannelList
        The list of channels returned by `read()`.

    Returns
    -------
    ChannelDict
        Dictionary of channels, grouped by channel names.

    """
    new_dict = OrderedDict()
    for ch in channels:
        if ch.name in new_dict:
            if isinstance(new_dict[ch.name], Channel):
                warn(f"Channel name is ambiguous: `{ch.name}`")
                new_dict[ch.name] = [new_dict[ch.name], ch]
            else:
                assert isinstance(new_dict[ch.name], list)
                new_dict[ch.name].append(ch)
        else:
            new_dict[ch.name] = ch
    return new_dict


def read(
    filepath: str,
    *,
    strip: bool = True,
    as_dict: bool = False,
) -> ChannelList | ChannelDict:
    """Read RPC file content.

    Parameters
    ----------
    filepath : str
        Path to file, may be relative or absolute.
    strip: bool
        Removing gap from last frame. Default is True.
    as_dict: bool
        Return a dictionary of channels with names used as keys, instead of a list.
        Default is False.

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

    def read_next_param(f: BinaryIO) -> tuple[str, str | int | float]:
        """Return next header parameter at read position.

        Value is type casted to float or int if possible.

        Parameters
        ----------
        f : BinaryIO
            File handle.

        Returns
        -------
        tuple[str, Any]
            Parameter as key/value pair.

        """
        key, value = (f.read(n).decode("latin-1").strip(" \0") for n in (32, 96))
        if key.replace("_", ".").split(".")[0] not in (
            "OPERATION",
            "PARENT",
            "DESC",
            "UNIT",
        ):
            with contextlib.suppress(ValueError):
                value = float(value) if "." in value else int(value)

        return key, value

    if not Path(filepath).exists():
        msg = f"File {filepath} not found."
        raise FileNotFoundError(msg)
    channels = []
    params = {"NUM_HEADER_BLOCKS": 1, "NUM_PARAMS": 3}
    with Path(filepath).open("rb") as f:
        # First 3 entries are specified:
        # Pos. 1: "FORMAT" is one of "BINARY_IEEE_LITTLE_END", "BINARY_IEEE_BIG_END",
        #         "BINARY" or "ASCII"
        # Pos. 2: "NUM_HEADER_BLOCKS", number of 512-byte blocks (max. 256 blocks)
        # Pos. 3: "NUM_PARAMS", number of parameters in header (max. 1024 parameters)
        i = 0
        while i < params["NUM_HEADER_BLOCKS"] * 4:
            if i < params["NUM_PARAMS"]:
                key, value = read_next_param(f)
                if i == 0:
                    # only Intel little endian format is supported
                    if key != "FORMAT":
                        msg = "First header key must be `FORMAT`."
                        raise FileFormatError(msg)
                    if value not in ("BINARY_IEEE_LITTLE_END", "BINARY"):
                        msg = "Only `BINARY_IEEE_LITTLE_END` and `BINARY` number formats supported."
                        raise FileFormatError(msg)
                elif i == 1:
                    # support large header (MTS(R) documentation "RPC3 file formats" specifies only max. 256 blocks)
                    if key != "NUM_HEADER_BLOCKS":
                        msg = "Second header key must be `NUM_HEADER_BLOCKS`."
                        raise FileFormatError(msg)
                    if not (0 < value <= 256 * 4):
                        msg = (
                            f"Too many header blocks (`NUM_HEADER_BLOCKS`={int(value)})."
                        )
                        raise FileFormatError(msg)
                elif i == 2:
                    # support large header (MTS(R) documentation "RPC3 file formats" specifies only max. 256
                    # blocks)
                    if key != "NUM_PARAMS":
                        msg = "Third header key must be `NUM_PARAMS`."
                        raise FileFormatError(msg)
                    if not (3 < value <= params["NUM_HEADER_BLOCKS"] * 4):
                        msg = f"Wrong number of parameters (`NUM_PARAMS`={int(value)})"
                        raise FileFormatError(msg)
                elif len(key) == 0:
                    continue
                params[key] = value
            else:
                f.seek(32 + 96, os.SEEK_CUR)
            i += 1

        if "FILE_TYPE" in params and params["FILE_TYPE"] != "TIME_HISTORY":
            msg = "Only data file type `TIME_HISTORY` is supported."
            raise FileFormatError(msg)

        if "TIME_TYPE" in params:
            if isinstance(params["TIME_TYPE"], int):
                if params["TIME_TYPE"] not in [1, 2, 3, 4]:
                    msg = (
                        "Only time types 1..4 supported "
                        f"(`TIME_TYPE`={int(params['TIME_TYPE'])})."
                    )
                    raise FileFormatError(msg)
            elif params["TIME_TYPE"] not in [
                "DRIVE",
                "RESPONSE",
                "MULT_DRIVE",
                "MULT_RESP",
            ]:
                msg = (
                    "Only `DRIVE`, `RESPONSE`, `MULT_DRIVE` and `MULT_RESP` supported "
                    f"(`TIME_TYPE`={int(params['TIME_TYPE'])})"
                )

        if len(params) != params["NUM_PARAMS"]:
            msg = "Wrong number of parameters in file header."
            raise FileFormatError(msg)

        if "DELTA_T" not in params or params["DELTA_T"] <= 0:
            msg = "Missing valid samplerate parameter `DELTA_T`."
            raise FileFormatError(msg)

        if "CHANNELS" not in params or params["CHANNELS"] < 0:
            msg = "Missing or wrong number of channels."
            raise FileFormatError(msg)

        CHANNELS = params["CHANNELS"]
        FRAMES = params["FRAMES"]
        SAMPLES = params.get("SAMPLES")
        PTS_PER_GROUP = params["PTS_PER_GROUP"]
        PTS_PER_FRAME = params["PTS_PER_FRAME"]
        if PTS_PER_FRAME <= 0 or PTS_PER_GROUP % PTS_PER_FRAME != 0:
            raise FileFormatError
        FRAMES_PER_GROUP = PTS_PER_GROUP // PTS_PER_FRAME
        if FRAMES_PER_GROUP <= 0 or FRAMES % FRAMES_PER_GROUP != 0:
            raise FileFormatError
        GROUPS = FRAMES // FRAMES_PER_GROUP
        SAMPLE_PTS = GROUPS * PTS_PER_GROUP
        if params.get("DATA_TYPE", "SHORT_INTEGER") == "SHORT_INTEGER":
            datatype = _INT16
        elif params["DATA_TYPE"] == "FLOATING_POINT":
            datatype = _FLOAT32
        else:
            msg = (
                "Only `SHORT_INTEGER` and `FLOATING_POINT` supported "
                f"({params['DATA_TYPE']})"
            )
            raise FileFormatError(msg)
        for i in range(1, CHANNELS + 1):
            ch = Channel(
                name=params[f"DESC.CHAN_{i}"],
                unit=params[f"UNITS.CHAN_{i}"],
                resolution=params[f"SCALE.CHAN_{i}"],
                minval=params[f"LOWER_LIMIT.CHAN_{i}"],
                maxval=params[f"UPPER_LIMIT.CHAN_{i}"],
                dt=params["DELTA_T"],
                data=empty(SAMPLE_PTS, datatype),
            )
            channels.append(ch)

        # Read multiplexed channels
        start = 0
        for _ in tqdm(
            range(GROUPS),
            unit="MB",
            unit_scale=PTS_PER_GROUP * CHANNELS * datatype.itemsize / 1024**2,
            disable=not progressbar,
        ):
            stop = start + PTS_PER_GROUP
            for ch in range(CHANNELS):
                channels[ch].data[start:stop] = fromfile(f, datatype, PTS_PER_GROUP)
            start = stop

        if SAMPLES is not None:
            for ch in channels:
                ch.data = ch.data[:SAMPLES]
        elif strip:
            # Remove gap from last group
            gap = nan
            for ch in channels:
                gap = min(argmax(ch.data[::-1] != ch.data[-1]), gap)
            if gap > 1:
                for ch in channels:
                    ch.data = ch.data[: 1 - gap]

        # Scale and type cast
        for ch in channels:
            if datatype == _INT16:
                # Rescale short int datatype
                ch.data = (ch.data * ch.resolution).astype(_FLOAT32)
                ch.minval = float(ch.data.min())
                ch.maxval = float(ch.data.max())
            else:
                ch.resolution = float(finfo(_FLOAT32).resolution)

    if as_dict:
        channels = to_dict(channels)

    return channels, params


def write(
    filepath: str,
    channels: ChannelList,
    *,
    datatype: type | None = int,
    pts_per_group: int | None = 2048,
    overwrite: bool | None = False,
    extra_params: dict | None = None,
):
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
    pts_per_group : Optional[int], optional
        The number of sample points per data group. The default is 2048.
    overwrite : bool, optional
        Overwrite existing file. The default is False.
    extra_params : dict, optional
        Additional parameters that can be stored in the file header.
        Caution: Only use if you know what you are doing, parameters can
        overwrite reserved key names.

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
        params["DATE"] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")  # ISO 8601
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

        params["NUM_PARAMS"] = len(params) + 2
        params["NUM_HEADER_BLOCKS"] = (params["NUM_PARAMS"] + 3) // 4

        return params

    if Path(filepath).suffix.lower() not in [".rpc", ".rpc3", ".rsp"]:
        filepath += ".rpc"
    if not overwrite and Path(filepath).exists():
        msg = f"File {filepath} already exists."
        raise FileExistsError(msg)

    params = make_params()
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
                key.encode().ljust(32, b"\0")
                + str(value).encode("latin-1").ljust(96, b"\0"),
            )
        # Write data
        CHANNELS = params["CHANNELS"]
        PTS_PER_FRAME = params["PTS_PER_FRAME"]
        PTS_PER_GROUP = params["PTS_PER_GROUP"]
        FRAMES_PER_GROUP = PTS_PER_GROUP // PTS_PER_FRAME
        GROUPS = (params["FRAMES"] + FRAMES_PER_GROUP - 1) // FRAMES_PER_GROUP
        scale = tuple(float(params[f"SCALE.CHAN_{i}"]) for i in range(1, CHANNELS + 1))
        start = 0
        for _ in tqdm(
            range(GROUPS),
            unit="MB",
            unit_scale=PTS_PER_GROUP * CHANNELS * datatype.itemsize / 1024**2,
            disable=not progressbar,
        ):
            stop = start + PTS_PER_GROUP
            for i, ch in enumerate(channels):
                buffer = ch.data[start:stop]  # view
                if buffer.size < PTS_PER_GROUP:
                    buffer = append(
                        buffer, ones(PTS_PER_GROUP - buffer.size) * ch.data[-1],
                    )
                if datatype == _INT16:
                    buffer = buffer / scale[i]  # copy
                f.write(buffer.astype(datatype))
            start = stop
