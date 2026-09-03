"""RPC3 file class unit test.

pytest --pyargs rpc3.tests

"""

# ruff: noqa: S101 PLR2004 E501

from itertools import product
import re

import numpy as np
import pytest
import rpc3

rng = np.random.default_rng()


def test_write_empty() -> None:
    """Write empty file."""
    rpc3.write("./test.rpc", channels=[], overwrite=True)


def test_overwrite(tmp_path) -> None:
    """Refuse to overwrite unless `overwrite=True`."""
    path = tmp_path / "exists.rpc"
    rpc3.write(str(path), channels=[], overwrite=True)
    with pytest.raises(FileExistsError, match="already exists"):
        rpc3.write(str(path), channels=[], overwrite=False)
    rpc3.write(str(path), channels=[], overwrite=True)


def test_write() -> None:
    """Write 4 channels."""
    fs = 500
    channels = []
    for _ in range(4):
        data = np.arange(0, 123456, 1)
        ch = rpc3.Channel("Accel", "m/s²", dt=1 / fs, data=data)
        channels.append(ch)
    rpc3.write("./test.rpc", channels, overwrite=True)


def test_read_strip() -> None:
    """Write 4 channels."""
    fs = 500
    channels = []
    for _ in range(4):
        data = np.linspace(0, 1, 57)
        ch = rpc3.Channel("Accel", "m/s²", dt=1 / fs, data=data)
        channels.append(ch)
    rpc3.write(
        "./test.rpc",
        channels,
        overwrite=True,
        pts_per_group=1024,
        omit_samples_param=True)
    read_channels, _ = rpc3.read("./test.rpc", strip=False)
    assert read_channels[0].data.size == 1024
    read_channels, _ = rpc3.read("./test.rpc", strip=True)
    assert read_channels[0].data.size == 57


def test_write_ambiguous() -> None:
    """Write 4 channels, two channels with unique name."""
    fs = 500
    channels = []
    channels.append(rpc3.Channel("First", "unit", dt=1 / fs, data=[1, 2, 3]))
    channels.append(rpc3.Channel("Second", "unit", dt=1 / fs, data=[4, 5, 6]))
    channels.append(rpc3.Channel("Second", "unit", dt=1 / fs, data=[7, 8, 9]))
    channels.append(rpc3.Channel("Third", "unit", dt=1 / fs, data=[10, 11, 12]))
    rpc3.write("./test.rpc", channels, overwrite=True)
    with pytest.warns(Warning, match="ambiguous.*`Second`"):
        channels_read, params = rpc3.read("./test.rpc", as_dict=True)
    assert params["DATA_TYPE"] == "SHORT_INTEGER"
    assert isinstance(channels_read, rpc3.OrderedDict)
    assert tuple(channels_read.keys()) == ("First", "Second", "Third")
    assert isinstance(channels_read["First"], rpc3.Channel)
    assert isinstance(channels_read["Second"], list)
    assert len(channels_read["Second"]) == 2
    assert isinstance(channels_read["Third"], rpc3.Channel)
    assert isinstance(channels_read["Second"][0], rpc3.Channel)
    assert isinstance(channels_read["Second"][1], rpc3.Channel)
    np.testing.assert_allclose(
        channels_read["First"].data,
        [1, 2, 3],
        atol=channels_read["First"].resolution,
    )
    np.testing.assert_allclose(
        channels_read["Second"][0].data,
        [4, 5, 6],
        atol=channels_read["Second"][0].resolution,
    )
    np.testing.assert_allclose(
        channels_read["Second"][1].data,
        [7, 8, 9],
        atol=channels_read["Second"][1].resolution,
    )
    np.testing.assert_allclose(
        channels_read["Third"].data,
        [10, 11, 12],
        atol=channels_read["Third"].resolution,
    )


def test_non_unique_lengths() -> None:
    """Write channels with non-unique lengths (shorter channels will be padded)."""
    fs = 500
    channels = []
    for i in range(4):
        data = np.arange(0, 123456 - 1000 * i, 1)
        ch = rpc3.Channel("Accel", "m/s²", dt=1 / fs, data=data)
        channels.append(ch)
    rpc3.write("./test.rpc", channels, overwrite=True)


def test_non_unique_dt() -> None:
    """Test non-unique channel sample rates (should fail)."""
    channels = []
    for i in range(4):
        data = np.arange(0, 1024, 1)
        ch = rpc3.Channel(f"Accel_{i}", "m/s²", dt=i, data=data)
        channels.append(ch)
    with pytest.raises(ValueError, match=".*general samplerate"):
        # This should fail
        rpc3.write("./test.rpc", channels, overwrite=True)


def test_extra_params() -> None:
    """Check extra parameters are stored (not portable!)."""
    rpc3.write(
        "./test.rpc",
        channels=[],
        overwrite=True,
        extra_params={"MY_STR": "Text", "MY_INT": 123456, "MY_FLOAT": 3.1415},
    )
    _, params = rpc3.read("./test.rpc")
    assert params["MY_STR"] == "Text"
    assert params["MY_INT"] == 123456
    assert params["MY_FLOAT"] == 3.1415


def test_write_reliably_int() -> None:
    """Test reliability on writing as SHORT_INTEGER and character encoding."""
    fs = 500
    channels = []
    data = rng.lognormal(33.3, 0.377, size=100_000)
    channels.append(rpc3.Channel("Random äüöß", "m/s²", dt=1 / fs, data=data))
    data = rng.lognormal(-33.3, 1.377, size=100_000)
    channels.append(rpc3.Channel("Random äüöß", "m/s²", dt=1 / fs, data=data))
    data = rng.normal(-33.3, 1000, size=100_000)
    channels.append(rpc3.Channel("Random äüöß", "m/s²", dt=1 / fs, data=data))
    rpc3.write(
        "./test.rpc",
        channels,
        overwrite=True)
    for file_mapping, batch_size in product([True, False], [-1, 0, 1, 2, 128]):
        channels_read, _ = rpc3.read(
            "./test.rpc",
            file_mapping=file_mapping,
            batch_size=batch_size)
        for lhs, rhs in zip(channels, channels_read):
            assert pytest.approx(rhs.minval, rhs.resolution) == rhs.data.min()
            assert pytest.approx(rhs.maxval, rhs.resolution) == rhs.data.max()
            assert rhs.name == "Random äüöß"
            assert np.abs(lhs.data - rhs.data).max() < 2 * rhs.resolution
            np.testing.assert_allclose(lhs.data, rhs.data, atol=rhs.resolution)


def test_write_reliably_float() -> None:
    """Test reliability on writing FLOATING_POINT and character encoding."""
    fs = 500
    channels = []
    data = rng.lognormal(33.3, 0.377, size=100_000)
    channels.append(rpc3.Channel("Random äüöß", "m/s²", dt=1 / fs, data=data))
    data = rng.lognormal(-33.3, 1.377, size=100_000)
    channels.append(rpc3.Channel("Random äüöß", "m/s²", dt=1 / fs, data=data))
    data = rng.normal(-33.3, 1000, size=100_000)
    channels.append(rpc3.Channel("Random äüöß", "m/s²", dt=1 / fs, data=data))
    rpc3.write("./test.rpc", channels, overwrite=True, datatype=float)
    for file_mapping in (False, True):
        channels_read, _ = rpc3.read("./test.rpc", file_mapping=file_mapping)
        for lhs, rhs in zip(channels, channels_read):
            rel = rhs.resolution * abs(rhs.minval)
            assert pytest.approx(rhs.minval, rel) == rhs.data.min()
            assert pytest.approx(rhs.maxval, rel) == rhs.data.max()
            assert rhs.name == "Random äüöß"
            assert rhs.unit == "m/s²"
            assert np.abs(lhs.data - rhs.data).max() < np.abs(rhs.data).max() * rhs.resolution
            np.testing.assert_allclose(lhs.data, rhs.data, rtol=rhs.resolution)


def test_plot() -> None:
    """Compare signals after write and reread from RPC3 graphically."""
    from pprint import PrettyPrinter

    import matplotlib.pyplot as plt

    fs = 500
    channels = []
    data = rng.normal(-33.3, 100, size=100_000)
    channels.append(
        rpc3.Channel(
            r"Random $\mu$=-33.3, $\sigma$=100",
            "unit",
            dt=1 / fs,
            data=data,
        ),
    )
    data = np.sin(np.arange(data.size) / fs) * 3 + 1.2
    channels.append(
        rpc3.Channel(r"Sinus Amplitude=3, Offset=1.2", "unit", dt=1 / fs, data=data),
    )
    rpc3.write("./test.rpc", channels, overwrite=True)
    channels_read, params = rpc3.read("./test.rpc")
    print()  # newline # noqa: T201
    PrettyPrinter().pprint(channels)
    PrettyPrinter().pprint(params)
    fig, ax = plt.subplots(2, 1, figsize=(12, 6))
    t = np.arange(channels[0].data.size) * channels[0].dt
    for i in range(len(channels)):
        ax[i].plot(
            t,
            channels[i].data,
            "k-",
            lw=0.8,
            label=f"{channels[i].name} [{channels[i].unit}] (Original)",
        )
        ax[i].plot(
            t,
            channels_read[i].data,
            "r--",
            lw=0.5,
            label=f"{channels[i].name} [{channels[i].unit}] (load<-write)",
        )
        ax[i].legend()
        ax[i].grid()
    plt.tight_layout()
    plt.show()

def _make_channels() -> rpc3.ChannelList:
    # two voltages (duplicate name), two temps with different units, one current
    return [
        rpc3.Channel("voltage", "V", dt=0.1, data=[1, 2, 3]),
        rpc3.Channel("voltage", "V", dt=0.1, data=[4, 5, 6]),
        rpc3.Channel("temp_1", "K", dt=0.1, data=[10, 11]),
        rpc3.Channel("temp_2", "°C", dt=0.1, data=[20, 21]),
        rpc3.Channel("current", "A", dt=0.1, data=[7, 8, 9]),
    ]

# ----------------------------- Sequence input ---------------------------------

def test_seq_exact_name_returns_all_matches() -> None:
    seq = _make_channels()
    hits = rpc3.find_channel(seq, name="voltage")
    assert isinstance(hits, list)
    assert len(hits) == 2
    assert all(ch.name == "voltage" for ch in hits)

def test_seq_regex_all_name_and_unit():
    seq = _make_channels()
    hits = rpc3.find_channel(seq, name=r"temp_\d+", unit=r"K|°C", regex=True, match="all")
    assert {ch.name for ch in hits} == {"temp_1", "temp_2"}

def test_seq_match_any_with_only_unit():
    seq = _make_channels()
    hits = rpc3.find_channel(seq, unit="A", match="any")
    assert len(hits) == 1 and hits[0].name == "current"

def test_seq_assert_once_zero_one_many():
    seq = _make_channels()
    # zero -> None
    assert rpc3.find_channel(seq, name="power", assert_once=True) is None
    # one -> Channel
    one = rpc3.find_channel(seq, name="current", assert_once=True)
    assert one is not None and one.name == "current"
    # many -> ValueError
    with pytest.raises(ValueError, match="More than one match"):
        rpc3.find_channel(seq, name="voltage", assert_once=True)

def test_seq_as_dict_groups_duplicates_and_keys_by_name():
    seq = _make_channels()
    out = rpc3.find_channel(seq, name=r"voltage|current", regex=True, as_dict=True)
    # dict keyed by name
    assert "voltage" in out and "current" in out
    # duplicate name becomes list[Channel]
    assert isinstance(out["voltage"], list) and len(out["voltage"]) == 2
    # single stays Channel
    assert out["current"].name == "current"

# ----------------------------- Mapping input ----------------------------------

def test_map_input_behaves_same_semantics():
    seq = _make_channels()
    mapping = rpc3.to_dict(seq)  # produces {'voltage': [.., ..], 'temp_1': Channel, ...}
    hits = rpc3.find_channel(mapping, unit="V", as_dict=True)
    assert "voltage" in hits
    assert isinstance(hits["voltage"], list) and len(hits["voltage"]) == 2

def test_map_assert_once_returns_channel_not_dict():
    seq = _make_channels()
    mapping = rpc3.to_dict(seq)
    ch = rpc3.find_channel(mapping, name="current", assert_once=True)
    assert isinstance(ch, rpc3.Channel) and ch.name == "current"

def test_assert_once_with_as_dict_returns_one_entry_dict():
    seq = _make_channels()
    out = rpc3.find_channel(seq, name="current", assert_once=True, as_dict=True)
    assert isinstance(out, rpc3.OrderedDict)
    assert list(out.keys()) == ["current"]
    assert isinstance(out["current"], rpc3.Channel) and out["current"].name == "current"
    # zero matches still None (assert_once semantics), even with as_dict
    assert rpc3.find_channel(seq, name="power", assert_once=True, as_dict=True) is None
    # many matches still raise
    with pytest.raises(ValueError, match="More than one match"):
        rpc3.find_channel(seq, name="voltage", assert_once=True, as_dict=True)

def test_map_regex_all_combination():
    seq = _make_channels()
    mapping = rpc3.to_dict(seq)
    hits = rpc3.find_channel(mapping, name=r"temp_\d+", unit=r"K|°C", regex=True, match="all")
    assert {c.name for c in hits} == {"temp_1", "temp_2"}

# ----------------------------- Stability / edge cases -------------------------

def test_as_dict_with_no_hits_returns_empty_ordered_dict():
    seq = _make_channels()
    out = rpc3.find_channel(seq, name="does_not_exist", as_dict=True)
    # to_dict([]) -> OrderedDict()
    from collections import OrderedDict
    assert isinstance(out, OrderedDict) and len(out) == 0

def test_type_errors_for_unsupported_inputs():
    with pytest.raises(TypeError):
        rpc3.find_channel("not a container", name="x")  # type: ignore[arg-type]

def test_find_channel_none_returns_none():
    assert rpc3.find_channel(None) is None
    assert rpc3.find_channel(None, name="x", as_dict=True) is None
    assert rpc3.find_channel(None, assert_once=True) is None

def test_find_channel_invalid_match_raises():
    seq = _make_channels()
    with pytest.raises(ValueError, match='match must be "any" or "all"'):
        rpc3.find_channel(seq, name="current", match="AND")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match='match must be "any" or "all"'):
        rpc3.find_channel(None, match="ALL")  # type: ignore[arg-type]

def test_no_filters_returns_all_for_any_and_all():
    seq = _make_channels()
    assert len(rpc3.find_channel(seq)) == len(seq)
    assert len(rpc3.find_channel(seq, match="all")) == len(seq)

def test_only_unit_with_any_works():
    seq = _make_channels()
    hits = rpc3.find_channel(seq, unit="A", match="any")
    assert [h.name for h in hits] == ["current"]

def test_match_any_with_both_name_and_unit_is_or():
    seq = _make_channels()
    hits = rpc3.find_channel(seq, name="voltage", unit="A", match="any")
    assert {h.name for h in hits} == {"voltage", "current"}
    # Contrast: AND requires both criteria on the same channel
    hits_all = rpc3.find_channel(seq, name="voltage", unit="A", match="all")
    assert hits_all == []

def test_invalid_regex_raises_re_error():
    seq = _make_channels()
    with pytest.raises(re.error):
        rpc3.find_channel(seq, name="[unterminated", regex=True)


def _write_custom_rpc3(
    path,
    data: np.ndarray,
    *,
    pts_per_frame: int,
    pts_per_group: int,
    extra_bytes: bytes = b"",
    omit_samples: bool = True,
) -> None:
    """Write a TIME_HISTORY file with independent frame and group sizes.

    `data` is shape ``(n_samples,)`` or ``(n_channels, n_samples)``.
    """
    data = np.atleast_2d(np.asarray(data, dtype=np.float32))
    n_ch, n_samples = data.shape
    frames = (n_samples + pts_per_frame - 1) // pts_per_frame
    frames_per_group = pts_per_group // pts_per_frame
    groups = (frames + frames_per_group - 1) // frames_per_group

    params: list[tuple[str, object]] = [
        ("DATA_TYPE", "FLOATING_POINT"),
        ("FILE_TYPE", "TIME_HISTORY"),
        ("CHANNELS", n_ch),
        ("DELTA_T", 0.001),
        ("PTS_PER_FRAME", pts_per_frame),
        ("PTS_PER_GROUP", pts_per_group),
        ("FRAMES", frames),
    ]
    if not omit_samples:
        params.append(("SAMPLES", n_samples))
    for i in range(1, n_ch + 1):
        ch = data[i - 1]
        params.extend(
            [
                (f"DESC.CHAN_{i}", f"ch{i}"),
                (f"UNITS.CHAN_{i}", "U"),
                (f"SCALE.CHAN_{i}", 1.0),
                (f"LOWER_LIMIT.CHAN_{i}", float(ch.min()) if ch.size else 0.0),
                (f"UPPER_LIMIT.CHAN_{i}", float(ch.max()) if ch.size else 0.0),
            ],
        )

    n_params = len(params) + 3
    n_blocks = (n_params + 3) // 4
    header: list[tuple[str, object]] = [
        ("FORMAT", "BINARY"),
        ("NUM_HEADER_BLOCKS", n_blocks),
        ("NUM_PARAMS", n_params),
        *params,
    ]

    with path.open("wb") as f:
        for i in range(n_blocks * 4):
            key, value = header[i] if i < n_params else ("", "")
            f.write(
                str(key).encode("latin-1").ljust(32, b"\0")
                + str(value).encode("latin-1").ljust(96, b"\0"),
            )
        start = 0
        for _ in range(groups):
            stop = start + pts_per_group
            for i in range(n_ch):
                buf = data[i, start:stop]
                if buf.size < pts_per_group:
                    pad = data[i, -1] if data[i].size else 0.0
                    buf = np.append(
                        buf,
                        np.ones(pts_per_group - buf.size, dtype=np.float32) * pad,
                    )
                f.write(np.asarray(buf, dtype=np.float32).tobytes())
            start = stop
        if extra_bytes:
            f.write(extra_bytes)


@pytest.mark.parametrize("file_mapping", [False, True])
def test_strip_does_not_trim_partial_last_group(tmp_path, file_mapping: bool) -> None:
    """Constant tail must be kept when unused last-group frames were not loaded."""
    pts_per_frame = 256
    pts_per_group = 1024
    n_frames = 10  # 2 leftover frames in the last of 3 groups
    hold = 400
    n_samples = n_frames * pts_per_frame
    data = np.concatenate(
        [
            np.arange(n_samples - hold, dtype=np.float32),
            np.full(hold, 999.0, dtype=np.float32),
        ],
    )
    path = tmp_path / "partial_group.rpc"
    _write_custom_rpc3(
        path,
        data,
        pts_per_frame=pts_per_frame,
        pts_per_group=pts_per_group,
        omit_samples=True,
    )
    with pytest.warns(Warning, match="Partially filled last group"):
        channels, _ = rpc3.read(str(path), strip=True, file_mapping=file_mapping)
    assert channels[0].data.size == n_samples
    np.testing.assert_allclose(channels[0].data, data, rtol=channels[0].resolution)


@pytest.mark.parametrize("file_mapping", [False, True])
def test_read_strip_last_group_padding(tmp_path, file_mapping: bool) -> None:
    """Last-value padding in a complete last group is still stripped."""
    fs = 500
    n = 57
    original = np.linspace(0, 1, n)
    path = tmp_path / "strip.rpc"
    rpc3.write(
        str(path),
        [rpc3.Channel("Accel", "m/s²", dt=1 / fs, data=original)],
        overwrite=True,
        pts_per_group=1024,
        omit_samples_param=True,
    )
    channels, _ = rpc3.read(str(path), strip=True, file_mapping=file_mapping)
    assert channels[0].data.size == n
    np.testing.assert_allclose(
        channels[0].data,
        original,
        atol=channels[0].resolution,
    )


@pytest.mark.parametrize("file_mapping", [False, True])
@pytest.mark.parametrize("extra_len", [1, 3, 127, 128, 129, 4001])
def test_trailing_extra_bytes_are_ignored(
    tmp_path,
    file_mapping: bool,
    extra_len: int,
) -> None:
    """Trailing bytes must not break mapped or buffered reads."""
    path = tmp_path / "extra.rpc"
    n = 3000
    channels = [
        rpc3.Channel("A", "U", dt=0.002, data=np.linspace(0, 1, n)),
        rpc3.Channel("B", "U", dt=0.002, data=np.linspace(1, 2, n)),
    ]
    rpc3.write(str(path), channels, overwrite=True)
    ref, _ = rpc3.read(str(path), file_mapping=False)
    with path.open("ab") as f:
        f.write(b"\xff" * extra_len)
    with pytest.warns(Warning, match="extra bytes"):
        got, _ = rpc3.read(str(path), file_mapping=file_mapping)
    assert len(got) == len(ref)
    for lhs, rhs in zip(ref, got):
        np.testing.assert_allclose(lhs.data, rhs.data, atol=rhs.resolution)


def test_channel_data_defaults_to_empty() -> None:
    ch = rpc3.Channel("A", "U", dt=0.1)
    assert ch.data.size == 0
    assert ch.data.dtype == np.float32


def test_write_empty_channel(tmp_path) -> None:
    """A channel with no samples can be written and read back."""
    path = tmp_path / "empty_ch.rpc"
    rpc3.write(
        str(path),
        [rpc3.Channel("Empty", "U", dt=0.1, data=None)],
        overwrite=True,
    )
    channels, params = rpc3.read(str(path))
    assert params["CHANNELS"] == 1
    assert channels[0].data.size == 0


def test_write_mixed_empty_channel_is_zero_padded(tmp_path) -> None:
    """Empty channels are padded to the longest channel with zeros."""
    path = tmp_path / "mixed.rpc"
    filled = np.array([1.0, 2.0, 3.0, 4.0])
    rpc3.write(
        str(path),
        [
            rpc3.Channel("Filled", "U", dt=0.1, data=filled),
            rpc3.Channel("Empty", "U", dt=0.1),
        ],
        overwrite=True,
        datatype=float,
    )
    channels, _ = rpc3.read(str(path))
    np.testing.assert_allclose(channels[0].data, filled, rtol=channels[0].resolution)
    assert channels[1].data.size == filled.size
    np.testing.assert_allclose(channels[1].data, 0.0, atol=channels[1].resolution)


def test_header_records_are_clipped(tmp_path) -> None:
    """Keys stay at 32 bytes and values at 96 so the header grid is preserved."""
    path = tmp_path / "clip.rpc"
    name = "N" * 120
    unit = "U" * 120
    extra_key = "K" * 40
    extra_val = "V" * 200
    rpc3.write(
        str(path),
        [rpc3.Channel(name, unit, dt=0.1, data=[1.0, 2.0, 3.0])],
        overwrite=True,
        extra_params={extra_key: extra_val},
    )
    channels, params = rpc3.read(str(path))
    assert channels[0].name == name[:96]
    assert channels[0].unit == unit[:96]
    assert extra_key[:32] in params
    assert extra_key not in params
    assert params[extra_key[:32]] == extra_val[:96]


@pytest.mark.parametrize("suffix", [".tim", ".rpc", ".rpc3", ".rsp", ".TIM"])
def test_write_keeps_known_suffixes(tmp_path, suffix: str) -> None:
    path = tmp_path / f"run{suffix}"
    rpc3.write(
        str(path),
        [rpc3.Channel("A", "U", dt=0.1, data=[1.0])],
        overwrite=True,
    )
    assert path.exists()
    assert not path.with_name(path.name + ".rpc").exists()


def test_write_appends_rpc_for_unknown_suffix(tmp_path) -> None:
    path = tmp_path / "run.txt"
    rpc3.write(
        str(path),
        [rpc3.Channel("A", "U", dt=0.1, data=[1.0])],
        overwrite=True,
    )
    assert not path.exists()
    assert path.with_name("run.txt.rpc").exists()


def test_scientific_notation_header_values(tmp_path) -> None:
    """Header numbers such as ``1e-3`` parse as floats, not leftover strings."""
    path = tmp_path / "sci.rpc"
    rpc3.write(
        str(path),
        [rpc3.Channel("A", "U", dt=0.001, data=[1.0, 2.0])],
        overwrite=True,
        extra_params={"MY_SCI": "1e-3", "DELTA_T": "2.5e-4"},
    )
    _, params = rpc3.read(str(path))
    assert params["MY_SCI"] == pytest.approx(1e-3)
    assert isinstance(params["MY_SCI"], float)
    assert params["DELTA_T"] == pytest.approx(2.5e-4)
    assert isinstance(params["DELTA_T"], float)


@pytest.mark.parametrize("unit", ["1", "1e-3"])
def test_numeric_looking_unit_stays_string(tmp_path, unit: str) -> None:
    """UNITS.CHAN_n must not be parsed as int/float."""
    path = tmp_path / "unit.rpc"
    rpc3.write(
        str(path),
        [rpc3.Channel("A", unit, dt=0.1, data=[1.0, 2.0])],
        overwrite=True,
    )
    channels, params = rpc3.read(str(path))
    assert channels[0].unit == unit
    assert isinstance(channels[0].unit, str)
    assert params["UNITS.CHAN_1"] == unit
    assert isinstance(params["UNITS.CHAN_1"], str)


def test_header_only_skips_channel_data(tmp_path) -> None:
    path = tmp_path / "header.rpc"
    data = np.arange(128, dtype=np.float32)
    rpc3.write(
        str(path),
        [rpc3.Channel("Accel", "m/s²", dt=0.002, data=data)],
        overwrite=True,
    )
    channels, params = rpc3.read(str(path), header_only=True)
    assert channels[0].data.size == 0
    assert channels[0].name == "Accel"
    assert channels[0].unit == "m/s²"
    assert params["CHANNELS"] == 1


def main() -> int:
    """Run pytest programmatically."""
    rpc3.progressbar = False
    return pytest.main()  # exit code



if __name__ == "__main__":
    main()
