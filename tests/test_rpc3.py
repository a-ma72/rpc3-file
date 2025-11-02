"""RPC3 file class unit test.

pytest --pyargs rpc3.tests

"""

# ruff: noqa: S101 PLR2004 E501

from itertools import product

import numpy as np
import pytest
import rpc3

rng = np.random.default_rng()


def test_write_empty() -> None:
    """Write empty file."""
    rpc3.write("./test.rpc", channels=[], overwrite=True)


def test_overwrite() -> None:
    """Test overwrite."""
    with pytest.raises(FileExistsError):
        rpc3.write("./test.rpc", channels=[], overwrite=False)


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
    channels_read, _ = rpc3.read("./test.rpc")
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
    # many -> AssertionError
    with pytest.raises(AssertionError):
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

def test_no_filters_returns_all_for_any_and_all():
    seq = _make_channels()
    assert len(rpc3.find_channel(seq)) == len(seq)
    assert len(rpc3.find_channel(seq, match="all")) == len(seq)

def test_only_unit_with_any_works():
    seq = _make_channels()
    hits = rpc3.find_channel(seq, unit="A", match="any")
    assert [h.name for h in hits] == ["current"]


def main() -> int:
    """Run pytest programmatically."""
    rpc3.progressbar = False
    return pytest.main()  # exit code



if __name__ == "__main__":
    main()
