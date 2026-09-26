# Portable NumPy semantics. Expectations checked against NumPy 2.5.3 on CPython 3.14.4.
# Scope: .npy and .npz round trips, byte-order handling, object refusal, and text load/save.

import io
import os
import tempfile

import numpy as np
import pytest
from numpy.testing import assert_array_equal


def round_trip(array):
    buffer = io.BytesIO()
    np.save(buffer, array)
    buffer.seek(0)
    return np.load(buffer)


ROUND_TRIP_SAMPLES = {
    "int64": [1, -2, 3],
    "int32": [1, -2, 3],
    "uint8": [0, 200, 255],
    "float64": [0.5, -1.25, 3.0],
    "float32": [0.5, -1.25, 3.0],
    "bool": [True, False, True],
    "complex128": [1 + 2j, -3j, 0.5],
    "str": ["ab", "cde", ""],
}


@pytest.mark.parametrize("dtype", ["int64", "int32", "uint8", "float64", "float32", "bool", "complex128", "str"])
def test_npy_round_trip_preserves_values_and_dtype(dtype):
    original = np.array(ROUND_TRIP_SAMPLES[dtype], dtype=dtype)
    loaded = round_trip(original)
    assert loaded.dtype == original.dtype
    assert loaded.shape == original.shape
    assert loaded.tolist() == original.tolist()


def test_npy_round_trip_two_dimensional():
    original = np.arange(12, dtype=np.float64).reshape(3, 4) / 4
    loaded = round_trip(original)
    assert loaded.shape == (3, 4)
    assert_array_equal(loaded, original)


def test_npy_round_trip_non_contiguous_view():
    original = np.arange(6).reshape(2, 3).T
    loaded = round_trip(original)
    assert loaded.shape == (3, 2)
    assert loaded.tolist() == [[0, 3], [1, 4], [2, 5]]


def test_npy_round_trip_zero_dimensional_and_empty():
    scalar = round_trip(np.array(3.5))
    assert scalar.shape == ()
    assert scalar == 3.5
    empty = round_trip(np.zeros((0, 3)))
    assert empty.shape == (0, 3)
    assert empty.dtype == np.float64


def test_npy_round_trip_string_array_keeps_width():
    loaded = round_trip(np.array(["ab", "cde"]))
    assert loaded.dtype == np.dtype("<U3")
    assert loaded.tolist() == ["ab", "cde"]


def test_npy_header_layout():
    buffer = io.BytesIO()
    np.save(buffer, np.arange(6, dtype=np.int32).reshape(2, 3))
    raw = buffer.getvalue()
    header = b"{'descr': '<i4', 'fortran_order': False, 'shape': (2, 3), }"
    assert raw[:10] == b"\x93NUMPY\x01\x00v\x00"
    assert raw[10 : 10 + len(header)] == header
    assert raw[127:128] == b"\n"
    assert raw[10 + len(header) : 127] == b" " * (127 - 10 - len(header))
    assert len(raw) == 128 + 6 * 4
    assert raw[128:132] == b"\x00\x00\x00\x00"
    assert raw[132:136] == b"\x01\x00\x00\x00"


def test_npy_single_byte_dtype_uses_pipe_descriptor():
    buffer = io.BytesIO()
    np.save(buffer, np.arange(4, dtype=np.uint8))
    raw = buffer.getvalue()
    assert b"'descr': '|u1'" in raw[:128]
    assert raw[-4:] == b"\x00\x01\x02\x03"


def test_npy_big_endian_integers_load_with_equal_values():
    buffer = io.BytesIO()
    np.save(buffer, np.arange(3).astype(">i4"))
    assert b"'descr': '>i4'" in buffer.getvalue()[:128]
    buffer.seek(0)
    loaded = np.load(buffer)
    assert loaded.tolist() == [0, 1, 2]
    assert loaded.dtype.kind == "i"
    assert loaded.dtype.itemsize == 4
    assert (loaded + 1).tolist() == [1, 2, 3]
    assert np.array_equal(loaded, np.array([0, 1, 2], dtype=np.int32))


def test_npy_big_endian_floats_load_with_equal_values():
    loaded = round_trip(np.array([1.5, -2.0], dtype=">f8"))
    assert loaded.tolist() == [1.5, -2.0]
    assert loaded.dtype.kind == "f"
    assert loaded.dtype.itemsize == 8
    assert (loaded * 2).tolist() == [3.0, -4.0]


def test_npy_object_array_requires_allow_pickle():
    buffer = io.BytesIO()
    np.save(buffer, np.array([1, "a"], dtype=object))
    buffer.seek(0)
    with pytest.raises(ValueError):
        np.load(buffer)


def test_save_to_path_appends_npy_extension():
    with tempfile.TemporaryDirectory() as directory:
        np.save(os.path.join(directory, "values"), np.arange(3))
        assert os.listdir(directory) == ["values.npy"]
        assert np.load(os.path.join(directory, "values.npy")).tolist() == [0, 1, 2]


def test_savez_keyword_arrays():
    buffer = io.BytesIO()
    np.savez(buffer, a=np.arange(3), b=np.eye(2))
    buffer.seek(0)
    archive = np.load(buffer)
    assert isinstance(archive, np.lib.npyio.NpzFile)
    assert sorted(archive.files) == ["a", "b"]
    assert "a" in archive
    assert archive["a"].tolist() == [0, 1, 2]
    assert archive["b"].tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert archive["b"].dtype == np.float64


def test_savez_positional_arrays_use_arr_names():
    buffer = io.BytesIO()
    np.savez(buffer, np.arange(3), np.ones(2))
    buffer.seek(0)
    archive = np.load(buffer)
    assert archive.files == ["arr_0", "arr_1"]
    assert archive["arr_0"].tolist() == [0, 1, 2]
    assert archive["arr_1"].tolist() == [1.0, 1.0]


def test_npz_missing_key_raises_key_error():
    buffer = io.BytesIO()
    np.savez(buffer, a=np.arange(3))
    buffer.seek(0)
    archive = np.load(buffer)
    with pytest.raises(KeyError):
        archive["missing"]


def test_savez_compressed_round_trip_and_context_manager():
    buffer = io.BytesIO()
    np.savez_compressed(buffer, x=np.zeros(1000), labels=np.array(["a", "bb"]))
    assert len(buffer.getvalue()) < 1000
    buffer.seek(0)
    with np.load(buffer) as archive:
        assert sorted(archive.files) == ["labels", "x"]
        assert archive["x"].shape == (1000,)
        assert archive["x"].sum() == 0.0
        assert archive["labels"].tolist() == ["a", "bb"]


def test_savez_to_path_appends_npz_extension():
    with tempfile.TemporaryDirectory() as directory:
        np.savez(os.path.join(directory, "bundle"), a=np.arange(2))
        assert os.listdir(directory) == ["bundle.npz"]
        with np.load(os.path.join(directory, "bundle.npz")) as archive:
            assert archive["a"].tolist() == [0, 1]


def test_savetxt_default_format():
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "data.txt")
        np.savetxt(path, np.array([[1.5, 2.0], [3.25, -4.0]]))
        with open(path) as handle:
            text = handle.read()
    assert text == (
        "1.500000000000000000e+00 2.000000000000000000e+00\n3.250000000000000000e+00 -4.000000000000000000e+00\n"
    )


def test_savetxt_fmt_and_delimiter():
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "data.csv")
        np.savetxt(path, np.array([[1, 2], [3, 4]]), fmt="%d", delimiter=",")
        with open(path) as handle:
            assert handle.read() == "1,2\n3,4\n"


def test_savetxt_header_uses_comment_prefix():
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "data.csv")
        np.savetxt(path, np.array([[1, 2], [3, 4]]), fmt="%d", delimiter=",", header="a,b")
        with open(path) as handle:
            assert handle.read() == "# a,b\n1,2\n3,4\n"
        np.savetxt(path, np.array([[1, 2], [3, 4]]), fmt="%d", delimiter=",", header="a,b", comments="")
        with open(path) as handle:
            assert handle.read() == "a,b\n1,2\n3,4\n"


def test_savetxt_header_footer_and_scientific_fmt():
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "data.txt")
        np.savetxt(path, np.array([[1.0, 2.5], [3.0, 4.0]]), fmt="%.3e", header="x y", footer="end")
        with open(path) as handle:
            assert handle.read() == "# x y\n1.000e+00 2.500e+00\n3.000e+00 4.000e+00\n# end\n"
        assert np.loadtxt(path).tolist() == [[1.0, 2.5], [3.0, 4.0]]


def test_savetxt_per_column_fmt():
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "data.tsv")
        np.savetxt(path, np.array([[1.0, 2.5], [3.0, 4.0]]), fmt=["%.1f", "%d"], delimiter="\t")
        with open(path) as handle:
            assert handle.read() == "1.0\t2\n3.0\t4\n"


def test_savetxt_one_dimensional_writes_one_value_per_line():
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "column.txt")
        np.savetxt(path, np.array([1.0, 2.5, 3.0]), fmt="%.2f")
        with open(path) as handle:
            assert handle.read() == "1.00\n2.50\n3.00\n"
        loaded = np.loadtxt(path)
        assert loaded.shape == (3,)
        assert loaded.tolist() == [1.0, 2.5, 3.0]


def test_savetxt_to_string_io():
    stream = io.StringIO()
    np.savetxt(stream, np.array([[1, 2], [3, 4]]), fmt="%d")
    assert stream.getvalue() == "1 2\n3 4\n"


def test_loadtxt_round_trip_through_file_with_skiprows_and_int_dtype():
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "data.csv")
        np.savetxt(path, np.array([[1, 2], [3, 4]]), fmt="%d", delimiter=",", header="a,b", comments="")
        loaded = np.loadtxt(path, delimiter=",", skiprows=1, dtype=int)
    assert loaded.dtype == np.int64
    assert loaded.tolist() == [[1, 2], [3, 4]]


def test_loadtxt_from_string_io_defaults_to_float():
    loaded = np.loadtxt(io.StringIO("1 2 3\n4 5 6\n"))
    assert loaded.dtype == np.float64
    assert loaded.shape == (2, 3)
    assert loaded.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_loadtxt_skips_default_comments():
    loaded = np.loadtxt(io.StringIO("# heading\n1,2\n# middle\n3,4\n"), delimiter=",")
    assert loaded.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_loadtxt_custom_comment_marker_strips_trailing_comment():
    loaded = np.loadtxt(io.StringIO("% c\n1 2\n3 4 % tail\n"), comments="%")
    assert loaded.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_loadtxt_skiprows():
    loaded = np.loadtxt(io.StringIO("h1\nh2\n1 2\n3 4\n"), skiprows=2)
    assert loaded.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_loadtxt_usecols():
    text = "1,2,3\n4,5,6\n"
    assert np.loadtxt(io.StringIO(text), delimiter=",", usecols=(0, 2)).tolist() == [[1.0, 3.0], [4.0, 6.0]]
    assert np.loadtxt(io.StringIO(text), delimiter=",", usecols=[2, 1]).tolist() == [[3.0, 2.0], [6.0, 5.0]]
    assert np.loadtxt(io.StringIO(text), delimiter=",", usecols=1).tolist() == [2.0, 5.0]


def test_loadtxt_int_dtype():
    loaded = np.loadtxt(io.StringIO("1,2,3\n4,5,6\n"), delimiter=",", dtype=int)
    assert loaded.dtype == np.int64
    assert loaded.tolist() == [[1, 2, 3], [4, 5, 6]]


def test_loadtxt_single_row_and_single_column_are_one_dimensional():
    assert np.loadtxt(io.StringIO("1 2\n")).shape == (2,)
    assert np.loadtxt(io.StringIO("1\n2\n3\n")).shape == (3,)


def test_loadtxt_ragged_rows_raise_value_error():
    with pytest.raises(ValueError):
        np.loadtxt(io.StringIO("1 2\n3\n"))


def test_loadtxt_non_numeric_raises_value_error():
    with pytest.raises(ValueError):
        np.loadtxt(io.StringIO("1 x\n"))


def test_genfromtxt_missing_value_becomes_nan():
    loaded = np.genfromtxt(io.StringIO("1,2,3\n4,,6\n"), delimiter=",")
    assert loaded.dtype == np.float64
    assert loaded.shape == (2, 3)
    assert np.isnan(loaded[1, 1])
    assert loaded[0].tolist() == [1.0, 2.0, 3.0]
    assert loaded[1, 0] == 4.0
    assert loaded[1, 2] == 6.0


def test_genfromtxt_skip_header():
    loaded = np.genfromtxt(io.StringIO("a,b\n1,2\n3,4\n"), delimiter=",", skip_header=1)
    assert loaded.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_genfromtxt_leading_missing_value_and_explicit_float_dtype():
    loaded = np.genfromtxt(io.StringIO("1.5,2\n,3\n"), delimiter=",", dtype=float)
    assert loaded[0].tolist() == [1.5, 2.0]
    assert np.isnan(loaded[1, 0])
    assert loaded[1, 1] == 3.0


def test_genfromtxt_skips_comments():
    loaded = np.genfromtxt(io.StringIO("# x\n1,2\n3,4\n"), delimiter=",")
    assert loaded.tolist() == [[1.0, 2.0], [3.0, 4.0]]
