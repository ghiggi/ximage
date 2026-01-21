import pytest

from ximage.patch import checks

# ############################################################################
# Tests for check_patch_size function
# ############################################################################


class TestCheckPatchSize:
    """Unit tests for check_patch_size function."""

    def test_patch_size_with_various_input_formats(self):
        """Accept patch size as int, list, tuple, or dict."""
        dimension_names = ["x", "y"]
        shape = (10, 10)
        patch_size_expected = {"x": 2, "y": 2}

        for patch_size in [2, [2, 2], (2, 2), {"x": 2, "y": 2}]:
            result = checks.check_patch_size(patch_size, dimension_names, shape)
            assert result == patch_size_expected

    def test_patch_size_with_full_dimension(self):
        """Use full dimension size when patch size is -1."""
        dimension_names = ["x", "y"]
        shape = (10, 10)
        patch_size_expected = {"x": 10, "y": 10}
        result = checks.check_patch_size(-1, dimension_names, shape)
        assert result == patch_size_expected

    def test_patch_size_raises_for_invalid_values(self):
        """Raise ValueError for invalid patch size values."""
        dimension_names = ["x", "y"]
        shape = (10, 10)
        invalid_values = [[2, 2, 2], {"x": 2, "z": 2}, {"x": 2}, 0, 1.5, 20]

        for patch_size in invalid_values:
            with pytest.raises(ValueError):
                checks.check_patch_size(patch_size, dimension_names, shape)

    def test_patch_size_raises_for_invalid_type(self):
        """Raise TypeError when patch size is invalid type."""
        dimension_names = ["x", "y"]
        shape = (10, 10)
        with pytest.raises(TypeError):
            checks.check_patch_size("2", dimension_names, shape)


# ############################################################################
# Tests for check_kernel_size function
# ############################################################################


class TestCheckKernelSize:
    """Unit tests for check_kernel_size function."""

    def test_kernel_size_with_various_input_formats(self):
        """Accept kernel size as int, list, tuple, or dict."""
        dimension_names = ["x", "y"]
        shape = (10, 10)
        kernel_size_expected = {"x": 2, "y": 2}

        for kernel_size in [2, [2, 2], (2, 2), {"x": 2, "y": 2}]:
            result = checks.check_kernel_size(kernel_size, dimension_names, shape)
            assert result == kernel_size_expected

    def test_kernel_size_with_full_dimension(self):
        """Use full dimension size when kernel size is -1."""
        dimension_names = ["x", "y"]
        shape = (10, 10)
        kernel_size_expected = {"x": 10, "y": 10}
        result = checks.check_kernel_size(-1, dimension_names, shape)
        assert result == kernel_size_expected

    def test_kernel_size_raises_for_invalid_values(self):
        """Raise ValueError for invalid kernel size values."""
        dimension_names = ["x", "y"]
        shape = (10, 10)
        invalid_values = [[2, 2, 2], {"x": 2, "z": 2}, {"x": 2}, 0, 1.5, 20]

        for kernel_size in invalid_values:
            with pytest.raises(ValueError):
                checks.check_kernel_size(kernel_size, dimension_names, shape)

    def test_kernel_size_raises_for_invalid_type(self):
        """Raise TypeError when kernel size is invalid type."""
        dimension_names = ["x", "y"]
        shape = (10, 10)
        with pytest.raises(TypeError):
            checks.check_kernel_size("2", dimension_names, shape)


# ############################################################################
# Tests for buffer and padding functions
# ############################################################################


class TestCheckBuffer:
    """Unit tests for check_buffer functions."""

    def test_check_buffer_with_various_input_formats(self):
        """Accept buffer as int, list, tuple, or dict."""
        dimension_names = ["x", "y"]
        shape = (10, 10)
        buffer_expected = {"x": 2, "y": 2}

        for buffer in [2, [2, 2], (2, 2), {"x": 2, "y": 2}]:
            result = checks.check_buffer(buffer, dimension_names, shape)
            assert result == buffer_expected

    def test_check_buffer_raises_for_non_integer(self):
        """Raise ValueError when buffer contains non-integer values."""
        dimension_names = ["x", "y"]
        shape = (10, 10)
        with pytest.raises(ValueError):
            checks.check_buffer(1.5, dimension_names, shape)


class TestCheckPadding:
    """Unit tests for check_padding functions."""

    def test_check_padding_with_various_input_formats(self):
        """Accept padding as int, list, tuple, or dict."""
        dimension_names = ["x", "y"]
        shape = (10, 10)
        padding_expected = {"x": 2, "y": 2}

        for padding in [2, [2, 2], (2, 2), {"x": 2, "y": 2}]:
            result = checks.check_padding(padding, dimension_names, shape)
            assert result == padding_expected

    def test_check_padding_raises_for_non_integer(self):
        """Raise ValueError when padding contains non-integer values."""
        dimension_names = ["x", "y"]
        shape = (10, 10)
        with pytest.raises(ValueError):
            checks.check_padding(1.5, dimension_names, shape)


def test_check_partitioning_method():
    """Validate partitioning method is None, 'sliding', or 'tiling'."""
    for method in [None, "sliding", "tiling"]:
        assert checks.check_partitioning_method(method) == method

    with pytest.raises(ValueError):
        checks.check_partitioning_method("invalid")

    with pytest.raises(TypeError):
        checks.check_partitioning_method(1)


class TestCheckStride:
    """Unit tests for check_stride function covering tiling and sliding."""

    def test_returns_none_when_no_partitioning_method(self):
        """Return None when partitioning method is None."""
        dimension_names = ["x", "y"]
        shape = (10, 10)
        result = checks.check_stride(None, dimension_names, shape, None)
        assert result is None

    def test_default_stride_for_tiling(self):
        """Default stride for 'tiling' is zero for all dims."""
        dimension_names = ["x", "y"]
        shape = (10, 10)
        expected = {"x": 0, "y": 0}
        result = checks.check_stride(None, dimension_names, shape, "tiling")
        assert result == expected

    def test_default_stride_for_sliding(self):
        """Default stride for 'sliding' is one for all dims."""
        dimension_names = ["x", "y"]
        shape = (10, 10)
        expected = {"x": 1, "y": 1}
        result = checks.check_stride(None, dimension_names, shape, "sliding")
        assert result == expected

    def test_given_stride_formats_for_both_methods(self):
        """Accept stride specified as int, list, tuple, or dict for both methods."""
        dimension_names = ["x", "y"]
        shape = (10, 10)
        expected = {"x": 2, "y": 2}

        for stride in [2, [2, 2], (2, 2), {"x": 2, "y": 2}]:
            for method in ["sliding", "tiling"]:
                result = checks.check_stride(stride, dimension_names, shape, method)
                assert result == expected

    def test_tiling_accepts_non_positive_integers(self):
        """Tiling accepts non-positive integer stride values like -1 and 0."""
        dimension_names = ["x", "y"]
        shape = (10, 10)
        checks.check_stride(-1, dimension_names, shape, "tiling")
        checks.check_stride(0, dimension_names, shape, "tiling")

    def test_tiling_rejects_non_integer_stride(self):
        """Tiling raises ValueError for non-integer stride values."""
        dimension_names = ["x", "y"]
        shape = (10, 10)
        with pytest.raises(ValueError):
            checks.check_stride(1.5, dimension_names, shape, "tiling")

    def test_sliding_rejects_non_positive_or_non_integer_stride(self):
        """Sliding requires strictly positive integer stride values."""
        dimension_names = ["x", "y"]
        shape = (10, 10)
        for stride in [-1, 0, 1.5]:
            with pytest.raises(ValueError):
                checks.check_stride(stride, dimension_names, shape, "sliding")
