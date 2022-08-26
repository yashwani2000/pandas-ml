#!/usr/bin/env python

import numpy as np
import pandas.util.testing as tm
from _pytest import warnings
from pandas import DataFrame
from pandas._libs.lib import no_default
from pandas._testing.asserters import _get_tol_from_less_precise, _check_isinstance, raise_assert_detail
from pandas.util._exceptions import find_stack_level

# from pandas_ml.compat import plotting
from pandas.util.testing import (assert_produces_warning,  # noqa
                                 close, RNGContext,  # noqa
                                 assert_index_equal,  # noqa
                                 assert_series_equal,  # noqa
                                 # assert_frame_equal,  # noqa
                                 assert_numpy_array_equal)  # noqa


# try:
# _flatten = plotting._flatten
# except AttributeError:
#     import pandas.plotting._tools
#     _flatten = pandas.plotting._tools._flatten


def assert_frame_equal(left,
                       right,
                       check_dtype=True,
                       check_index_type="equiv",
                       check_column_type="equiv",
                       check_frame_type=True,
                       check_less_precise=no_default,
                       check_names=True,
                       by_blocks=False,
                       check_exact=False,
                       check_datetimelike_compat=False,
                       check_categorical=True,
                       check_like=False,
                       check_freq=True,
                       check_flags=True,
                       rtol=1.0e-5,
                       atol=1.0e-8,
                       obj="DataFrame",
                       ):
    """
    Check that left and right DataFrame are equal.

    This function is intended to compare two DataFrames and output any
    differences. Is mostly intended for use in unit tests.
    Additional parameters allow varying the strictness of the
    equality checks performed.

    Parameters
    ----------
    left : DataFrame
        First DataFrame to compare.
    right : DataFrame
        Second DataFrame to compare.
    check_dtype : bool, default True
        Whether to check the DataFrame dtype is identical.
    check_index_type : bool or {'equiv'}, default 'equiv'
        Whether to check the Index class, dtype and inferred_type
        are identical.
    check_column_type : bool or {'equiv'}, default 'equiv'
        Whether to check the columns class, dtype and inferred_type
        are identical. Is passed as the ``exact`` argument of
        :func:`assert_index_equal`.
    check_frame_type : bool, default True
        Whether to check the DataFrame class is identical.
    check_less_precise : bool or int, default False
        Specify comparison precision. Only used when check_exact is False.
        5 digits (False) or 3 digits (True) after decimal points are compared.
        If int, then specify the digits to compare.

        When comparing two numbers, if the first number has magnitude less
        than 1e-5, we compare the two numbers directly and check whether
        they are equivalent within the specified precision. Otherwise, we
        compare the **ratio** of the second number to the first number and
        check whether it is equivalent to 1 within the specified precision.

        .. deprecated:: 1.1.0
           Use `rtol` and `atol` instead to define relative/absolute
           tolerance, respectively. Similar to :func:`math.isclose`.
    check_names : bool, default True
        Whether to check that the `names` attribute for both the `index`
        and `column` attributes of the DataFrame is identical.
    by_blocks : bool, default False
        Specify how to compare internal data. If False, compare by columns.
        If True, compare by blocks.
    check_exact : bool, default False
        Whether to compare number exactly.
    check_datetimelike_compat : bool, default False
        Compare datetime-like which is comparable ignoring dtype.
    check_categorical : bool, default True
        Whether to compare internal Categorical exactly.
    check_like : bool, default False
        If True, ignore the order of index & columns.
        Note: index labels must match their respective rows
        (same as in columns) - same labels must be with the same data.
    check_freq : bool, default True
        Whether to check the `freq` attribute on a DatetimeIndex or TimedeltaIndex.

        .. versionadded:: 1.1.0
    check_flags : bool, default True
        Whether to check the `flags` attribute.
    rtol : float, default 1e-5
        Relative tolerance. Only used when check_exact is False.

        .. versionadded:: 1.1.0
    atol : float, default 1e-8
        Absolute tolerance. Only used when check_exact is False.

        .. versionadded:: 1.1.0
    obj : str, default 'DataFrame'
        Specify object name being compared, internally used to show appropriate
        assertion message.

    See Also
    --------
    assert_series_equal : Equivalent method for asserting Series equality.
    DataFrame.equals : Check DataFrame equality.

    Examples
    --------
    This example shows comparing two DataFrames that are equal
    but with columns of differing dtypes.

    >>> from pandas.testing import assert_frame_equal
    >>> df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    >>> df2 = pd.DataFrame({'a': [1, 2], 'b': [3.0, 4.0]})

    df1 equals itself.

    >>> assert_frame_equal(df1, df1)

    df1 differs from df2 as column 'b' is of a different type.

    >>> assert_frame_equal(df1, df2)
    Traceback (most recent call last):
    ...
    AssertionError: Attributes of DataFrame.iloc[:, 1] (column name="b") are different

    Attribute "dtype" are different
    [left]:  int64
    [right]: float64

    Ignore differing dtypes in columns with check_dtype.

    >>> assert_frame_equal(df1, df2, check_dtype=False)
    """
    __tracebackhide__ = True

    if check_less_precise is not no_default:
        warnings.warn(
            "The 'check_less_precise' keyword in testing.assert_*_equal "
            "is deprecated and will be removed in a future version. "
            "You can stop passing 'check_less_precise' to silence this warning.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        rtol = atol = _get_tol_from_less_precise(check_less_precise)

    # instance validation
    _check_isinstance(left, right, DataFrame)

    if check_frame_type:
        assert isinstance(left, type(right))
        # assert_class_equal(left, right, obj=obj)

    # shape comparison
    if left.shape != right.shape:
        raise_assert_detail(
            obj, f"{obj} shape mismatch", f"{repr(left.shape)}", f"{repr(right.shape)}"
        )

    if check_flags:
        assert left.flags == right.flags, f"{repr(left.flags)} != {repr(right.flags)}"

    # index comparison
    assert_index_equal(
        left.index,
        right.index,
        exact=check_index_type,
        check_names=check_names,
        check_exact=check_exact,
        check_categorical=check_categorical,
        check_order=not check_like,
        rtol=rtol,
        atol=atol,
        obj=f"{obj}.index",
    )

    # column comparison
    assert_index_equal(
        left.columns,
        right.columns,
        exact=check_column_type,
        check_names=check_names,
        check_exact=check_exact,
        check_categorical=check_categorical,
        check_order=not check_like,
        rtol=rtol,
        atol=atol,
        obj=f"{obj}.columns",
    )

    if check_like:
        left, right = left.reindex_like(right), right

    # compare by blocks
    if by_blocks:
        rblocks = right._to_dict_of_blocks()
        lblocks = left._to_dict_of_blocks()
        for dtype in list(set(list(lblocks.keys()) + list(rblocks.keys()))):
            assert dtype in lblocks
            assert dtype in rblocks
            assert_frame_equal(
                lblocks[dtype], rblocks[dtype], check_dtype=check_dtype, obj=obj
            )

    # compare by columns
    else:
        for i, col in enumerate(left.columns):
            # We have already checked that columns match, so we can do
            #  fast location-based lookups
            lcol = left._ixs(i, axis=1)
            rcol = right._ixs(i, axis=1)

            # GH #38183
            # use check_index=False, because we do not want to run
            # assert_index_equal for each column,
            # as we already checked it for the whole dataframe before.
            assert_series_equal(
                lcol,
                rcol,
                check_dtype=check_dtype,
                check_series_type=False,
                check_index_type=check_index_type,
                check_exact=check_exact,
                check_names=check_names,
                check_datetimelike_compat=check_datetimelike_compat,
                check_categorical=check_categorical,
                check_freq=check_freq,
                obj=f'{obj}.iloc[:, {i}] (column name="{col}")',
                rtol=rtol,
                atol=atol,
                check_index=False,
                check_flags=False,
            )


class TestCase(object):

    @property
    def random_state(self):
        return np.random.RandomState(1234)

    def format(self, val):
        return '{} (type: {})'.format(val, type(val))

    def format_values(self, left, right):
        fmt = """Input vaues are different:
Left: {}
Right: {}
"""
        return fmt.format(self.format(left), self.format(right))

    def assertEqual(self, left, right):
        assert left == right, self.format_values(left, right)

    def assertIs(self, left, right):
        assert left is right, self.format_values(left, right)

    def assertAlmostEqual(self, left, right):
        assert tm.assert_almost_equal(left, right), self.format_values(left, right)

    def assertIsNone(self, left):
        assert left is None, self.format(left)

    def assertTrue(self, left):
        assert left is True or left is np.bool_(True), self.format(left)

    def assertFalse(self, left):
        assert left is False or left is np.bool_(False), self.format(left)

    def assertIsInstance(self, instance, klass):
        assert isinstance(instance, klass), self.format(instance)

    def assert_numpy_array_almost_equal(self, a, b):
        return np.testing.assert_array_almost_equal(a, b)


class PlottingTestCase(TestCase):

    def teardown_method(self):
        tm.close()

    def _check_axes_shape(self, axes, axes_num=None, layout=None, figsize=(8.0, 6.0)):
        """
        Check expected number of axes is drawn in expected layout

        Parameters
        ----------
        axes : matplotlib Axes object, or its list-like
        axes_num : number
            expected number of axes. Unnecessary axes should be set to invisible.
        layout :  tuple
            expected layout, (expected number of rows , columns)
        figsize : tuple
            expected figsize. default is matplotlib default
        """

        # derived from pandas.tests.test_graphics
        visible_axes = self._flatten_visible(axes)

        if axes_num is not None:
            self.assertEqual(len(visible_axes), axes_num)
            for ax in visible_axes:
                # check something drawn on visible axes
                self.assertTrue(len(ax.get_children()) > 0)

        if layout is not None:
            result = self._get_axes_layout(axes.flatten())
            self.assertEqual(result, layout)

        if figsize is not None:
            self.assert_numpy_array_equal(np.round(visible_axes[0].figure.get_size_inches()),
                                          np.array(figsize))

    def _get_axes_layout(self, axes):
        x_set = set()
        y_set = set()
        for ax in axes:
            # check axes coordinates to estimate layout
            points = ax.get_position().get_points()
            x_set.add(points[0][0])
            y_set.add(points[0][1])
        return (len(y_set), len(x_set))

    def _flatten_visible(self, axes):
        """
        Flatten axes, and filter only visible

        Parameters
        ----------
        axes : matplotlib Axes object, or its list-like

        """
        axes = axes.flatten()
        axes = [ax for ax in axes if ax.get_visible()]
        return axes
