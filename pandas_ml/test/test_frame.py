#!/usr/bin/env python
import pytest

import datetime
import warnings

import numpy as np
import pandas as pd

import sklearn.datasets as datasets
import sklearn.svm as svm

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestModelFrame(tm.TestCase):

    def test_version(self):
        self.assertTrue(len(pdml.__version__) > 1)

    def test_frame_instance(self):

        df = pdml.ModelFrame(datasets.load_digits())
        self.assertIsInstance(df, pdml.ModelFrame)

        train_df, test_df = df.model_selection.train_test_split()

        self.assertIsInstance(train_df, pdml.ModelFrame)
        self.assertIsInstance(test_df, pdml.ModelFrame)
        self.assertIsInstance(train_df.iloc[:, 2:3], pdml.ModelFrame)

    def test_frame_init_df_series(self):
        # initialization by dataframe and no-named series
        df = pd.DataFrame({'A': [1, 2, 3],
                           'B': [4, 5, 6],
                           'C': [7, 8, 9]},
                          index=['a', 'b', 'c'],
                          columns=['A', 'B', 'C'])
        s = pd.Series([1, 2, 3], index=['a', 'b', 'c'], name='.target')

        mdf = pdml.ModelFrame(df, target=s)
        self.assertIsInstance(mdf, pdml.ModelFrame)
        self.assertEqual(mdf.shape, (3, 4))
        tm.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        tm.assert_index_equal(mdf.columns, pd.Index(['.target', 'A', 'B', 'C']))
        tm.assert_frame_equal(mdf.data, df)
        tm.assert_series_equal(mdf.target, s, check_series_type=False)
        self.assertEqual(mdf.target.name, '.target')
        self.assertEqual(mdf.target_name, '.target')

        s = pd.Series([1, 2, 3])
        with pytest.raises(ValueError, match='data and target must have equal index'):
            mdf = pdml.ModelFrame(df, target=s)

        s = pd.Series([1, 2, 3], index=['a', 'b', 'c'], name='XXX')
        mdf = pdml.ModelFrame(df, target=s)
        self.assertIsInstance(mdf, pdml.ModelFrame)
        self.assertEqual(mdf.shape, (3, 4))
        tm.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        tm.assert_index_equal(mdf.columns, pd.Index(['XXX', 'A', 'B', 'C']))
        tm.assert_frame_equal(mdf.data, df)
        tm.assert_series_equal(mdf.target, s, check_series_type=False)
        self.assertEqual(mdf.target.name, 'XXX')
        self.assertEqual(mdf.target_name, 'XXX')

    def test_frame_init_df_str(self):
        # initialization by dataframe and str
        df = pd.DataFrame({'A': [1, 2, 3],
                           'B': [4, 5, 6],
                           'C': [7, 8, 9]},
                          index=['a', 'b', 'c'],
                          columns=['A', 'B', 'C'])

        mdf = pdml.ModelFrame(df, target='A')
        self.assertIsInstance(mdf, pdml.ModelFrame)
        self.assertEqual(mdf.shape, (3, 3))
        tm.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        tm.assert_index_equal(mdf.columns, pd.Index(['A', 'B', 'C']))
        tm.assert_frame_equal(mdf.data, df[['B', 'C']])
        tm.assert_series_equal(mdf.target, df['A'], check_series_type=False)
        self.assertEqual(mdf.target.name, 'A')
        self.assertEqual(mdf.target_name, 'A')

        msg = "Specified target 'X' is not included in data"
        with pytest.raises(ValueError, match=msg):
            mdf = pdml.ModelFrame(df, target='X')

    def test_frame_init_dict_list(self):
        # initialization by dataframe and list
        df = pd.DataFrame({'A': [1, 2, 3],
                           'B': [4, 5, 6],
                           'C': [7, 8, 9]},
                          index=['a', 'b', 'c'],
                          columns=['A', 'B', 'C'])
        s = [1, 2, 3]
        mdf = pdml.ModelFrame(df, target=s)
        self.assertIsInstance(mdf, pdml.ModelFrame)
        self.assertEqual(mdf.shape, (3, 4))
        tm.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        tm.assert_index_equal(mdf.columns, pd.Index(['.target', 'A', 'B', 'C']))
        tm.assert_frame_equal(mdf.data, df)
        expected = pd.Series([1, 2, 3], index=['a', 'b', 'c'], name='.target')
        tm.assert_series_equal(mdf.target, expected, check_series_type=False)
        self.assertEqual(mdf.target.name, '.target')
        self.assertEqual(mdf.target_name, '.target')

        df = {'A': [1, 2, 3],
              'B': [4, 5, 6],
              'C': [7, 8, 9]}
        s = [1, 2, 3]
        mdf = pdml.ModelFrame(df, target=s)
        self.assertIsInstance(mdf, pdml.ModelFrame)
        self.assertEqual(mdf.shape, (3, 4))
        tm.assert_index_equal(mdf.index, pd.Index([0, 1, 2]))
        tm.assert_index_equal(mdf.columns, pd.Index(['.target', 'A', 'B', 'C']))
        expected = pd.DataFrame(df)
        tm.assert_frame_equal(mdf.data, expected)
        expected = pd.Series([1, 2, 3], index=[0, 1, 2], name='.target')
        tm.assert_series_equal(mdf.target, expected, check_series_type=False)
        self.assertEqual(mdf.target.name, '.target')
        self.assertEqual(mdf.target_name, '.target')

        mdf = pdml.ModelFrame(df, target='A')
        self.assertIsInstance(mdf, pdml.ModelFrame)
        self.assertEqual(mdf.shape, (3, 3))
        tm.assert_index_equal(mdf.index, pd.Index([0, 1, 2]))
        tm.assert_index_equal(mdf.columns, pd.Index(['A', 'B', 'C']))
        expected = pd.DataFrame(df)
        tm.assert_frame_equal(mdf.data, expected[['B', 'C']])
        tm.assert_series_equal(mdf.target, expected['A'], check_series_type=False)
        self.assertEqual(mdf.target.name, 'A')
        self.assertEqual(mdf.target_name, 'A')

        mdf = pdml.ModelFrame({'A': [1, 2, 3],
                               'B': [4, 5, 6],
                               'C': [7, 8, 9]},
                              index=['a', 'b', 'c'],
                              columns=['A', 'B', 'C'])
        self.assertIsInstance(mdf, pdml.ModelFrame)
        self.assertEqual(mdf.shape, (3, 3))
        tm.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        tm.assert_index_equal(mdf.columns, pd.Index(['A', 'B', 'C']))
        tm.assert_frame_equal(mdf.data, mdf)
        self.assertEqual(mdf.target_name, '.target')

    def test_frame_init_df_array_series(self):
        s = pd.Series([1, 2, 3], index=['a', 'b', 'c'], name='.target')
        mdf = pdml.ModelFrame(np.array([[1, 2, 3], [4, 5, 6],
                                        [7, 8, 9]]), target=s,
                              index=['a', 'b', 'c'], columns=['A', 'B', 'C'])

        self.assertIsInstance(mdf, pdml.ModelFrame)
        self.assertEqual(mdf.shape, (3, 4))
        tm.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        tm.assert_index_equal(mdf.columns, pd.Index(['.target', 'A', 'B', 'C']))

        expected = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6],
                                          [7, 8, 9]]),
                                index=['a', 'b', 'c'], columns=['A', 'B', 'C'])
        tm.assert_frame_equal(mdf.data, expected)
        tm.assert_series_equal(mdf.target, s, check_series_type=False)
        self.assertEqual(mdf.target.name, '.target')
        self.assertEqual(mdf.target_name, '.target')

    def test_frame_init_dict_list_series_index(self):
        # initialization by dataframe and list
        df = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
        target = pd.Series([9, 8, 7], name='X', index=['a', 'b', 'c'])
        mdf = pdml.ModelFrame(df, target=target)

        self.assertIsInstance(mdf, pdml.ModelFrame)
        self.assertEqual(mdf.shape, (3, 4))
        tm.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        tm.assert_index_equal(mdf.columns, pd.Index(['X', 'A', 'B', 'C']))
        expected = pd.DataFrame(df, index=['a', 'b', 'c'])
        tm.assert_frame_equal(mdf.data, expected)
        tm.assert_series_equal(mdf.target, target, check_series_type=False)
        self.assertEqual(mdf.target.name, 'X')
        self.assertEqual(mdf.target_name, 'X')

    def test_frame_init_df_none(self):
        # initialization by dataframe and none
        df = pd.DataFrame({'A': [1, 2, 3],
                           'B': [4, 5, 6],
                           'C': [7, 8, 9]},
                          index=['a', 'b', 'c'],
                          columns=['A', 'B', 'C'])

        mdf = pdml.ModelFrame(df, target=None)
        self.assertIsInstance(mdf, pdml.ModelFrame)
        self.assertEqual(mdf.shape, (3, 3))
        tm.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        tm.assert_index_equal(mdf.columns, pd.Index(['A', 'B', 'C']))
        tm.assert_frame_equal(mdf.data, df)
        self.assertTrue(mdf.has_data())
        self.assertTrue(mdf.target is None)
        self.assertEqual(mdf.target_name, '.target')

    def test_frame_init_df_duplicated(self):
        # initialization by dataframe and duplicated target
        df = pd.DataFrame({'A': [1, 2, 3],
                           'B': [4, 5, 6],
                           'C': [7, 8, 9]},
                          columns=['A', 'B', 'C'])
        s = pd.Series([10, 11, 12], name='A')

        msg = "data and target must have unique names"
        with pytest.raises(ValueError, match=msg):
            pdml.ModelFrame(df, target=s)

        df = pdml.ModelFrame({'A': [1, 2, 3],
                              'B': [4, 5, 6],
                              'C': [7, 8, 9]},
                             columns=['A', 'B', 'C'])
        df.target = pd.Series([10, 11, 12], name='A')
        expected = pdml.ModelFrame({'A': [10, 11, 12],
                                    'B': [4, 5, 6],
                                    'C': [7, 8, 9]},
                                   columns=['A', 'B', 'C'])
        tm.assert_frame_equal(df, expected)

    def test_frame_data_none(self):
        msg = "ModelFrame must have either data or target"
        with pytest.raises(ValueError, match=msg):
            mdf = pdml.ModelFrame(None)

        msg = "target must be list-like when data is None"
        with pytest.raises(ValueError, match=msg):
            mdf = pdml.ModelFrame(None, target='X')

        # initialization without data
        s = pd.Series([1, 2, 3], index=['a', 'b', 'c'], name='.target')
        mdf = pdml.ModelFrame(None, target=s)

        self.assertIsInstance(mdf, pdml.ModelFrame)
        self.assertEqual(mdf.shape, (3, 1))
        tm.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        tm.assert_index_equal(mdf.columns, pd.Index(['.target']))
        self.assertTrue(mdf.data is None)
        self.assertFalse(mdf.has_data())
        tm.assert_series_equal(mdf.target, s, check_series_type=False)
        self.assertEqual(mdf.target.name, '.target')
        self.assertEqual(mdf.target_name, '.target')

    def test_frame_slice(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        expected = ['.target'] + iris.feature_names
        self.assertEqual(df.columns.tolist(), expected)

        s = df['.target']
        self.assertIsInstance(s, pdml.ModelSeries)
        s = s[1:5]
        self.assertIsInstance(s, pdml.ModelSeries)

        s = df[['.target']]
        self.assertIsInstance(s, pdml.ModelFrame)

    def test_frame_data_property(self):
        df = pd.DataFrame({'A': [1, 2, 3],
                           'B': [4, 5, 6],
                           'C': [7, 8, 9]},
                          index=['a', 'b', 'c'],
                          columns=['A', 'B', 'C'])
        s = pd.Series([1, 2, 3], index=['a', 'b', 'c'], name='.target')

        mdf = pdml.ModelFrame(df, target=s)
        self.assertIsInstance(mdf, pdml.ModelFrame)

        new = pd.DataFrame({'X': [1, 2, 3],
                            'Y': [4, 5, 6]},
                           index=['a', 'b', 'c'],
                           columns=['X', 'Y'])
        # set data property
        mdf.data = new

        self.assertIsInstance(mdf, pdml.ModelFrame)
        self.assertEqual(mdf.shape, (3, 3))
        tm.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        tm.assert_index_equal(mdf.columns, pd.Index(['.target', 'X', 'Y']))
        tm.assert_frame_equal(mdf.data, new)
        tm.assert_series_equal(mdf.target, s, check_series_type=False)
        self.assertEqual(mdf.target.name, '.target')
        self.assertEqual(mdf.target_name, '.target')

        new = pdml.ModelFrame({'M': [1, 2, 3],
                               'N': [4, 5, 6]},
                              index=['a', 'b', 'c'],
                              columns=['M', 'N'])

        # set data property
        mdf.data = new

        self.assertIsInstance(mdf, pdml.ModelFrame)
        self.assertEqual(mdf.shape, (3, 3))
        tm.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        tm.assert_index_equal(mdf.columns, pd.Index(['.target', 'M', 'N']))
        tm.assert_frame_equal(mdf.data, new)
        tm.assert_series_equal(mdf.target, s, check_series_type=False)
        self.assertEqual(mdf.target.name, '.target')
        self.assertEqual(mdf.target_name, '.target')

        new = pd.DataFrame({'.target': [1, 2, 3],
                            'K': [4, 5, 6]},
                           index=['a', 'b', 'c'])

        # unable to set data if passed value has the same column as the target
        msg = "Passed data has the same column name as the target '.target'"
        with pytest.raises(ValueError, match=msg):
            mdf.data = new

        # unable to set ModelFrame with target attribute
        msg = "Cannot update with ModelFrame which has target attribute"
        with pytest.raises(ValueError, match=msg):
            mdf.data = mdf

        # set delete property
        del mdf.data
        self.assertIsInstance(mdf, pdml.ModelFrame)
        self.assertEqual(mdf.shape, (3, 1))
        tm.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        tm.assert_index_equal(mdf.columns, pd.Index(['.target']))
        self.assertTrue(mdf.data is None)
        tm.assert_series_equal(mdf.target, s, check_series_type=False)
        self.assertEqual(mdf.target.name, '.target')
        self.assertEqual(mdf.target_name, '.target')

    def test_frame_data_property_series(self):
        df = pdml.ModelFrame({'A': [1, 2, 3],
                              'B': [4, 5, 6]},
                             target=[7, 8, 9],
                             index=['a', 'b', 'c'])
        df.data = df['A']
        exp = pdml.ModelFrame({'A': [1, 2, 3]},
                              target=[7, 8, 9],
                              index=['a', 'b', 'c'])
        tm.assert_frame_equal(df, exp)

        df = pdml.ModelFrame({'A': [1, 2, 3],
                              'B': [4, 5, 6]},
                             target=[7, 8, 9],
                             index=['a', 'b', 'c'])
        df.data = pd.Series([1, 2, 3], name='x', index=['a', 'b', 'c'])
        exp = pdml.ModelFrame({'x': [1, 2, 3]},
                              target=[7, 8, 9],
                              index=['a', 'b', 'c'])
        tm.assert_frame_equal(df, exp)

        df = pdml.ModelFrame({'A': [1, 2, 3],
                              'B': [4, 5, 6]},
                             target=[7, 8, 9],
                             index=['a', 'b', 'c'])
        with pytest.raises(TypeError):
            df.data = [1, 2, 3]

    def test_frame_target_property(self):
        df = pd.DataFrame({'A': [1, 2, 3],
                           'B': [4, 5, 6],
                           'C': [7, 8, 9]},
                          index=['a', 'b', 'c'],
                          columns=['A', 'B', 'C'])
        s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
        mdf = pdml.ModelFrame(df, target=s)

        new = pd.Series([4, 5, 6], index=['a', 'b', 'c'], name='.target')
        # set target property
        mdf.target = new

        self.assertIsInstance(mdf, pdml.ModelFrame)
        self.assertEqual(mdf.shape, (3, 4))
        tm.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        tm.assert_index_equal(mdf.columns, pd.Index(['.target', 'A', 'B', 'C']))
        tm.assert_frame_equal(mdf.data, df)
        tm.assert_series_equal(mdf.target, new, check_series_type=False)
        self.assertEqual(mdf.target.name, '.target')

        with tm.assert_produces_warning(UserWarning):
            new = pd.Series([4, 5, 6], index=['a', 'b', 'c'], name='xxx')
            # set target property
            mdf.target = new

            self.assertIsInstance(mdf, pdml.ModelFrame)
            self.assertEqual(mdf.shape, (3, 4))
            tm.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
            tm.assert_index_equal(mdf.columns, pd.Index(['.target', 'A', 'B', 'C']))
            tm.assert_frame_equal(mdf.data, df)

            exp_target = pd.Series(new, name='.target')
            tm.assert_series_equal(mdf.target, exp_target, check_series_type=False)
            self.assertEqual(mdf.target.name, '.target')
            self.assertEqual(mdf.target_name, '.target')

        new = pd.Series([4, 5, 6], name='.target')
        with pytest.raises(ValueError, match='data and target must have equal index'):
            mdf.target = new

        # set target property
        mdf.target = [7, 8, 9]

        self.assertIsInstance(mdf, pdml.ModelFrame)
        self.assertEqual(mdf.shape, (3, 4))
        tm.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        tm.assert_index_equal(mdf.columns, pd.Index(['.target', 'A', 'B', 'C']))
        tm.assert_frame_equal(mdf.data, df)
        expected = pd.Series([7, 8, 9], index=['a', 'b', 'c'], name='.target')
        tm.assert_series_equal(mdf.target, expected, check_series_type=False)
        self.assertEqual(mdf.target.name, '.target')

        # if pdml.compat._PANDAS_ge_023:
        #     msg = 'Length of passed values is 2, index implies 3'
        # else:
        #     msg = 'Wrong number of items passed 2, placement implies 3'

        try:  # Tests if incorrect number of values is passed for target
            mdf.target = [1, 2]
        except ValueError as e:
            pass

        # with pytest.raises(ValueError, match=msg):
        #     mdf.target = [1, 2]

        # set target property
        mdf.target = None

        self.assertIsInstance(mdf, pdml.ModelFrame)
        self.assertEqual(mdf.shape, (3, 3))
        tm.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        tm.assert_index_equal(mdf.columns, pd.Index(['A', 'B', 'C']))
        tm.assert_frame_equal(mdf.data, df)
        self.assertEqual(mdf.target_name, '.target')

    def test_frame_delete_proparty(self):
        mdf = pdml.ModelFrame(None, target=[1, 2, 3])
        msg = 'ModelFrame must have either data or target'

        with pytest.raises(ValueError, match=msg):
            del mdf.target

        with pytest.raises(ValueError, match=msg):
            mdf.target = None

        mdf = pdml.ModelFrame([1, 2, 3])
        msg = 'ModelFrame must have either data or target'

        with pytest.raises(ValueError, match=msg):
            del mdf.data

        with pytest.raises(ValueError, match=msg):
            mdf.data = None

    def test_frame_target_object(self):
        df = pd.DataFrame({datetime.datetime(2014, 1, 1): [1, 2, 3],
                           datetime.datetime(2015, 1, 1): [4, 5, 6],
                           datetime.datetime(2016, 1, 1): [7, 8, 9]},
                          index=['a', 'b', 'c'])
        mdf = pdml.ModelFrame(df, target=datetime.datetime(2016, 1, 1))

        self.assertIsInstance(mdf, pdml.ModelFrame)
        self.assertEqual(mdf.shape, (3, 3))
        tm.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        expected = pd.DatetimeIndex(['2014-01-01', '2015-01-01', '2016-01-01'])
        tm.assert_index_equal(mdf.columns, expected)
        tm.assert_frame_equal(mdf.data, df.iloc[:, :2])
        expected = pd.Series([7, 8, 9], index=['a', 'b', 'c'],
                             name=pd.Timestamp('2016-01-01'))
        tm.assert_series_equal(mdf.target, expected, check_series_type=False)
        self.assertEqual(mdf.target.name, datetime.datetime(2016, 1, 1))
        self.assertEqual(mdf.target_name, datetime.datetime(2016, 1, 1))

    def test_frame_target_object_set(self):
        df = pd.DataFrame({datetime.datetime(2014, 1, 1): [1, 2, 3],
                           datetime.datetime(2015, 1, 1): [4, 5, 6],
                           datetime.datetime(2016, 1, 1): [7, 8, 9]},
                          index=['a', 'b', 'c'])
        mdf = pdml.ModelFrame(df)
        mdf.target = pd.Series(['A', 'B', 'C'], index=['a', 'b', 'c'], name=5)

        self.assertIsInstance(mdf, pdml.ModelFrame)
        self.assertEqual(mdf.shape, (3, 4))

        tm.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))

        expected = pd.Index([5, datetime.datetime(2014, 1, 1),
                             datetime.datetime(2015, 1, 1), datetime.datetime(2016, 1, 1)])
        tm.assert_index_equal(mdf.columns, expected)
        # tm.assert_frame_equal(mdf.data, df)
        expected = pd.Series(['A', 'B', 'C'], index=['a', 'b', 'c'], name=5)
        tm.assert_series_equal(mdf.target, expected, check_series_type=False)
        self.assertEqual(mdf.target.name, 5)

        # name will be ignored if ModelFrame already has a target
        mdf.target = pd.Series([10, 11, 12], index=['a', 'b', 'c'], name='X')
        self.assertIsInstance(mdf, pdml.ModelFrame)
        self.assertEqual(mdf.shape, (3, 4))
        tm.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        expected = pd.Index([5,
                             datetime.datetime(2014, 1, 1),
                             datetime.datetime(2015, 1, 1),
                             datetime.datetime(2016, 1, 1)])
        tm.assert_index_equal(mdf.columns, expected)
        # tm.assert_frame_equal(mdf.data, df)
        expected = pd.Series([10, 11, 12], index=['a', 'b', 'c'], name=5)
        tm.assert_series_equal(mdf.target, expected, check_series_type=False)
        self.assertEqual(mdf.target.name, 5)

    def test_predict_proba(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        models = ['SVC']
        for model in models:
            mod1 = getattr(df.svm, model)(probability=True,
                                          random_state=self.random_state)
            mod2 = getattr(svm, model)(probability=True,
                                       random_state=self.random_state)

            df.fit(mod1)
            mod2.fit(iris.data, iris.target)

            result = df.predict(mod1)
            expected = mod2.predict(iris.data)

            self.assertIsInstance(result, pdml.ModelSeries)
            self.assert_numpy_array_almost_equal(result.values, expected)

            result = df.predict_proba(mod1)
            expected = mod2.predict_proba(iris.data)

            self.assertIsInstance(result, pdml.ModelFrame)
            tm.assert_index_equal(result.index, df.index)
            self.assert_numpy_array_almost_equal(result.values, expected)
            self.assert_numpy_array_almost_equal(df.proba.values, expected)

            result = df.predict_log_proba(mod1)
            expected = mod2.predict_log_proba(iris.data)

            self.assertIsInstance(result, pdml.ModelFrame)
            tm.assert_index_equal(result.index, df.index)
            self.assert_numpy_array_almost_equal(result.values, expected)
            self.assert_numpy_array_almost_equal(df.log_proba.values, expected)

            result = df.decision_function(mod1)
            expected = mod2.decision_function(iris.data)

            self.assertIsInstance(result, pdml.ModelFrame)
            tm.assert_index_equal(result.index, df.index)
            self.assert_numpy_array_almost_equal(result.values, expected)
            self.assert_numpy_array_almost_equal(df.decision.values, expected)

            # not reset if estimator is identical
            df.fit(mod1)
            self.assertFalse(df._predicted is None)
            self.assertFalse(df._proba is None)
            self.assertFalse(df._log_proba is None)
            self.assertFalse(df._decision is None)

            # reset estimator
            mod3 = getattr(df.svm, model)(probability=True,
                                          random_state=self.random_state)
            df.fit(mod3)
            self.assertTrue(df._predicted is None)
            self.assertTrue(df._proba is None)
            self.assertTrue(df._log_proba is None)
            self.assertTrue(df._decision is None)

    def test_predict_automatic(self):
        with warnings.catch_warnings():
            warnings.simplefilter("always", UserWarning)

            iris = datasets.load_iris()
            df = pdml.ModelFrame(iris)

            model = 'SVC'

            df = pdml.ModelFrame(iris)
            mod1 = getattr(df.svm, model)(probability=True,
                                          random_state=self.random_state)
            mod2 = getattr(svm, model)(probability=True,
                                       random_state=self.random_state)

            df.fit(mod1)
            mod2.fit(iris.data, iris.target)

            # test automatically calls related methods
            with tm.assert_produces_warning(UserWarning):
                result = df.predicted
            expected = mod2.predict(iris.data)

            self.assertIsInstance(result, pdml.ModelSeries)
            self.assert_numpy_array_almost_equal(result.values, expected)

            # with tm.assert_produces_warning(UserWarning):
            result = df.proba
            expected = mod2.predict_proba(iris.data)

            self.assertIsInstance(result, pdml.ModelFrame)
            tm.assert_index_equal(result.index, df.index)
            self.assert_numpy_array_almost_equal(result.values, expected)

            with tm.assert_produces_warning(UserWarning):
                result = df.log_proba
            expected = mod2.predict_log_proba(iris.data)

            self.assertIsInstance(result, pdml.ModelFrame)
            tm.assert_index_equal(result.index, df.index)
            self.assert_numpy_array_almost_equal(result.values, expected)

            # with tm.assert_produces_warning(UserWarning):
            result = df.decision
            expected = mod2.decision_function(iris.data)

            self.assertIsInstance(result, pdml.ModelFrame)
            tm.assert_index_equal(result.index, df.index)
            self.assert_numpy_array_almost_equal(result.values, expected)

        warnings.simplefilter("default")

    def test_frame_metadata(self):
        df = pd.DataFrame({'A': [1, 2, 3],
                           'B': [4, 5, 6],
                           'C': [7, 8, 9]},
                          index=['a', 'b', 'c'],
                          columns=['A', 'B', 'C'])
        s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
        mdf = pdml.ModelFrame(df, target=s)

        self.assertEqual(mdf.target_name, '.target')
        sliced = mdf.iloc[1:2, ]
        self.assertIsInstance(sliced, pdml.ModelFrame)
        self.assertTrue(sliced.has_target())
        self.assertEqual(sliced.target_name, '.target')

        df = pd.DataFrame({'A': [1, 2, 3],
                           'B': [4, 5, 6],
                           'C': [7, 8, 9]},
                          index=['a', 'b', 'c'],
                          columns=['A', 'B', 'C'])
        mdf = pdml.ModelFrame(df, target='A')

        self.assertEqual(mdf.target_name, 'A')
        sliced = mdf.loc[['a', 'b'], :]
        self.assertIsInstance(sliced, pdml.ModelFrame)
        self.assertTrue(sliced.has_target())
        self.assertEqual(sliced.target_name, 'A')

        sliced = mdf.iloc[[1, 2], :]
        self.assertIsInstance(sliced, pdml.ModelFrame)
        self.assertTrue(sliced.has_target())
        self.assertEqual(sliced.target_name, 'A')


class TestModelFrameMultiTarges(tm.TestCase):

    def test_frame_instance(self):
        df = pdml.ModelFrame(datasets.load_digits())
        self.assertIsInstance(df, pdml.ModelFrame)

        train_df, test_df = df.model_selection.train_test_split()

        self.assertIsInstance(train_df, pdml.ModelFrame)
        self.assertIsInstance(test_df, pdml.ModelFrame)
        self.assertIsInstance(train_df.iloc[:, 2:3], pdml.ModelFrame)

    def test_frame_init_df_df(self):
        # initialization by dataframe and dataframe
        df = pd.DataFrame({'A': [1, 2, 3],
                           'B': [4, 5, 6],
                           'C': [7, 8, 9]},
                          index=['a', 'b', 'c'],
                          columns=['A', 'B', 'C'])
        target = pd.DataFrame({'t1': [10, 11, 12],
                               't2': [13, 14, 15]},
                              index=['a', 'b', 'c'])
        mdf = pdml.ModelFrame(df, target=target)

        self.assertIsInstance(mdf, pdml.ModelFrame)
        self.assertEqual(mdf.shape, (3, 5))
        expected = pd.DataFrame({'t1': [10, 11, 12],
                                 't2': [13, 14, 15],
                                 'A': [1, 2, 3],
                                 'B': [4, 5, 6],
                                 'C': [7, 8, 9]},
                                index=['a', 'b', 'c'],
                                columns=['t1', 't2', 'A', 'B', 'C'])
        tm.assert_frame_equal(mdf, expected)
        tm.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        tm.assert_index_equal(mdf.columns,
                              pd.Index(['t1', 't2', 'A', 'B', 'C']))
        tm.assert_frame_equal(mdf.data, df)
        tm.assert_frame_equal(mdf.target, target)
        tm.assert_index_equal(mdf.target.columns, pd.Index(['t1', 't2']))
        tm.assert_index_equal(mdf.target_name, pd.Index(['t1', 't2']))
        self.assertTrue(mdf.has_multi_targets())

        target = pd.DataFrame({'t1': [10, 11, 12], 't2': [13, 14, 15]})
        msg = 'data and target must have equal index'
        with pytest.raises(ValueError, match=msg):
            mdf = pdml.ModelFrame(df, target=target)

        # single column DataFrame will results in single target column
        target = pd.DataFrame({'t1': [10, 11, 12]}, index=['a', 'b', 'c'])
        mdf = pdml.ModelFrame(df, target=target)
        self.assertIsInstance(mdf, pdml.ModelFrame)
        self.assertEqual(mdf.shape, (3, 4))
        tm.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        tm.assert_index_equal(mdf.columns, pd.Index(['t1', 'A', 'B', 'C']))
        tm.assert_frame_equal(mdf.data, df)

        target = pd.Series([10, 11, 12], name='t1', index=['a', 'b', 'c'])
        tm.assert_series_equal(mdf.target, target, check_series_type=False)
        self.assertEqual(mdf.target_name, 't1')

    def test_frame_init_df_target_setter(self):
        # initialization by dataframe and dataframe
        df = pd.DataFrame({'A': [1, 2, 3],
                           'B': [4, 5, 6],
                           'C': [7, 8, 9]},
                          index=['a', 'b', 'c'],
                          columns=['A', 'B', 'C'])
        mdf = pdml.ModelFrame(df)
        self.assertFalse(mdf.has_target())
        target = pd.DataFrame({'t1': [10, 11, 12],
                               't2': [13, 14, 15]},
                              index=['a', 'b', 'c'])
        mdf.target = target

        self.assertIsInstance(mdf, pdml.ModelFrame)
        self.assertEqual(mdf.shape, (3, 5))
        expected = pd.DataFrame({'t1': [10, 11, 12],
                                 't2': [13, 14, 15],
                                 'A': [1, 2, 3],
                                 'B': [4, 5, 6],
                                 'C': [7, 8, 9]},
                                index=['a', 'b', 'c'],
                                columns=['t1', 't2', 'A', 'B', 'C'])
        tm.assert_frame_equal(mdf, expected)
        tm.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        tm.assert_index_equal(mdf.columns,
                              pd.Index(['t1', 't2', 'A', 'B', 'C']))
        tm.assert_frame_equal(mdf.data, df)
        tm.assert_frame_equal(mdf.target, target)
        tm.assert_index_equal(mdf.target.columns, pd.Index(['t1', 't2']))
        tm.assert_index_equal(mdf.target_name, pd.Index(['t1', 't2']))
        self.assertTrue(mdf.has_multi_targets())

        target = pd.DataFrame({'x1': [20, 21, 22],
                               'x2': [23, 24, 25]},
                              index=['a', 'b', 'c'])

        with tm.assert_produces_warning(UserWarning):
            # when the target has the same length as the target_name,
            # is renamed to existing target ['t1', 't2']
            mdf.target = target

        self.assertIsInstance(mdf, pdml.ModelFrame)
        self.assertEqual(mdf.shape, (3, 5))
        expected = pd.DataFrame({'t1': [20, 21, 22],
                                 't2': [23, 24, 25],
                                 'A': [1, 2, 3],
                                 'B': [4, 5, 6],
                                 'C': [7, 8, 9]},
                                index=['a', 'b', 'c'],
                                columns=['t1', 't2', 'A', 'B', 'C'])
        tm.assert_frame_equal(mdf, expected)
        tm.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        tm.assert_index_equal(mdf.columns,
                              pd.Index(['t1', 't2', 'A', 'B', 'C']))
        tm.assert_frame_equal(mdf.data, df)
        expected = pd.DataFrame({'t1': [20, 21, 22],
                                 't2': [23, 24, 25]},
                                index=['a', 'b', 'c'])
        tm.assert_frame_equal(mdf.target, expected)
        tm.assert_index_equal(mdf.target.columns, pd.Index(['t1', 't2']))
        tm.assert_index_equal(mdf.target_name, pd.Index(['t1', 't2']))
        self.assertTrue(mdf.has_multi_targets())

        target = pd.DataFrame({'x1': [20, 21, 22],
                               'x2': [23, 24, 25],
                               'x3': [25, 26, 27]},
                              index=['a', 'b', 'c'])

        # when the target has the different length as the target_name,
        # target is being replaced
        mdf.target = target

        self.assertIsInstance(mdf, pdml.ModelFrame)
        self.assertEqual(mdf.shape, (3, 6))
        expected = pd.DataFrame({'x1': [20, 21, 22],
                                 'x2': [23, 24, 25],
                                 'x3': [25, 26, 27],
                                 'A': [1, 2, 3],
                                 'B': [4, 5, 6],
                                 'C': [7, 8, 9]},
                                index=['a', 'b', 'c'],
                                columns=['x1', 'x2', 'x3', 'A', 'B', 'C'])
        tm.assert_frame_equal(mdf, expected)
        tm.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        tm.assert_index_equal(mdf.columns,
                              pd.Index(['x1', 'x2', 'x3', 'A', 'B', 'C']))
        tm.assert_frame_equal(mdf.data, df)
        tm.assert_frame_equal(mdf.target, target)
        tm.assert_index_equal(mdf.target.columns,
                              pd.Index(['x1', 'x2', 'x3']))
        tm.assert_index_equal(mdf.target_name, pd.Index(['x1', 'x2', 'x3']))
        self.assertTrue(mdf.has_multi_targets())

    def test_frame_init_df_duplicated_columns(self):
        # initialization by dataframe and dataframe which have same columns
        df = pd.DataFrame({'A': [1, 2, 3],
                           'B': [4, 5, 6],
                           'C': [7, 8, 9]})
        target = pd.DataFrame({'A': [10, 11, 12],
                               'B': [13, 14, 15]})

        mdf = pdml.ModelFrame(df, target=target)

        cols = pd.MultiIndex.from_tuples([('.target', 'A'),
                                          ('.target', 'B'),
                                          ('.data', 'A'),
                                          ('.data', 'B'),
                                          ('.data', 'C')])
        expected = pd.DataFrame({('.target', 'A'): [10, 11, 12],
                                 ('.target', 'B'): [13, 14, 15],
                                 ('.data', 'A'): [1, 2, 3],
                                 ('.data', 'B'): [4, 5, 6],
                                 ('.data', 'C'): [7, 8, 9]},
                                columns=cols)
        tm.assert_frame_equal(mdf, expected)
