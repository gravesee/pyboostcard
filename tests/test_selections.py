# type: ignore

from pyboostcard.selections import *
import numpy as np


class TestIntervalSelection:

    x = np.array(range(5))

    def test_interval_cc(self):
        i = Interval((0.0, 4.0), (True, True))
        np.testing.assert_equal(i.in_selection(self.x), np.array([True, True, True, True, True]))

    def test_interval_oo(self):
        i = Interval((0.0, 4.0), (False, False))
        np.testing.assert_equal(i.in_selection(self.x), np.array([False, True, True, True, False]))

    def test_interval_oc(self):
        i = Interval((0.0, 4.0), (False, True))
        np.testing.assert_equal(i.in_selection(self.x), np.array([False, True, True, True, True]))

    def test_interval_co(self):
        i = Interval((0.0, 4.0), (True, False))
        np.testing.assert_equal(i.in_selection(self.x), np.array([True, True, True, True, False]))


class TestOverrideSelection:
    def test_override(self):
        z = Override(-1)
        np.testing.assert_equal(z.in_selection(np.array([-1.0])), np.array([True]))
        np.testing.assert_equal(z.in_selection(np.array([0.0])), np.array([False]))
        np.testing.assert_equal(z.in_selection(np.array([0.0, -1.0])), np.array([False, True]))


class TestMissingSelection:
    def test_missing(self):
        z = Missing()
        np.testing.assert_equal(z.in_selection(np.array([-1.0])), np.array([False]))
        np.testing.assert_equal(z.in_selection(np.array([np.nan])), np.array([True]))
        np.testing.assert_equal(z.in_selection(np.array([0.0, np.nan])), np.array([False, True]))


class TestIdentitySelection:
    def test_identity(self):
        z = Identity()
        np.testing.assert_equal(z.in_selection(np.array([0, 1, 2, 3])), np.array([True, True, True, True]))


class TestStaticMethods:
    def test_bounds_from_string(self):
        assert Selection.bounds_from_string("[]") == (True, True)
        assert Selection.bounds_from_string("[)") == (True, False)
        assert Selection.bounds_from_string("(]") == (False, True)
        assert Selection.bounds_from_string("()") == (False, False)
    
    def test_selection_from_dict(self):
        i = Selection.from_dict({"type":"interval", "values":(0.0, 10.0), "bounds":"[]", "order":0, "mono":1})
        assert i.__dict__ == Interval((0.0, 10.0), (True, True), 0, 1).__dict__
    
        o = Selection.from_dict({"type":"override", "override":-1., "order":0})
        assert o.__dict__ == Override(-1., 0).__dict__

        m = Selection.from_dict({"type":"missing", "order":0})
        assert m.__dict__ == Missing(0).__dict__
    
    def test_selection_from_json(self):
        i = Selection.from_json('{"type":"interval", "bounds":"[]", "values": [0.0, 10.0], "order":0, "mono":1}')
        assert i.__dict__ == Interval((0.0, 10.0), (True, True), 0, 1).__dict__
    
        o = Selection.from_json('{"type":"override", "override":-1.0, "order":0}')
        assert o.__dict__ == Override(-1., 0).__dict__

        m = Selection.from_json('{"type":"missing", "order":0}')
        assert m.__dict__ == Missing(0).__dict__
fa