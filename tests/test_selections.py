# type: ignore

from pyboostcard.selections import *
import numpy as np

class TestIntervalSelection:

    x = np.array(list(range(5)))

    def test_interval_oo(self):
        i = Interval((0.0, 4.0), (True, True))        
        np.testing.assert_equal(i.in_selection(self.x), np.array([True, True, True, True, True]))

    def test_interval_cc(self):
        i = Interval((0.0, 4.0), (False, False))        
        np.testing.assert_equal(i.in_selection(self.x), np.array([False, True, True, True, False]))
    
    def test_interval_oc(self):
        i = Interval((0.0, 4.0), (False, True))        
        np.testing.assert_equal(i.in_selection(self.x), np.array([False, True, True, True, True]))
    
    def test_interval_co(self):
        i = Interval((0.0, 4.0), (True, False))
        np.testing.assert_equal(i.in_selection(self.x), np.array([True, True, True, True, False]))

class TestExceptionSelection:
    def test_exception(self):
        z = Exception(-1)
        np.testing.assert_equal(z.in_selection(np.array([-1.])), np.array([True]))
        np.testing.assert_equal(z.in_selection(np.array([0.])), np.array([False]))
        np.testing.assert_equal(z.in_selection(np.array([0., -1.])), np.array([False, True]))

class TestMissingSelection:
    def test_missing(self):
        z = Missing()
        np.testing.assert_equal(z.in_selection(np.array([-1.])), np.array([False]))
        np.testing.assert_equal(z.in_selection(np.array([np.nan])), np.array([True]))
        np.testing.assert_equal(z.in_selection(np.array([0., np.nan])), np.array([False, True]))