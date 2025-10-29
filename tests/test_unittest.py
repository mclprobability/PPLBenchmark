import unittest
import inspect
from datetime import date

from ppl_benchmark.utils import log

logger = log.getLogger("unittest")


__author__ = "Christian Findenig"
__email__ = "christian.findenig@mcl.at"
__copyright__ = "Copyright 2025, " "Materials Center Leoben Forschung GmbH"
__license__ = "MIT"
__status__ = "Development"


def add(x, y):
    logger.info("We're in custom made function : " + inspect.stack()[0][3])
    return x + y


class TestClass(unittest.TestCase):

    @unittest.skip("you can skip a test case")
    def test_dummy(self):
        logger.info("\nRunning Test Method : " + inspect.stack()[0][3])
        self.assertEqual(add(2, 3), 5)

    def test_case02(self):
        logger.info("\nRunning Test Method : " + inspect.stack()[0][3])
        my_var = 3.14
        self.assertTrue(isinstance(my_var, float))

    def test_case03(self):
        logger.info("\nRunning Test Method : " + inspect.stack()[0][3])
        self.assertEqual(add(2, 2), 4)

    def test_case04(self):
        logger.info("\nRunning Test Method : " + inspect.stack()[0][3])
        my_var = 3.14
        self.assertTrue(isinstance(my_var, float))
