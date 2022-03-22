import logging

import numpy  # noqa, avoids segfault if torch is imported before numpy
import pyro

logging.basicConfig(format="%(message)s", level=logging.INFO)


def pytest_runtest_setup(item):
    pyro.clear_param_store()
    pyro.set_rng_seed(12345)
