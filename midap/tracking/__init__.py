# we set the __all__ list to all python files except init, such that we can get all subclasses from the
# Tracking class with the __subclasses__ method after calling using wildcard import: from tracking import *
import glob
import os
modules = glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))
__all__ = [os.path.basename(f)[:-3] for f in modules if os.path.isfile(f) and not f.endswith('__init__.py')]
