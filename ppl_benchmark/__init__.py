"""
This __init__.py file defines the whole project source code folder as a python
package. It can be empty.
In this case, we just showcase, that beside variables (see .utils/__init__.py),
even functions can be broadcasted here for well structured and clearer imports.
"""

version = "release_1"
try:
    from setuptools_scm import get_version

    version = get_version(root="..", relative_to=__file__, git_describe_command="git describe --tags --match v[0-9]*")
except:
    pass

from .utils import PROJECT_ROOT

# Importing here in the __init__.py like below makes it possible to import with
# >>> from core import main
# instead of having to know, in which module (.py file) of the core package,
# the function "main" is implemented.
# This means, we make module level objects importable at package level.
# from .core.core import main

from .utils import CONFIG
from .utils import PARAMETERS
from .utils import PROJECT_ROOT
