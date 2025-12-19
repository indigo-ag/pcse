# -*- coding: utf-8 -*-
# Copyright (c) 2004-2018 Alterra, Wageningen-UR
# Allard de Wit (allard.dewit@wur.nl), April 2018
"""
The Python Crop Simulation Environment (PCSE) has been developed
to facilitate implementing crop simulation models that were
developed in Wageningen. PCSE provides a set of building blocks
that on the one hand facilitates implementing the crop simulation
models themselves and other hand allows to interface these models with
external inputs and outputs (files, databases, webservers)

PCSE builds on existing ideas implemented in the FORTRAN
Translator (FST) and its user interface FSE. It inherits ideas
regarding the rigid distinction between rate calculation
and state integration and the initialization of parameters
in a PCSE model. Moreover PCSE provides support for reusing
input files and weather files that are used by FST models.

PCSE currently provides an implementation of the WOFOST and LINTUL crop
simulation models and variants of WOFOST with extended
capabilities.

See Also
--------
* http://www.wageningenur.nl/wofost
* http://github.com/ajwdewit/pcse
* http://pcse.readthedocs.org
"""
from __future__ import print_function

__author__ = "Allard de Wit <allard.dewit@wur.nl>"
__license__ = "European Union Public License"
__stable__ = True
__version__ = "0.6.0"

# WARNING: Avoid heavy imports or side-effects at import time.
# This module now initializes lazily to prevent requiring optional
# dependencies (e.g., sqlalchemy) just to read __version__.

import logging
import logging.config
import os
import sys
from importlib import import_module

from . import init_utils as util

# Public API exposed from the package root (lazily loaded)
__all__ = [
    "settings",
    "db",
    "fileinput",
    "agromanager",
    "soil",
    "crop",
    "start_wofost",
    "test",
    "setup",  # kept for backward compatibility
    "initialize",
]


# Backward-compatible setup function (no longer executed automatically)
def setup():
    """
    Prepare the ~/.pcse folder and user settings file, and add ~/.pcse to sys.path.
    This function is left for backward compatibility; use initialize() to
    fully initialize PCSE (logging, demo DB creation) when needed.
    """
    user_home = util.get_user_home()
    pcse_user_home = os.path.join(user_home, ".pcse")
    if not os.path.exists(pcse_user_home):
        os.makedirs(pcse_user_home, exist_ok=True)

    # Add PCSE home to python PATH (for user_settings.py)
    if pcse_user_home not in sys.path:
        sys.path.append(pcse_user_home)

    # Check existence of user settings file. If not exists, create it.
    user_settings_file = os.path.join(pcse_user_home, "user_settings.py")
    if not os.path.exists(user_settings_file):
        pcse_dir = os.path.dirname(__file__)
        default_settings_file = os.path.join(
            pcse_dir, "settings", "default_settings.py"
        )
        with open(default_settings_file) as fsrc, open(user_settings_file, "w") as fdst:
            for line in fsrc:
                if line.startswith(("#", '"', "'", "import")):
                    cline = line
                elif len(line.strip()) == 0:  # empty line
                    cline = line
                else:
                    cline = "# " + line
                fdst.write(cline)


# Internal guard to ensure one-time initialization
__initialized = False
settings = None


def _ensure_initialized():
    global __initialized
    if __initialized:
        return

    # Minimal environment setup
    setup()

    global settings
    if settings is None:
        try:
            from .settings import settings as _settings

            settings = _settings
        except Exception:
            pass

    # Configure logging using settings
    if settings is not None:
        logging.config.dictConfig(settings.LOG_CONFIG)
    # If logging configuration fails, fall back to basicConfig
    else:
        logging.basicConfig(level=logging.INFO)

    # If no PCSE demo database, build it (best-effort)
    if settings is not None:
        try:
            pcse_db_file = os.path.join(settings.PCSE_USER_HOME, "pcse.db")
            if not os.path.exists(pcse_db_file):
                print("Building PCSE demo database at: %s ..." % pcse_db_file, end=" ")
                pcse_home = os.path.dirname(__file__)
                pcse_db_dump_file = os.path.join(
                    pcse_home, "db", "pcse", "pcse_dump.sql"
                )
                try:
                    util.load_SQLite_dump_file(pcse_db_dump_file, pcse_db_file)
                    print("OK")
                except Exception as e:
                    logger = logging.getLogger()
                    msg1 = "Failed to create the PCSE demo database: %s" % e
                    msg2 = "PCSE will likely be functional, but some tests and demos may fail."
                    logger.warning(msg1)
                    logger.warning(msg2)
        finally:
            __initialized = True

    if not __stable__:
        print("Warning: You are running a PCSE development version:  %s" % __version__)


# Public function to initialize on demand
def initialize():
    """Initialize PCSE environment, logging and demo database on demand."""
    _ensure_initialized()


# Provide a lightweight proxy for test to satisfy static checkers while keeping laziness
# The real implementation is provided via __getattr__ when invoked.
def test(dsn=None):
    _test = __getattr__("test")
    return _test(dsn)


# Lazy attribute access for heavy submodules and objects
def __getattr__(name):
    # Map attribute names to import targets and optional attribute to fetch
    targets = {
        "db": (".db", None),
        "fileinput": (".fileinput", None),
        "agromanager": (".agromanager", None),
        "soil": (".soil", None),
        "crop": (".crop", None),
        "start_wofost": (".start_wofost", "start_wofost"),
    }
    if name in targets:
        _ensure_initialized()
        mod_name, attr = targets[name]
        mod = import_module(mod_name, package=__name__)
        return getattr(mod, attr) if attr else mod
    if name == "test":
        # Lazily expose test entry-point
        def _test(dsn=None):
            _ensure_initialized()
            from . import tests

            tests.test_all(dsn)

        return _test
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Ensure minimal compatibility for direct submodule imports (e.g., pcse.base)
# This has no heavy dependencies and does not configure logging/DB.
setup()
