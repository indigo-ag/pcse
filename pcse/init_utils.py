import logging
import os
import platform
import sqlite3
import tempfile


def get_user_home():
    """A reasonable platform independent way to get the user home folder.
    If PCSE runs under a system user then return the temp directory as returned
    by tempfile.gettempdir()
    """
    user_home = None
    if platform.system() == "Windows":
        user = os.getenv("USERNAME")
        if user is not None:
            user_home = os.path.expanduser("~")
    elif platform.system() == "Linux" or platform.system() == "Darwin":
        user = os.getenv("USER")
        if user is not None:
            user_home = os.path.expanduser("~")
    else:
        msg = "Platform not recognized, using system temp directory for PCSE settings."
        logger = logging.getLogger("pcse")
        logger.warning(msg)

    if user_home is None:
        user_home = tempfile.gettempdir()

    return user_home


def load_SQLite_dump_file(dump_file_name, SQLite_db_name):
    """Build an SQLite database <SQLite_db_name> from dump file <dump_file_name>."""

    with open(dump_file_name) as fp:
        sql_dump = fp.readlines()
    str_sql_dump = "".join(sql_dump)
    con = sqlite3.connect(SQLite_db_name)
    con.executescript(str_sql_dump)
    con.close()
