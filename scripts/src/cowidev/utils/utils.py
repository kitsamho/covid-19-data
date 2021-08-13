import os
import pytz
import ntpath
from datetime import datetime, timedelta
from dotenv import load_dotenv


def get_project_dir(err: bool = False):
    load_dotenv()
    project_dir = os.environ.get("OWID_COVID_PROJECT_DIR")
    if project_dir is None:  # err and
        raise ValueError("Please have ${OWID_COVID_PROJECT_DIR}.")
    return project_dir


def export_timestamp(timestamp_filename: str):
    timestamp_filename = os.path.join(
        get_project_dir(), "public", "data", "internal", "timestamp", timestamp_filename
    )
    with open(timestamp_filename, "w") as timestamp_file:
        timestamp_file.write(datetime.utcnow().replace(microsecond=0).isoformat())


def time_str_grapher():
    return (
        (datetime.now() - timedelta(minutes=10))
        .astimezone(pytz.timezone("Europe/London"))
        .strftime("%-d %B %Y, %H:%M")
    )


def get_filename(filepath: str, remove_extension: bool = True):
    filename = ntpath.basename(filepath)
    if remove_extension:
        return filename.split(".")[0]
    return filename
