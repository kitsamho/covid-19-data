import os

from cowidev.utils.utils import get_project_dir
from .etl import run_etl
from .grapher import run_grapheriser, run_explorerizer, run_db_updater
from ._parser import _parse_args


FILE_DS = os.path.join(
    get_project_dir(), "public", "data", "variants", "covid-variants.csv"
)
FILE_GRAPHER = os.path.join(
    get_project_dir(), "scripts", "grapher", "COVID-19 - Variants.csv"
)
FILE_EXPLORER = os.path.join(
    get_project_dir(), "public", "data", "internal", "megafile--variants.json"
)


def run_step(step: str):
    if step == "etl":
        run_etl(FILE_DS)
    elif step == "grapher-file":
        run_grapheriser(FILE_DS, FILE_GRAPHER)
    elif step == "explorer-file":
        run_explorerizer(FILE_DS, FILE_EXPLORER)
    elif step == "grapher-db":
        run_db_updater(FILE_GRAPHER)


if __name__ == "__main__":
    args = _parse_args()
    run_step(args.step)
