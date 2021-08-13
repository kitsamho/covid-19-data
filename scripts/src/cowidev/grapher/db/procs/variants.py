from cowidev.grapher.db.base import GrapherBaseUpdater


class GrapherVariantsUpdater(GrapherBaseUpdater):
    def __init__(self) -> None:
        super().__init__(
            dataset_name="COVID-19 - Variants",
            source_name=(
                f"CoVariants.org and GISAID – Last updated {self.time_str} (London time)"
            ),
            zero_day="2020-01-21",
            unit="%",
            unit_short="%",
        )
