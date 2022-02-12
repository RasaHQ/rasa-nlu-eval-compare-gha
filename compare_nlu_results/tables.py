import logging
from typing import Any, List, Optional, Text

import pandas as pd

logger = logging.getLogger(__file__)

from compare_nlu_results.dataframes import ResultDf, ResultSetDf, ResultSetDiffDf


class ResultTable:
    def __init__(
        self,
        df: ResultDf,
        metric_to_sort_by: Text,
        metrics_to_display: Optional[List[Text]] = None,
        labels: Optional[List[Text]] = None,
    ):
        sorted_labels = df.get_sorted_labels(
            metric_to_sort_by=metric_to_sort_by, labels=labels
        )
        if not metrics_to_display:
            metrics_to_display = df.columns
        self.df = df.loc[sorted_labels, metrics_to_display]
        self.df = df

    def style_table(self) -> pd.DataFrame.style:
        borders = {
            "selector": "th, td",
            "props": [
                ("border-style", "solid"),
                ("border-width", "1px"),
                ("border-color", "black"),
                ("padding", "5px"),
            ],
        }
        border_collapse = {"selector": "", "props": [("border-collapse", "collapse")]}
        row_hover = {
            "selector": "tr:hover",
            "props": [("background-color", "gainsboro")],
        }
        index_names = {"selector": ".index_name", "props": [("color", "white")]}
        row_headers = {"selector": "th.row_heading", "props": [("text-align", "right")]}
        column_headers = {
            "selector": "th.col_heading",
            "props": [("text-align", "center")],
        }
        value_cells = {"selector": "td", "props": [("text-align", "center")]}
        styler = self.df.style.set_table_styles(
            [
                borders,
                border_collapse,
                row_hover,
                index_names,
                row_headers,
                column_headers,
                value_cells,
            ]
        )

        styler.format(
            na_rep="N/A",
            formatter={
                col: ("{:.0f}" if "support" in col else "{:.2f}")
                for col in self.df.columns
                if not "confused_with" in col
            },
        )
        return styler

    @property
    def styled_table(self) -> Text:
        """Styled HTML table"""
        styled_table = self.style_table()
        return styled_table.render()


class ResultSetTable(ResultTable):
    def __init__(
        self,
        df: ResultSetDf,
        metric_to_sort_by: Text,
        metrics_to_display: Optional[List[Text]] = None,
        labels: Optional[List[Text]] = None,
    ):
        sorted_labels = df.get_sorted_labels(
            metric_to_sort_by=metric_to_sort_by, labels=labels
        )
        if not metrics_to_display:
            metrics_to_display = df.columns.get_level_values(0)
        self.df = df.loc[sorted_labels, metrics_to_display]
        self.df = df

    def style_table(self):
        styler = super().style_table()
        metrics_column_headers = {
            "selector": "th.col_heading.level0",
            "props": [("font-size", "1.5em")],
        }

        styler.set_table_styles([metrics_column_headers], overwrite=False)
        return styler


class ResultSetDiffTable(ResultSetTable):
    def __init__(
        self,
        result_set_df: ResultSetDf,
        diff_df: ResultSetDiffDf,
        metric_to_sort_by: Text,
        metrics_to_display: Optional[List[Text]] = None,
        display_only_diff: bool = False,
        diff_columns: Optional[List[Any]] = None,
    ):
        labels = None
        if display_only_diff:
            labels = diff_df.find_labels_with_changes()
        sorted_labels = result_set_df.get_sorted_labels(
            metric_to_sort_by=metric_to_sort_by, labels=labels
        )
        if not metrics_to_display:
            metrics_to_display = result_set_df.columns.get_level_values(0)
        self.df = pd.concat([result_set_df, diff_df], axis=1).loc[sorted_labels, metrics_to_display]

        if not diff_columns:
            diff_columns = []
        diff_columns_in_table = [col for col in diff_columns if col in self.df.columns]
        self.diff_columns = diff_columns_in_table

    def style_table(self):
        def style_negative(value):
            return [("color", "red"), ("font-weight", "bold")] if value < 0 else None

        def style_positive(value):
            return [("color", "green"), ("font-weight", "bold")] if value > 0 else None

        styler = super().style_table()

        styler.applymap(style_negative, subset=self.diff_columns)
        styler.applymap(style_positive, subset=self.diff_columns)

        return styler
