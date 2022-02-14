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
        title: Optional[Text] = "NLU Evaluation Results"
    ):
        sorted_labels = df.get_sorted_labels(
            metric_to_sort_by=metric_to_sort_by, labels=labels
        )
        if not metrics_to_display:
            metrics_to_display = df.columns
        self.df = df.loc[sorted_labels, metrics_to_display]
        self.df = df
        self.title = title

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

    def get_table(self, styled: bool=False) -> Text:
        """Styled HTML table"""
        if styled:
            return self.style_table().render()
        else:
            return self.df.to_html(na_rep="N/A", formatters={
                col: ("{:.0f}".format if "support" in col else "{:.2f}".format)
                for col in self.df.columns
                if not "confused_with" in col
            },
        )

    def write_to_file(self, html_outfile: Text, append_table: bool=False, style_table: bool=False):
        mode = "w+"
        if append_table:
            mode = "a+"
        with open(html_outfile, mode) as fh:
            fh.write(f"<h1>{self.title}</h1>")
            fh.write(self.get_table(styled=style_table))
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
        metrics_to_diff: Optional[List[Text]] = None,
        display_only_diff: bool = False,
        title: Optional[Text] = "Compared NLU Evaluation Results",
        label_name: Optional[Text] = "label"
    ):
        self.display_only_diff = display_only_diff
        self.title = title
        self.label_name = label_name
        self.metrics_to_diff = metrics_to_diff
        labels = None
        if self.display_only_diff:
            labels = diff_df.find_labels_with_changes()
        sorted_labels = result_set_df.get_sorted_labels(
            metric_to_sort_by=metric_to_sort_by, labels=labels
        )
        if not metrics_to_display:
            metrics_to_display = result_set_df.columns.get_level_values(0)
        self.df = pd.concat([result_set_df, diff_df], axis=1).loc[sorted_labels, metrics_to_display]

        diff_columns = diff_df.columns.tolist()
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

    def write_to_file(self, html_outfile: Text, append_table: bool=False, style_table: bool=False):
        mode = "w+"
        if append_table:
            mode = "a+"
        with open(html_outfile, mode) as fh:
            fh.write(f"<h1>{self.title}</h1>")
            if self.display_only_diff:
                fh.write(
                    f"<body>Only averages and the {self.label_name}(s) that show "
                    f"differences in at least one of the following metrics: "
                    f"{self.metrics_to_diff} are displayed.</body>"
                )
            fh.write(self.get_table(styled=style_table))
