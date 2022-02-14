from email.mime import base
import logging
from typing import List, Optional, Text

import pandas as pd

logger = logging.getLogger(__file__)


class ResultDf(pd.DataFrame):
    excluded_labels = ["accuracy"]
    columns_names = ["metrics"]

    @property
    def _constructor(self):
        return ResultDf

    def clean(
        self,
        label_name: Text = "label",
    ):
        self.label_name = label_name
        self.drop_excluded_labels()
        self.set_index_names()

    def drop_excluded_labels(self):
        """
        Drop the labels that don't follow the same structure
        as all other labels i.e. `accuracy`.
        """
        for excluded_label in self.excluded_labels:
            try:
                self.drop(excluded_label, inplace=True)
            except KeyError:
                pass

    def drop_non_numeric_metrics(self):
        """
        Drop metrics that are not numeric
        """
        for non_numeric_metric in ["confused_with"]:
            try:
                self.drop(columns=non_numeric_metric, level="metric", inplace=True)
            except:
                pass

    def set_index_names(self):
        """Set names of indices of dataframe.

        Columns will be labeled "metric".
        Index will be labeled with `self.label_name`
        """
        self.columns.set_names(self.columns_names, inplace=True)
        self.index.set_names(self.label_name, inplace=True)

    def drop_non_numeric_metrics(self):
        """
        Drop metrics that are not numeric
        """
        for non_numeric_metric in ["confused_with"]:
            try:
                self.drop(columns=non_numeric_metric, inplace=True)
            except KeyError:
                pass

    def sorted_by_metric(self, metric_to_sort_by: Text) -> pd.DataFrame:
        """
        Sort in descending order
        by the metric provided
        """
        return self.sort_values(by=metric_to_sort_by, ascending=False)

    def get_sorted_labels(
        self,
        metric_to_sort_by: Text,
        labels: Optional[List[Text]] = None,
    ) -> List[Text]:
        """
        Return all avg metrics followed by all other labels sorted by the
        metric provided.

        If a list of `labels` is provided, it will include only avg metrics and
        those specific labels. If no `labels` are provided, all labels will be
        included.
        """
        sorted_labels = self.sorted_by_metric(
            metric_to_sort_by=metric_to_sort_by
        ).index.tolist()
        avg_labels = ["macro avg", "micro avg", "weighted avg"]
        labels_avg_first = [
            label for label in self.index.tolist() if label in avg_labels
        ] + [
            label
            for label in sorted_labels
            if label not in avg_labels and (labels is None or label in labels)
        ]
        return labels_avg_first


class ResultSetDf(ResultDf):
    columns_names = ["metric", "result_set"]

    @property
    def _constructor(self):
        return ResultSetDf

    def drop_non_numeric_metrics(self):
        """
        Drop metrics that are not numeric
        """
        for non_numeric_metric in ["confused_with"]:
            try:
                self.drop(columns=non_numeric_metric, level="metric", inplace=True)
            except:
                pass

    def set_index_names(self):
        """Set names of indices of dataframe.

        Columns have a hierarchical index, the levels will be labeled `metric`
        and `result_set`. Index will be labeled with `self.label_name`
        """
        self.columns.set_names(self.columns_names, inplace=True)
        self.index.set_names([self.label_name], inplace=True)

    def sorted_by_metric(self, metric_to_sort_by: Text) -> pd.DataFrame:
        return self.sort_values(
            by=[(metric_to_sort_by, self[metric_to_sort_by].iloc[:, 0].name)],
            ascending=False,
        )


class ResultSetDiffDf(ResultSetDf):
    @property
    def _constructor(self):
        return ResultSetDiffDf

    def find_labels_with_changes(self):
        """Return labels with changes across at least one metric"""
        df_with_only_changes = self[self.apply(lambda x: x.any(), axis=1)]
        return df_with_only_changes.index.values.tolist()

    def clean(self, base_result_set_name: Text):
        self.drop(columns=base_result_set_name, level=1, inplace=True)
        self.drop_non_numeric_metrics()
    
    def name_diff_cols(self, base_result_set_name: Text):
        self.rename(
            lambda col: f"({col} - {base_result_set_name})",
            axis=1,
            level=1,
            inplace=True,
        )

    @classmethod
    def from_df(
        cls,
        df: ResultSetDf,
        base_result_set_name: Text,
        metrics_to_diff: Optional[List[Text]] = None,
    ) -> "ResultSetDf":
        """Initialize Dataframe of differences in each metric across result sets from undiffed dataframe."""

        all_numeric_metrics = [metric for metric in list(set(df.columns.get_level_values("metric"))) if not metric=="confused_with"]
        if not metrics_to_diff:
            metrics_to_diff = all_numeric_metrics
        else:
            try:
                assert all([metric in all_numeric_metrics for metric in metrics_to_diff])
            except AssertionError:
                logger.error(f"ERROR: You have specified an invalid metric to diff by. Valid metrics for these reports are {all_numeric_metrics}. You specified {metrics_to_diff}")
                raise

        def diff_from_base(x):
            metric = x.name[0]
            try:
                base_result = df[(metric, base_result_set_name)]
            except KeyError:
                difference = pd.Series(None, index=x.index, dtype="float64")
                return difference
            if metric == "support":
                difference = x.fillna(0) - base_result.fillna(0)
            else:
                difference = x - base_result
            return difference

        diff_df = cls(df[metrics_to_diff].apply(diff_from_base))
        diff_df.clean(base_result_set_name=base_result_set_name)
        diff_df.name_diff_cols(base_result_set_name=base_result_set_name)

        return diff_df
