import argparse
import json
import logging
from typing import Dict, List, NamedTuple, Optional, Text

import pandas as pd

logger = logging.getLogger(__file__)


class NamedResultFile(NamedTuple):
    """Holds a filepath and the name associated with it."""

    filepath: Text
    name: Text


class NLUEvaluationResult:
    """
    Represents a single set of Rasa NLU evaluation results
    i.e. the content of a single <report-type>_report.json file
    e.g. intent_report.json, DIETClassifier_report.json, response_selection_report.json
    """

    def __init__(
        self,
        name: Text = "Evaluation Result",
        label_name: Text = "label",
        json_report_filepath: Optional[Text] = None,
    ):
        self.from_json_report(json_report_filepath)
        self.name = name
        self.label_name = label_name
        self.df = self.report_to_df()
        self.drop_excluded_classes()
        self.set_index_names()

    def from_json_report(self, filepath) -> Dict:
        """
        Load report from a <report-type>_report.json file.
        """
        if filepath:
            with open(filepath, "r") as f:
                report = json.loads(f.read())
        else:
            report = {}
        self.json_report_filepath = filepath
        self.report = report

    def df_to_report(self) -> Dict:
        """Convert dataframe to dict representation"""
        report = self.df.T.to_dict()
        return report

    def report_to_df(self):
        """Convert dict representation to dataframe"""
        df = pd.DataFrame.from_dict(self.report).transpose()
        df.name = self.name
        return df

    def set_index_names(self):
        """Set names of indices of dataframe.

        Columns will be labeled "metric".
        Index will be labeled with `self.label_name`
        """
        self.df.columns.set_names("metric", inplace=True)
        self.df.index.set_names(self.label_name, inplace=True)

    def drop_excluded_classes(self):
        """
        Drop the labels that don't follow the same structure
        as all other labels i.e. `accuracy`.
        """
        for excluded_class in ["accuracy"]:
            try:
                self.df.drop(excluded_class, inplace=True)
            except KeyError:
                pass

    @classmethod
    def drop_non_numeric_metrics(cls, df):
        """
        Drop metrics that are not numeric i.e. `confused_with` in intent reports.
        """
        for non_numeric_metric in ["confused_with"]:
            try:
                df = df.drop(columns=non_numeric_metric)
            except KeyError:
                pass
        return df

    def sort_by_metric(self, sort_by_metric: Text):
        """Return self.df sorted in descending order by the metric provided"""
        return self.df.sort_values(by=sort_by_metric, ascending=False)

    def get_sorted_labels(self, sort_by_metric: Text, labels: List[Text] = None):
        """Return all avg metrics followed by all other labels sorted by the metric provided.

        If a list of `labels` is provided, it will include only avg metrics and those specific labels.
        If no `labels` are provided, all labels will be included.
        """
        sorted_labels = self.sort_by_metric(
            sort_by_metric=sort_by_metric
        ).index.tolist()
        avg_labels = ["macro avg", "micro avg", "weighted avg"]
        labels_avg_first = [
            label for label in self.df.index.tolist() if label in avg_labels
        ] + [
            label
            for label in sorted_labels
            if label not in avg_labels and (labels is None or label in labels)
        ]
        return labels_avg_first

    def create_html_table(self, columns=None, labels=None, sort_by_metric="support"):
        """Create an HTML table of the results sorted by the metric specified.

        If `columns` or `labels` is provided, only those columns/rows will be included.
        Otherwise all columns/rows will be included.
        """
        df = self.df
        labels = self.get_sorted_labels(sort_by_metric=sort_by_metric, labels=labels)
        if not columns:
            columns = df.columns
        df_for_table = df.loc[labels, columns]
        df_for_table.columns.set_names([None], inplace=True)
        df_for_table.index.set_names([None], inplace=True)
        html_table = df_for_table.to_html(na_rep="N/A")
        return html_table


class CombinedNLUEvaluationResults(NLUEvaluationResult):
    """
    Combine and compare multiple sets of Rasa NLU evaluation results of the same kind
    (e.g. intent classification, entity extraction).
    """

    def __init__(
        self,
        name: Text = "Combined Evaluation Results",
        label_name="label",
        result_sets: Optional[List[NLUEvaluationResult]] = None,
        base_result_set_name=None,
        metrics_to_diff=None,
    ):
        self.name = name
        if not result_sets:
            result_sets = []
        self.result_sets = result_sets
        self.label_name = label_name
        self.df = self.result_sets_to_df()
        self.set_index_names()
        self.drop_excluded_classes()
        self.report = self.df_to_report()
        self.base_result_set_name = base_result_set_name
        self.metrics_to_diff = metrics_to_diff

    def result_sets_to_df(self) -> pd.DataFrame:
        """Combine multiple sets of evaluation results into a single dataframe"""
        if not self.result_sets:
            columns = pd.MultiIndex(levels=[[], []], codes=[[], []])
            index = pd.Index([])
            joined_df = pd.DataFrame(index=index, columns=columns)
        else:
            joined_df = pd.concat(
                [result.df for result in self.result_sets],
                axis=1,
                keys=[result.name for result in self.result_sets],
            )
        return joined_df

    def set_index_names(self):
        """Set names of indices of dataframe.

        Columns have a hierarchical index, the levels will be labeled `metric` and `result_set`.
        Index will be labeled with `self.label_name`
        """
        self.df = self.df.swaplevel(axis="columns")
        self.df.columns.set_names(["metric", "result_set"], inplace=True)
        self.df.index.set_names([self.label_name], inplace=True)

    def df_to_report(self):
        """Convert dataframe to dict representation"""
        report = {
            label: {
                metric: self.df.loc[label].xs(metric).to_dict()
                for metric in self.df.loc[label].index.get_level_values("metric")
                if label
            }
            for label in self.df.index
        }
        return report

    def df_to_result_sets(self):
        """Split dataframe out into the component result sets."""
        result_sets = []
        for result_set_name in self.df.columns.get_level_values("result_set"):
            result = NLUEvaluationResult(
                name=result_set_name, label_name=self.label_name
            )
            result.df = self.df.swaplevel(axis=1)[result_set_name]
            result.report = result.df_to_report()
            result_sets.append(result)
        return result_sets

    @classmethod
    def drop_non_numeric_metrics(cls, df):
        """
        Drop metrics that are not numeric i.e. `confused_with` in intent reports.
        """
        for non_numeric_metric in ["confused_with"]:
            try:
                df = df.drop(columns=non_numeric_metric, level="metric")
            except:
                pass
        return df

    def sort_by_metric(self, sort_by_metric: Text):
        return self.df.sort_values(
            by=[(sort_by_metric, self.df[sort_by_metric].iloc[:, 0].name)],
            ascending=False,
        )

    def write_json_report(self, filepath):
        """Write combined report of all result sets to json file."""
        with open(filepath, "w+") as fh:
            json.dump(self.report, fh, indent=2)

    def from_json_report(self, filepath):
        """Load dataframe, report & result sets from json report of combined result sets."""
        with open(filepath, "r") as fh:
            self.report = json.load(fh)
        self.df = self.report_to_df()
        self.drop_excluded_classes()
        self.set_index_names()
        self.result_sets = self.df_to_result_sets()

    def report_to_df(self):
        """Load dataframe from dict report."""
        joined_df = pd.DataFrame.from_dict(
            {
                (label, metric): self.report[label][metric]
                for label in self.report.keys()
                for metric in self.report[label].keys()
            },
            orient="index",
        ).unstack()
        return joined_df

    @property
    def diff_df(self):
        """Dataframe of differences in each metric across result sets."""
        if not self.base_result_set_name:
            self.base_result_set_name = self.result_sets[0].name
        if not self.metrics_to_diff:
            self.metrics_to_diff = list(set(self.df.columns.get_level_values("metric")))

        def diff_from_base(x):
            metric = x.name[0]
            if metric == "confused_with":
                difference = pd.Series(None, index=x.index, dtype="float64")
                return difference
            try:
                base_result = self.df[(metric, self.base_result_set_name)]
            except KeyError:
                difference = pd.Series(None, index=x.index, dtype="float64")
                return difference
            if metric == "support":
                difference = x.fillna(0) - base_result.fillna(0)
            else:
                difference = x - base_result
            return difference

        diff_df = self.df[self.metrics_to_diff].apply(diff_from_base)
        diff_df.drop(columns=self.base_result_set_name, level=1, inplace=True)
        diff_df = self.drop_non_numeric_metrics(diff_df)
        diff_df.rename(
            lambda col: f"({col} - {self.base_result_set_name})",
            axis=1,
            level=1,
            inplace=True,
        )
        return pd.DataFrame(diff_df)

    @property
    def combined_df(self):
        """Return combined dataframe of original reports & diff columns"""
        return pd.concat([self.df, self.diff_df], axis=1)

    def find_labels_with_changes(self):
        """Return labels with changes across at least one metric"""
        return self.diff_df.apply(lambda x: x.any(), axis=1)

    def create_html_table(
        self, columns=None, labels=None, sort_by_metric="support", include_diffs=True
    ):
        """Create an HTML table of the combined results sorted by the metric specified.

        If `include_diffs` is True, columns from self.diff_df will be included/available for selection.
        Otherwise, only columns from self.df will be included/available for selection.
        If `columns` or `labels` is provided, only those columns/rows will be included.
        Otherwise all columns/rows will be included.
        """
        if include_diffs:
            df = self.combined_df
        else:
            df = self.df
        labels = self.get_sorted_labels(sort_by_metric=sort_by_metric, labels=labels)
        if not columns:
            columns = df.columns
        df_for_table = df.loc[labels, columns]
        df_for_table.columns.set_names([None, None], inplace=True)
        df_for_table.index.set_names([None], inplace=True)
        html_table = df_for_table.to_html(na_rep="N/A")
        return html_table


def combine_results(
    nlu_result_files: List[NamedResultFile],
    label_name: Optional[Text] = "label",
    table_title="Combined NLU Evaluation Results",
    metrics_to_diff=None,
) -> CombinedNLUEvaluationResults:
    """Combine multiple NLU evaluation result files into a CombinedNLUEvaluationResults instance"""
    result_sets = [
        NLUEvaluationResult(
            name=result_file.name,
            label_name=label_name,
            json_report_filepath=result_file.filepath,
        )
        for result_file in nlu_result_files
    ]
    combined_results = CombinedNLUEvaluationResults(
        name=table_title,
        result_sets=result_sets,
        label_name=label_name,
        metrics_to_diff=metrics_to_diff,
    )
    return combined_results


def parse_var(s):
    """
    Parse a key, value pair, separated by '='

    On the command line (argparse) a declaration will typically look like:
        foo=hello
    or
        foo="hello world"
    """
    items = s.split("=")
    key = items[0].strip()
    if len(items) > 1:
        value = "=".join(items[1:])
    return (key, value)


def parse_vars(items):
    """
    Parse a series of key-value pairs and return a dictionary
    """
    d = {}

    if items:
        for item in items:
            key, value = parse_var(item)
            d[key] = value
    return d


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare multiple sets of Rasa NLU evaluation results of the same type "
        "(e.g. intent classification, entity extraction). "
        "Writes results to an HTML table and to a json file."
    )
    parser.add_argument(
        "--nlu_result_files",
        required=True,
        metavar="RESULT_FILEPATH_1=RESULT_LABEL1 RESULT_FILEPATH_2=RESULT_LABEL2 ...",
        nargs="+",
        help="The json report files that should be compared and the labels to associate with each of them. "
        "The report from which diffs should be calculated should be listed first. "
        "All results must be of the same type (e.g. intent classification, entity extraction)"
        "Labels for files should be unique."
        "For example: "
        "'intent_report.json=1 second_intent_report.json=2'. "
        "Do not put spaces before or after the = sign. "
        "Label values with spaces should be put in double quotes. "
        "For example: "
        '\'previous_results/DIETClassifier_report.json="Previous Stable Results" results/DIETClassifier_report.json="New Results"\'',
    )

    parser.add_argument(
        "--html_outfile",
        help=(
            "File to which to write HTML table. File will be overwritten unless --append_table is specified."
        ),
        default="formatted_compared_results.html",
    )

    parser.add_argument(
        "--append_table",
        help=("Append to html_outfile instead of overwriting it."),
        action="store_true",
    )

    parser.add_argument(
        "--json_outfile",
        help=("File to which to write combined json report."),
        default="combined_results.json",
    )

    parser.add_argument(
        "--table_title",
        help=("Title of HTML table."),
        default="Compared NLU Evaluation Results",
    )

    parser.add_argument(
        "--label_name",
        help=("Type of labels predicted e.g. 'intent', 'entity', 'retrieval intent'"),
        default="label",
    )

    parser.add_argument(
        "--metrics_to_diff",
        help=("Metrics to consider when determining changes across result sets."),
        nargs="+",
        default=["support", "f1-score"],
    )

    parser.add_argument(
        "--metrics_to_display",
        help=("Metrics to display in resulting HTML table."),
        nargs="+",
        default=["support", "f1-score"],
    )

    parser.add_argument(
        "--sort_by_metric",
        help=("Metrics to sort by (descending) in resulting HTML table."),
        default="support",
    )

    parser.add_argument(
        "--display_only_diff",
        help=(
            "Display only labels with a change in at least one metric from the first listed result set. Default is False"
        ),
        action="store_true",
    )

    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    nlu_result_files = [
        NamedResultFile(filepath=filepath, name=name)
        for filepath, name in parse_vars(args.nlu_result_files).items()
    ]
    combined_results = combine_results(
        nlu_result_files=nlu_result_files,
        label_name=args.label_name,
        metrics_to_diff=args.metrics_to_diff,
    )
    combined_results.write_json_report(args.json_outfile)

    if args.display_only_diff:
        labels_with_changes = combined_results.find_labels_with_changes()

        table = combined_results.create_html_table(
            labels=labels_with_changes,
            columns=args.metrics_to_display,
            sort_by_metric=args.sort_by_metric,
        )

    else:
        table = combined_results.create_html_table(
            columns=args.metrics_to_display, sort_by_metric=args.sort_by_metric
        )

    mode = "w+"
    if args.append_table:
        mode = "a+"
    with open(args.html_outfile, mode) as fh:
        fh.write(f"<h1>{args.table_title}</h1>")
        if args.display_only_diff:
            fh.write(
                f"<body>Only averages and the {args.label_name}(s) that show differences in at least one of the following metrics: {args.metrics_to_diff} are displayed.</body>"
            )
        fh.write(table)


if __name__ == "__main__":
    main()
