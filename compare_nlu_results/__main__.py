import argparse
import logging
from typing import List, Optional, Text

import pandas as pd

from compare_nlu_results.nlu_result_dataframes import ResultSetDiffDf
from compare_nlu_results.nlu_results import (
    NamedResultFile,
    EvaluationResult,
    EvaluationResultSet,
)

logger = logging.getLogger(__file__)


def combine_results(
    nlu_result_files: List[NamedResultFile],
    label_name: Optional[Text] = "label",
) -> EvaluationResultSet:
    """
    Combine multiple NLU evaluation result files into a
    EvaluationResultSet instance
    """
    result_sets = [
        EvaluationResult(
            name=result_file.name,
            label_name=label_name,
            json_report_filepath=result_file.filepath,
        )
        for result_file in nlu_result_files
    ]
    combined_results = EvaluationResultSet(result_sets=result_sets, label_name=label_name)
    return combined_results


def parse_cli_arg_pair(input_string):
    """
    Parse a key, value pair, separated by '='

    On the command line (argparse) a declaration will typically look like:
        foo=hello
    or
        foo="hello world"
    """
    items = input_string.split("=")
    try:
        assert len(items)==2
    except AssertionError:
        logger.error(f"ERROR: argument '{input_string}' is not parseable. When passing key-value command line arguments, "
                     f"separate key and value with = and surround values with spaces with quotes. "
                     f"e.g. --cli_flag key=value key2=\"value 2\"")
        exit()
    key = items[0].strip()
    value = items[1].strip()
    return (key, value)


def parse_cli_arg_pairs(items):
    """
    Parse a series of key-value pairs
    """
    return [parse_cli_arg_pair(item) for item in items]


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare multiple sets of Rasa NLU evaluation "
        "results of the same type "
        "(e.g. intent classification, entity extraction). "
        "Writes results to an HTML table and to a json file."
    )
    parser.add_argument(
        "--nlu_result_files",
        required=True,
        metavar="RESULT_FILEPATH_1=RESULT_LABEL1 RESULT_FILEPATH_2=RESULT_LABEL2 ...",
        nargs="+",
        help="The json report files that should be compared and the labels to "
        "associate with each of them. "
        "The report from which diffs should be calculated should be listed first. "
        "All results must be of the same type "
        "(e.g. intent classification, entity extraction)"
        "Labels for files should be unique."
        "For example: "
        "'intent_report.json=1 second_intent_report.json=2'. "
        "Do not put spaces before or after the = sign. "
        "Label values with spaces should be put in double quotes. "
        "For example: "
        'previous_results/DIETClassifier_report.json="Previous Stable Results"'
        'results/DIETClassifier_report.json="New Results"',
    )

    parser.add_argument(
        "--html_outfile",
        help=(
            "File to which to write HTML table. File will be overwritten "
            "unless --append_table is specified."
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
        "--metric_to_sort_by",
        help=("Metrics to sort by (descending) in resulting HTML table."),
        default="support",
    )

    parser.add_argument(
        "--display_only_diff",
        help=(
            "Display only labels with a change in at least one metric"
            "from the first listed result set. Default is False"
        ),
        action="store_true",
    )

    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    nlu_result_files = [
        NamedResultFile(filepath=filepath, name=name)
        for filepath, name in parse_cli_arg_pairs(args.nlu_result_files)
    ]
    base_result_set_name = nlu_result_files[0].name
    combined_results = combine_results(
        nlu_result_files=nlu_result_files, label_name=args.label_name
    )
    EvaluationResultSet.write_json_report_to_file(
        combined_results.report, args.json_outfile
    )
    diff_df = ResultSetDiffDf.from_df(
        combined_results.df, base_result_set_name, args.metrics_to_diff
    )
    combined_diffed_df = pd.concat([combined_results.df, diff_df], axis=1)
    labels = []

    if args.display_only_diff:
        labels = diff_df.find_labels_with_changes()

    table = combined_results.create_html_table(
        df=combined_diffed_df,
        labels=labels,
        columns=args.metrics_to_display,
        metric_to_sort_by=args.metric_to_sort_by,
    )

    mode = "w+"
    if args.append_table:
        mode = "a+"
    with open(args.html_outfile, mode) as fh:
        fh.write(f"<h1>{args.table_title}</h1>")
        if args.display_only_diff:
            fh.write(
                f"<body>Only averages and the {args.label_name}(s) that show"
                f"differences in at least one of the following metrics: "
                f"{args.metrics_to_diff} are displayed.</body>"
            )
        fh.write(table)


if __name__ == "__main__":
    main()
