import argparse
import logging
from unittest import result

from compare_nlu_results.results import NamedResultFile

logger = logging.getLogger(__file__)

class parse_nlu_result_files(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        result_file_pairs = parse_cli_arg_pairs(values)
        nlu_result_files = [
            NamedResultFile(filepath=filepath, name=name)
            for filepath, name in result_file_pairs
        ]
        setattr(args, self.dest, nlu_result_files)

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
        assert len(items) == 2
    except AssertionError:
        logger.error(
            f"ERROR: argument '{input_string}' is not parseable. When passing key-value command line arguments, "
            f"separate key and value with = and surround values with spaces with quotes. "
            f'e.g. --cli_flag key=value key2="value 2"'
        )
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
        action=parse_nlu_result_files,
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
        "--json_outfile",
        help=("File to which to write combined json report."),
        default="combined_results.json",
    )

    parser.add_argument(
        "--metrics_to_diff",
        help=("Metrics to consider when determining changes across result sets."),
        nargs="+",
        required=False
    )

    parser.add_argument(
        "--metrics_to_display",
        help=("Metrics to display in resulting HTML table."),
        nargs="+",
        required=False
    )

    parser.add_argument(
        "--metric_to_sort_by",
        help=("Metrics to sort by (descending) in resulting HTML table."),
        default="support",
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
        "--append_table",
        help=("Append to html_outfile instead of overwriting it."),
        action="store_true",
    )

    parser.add_argument(
        "--display_only_diff",
        help=(
            "Display only labels with a change in at least one metric"
            "from the first listed result set. Default is False"
        ),
        action="store_true",
    )

    parser.add_argument(
        "--style_table",
        help=(
            "Adds CSS style tags to table to highlight changed values. "
            "Not compatible with Github Markdown format. Default is False."
        ),
        action="store_true",
    )

    return parser
