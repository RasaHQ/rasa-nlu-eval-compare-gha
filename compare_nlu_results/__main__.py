from cgitb import html
import logging
from typing import List, Text, Optional

from click import style

from compare_nlu_results.dataframes import ResultSetDiffDf
from compare_nlu_results.results import (
    NamedResultFile,
    EvaluationResultSet
)
from compare_nlu_results.tables import ResultSetDiffTable
from compare_nlu_results import cli


logger = logging.getLogger(__file__)

def create_comparison(
        base_result_file: NamedResultFile,
        other_result_files: List[NamedResultFile],
        json_outfile: Optional[Text]=None,
        html_outfile: Optional[Text]=None,
        metrics_to_display: Optional[List]=None,
        metrics_to_diff: Optional[List]=None,
        metric_to_sort_by: Optional[List]=None,
        label_name: Optional[Text]=None,
        table_title: Optional[Text]=None,
        display_only_diff: Optional[bool]=None,
        append_table: Optional[bool]=None,
        style_table: Optional[bool]=None
    ):

    combined_results = EvaluationResultSet.from_result_files(
        nlu_result_files=[base_result_file]+other_result_files, label_name=label_name
    )
    diff_df = ResultSetDiffDf.from_df(
        combined_results.df, base_result_file.name, metrics_to_diff
    )
    table = ResultSetDiffTable(
        result_set_df=combined_results.df,
        diff_df=diff_df,
        metrics_to_display=metrics_to_display,
        metric_to_sort_by=metric_to_sort_by,
        display_only_diff=display_only_diff,
        title=table_title,
        label_name=label_name
    )

    table.write_to_file(html_outfile=html_outfile, append_table=append_table, style_table=style_table)
    combined_results.write_json_report_to_file(json_outfile)

def main():
    parser = cli.create_argument_parser()
    args = parser.parse_args()
    base_result_file = args.nlu_result_files.pop(0)
    create_comparison(
        base_result_file=base_result_file,
        other_result_files=args.nlu_result_files,
        json_outfile=args.json_outfile,
        html_outfile=args.html_outfile,
        metrics_to_display=args.metrics_to_display,
        metrics_to_diff=args.metrics_to_diff,
        metric_to_sort_by=args.metric_to_sort_by,
        label_name=args.label_name,
        table_title=args.table_title,
        display_only_diff=args.display_only_diff,
        append_table=args.append_table,
        style_table=args.style_table
    )


if __name__ == "__main__":
    main()
