import json
import logging
from typing import Any, Dict, List, NamedTuple, Optional, Text

import pandas as pd

from compare_nlu_results.dataframes import ResultSetDf, ResultDf

logger = logging.getLogger(__file__)


class NamedResultFile(NamedTuple):
    """Holds a filepath and the name associated with it."""

    filepath: Text
    name: Text


class EvaluationResult:
    """
    Represents a single set of Rasa NLU evaluation results
    i.e. the content of a single <report-type>_report.json file
    e.g. intent_report.json, DIETClassifier_report.json,
    response_selection_report.json
    """

    def __init__(
        self,
        name: Text = "Evaluation Result",
        label_name: Text = "label",
        json_report_filepath: Optional[Text] = None,
    ):
        self.report = self.load_json_report_from_file(json_report_filepath)
        self.name = name
        self.label_name = label_name
        self.df = self.convert_report_to_df(self.report, label_name)

    @classmethod
    def load_json_report_from_file(cls, filepath: Text) -> Dict:
        """
        Load report from a <report-type>_report.json file.
        """
        if filepath:
            with open(filepath, "r") as f:
                report = json.loads(f.read())
        else:
            report = {}
        return report

    @classmethod
    def write_json_report_to_file(cls, report: Dict, filepath: Text):
        """Write report to json file."""
        with open(filepath, "w+") as fh:
            json.dump(report, fh, indent=2)

    @classmethod
    def convert_df_to_report(cls, df: ResultDf) -> Dict:
        """Convert dataframe to dict representation"""
        report = df.T.to_dict()
        return report

    @classmethod
    def convert_report_to_df(cls, report: Dict, label_name: Text = "label") -> ResultDf:
        """Convert dict representation to dataframe"""
        df = ResultDf(pd.DataFrame.from_dict(report).transpose())
        df.clean(label_name=label_name)
        return df


class EvaluationResultSet(EvaluationResult):
    """
    Combine and compare multiple sets of Rasa NLU evaluation results of the
    same kind (e.g. intent classification, entity extraction).
    """

    def __init__(
        self,
        name: Text = "Combined Evaluation Results",
        label_name: Text = "label",
        result_sets: Optional[List[EvaluationResult]] = None,
    ):
        self.name = name
        if not result_sets:
            result_sets = []
        self.result_sets = result_sets
        self.label_name = label_name
        self.df = self.convert_result_sets_to_df(self.result_sets, label_name)
        self.report = self.convert_df_to_report(self.df)

    @classmethod
    def convert_result_sets_to_df(
        cls,
        result_sets: Optional[List[EvaluationResult]] = None,
        label_name: Text = "label",
    ) -> ResultSetDf:
        """Combine multiple sets of evaluation results into a single dataframe"""
        if not result_sets:
            columns = pd.MultiIndex(levels=[[], []], codes=[[], []])
            index = pd.Index([])
            joined_df = pd.DataFrame(index=index, columns=columns)
        else:
            joined_df = pd.concat(
                [pd.DataFrame(result_set.df) for result_set in result_sets],
                axis=1,
                keys=[result.name for result in result_sets],
            )
            joined_df.columns = joined_df.columns.swaplevel()
            joined_df = ResultSetDf(joined_df)
            joined_df.clean(label_name=label_name)
        return joined_df

    @classmethod
    def convert_df_to_report(cls, df: ResultSetDf) -> Dict[Text, Any]:
        """Convert dataframe to dict representation"""
        report = {
            label: {
                metric: df.loc[label].xs(metric).to_dict()
                for metric in df.loc[label].index.get_level_values("metric")
                if label
            }
            for label in df.index
        }
        return report

    @classmethod
    def convert_report_to_df(cls, report: Dict, label_name="label") -> ResultSetDf:
        """Load dataframe from dict report."""
        joined_df = pd.DataFrame.from_dict(
            {
                (label, metric): report[label][metric]
                for label in report.keys()
                for metric in report[label].keys()
            },
            orient="index",
        ).unstack()
        joined_df = ResultSetDf(joined_df)
        joined_df.clean(label_name=label_name)
        return joined_df

    @classmethod
    def convert_df_to_result_sets(
        cls, df: ResultSetDf, label_name: Text
    ) -> List[EvaluationResult]:
        """Split dataframe out into the component result sets."""
        result_sets = []
        for result_set_name in df.columns.get_level_values("result_set"):
            result = EvaluationResult(name=result_set_name, label_name=label_name)
            result.df = df.swaplevel(axis=1)[result_set_name]
            result.report = result.convert_df_to_report()
            result_sets.append(result)
        return result_sets
