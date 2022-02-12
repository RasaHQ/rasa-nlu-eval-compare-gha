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
        self.df = self.convert_report_to_df()

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

    def write_json_report_to_file(self, filepath: Text):
        """Write report to json file."""
        with open(filepath, "w+") as fh:
            json.dump(self.report, fh, indent=2)

    def convert_df_to_report(self) -> Dict:
        """Convert dataframe to dict representation"""
        report = self.df.T.to_dict()
        return report

    def convert_report_to_df(self) -> ResultDf:
        """Convert dict representation to dataframe"""
        df = ResultDf(pd.DataFrame.from_dict(self.report).transpose())
        df.clean(label_name=self.label_name)
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
        self.df = self.convert_result_sets_to_df()
        self.report = self.convert_df_to_report()

    def convert_result_sets_to_df(self) -> ResultSetDf:
        """Combine multiple sets of evaluation results into a single dataframe"""
        if not self.result_sets:
            columns = pd.MultiIndex(levels=[[], []], codes=[[], []])
            index = pd.Index([])
            joined_df = pd.DataFrame(index=index, columns=columns)
        else:
            joined_df = pd.concat(
                [pd.DataFrame(result_set.df) for result_set in self.result_sets],
                axis=1,
                keys=[result.name for result in self.result_sets],
            )
            joined_df.columns = joined_df.columns.swaplevel()
            joined_df = ResultSetDf(joined_df)
            joined_df.clean(label_name=self.label_name)
        return joined_df

    def convert_df_to_report(self) -> Dict[Text, Any]:
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

    def convert_report_to_df(self) -> ResultSetDf:
        """Load dataframe from dict report."""
        joined_df = pd.DataFrame.from_dict(
            {
                (label, metric): self.report[label][metric]
                for label in self.report.keys()
                for metric in self.report[label].keys()
            },
            orient="index",
        ).unstack()
        joined_df = ResultSetDf(joined_df)
        joined_df.clean(label_name=self.label_name)
        return joined_df

    def convert_df_to_result_sets(self) -> List[EvaluationResult]:
        """Split dataframe out into the component result sets."""
        result_sets = []
        for result_set_name in self.df.columns.get_level_values("result_set"):
            result = EvaluationResult(name=result_set_name, label_name=self.label_name)
            result.df = self.df.swaplevel(axis=1)[result_set_name]
            result.report = result.convert_df_to_report()
            result_sets.append(result)
        return result_sets
