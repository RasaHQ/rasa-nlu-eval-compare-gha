# Rasa NLU Evaluation Result Comparison
This repository contains code to compare multiple sets of Rasa NLU evaluation results. It can be used locally or as a Github Action.
## Use as a Github Action

This Github action compares NLU evaluation results using the command `python -m compare_nlu_results` with the [input arguments](#input-arguments) provided to it.

You can find more information about Rasa NLU evaluation in [the Rasa Open Source docs](https://rasa.com/docs/rasa/testing-your-assistant#comparing-nlu-performance).

### Action Output

There are no output parameters returned by this Github Action, however two files are written:

It writes a json report of all result sets combined to `json_outfile`.
It writes a formatted table of the compared results to `html_outfile`. 
For example:

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>metric</th>
      <th colspan="3" halign="left">precision</th>
      <th colspan="3" halign="left">recall</th>
    </tr>
    <tr>
      <th>result_set</th>
      <th>old</th>
      <th>new</th>
      <th>(new - old)</th>
      <th>old</th>
      <th>new</th>
      <th>(new - old)</th>
    </tr>
    <tr>
      <th>entity</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>micro avg</th>
      <td>0.994698</td>
      <td>0.998004</td>
      <td>0.003306</td>
      <td>0.999334</td>
      <td>0.998668</td>
      <td>-0.000666</td>
    </tr>
    <tr>
      <th>macro avg</th>
      <td>0.997904</td>
      <td>0.998714</td>
      <td>0.00081</td>
      <td>0.998967</td>
      <td>0.994012</td>
      <td>-0.004955</td>
    </tr>
    <tr>
      <th>weighted avg</th>
      <td>0.994733</td>
      <td>0.99802</td>
      <td>0.003287</td>
      <td>0.999334</td>
      <td>0.998668</td>
      <td>-0.000666</td>
    </tr>
    <tr>
      <th>product</th>
      <td>0.989286</td>
      <td>0.998198</td>
      <td>0.008912</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>language</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.996633</td>
      <td>-0.003367</td>
    </tr>
    <tr>
      <th>company</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.988636</td>
      <td>1.0</td>
      <td>0.011364</td>
    </tr>
  </tbody>
</table>

### Input arguments

You can set the following options using [`with`](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions#jobsjob_idstepswith) in the step running this action. **The `nlu_result_files` argument is required.**



|           Input            |                                                           Description                                                           |        Default         |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| `nlu_result_files`        | The Rasa NLU evaluation report files that should be compared and the labels to associate with each of them. For example: `intent_report.json=stable second_intent_report.json=incoming`. The report from which diffs should be calculated should be listed first. All results must be of the same type (e.g. intent classification, entity extraction). Labels for files should be unique. Do not put spaces before or after the = sign. Label values with spaces should be put in double quotes. For example: `previous_results/DIETClassifier_report.json="Previous Stable Results" current_results/DIETClassifier_report.json="New Results"` | |
| `json_outfile`            | File to which to write combined json report (contents of all result files). | combined_results.json  |
| `html_outfile`            | File to which to write HTML table. File will be overwritten unless `append_table` is specified. | formatted_compared_results.html |
| `table_title`             | Title of HTML table. | Compared NLU Evaluation Results |
| `label_name`              | Type of labels predicted in the provided NLU result files e.g. 'intent', 'entity', 'retrieval intent'. | label |
| `metrics_to_diff`         | Space-separated list of numeric metrics to consider when determining changes across result sets e.g. "support, f1-score". | All numeric metrics found in input reports |
| `metrics_to_display`         | Space-separated list of metrics to display in resulting HTML table e.g. "support, f1-score, confused_with" | All metrics found in input reports |
| `metric_to_sort_by`       | Metrics to sort by (descending) in resulting HTML table. | `support` |
| `display_only_diff`       | Display only labels (e.g. intents or entities) where there is a difference in at least one of the `metrics_to_diff` between the first listed result set and the other result set(s). Set to `true` to use. | |
| `append_table`            | Whether to append the comparison table to the html output file, instead of overwriting it. If not specified, html_outfile will be overwritten. Set to `true` to use. | |
| `style_table`            | Whether to add CSS style tags to the html table to highlight changed values. Not compatible with Github Markdown format. Set to `true` to use. | |


### Example Usage


You can use this Github Aciton in a CI/CD pipeline for a Rasa assistant which e.g.:
1. Runs NLU cross-validation
2. Refers to previous stable results (e.g. download these from a remote storage bucket, the example below assumes the results are already in the repo path for demonstration purposes)
3. Runs this action to compare the output of incoming cross-validation results to the previous stable results
4. Posts the HTML table as a comment to the pull request to more easily review changes

For example:
```yaml
on:
  pull_request: {}

jobs:
  run_cross_validation:
    runs-on: ubuntu-latest
    name: Cross-validate
    steps:
    - name: Setup python
      uses: actions/setup-python@v1
      with:
        python-version: '3.8'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt

    - name: Run cross-validation
      run: |
        rasa test nlu --cross-validation

    - name: Compare Intent Results
      uses: RasaHQ/rasa-nlu-eval-compare-gha@1.0.0
      with:
        nlu_result_files: last_stable_results/intent_report.json="Stable" results/intent_report.json="Incoming"
        table_title: Intent Classification Results
        json_outfile: results/compared_intent_classification.json
        html_outfile: results/compared_results.html
        display_only_diff: false
        label_name: intent
        metrics_to_display: support f1-score
        metrics_to_diff: support f1-score
        metric_to_sort_by: support

    - name: Compare Intent Results
      uses: RasaHQ/rasa-nlu-eval-compare-gha@1.0.0
      with:
        nlu_result_files: last_stable_results/DIETClassifier_report.json="Stable" results/DIETClassifier_report.json="Incoming"
        table_title: Entity Extraction Results
        json_outfile: results/compared_DIETClassifier.json
        html_outfile: results/compared_results.html
        append_table: true
        display_only_diff: true
        label_name: entity
        metrics_to_display: support f1-score precision recall
        metrics_to_diff: precision recall
        metric_to_sort_by: recall

    - name: Post cross-val comparison to PR
      uses: amn41/comment-on-pr@comment-file-contents
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        msg: results/compared_results.html
```


## Local Use

To compare NLU evaluation results locally, run e.g.

```bash
python -m compare_nlu_results --nlu_result_files results/intent_report.json=Base new_results/intent_report.json=New
```

See `python -m compare_nlu_results --help` for all options; the descriptions can also be found in the [input arguments](#input-arguments) section.

You can also use the package in a Python script to load, compare and further analyse results:

```
from compare_nlu_results.results import (
    EvaluationResult,
    EvaluationResultSet
)

# view just a result set
old_results = EvaluationResult(json_report_filepath="tests/data/results/intent_report.json", name="old")
print(old_results.df)

# combine two result sets
new_results = EvaluationResult(json_report_filepath="tests/data/second_results/intent_report.json", name="new")
combined_results = EvaluationResultSet(result_sets=[old_results, new_results], label_name="intents")
print(combined_results.df)

# See differences between result sets
combined_results.get_diffs_between_sets()
```
