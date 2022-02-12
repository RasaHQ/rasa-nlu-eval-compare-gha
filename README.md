# Rasa NLU Evaluation Result Comparison
This repository contains code to combine and compare multiple sets of Rasa NLU evaluation results. It can be used locally or as a Github Action.
## Use as a Github Action

This Github action compares multiple sets of Rasa NLU evaluation results. It runs the command `python -m compare_nlu_results` with the [input arguments](#input-arguments) provided to it.

It outputs a formatted HTML table of the compared results and a json report of all result sets combined. 

You can find more information about Rasa NLU evaluation in [the Rasa Open Source docs](https://rasa.com/docs/rasa/testing-your-assistant#comparing-nlu-performance).

### Input arguments

You can set the following options using [`with`](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions#jobsjob_idstepswith) in the step running this action. **The `nlu_result_files` argument is required.**



|           Input            |                                                           Description                                                           |        Default         |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| `nlu_result_files`        | The Rasa NLU evaluation report files that should be compared and the labels to associate with each of them. For example: `intent_report.json=stable second_intent_report.json=incoming`. The report from which diffs should be calculated should be listed first. All results must be of the same type (e.g. intent classification, entity extraction). Labels for files should be unique. Do not put spaces before or after the = sign. Label values with spaces should be put in double quotes. For example: `previous_results/DIETClassifier_report.json="Previous Stable Results" current_results/DIETClassifier_report.json="New Results"` | |
| `json_outfile`            | File to which to write combined json report (contents of all result files). | combined_results.json  |
| `html_outfile`            | File to which to write HTML table. File will be overwritten unless `append_table` is specified. | formatted_compared_results.html |
| `append_table`            | Whether to append the comparison table to the html output file, instead of overwriting it. If not specified, html_outfile will be overwritten. | false |
| `table_title`             | Title of HTML table. | Compared NLU Evaluation Results |
| `label_name`              | Type of labels predicted in the provided NLU result files e.g. 'intent', 'entity', 'retrieval intent'. | label |
| `metrics_to_diff`         | Space-separated list of metrics to consider when determining changes across result sets. Valid values are support, f1-score, precision, and recall | support f1-score |
| `metrics_to_display`         | Space-separated list of metrics to display in resulting HTML table. Valid values are support, f1-score, precision, recall, and confused_with (for intent classification and response selection only) | support f1-score |
| `metric_to_sort_by`       | Metrics to sort by (descending) in resulting HTML table. | support |
| `display_only_diff`       | Display only labels with a change in at least one metric from the first listed result set. | false |

## Outputs

The list of available output variables:

|          Output          |                                                          Description                                                          |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| `docker_image_name`      | Docker image name, the name contains the registry address and the image name, e.g., `docker.io/my_account/my_image_name`      |
| `docker_image_tag`       | Tag of the image, e.g., `v1.0`                                                                                                |
| `docker_image_full_name` | Docker image name (contains an address to the registry, image name, and tag), e.g., `docker.io/my_account/my_image_name:v1.0` |

_GitHub Actions that run later in a workflow can use the output parameters returned by the Rasa GitHub Action, see [the example](examples/upgrade-deploy-rasa-x.yml) of output parameters usage._

### Example Usage

The following is an example use-case in a CI/CD pipeline for a Rasa assistant which:
1. Runs NLU cross-validation
2. Refers to previous stable results kept in the repository (you could also e.g. download these from a remote storage bucket)
3. Runs this action to compare the output of incoming cross-validation results to the previous stable results
4. Posts the HTML table as a comment to the pull request to more easily review changes

```yaml

```

## Local Use
