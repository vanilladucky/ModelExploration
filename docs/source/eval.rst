Evaluation
###########

.. code-block:: Python

   model.plot('metric_name')

There are several different evaluation metrics you can choose from for regression and classification tasks. 
They are derived from Scikit-Learn's `own metrics <https://scikit-learn.org/stable/modules/model_evaluation.html>`_. 
All you need to is type in the metric name in *string* format when using the *.plot* method.

Regression metrics
==================

.. list-table:: 
   :widths: 30 70
   :header-rows: 1

   * - Parameter
     - String format
   * - `explained variance <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html#sklearn.metrics.explained_variance_score>`_
     - 'explained_variance'
   * - `max error <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.max_error.html#sklearn.metrics.max_error>`_
     - 'max_error'
   * - `mean absolute error <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error>`_
     - 'mean_absolute_error'
   * - `mean squared error <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error>`_
     - 'mean_squared_error'
   * - `mean squared log error <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_log_error.html#sklearn.metrics.mean_squared_log_error>`_
     - 'mean_squared_log_error'
   * - `median absolute error <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.median_absolute_error.html#sklearn.metrics.median_absolute_error>`_
     - 'median_absolute_error'
   * - `r2 score <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score>`_
     - 'r2_score'
   * - `mean absolute error score <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html#sklearn.metrics.explained_variance_score>`_
     - 'mean_absolute_error_score'
   * - `d2 absolute error <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.d2_absolute_error_score.html#sklearn.metrics.d2_absolute_error_score>`_
     - 'd2_absolute_error_score'
   * - `d2 pinball score <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.d2_pinball_score.html#sklearn.metrics.d2_pinball_score>`_
     - 'd2_pinball_score'
   * - `d2 tweedie score <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.d2_tweedie_score.html#sklearn.metrics.d2_tweedie_score>`_
     - 'd2_tweedie_score'

Classification metrics
=======================

.. list-table:: 
   :widths: 30 70
   :header-rows: 1

   * - Parameter
     - String format
   * - `accuracy score <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score>`_
     - 'accuracy_score'
   * - `balanced accuracy score <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score>`_
     - 'balanced_accuracy_score'
   * - `f1 score <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score>`_
     - 'f1_score'
   * - `precision score <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score>`_
     - 'precision_score'
   * - `recall score <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score>`_
     - 'recall_score'
   * - `jaccard score <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html#sklearn.metrics.jaccard_score>`_
     - 'jaccard_score'
   * - `roc auc score <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score>`_
     - 'roc_auc_score'
   * - `log loss <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html#sklearn.metrics.log_loss>`_
     - 'log_loss'
   * - `average precision score <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html#sklearn.metrics.average_precision_score>`_
     - 'average_precision_score'
   * - `brier score loss <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html#sklearn.metrics.brier_score_loss>`_
     - 'brier_score_loss'