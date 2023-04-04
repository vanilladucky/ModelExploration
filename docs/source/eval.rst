Evaluation
###########

.. code-block:: Python

   model.plot('metric_name')

There are several different evaluation metrics you can choose from for regression and classification tasks. 
They are derived from Scikit-Learn's `own metrics <https://scikit-learn.org/stable/modules/model_evaluation.html>`_. 

Regression metrics
==================

.. list-table:: 
   :widths: 30 70
   :header-rows: 1

   * - Parameter
     - Information
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