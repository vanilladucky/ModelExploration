Getting started
################

A guide on how to get started with simplysklearn 

Dependencies
============

* Programming Language: Python
* License: MIT
* Operating System: Independent
* Dependencies: `here <https://github.com/vanilladucky/simplysklearn/blob/main/requirements.txt>`_

Installing
==========

.. code-block:: Python

   pip install simplysklearn

The python package website can be accessed `here <https://pypi.org/project/simplysklearn/>`_

Execution
=========

.. code-block:: Python

   from simplysklearn import Classification
   from simplysklearn import Regression


Example
=======

Classification
***************

.. code-block:: Python

    df = pd.read_csv('titanic.csv')
    model = Classification(df, feature_columns_list, target_column)
    model.fit()
    model.plot('accuracy_score')
    print(model.outlier_values)

Regression
************

.. code-block:: Python

    df = pd.read_csv('housing_prices.csv')
    model = Regression(df, feature_columns_list, target_column)
    model.fit()
    model.plot('mean_squared_error')
    print(model.outlier_values)