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

   from simplysklearn.model import ExploreModel


Example
=======

Classification
***************

.. code-block:: Python

    import pandas as pd
    df = pd.read_csv('titanic.csv')
    model = ExploreModel(df, feature_columns_list, 'Survived', PredictProba = False, 
                        EnsembleBoolean=True, NeuralBoolean=True, SplitRatio=0.3, 
                        OutputType='classification')
    model.fit()
    model.calculate_accuracy()
    model.plot('accuracy_score')
    print(model.outlier_values)

Regression
************

.. code-block:: Python

    import pandas as pd
    df = pd.read_csv('housing_prices.csv')
    model = ExploreModel(df, feature_columns_list, 'SalePrice', PredictProba = False, 
                        EnsembleBoolean=True, NeuralBoolean=True, SplitRatio=0.3, 
                        OutputType='regression')
    model.fit()
    model.calculate_accuracy()
    model.plot('mean_squared_error')
    print(model.outlier_values)