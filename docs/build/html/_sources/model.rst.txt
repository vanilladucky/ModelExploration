Model
#####


.. code-block:: Python

   from simplysklearn.model import ExploreModel
   ExploreModel(DataFrame, FeatureList, Target, PredictProba = False, EnsembleBoolean=True, 
   NeuralBoolean=True, SplitRatio=0.3, OutputType='regression', Randomstate=42)

Parameters 
===========

.. list-table:: 
   :widths: 30 70
   :header-rows: 1

   * - Parameter
     - Information
   * - DataFrame (*pandas*)
     - data to be analyzed - preferably imported via pd.read_csv('filename')
   * - FeatureList (*list*)
     - names of the columns of the dataframe to be used as features, in the form of a list
   * - Target (*string*)
     - name of the column of the target variable
   * - PredictProba (*boolean*)
     - whether we wish to use the predict_proba method. If model doesn't have predict_proba method, it won't produce any value for the metric suggested
   * - EnsembleBoolean (*boolean*)
     - whether you wish to include ensembling models 
   * - NeuralBoolean (*boolean*)
     - whether you wish to include neural network models
   * - SplitRatio (*int; [0,1]*)
     - the split ratio for train_test_split 
   * - OutputType (*regression/classification*)
     - whether you wish to test for a regression task or a classification task 
   * - Randomstate (*int*)
     - seed value for the random state used throughout the process


Attributes
==========

.. list-table:: 
   :widths: 30 70
   :header-rows: 1

   * - Attributes
     - Information
   * - .fit()
     - will prepare the dataset using one-hot-encoding, standard scaler and simple imputer. It will then fit the prepared dataset to the models and return the prediction values
   * - .calculate_accuracy()
     - will calculate the scores of models 
   * - .plot(metric_name(*string*))
     - will take in the name of the metric and will plot the values of the models accordingly using seaborn
   * - .outlier_values
     - will return a dictionary where key is the name of the models and value is the value of the error. These models are left out because their values are too large to be plotted or the models have no predict_proba method when the user requested for it

   