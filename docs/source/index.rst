.. simplysklearn documentation master file, created by
   sphinx-quickstart on Sun Jan 29 18:34:48 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to simplysklearn's documentation!
=========================================

.. toctree::
   :maxdepth: 2

   install
   model
   eval
   future


Description
==================

As Bill Gates once said, “I choose a lazy person to do a hard job. Because a lazy person will find an easy way to do it.”.

After numerous hours, days, weeks and months spent on Kaggle, I was inspired to sum up all the repetitive and tedious process to one single package. For any machine learning projects, the user must perform data manipulations, check the dataset is all set for models. After that, they must fit it into models and take note of how the model performs.

With this package, I want to eliminate that tedious stage for any user. This package would mainly be beneficial to those recently starting out in the field of machine learning or intermediate users. It is probably unsuitable for highly complex models or datasets.

The main usage should be to assist the user by assessing the performances of myriad of models.

What it provides
==================

* Prepares datatset by takeing care of categorical feature by one-hot-encoding and numerical feature by scaling.
* Fits the engineered dataset on myriad of Scikit-Learn models.
* Depending on what metric you want to observe, will plot you a colorful comparison of metrics of different models.

Future improvements
=====================

* Automated hyperparameter tuning process
* Option to display feature importances

License
========

This project is licensed under the MIT License - see the `LICENSE file <https://github.com/vanilladucky/simplysklearn/blob/main/LICENSE>`_ for details

Website 
=======

* `Pypi package <https://pypi.org/project/simplysklearn/>`_
* `Github <https://github.com/vanilladucky/simplysklearn>`_