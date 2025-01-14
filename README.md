<div align="center">
<h1>SIMPLYSKLEARN</h1>
<br>

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

<br>

<p>Time to even automate your machine learning process!

<br>
<br>
</div>

## Description

As Bill Gates once said, “I choose a lazy person to do a hard job. Because a lazy person will find an easy way to do it.”.

After numerous hours, days, weeks and months spent on Kaggle, I was inspired to sum up all the repetitive and tedious process to one single package. 
For any machine learning projects, the user must perform data manipulations, check the dataset is all set for models.
After that, they must fit it into models and take note of how the model performs. 

With this package, I want to eliminate that tedious stage for any user. 
This package would mainly be beneficial to those recently starting out in the field of machine learning or intermediate users. 
It is probably unsuitable for highly complex models or datasets. 

The main usage should be to assist the user by assessing the performances of myriad of models.

## What it provides

![image](https://github.com/vanilladucky/ModelExploration/blob/main/images/modelflow.png)

* Prepares datatset by taking care of categorical feature by one-hot-encoding and numerical feature by scaling.
* Automatically fits the engineered dataset on myriad of Scikit-Learn models. 
* Depending on what metric you want to observe, will plot you a colorful comparison of metrics of different models. 

## Future Improvements 

* Automated hyperparameter tuning process
* Option to display feature importances 

## Getting Started

### Dependencies

* Programming Language: Python
* License: MIT
* Operating System: Independent
* [Dependencies](https://github.com/vanilladucky/simplysklearn/blob/main/requirements.txt)

### Installing

```
pip install simplysklearn
```
The python package website can be accessed [here](https://pypi.org/project/simplysklearn/) 

### Executing program

```
from simplysklearn import Classification
from simplysklearn import Regression
```

## Classification Example 
```
df = pd.read_csv('titanic.csv')
model = Classification(df, feature_columns_list, target_column)
model.fit()
model.plot('accuracy_score')
print(model.outlier_values)
```
![image](https://user-images.githubusercontent.com/77542415/215261264-a14ed13e-9bdc-4d76-b280-b1040c7ab74c.png)

## Regression Example
```
df = pd.read_csv('housing_prices.csv')
model = Regression(df, feature_columns_list, target_column)
model.fit()
model.plot('mean_squared_error')
print(model.outlier_values)
```
![image](https://user-images.githubusercontent.com/77542415/215264939-2daae110-f53b-4538-9be3-1f0d936dc9b9.png)

## Documentation

[Documentation](https://simplysklearn.readthedocs.io/en/latest/) can be accessed here.

## Contributing

Contributions are very much welcomed from everyone! 

Just make sure to fork this repo, make changes and open a pull request.

Or you are always welcome to contact me via email!

When you have created a pull request, we can take a look and improve on this package! ^_^

## Authors

* Kim Hyun Bin 
* [Email](KIMH0004@e.ntu.edu.sg)
* [Kaggle](https://www.kaggle.com/kimmik123)

## Version History

* 0.2.0 
    * Major overhaul for organization (2023.04.04)
    
* 0.1.0
    * Fixes for Classification Module (2023.04.03)

* 0.0.11
    * Separation of Regression and Classification for clarity (2023.02.04)

* 0.0.5
    * Various regression bug fixes(2023.01.29)

* 0.0.1
    * Initial Release (2023.01.28)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
