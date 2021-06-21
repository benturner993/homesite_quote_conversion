## Objective

Before asking someone on a date or skydiving, it's important to know your likelihood of success. The same goes for quoting home insurance prices to a potential customer. Homesite, a leading provider of homeowners insurance, does not currently have a dynamic conversion rate model that can give them confidence a quoted price will lead to a purchase. 

Using an anonymized database of information on customer and sales activity, including property and coverage information, Homesite is challenging you to predict which customers will purchase a given quote. Accurately predicting conversion would help Homesite better understand the impact of proposed pricing changes and maintain an ideal portfolio of customer segments. 

Kaggle - https://www.kaggle.com/c/homesite-quote-conversion/

## Contents
This repository contains:

- *Kaggle Submissions.ipynb* - Notebook containing Logistic Regression, Elastic Net, Decision Tree, Random Forest, XGB HyperOpt, XGB SkOpt and XGB Ax and methodology to stack all models
- *tools.py* - python file containing useful python functions

## Results

Using only 5000 records, no feature engineering and no categorical features, this output ranks 1574/1755.

## Future Developments
Developments to investigate / consider:

- Add in categorical variables and add suitable transformation
- Need to check num_boost_round and n_estimators and early stopping

## Author
Ben Turner
Q2 2021