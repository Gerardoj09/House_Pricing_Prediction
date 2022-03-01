# House_Pricing_Prediction

## Overview of the Project

Buying a house can be a very exhausting process, there is a great variety of houses, many factors that can cause the price of a house to increase or decrease, it may be located in an area that you like or that you do not like so much, there are simply so many factors that must be taken into consideration before being able to buy a house according to the budget that one has. For this project we will try to predict the value of houses in the state of Iowa and more specifically in the Ames area. Our dataset consists of 79 variables that describe in detail the characteristics of the houses in this state. We will apply different machine learning models with the hope of having the best possible accuracy when predicting the value of houses. The main reasons for choosing this topic was wanting to know a little more about how the home market works and to test our knowledge of machine learning, we believe that it is a good test for us to know what kind of models we should work with and In addition to this, what kind of treatment should we give to the data.

## List Members

- Josue Emmanuel Aviles Ledezma
- Humberto Rodriguez
- Gerardo Jimenez

## Resources
 
- Dataset obtained from Kaggle https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data 
- Datasets: test.csv, train.csv, Data_Description.txt
- Software: Python 3.7.0, HTML, CSS, SQL, MongoDB, JupyterNotebook

#### Presentation: https://docs.google.com/presentation/d/1RP4-LQ4zja44UlSFVeQPNpQE2bsh4QSUvL4yCp1ztZA/edit?usp=sharing

### Questions to answer:

- What neighborhoods have the highest value?
- What variables most influence the increase or decrease in the price of a house?
- What variables have less influence on the increase or decrease in the price of a house?
- What is the average size of a house in Iowa?
- What is the average price of the value of the houses?

## ETL
### To clean and organize the raw data downloaded from Kaggle, the number of features were reduced by dropping redundant columns and identify the categorical and numerical data. Some minor changes were conducted to adapt the data before start the transformation phase. To process categorical data, a get_dummies and LabelEnconder functions were applied.

## Description about the database
### The dataset is stored in MongoDB which is classified as a NoSQL database which uses JSON-like files and schemas.

## Machine Learning Analysis
### To start our analysis, the first thing we do is check the distribution of our target variable and from what we can see is skewed to the right, we will have to use the logarithmic function to mitigate its skewness.

![image](https://user-images.githubusercontent.com/66183125/155928748-00b5a180-669b-435c-a258-4d615d54cb66.png)


## After this we have to check the correlation that the variables have with each other and even more important is to check what is the correlation that it has with the target variable, since here it will depend on which will be the most important variables to add to the model.

![image](https://user-images.githubusercontent.com/66183125/155927411-b19e09de-7088-44fc-990f-65780a12399a.png)

## As mentioned before, having chosen the most important variables or those with the most impact in the correlation analysis, we have to see how their information is distributed graphically and, from this, know what to do with these variables.

![image](https://user-images.githubusercontent.com/66183125/155928886-8010157e-371b-49d4-b834-d71bb9b4f605.png)
![image](https://user-images.githubusercontent.com/66183125/155928924-c98eb06c-300e-4172-bb05-bc17006ce68b.png)

