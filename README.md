# House_Pricing_Prediction

## Overview of the Project

Buying a house can be a very exhausting process, there is a great variety of houses, many factors that can cause the price of a house to increase or decrease, it may be located in an area that you like or that you do not like so much, there are simply so many factors that must be taken into consideration before being able to buy a house according to the budget that one has. For this project we will try to predict the value of houses in the state of Iowa and more specifically in the Ames area. Our dataset consists of 79 variables that describe in detail the characteristics of the houses in this state. 

We will apply different machine learning models with the hope of having the best possible accuracy when predicting the value of houses. The main reasons for choosing this topic was wanting to know a little more about how the home market works and to test our knowledge of machine learning, we believe that it is a good test for us to know what kind of models we should work with and in addition to this, what kind of treatment should we give to the data.

## List Members

- Josue Emmanuel Aviles Ledezma
- Humberto Rodriguez
- Gerardo Jimenez

## Resources
 
- Dataset obtained from Kaggle https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data 
- Datasets: test.csv, train.csv, Data_Description.txt
- Software: Python 3.7.0, HTML, CSS, SQL, MongoDB, JupyterNotebook

#### Presentation: https://docs.google.com/presentation/d/1RP4-LQ4zja44UlSFVeQPNpQE2bsh4QSUvL4yCp1ztZA/edit?usp=sharing
#### Interactive presentation: https://ml-model-house-pricing-predict.herokuapp.com/database

### Questions to answer:

- What variables most influence the increase or decrease in the price of a house?
- What variables have less influence on the increase or decrease in the price of a house?
- What is the average size of a house in Iowa?
- What is the average price of the value of the houses?


To clean and organize the raw data downloaded from Kaggle, the number of features were reduced by dropping redundant columns and identifying categorical and numerical data. Some minor changes were conducted to adapt the data before start the transformation phase. To process categorical data, a get_dummies and LabelEnconder functions were applied.

## Database description  

Our dataset consists of a total of 81 variables, both numerical and categorical, that focus on the description of the characteristics of a house, such as what material is it built on? What neighborhood is it in? How many bedrooms does it have? What does a garage have and what conditions is it in? . The most important variable within the whole set is 'SalePrice' since this will be our target variable where we hope to generate a model that can predict its value as correctly as possible.

The dataset is stored in MongoDB which is classified as a NoSQL database which uses JSON-like files and schemas.

## ETL

The first thing that was done was to take a quick look at our database, since it is important to know what type of data you are going to work with. If they are numerical data such as floating point or integer, or failing that, if they are categorical data, they could become nominal or ordinal. Each type of data must be given a different analysis and treatment.

![image](https://user-images.githubusercontent.com/66183125/156894833-50f38fd6-644c-4ac1-b956-1e5ded995833.png)

### Data Distribution
After seeing how our database was made up, we started by knowing the target variable 'SalePrice', the first thing we did was checking the distribution of our target variable and from what we can see is skewed to the right, we will have to use the logarithmic function to mitigate its skewness.

Before applying the natural logarithm

![image](https://user-images.githubusercontent.com/66183125/155928748-00b5a180-669b-435c-a258-4d615d54cb66.png)

After applying the natural logarithm

![image](https://user-images.githubusercontent.com/66183125/157169707-8ccfc92c-2180-4db8-8b04-fe90ad6e65bc.png)

### Correlation
As previously mentioned, if we do not count the objective variable, we have a dataset made up of 80 columns from which we have to correctly choose the ones that support the model that we will generate later in order to predict the price of houses. For this reason, we made a heatmap which will show us all the variables that have a correlation greater than .46 with our dependent variable, a correlation of .40 or .50 can be considered as a median correlation, any value below this was considered such as a low correlation or a null correlation between our variables, in addition to choosing variables with no correlation could affect the result of our model. For the same reason, they only chose 13 variables with the highest possible correlation within the numerical variables.

![image](https://user-images.githubusercontent.com/66183125/155927411-b19e09de-7088-44fc-990f-65780a12399a.png)

### Distribution of the most correlated variables and scatter plots against the target variable

Once the most important numerical variables have been chosen, now it is time to analyze them one by one to know how the distribution of their data is, and thus know if any of them are skewed to the right or left, and if it is necessary also apply the natural logarithm to nullify its skweness as much as possible

![image](https://user-images.githubusercontent.com/66183125/155928886-8010157e-371b-49d4-b834-d71bb9b4f605.png)
![image](https://user-images.githubusercontent.com/66183125/155928924-c98eb06c-300e-4172-bb05-bc17006ce68b.png)

All the chosen variables having a high correlation, we could expect that graphically they would have a good interaction with the price of the houses. So we will plot the relationship they have with our target variable through scatter plots.

![image](https://user-images.githubusercontent.com/66183125/156906422-30065499-85be-40a3-84f6-ad46689cc078.png)
![image](https://user-images.githubusercontent.com/66183125/156906430-fd00ef2e-63e6-4818-aea3-3f674503ca34.png)

Apparently there is a good relationship between 'SalePrice' and the variables that were chosen, there is really nothing out of the ordinary,depending on how high the correlation is, the scatter plot will look better or worst. 

### Missing data, Dropping columns and Replacing values

Now is the time to review the variables  in more detail, to do this in the notebook the dataframe was divided in to 4 parts  so  in this way we can pay more attention to each variable. We are trying to look for  missing values, different type of data, and any other variable that seems useless to us.

This is just one of the parts that was divided , and it was applied the info function to see missing values and the type of variables it has, but we will cover more examples of the decisions made in this section.

A clear example that is seen for the treatment of missing values ​​is that there are variables that are not worth considering for the simple fact that the amount of data they have is minimal. In the image above you can see how the 'Alley' variable only has 198 values ​​out of 2919, so variables like this that only have a very small amount of data have to be removed from the dataset.

![image](https://user-images.githubusercontent.com/66183125/156907203-be878460-9d18-42c5-9fb7-e7ff98165fbe.png)

We were also able to observe variables that mostly had only one of two possible values, since the difference is too great, it is best to remove the variable since it will not have any analysis effect, a variable that almost entirely has 100% of a value.

![image](https://user-images.githubusercontent.com/66183125/156953307-70afe4c0-d4e1-4418-9129-652e7f9870c7.png)
![image](https://user-images.githubusercontent.com/66183125/156953359-c50818aa-5319-4b50-a0d1-65927c1195c5.png)

Another problem we encountered was making the decision of whether it was best to remove the rows with missing values ​​from certain columns or whether it was best to replace those values ​​to avoid losing information, but to be honest, it is not a decision or a rule that can be applied to all cases. for example there are variables that have missing data for a reason, such as firplaces or basement, they appear to us as if certain rows had no information but this is not by mistake, it is simply that these houses do not have any of the characteristics mentioned, for the same reason all the variables that had the similar problem, all those values ​​had to be replaced with some other that would show that there is no such characteristic in the house.

![image](https://user-images.githubusercontent.com/66183125/156956068-15099e39-9894-4239-a24c-cf6062714e06.png)

### Transforming categorical data

At the moment of wanting to transform the categorical data to numerical data, our first problem to analyze is, what kind of categorical data do we have? , since we can have nominal categorical data and ordinal categorical data, so the same treatment cannot be applied to both cases.

- Nominal data is data that we assign individual values ​​to named categories that do not have a value or range
- Ordinal data have values that are assigned to categories that have some sort of order

The examples that we can give would be the neighborhood variable, which would be categorized as a nominal variable since no matter which neighborhood is chosen, it will not have any order as such. But instead the variable 'BsmtQual', shows us different types of values ​​to which we can give a certain order, for example if it does not have a basement it will be given a value of 0, however if it has a value of 'Excellent' that is the highest value, it will be given a value of 5

![image](https://user-images.githubusercontent.com/66183125/156962216-a603152d-328f-4969-97ef-72e3cf099fca.png)

![image](https://user-images.githubusercontent.com/66183125/156962128-92e29e5d-72e7-475e-9f9b-de9f1a8f0180.png)

Then, for all the nominal variables, their transformation to numerics was through creating dummy variables in each of their values. and in the case of ordinal variables, we gave them a specific value depending on the order that each value will carry

![image](https://user-images.githubusercontent.com/66183125/156962305-2fd3edce-99ff-4a34-bc45-79d91da1e8aa.png)
![image](https://user-images.githubusercontent.com/66183125/156962341-22f62a1f-8f83-4c3b-9382-3dc0a4972e1a.png)

### Applying Machine Learning Models

Having our database ready, it is time to apply machine learning models to find out which of all can best predict the price of houses.Four models were chosen to be able to assess which of them best predicted the price of houses. The models were the following:

- RandomForestRegressor
- XGB Regressor
- Multiple Linear Regression
- Gradient Boosting Regressor

For the 4 models the conditions were the same, the dataset was divided into 80% training set and 20% testing set, and the values ​​of the database would be randomly chosen to conform both bases.

![image](https://user-images.githubusercontent.com/66183125/156967455-ab2bd132-8d12-48a7-8efc-2183684cd53d.png)

And the way in which the best model would be chosen would be through two metrics:

- The first evaluation would be through 'mean_squared_error', where we would seek to have the lowest error

- The second evaluation would be through R2, where what we are looking for is to have the highest possible value

## Results

These were the results obtained from the 4 selected models:

### RandomForestRegressor
![image](https://user-images.githubusercontent.com/66183125/156968657-74989954-401f-48db-8ce4-5bc0b10d1fa0.png)

### XGB Regressor
![image](https://user-images.githubusercontent.com/66183125/156968704-05f62d7d-7ad9-4ea3-b613-ceb44be5313b.png)

### Multiple Linear Regression
![image](https://user-images.githubusercontent.com/66183125/156968735-76f16eaf-56d7-4124-a342-7509159e73f1.png)

### Gradient Boosting Regressor
![image](https://user-images.githubusercontent.com/66183125/156968815-1b91c5e6-aa64-49b6-99aa-9f88603f412c.png)

As we can see, Multiple Linear Regression was the one that obtained the best results both in getting a lower mean square error and a higher R2, so it is the model chosen to predict the price of houses.

### Conclusions

Having carried out this analysis makes it very clear that preparing the database before running the models is too important, since previous tests were carried out before data cleaning and the results were horrible, each variable is different from one another and You have to know how to carefully evaluate which variables are useful for the project you are working on and which ones are not. It is also important to be very careful when making modifications to the variables since this can generate results that are quite different from what you would expect to come out. The most remarkable thing we can say about this first team project is that the decision-making and the way the project develops is not always as one would expect.

A clear example of what was mentioned is that having a normal distribution of your data can be vital for regression analysis, since as we saw in the graph our target variable was skewed to the right and we had to make the decision to apply the logarithmic function in it so that it could have a normal distribution, but in exchange for this the correlation it had with the other variables was favored.

Decisions also had to be made about how much data you can leave, usually when you work with models the amount of data you have is essential to be able to generate models with good results. For this project, decisions were made such as eliminating missing data, useless columns, information had to be replaced so that in this way we would avoid having to eliminate the variable completely. You have to be smart when knowing what data to work with and what data not to.

There were many times when we thought we had the correct result and in the end it was not so, so we had to go back to the beginning to make more modifications and wait to see a new result, this was the process several times until we obtained a result that better convinced us.

Finally, the choice of the model was also somehow difficult since the 4 models performed very well, in addition to this, as the decision was made that the information in the training database and the testing database was out of randomly, each time the models were run they gave different results . So after several attempts on each model, the one that had performed the best most of the time, that would be the model of choice to predict house prices.
