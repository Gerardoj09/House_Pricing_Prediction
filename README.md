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

### Questions to answer:

- What neighborhoods have the highest value?
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

After seeing how our database was made up, we started by knowing the target variable 'SalePrice', the first thing we did was checking the distribution of our target variable and from what we can see is skewed to the right, we will have to use the logarithmic function to mitigate its skewness.

![image](https://user-images.githubusercontent.com/66183125/155928748-00b5a180-669b-435c-a258-4d615d54cb66.png)


As previously mentioned, if we do not count the objective variable, we have a dataset made up of 80 columns from which we have to correctly choose the ones that support the model that we will generate later in order to predict the price of houses. For this reason, we made a heatmap which will show us all the variables that have a correlation greater than .46 with our dependent variable, a correlation of .40 or .50 can be considered as a median correlation, any value below this was considered such as a low correlation or a null correlation between our variables, in addition to choosing variables with no correlation could affect the result of our model. For the same reason, they only chose 13 variables with the highest possible correlation within the numerical variables.

![image](https://user-images.githubusercontent.com/66183125/155927411-b19e09de-7088-44fc-990f-65780a12399a.png)


Once the most important numerical variables have been chosen, now it is time to analyze them one by one to know how the distribution of their data is, and thus know if any of them are skewed to the right or left, and if it is necessary also apply the natural logarithm to nullify its skweness as much as possible

![image](https://user-images.githubusercontent.com/66183125/155928886-8010157e-371b-49d4-b834-d71bb9b4f605.png)
![image](https://user-images.githubusercontent.com/66183125/155928924-c98eb06c-300e-4172-bb05-bc17006ce68b.png)

All the chosen variables having a high correlation, we could expect that graphically they would have a good interaction with the price of the houses. So we will plot the relationship they have with our target variable through scatter plots.

![image](https://user-images.githubusercontent.com/66183125/156906422-30065499-85be-40a3-84f6-ad46689cc078.png)
![image](https://user-images.githubusercontent.com/66183125/156906430-fd00ef2e-63e6-4818-aea3-3f674503ca34.png)

Apparently there is a good relationship between 'SalePrice' and the variables that were chosen, there is really nothing out of the ordinary,depending on how high the correlation is, the scatter plot will look better or worst. So now is the time to review the variables  in more detail, to do this in the notebook the dataframe was divided in to 4 parts  so  in this way we can pay more attention to each variable. We are trying to look for  missing values, different type of data, and any other variable that seems useless to us.

This is just one of the parts that was divided , and it was applied the info function to see missing values and the type of variables it has, but we will cover more examples of the decisions made in this section.

A clear example that is seen for the treatment of missing values ​​is that there are variables that are not worth considering for the simple fact that the amount of data they have is minimal. In the image above you can see how the 'Alley' variable only has 198 values ​​out of 2919, so variables like this that only have a very small amount of data have to be removed from the dataset.

![image](https://user-images.githubusercontent.com/66183125/156907203-be878460-9d18-42c5-9fb7-e7ff98165fbe.png)

We were also able to observe variables that mostly had only one of two possible values, since the difference is too great, it is best to remove the variable since it will not have any analysis effect, a variable that almost entirely has 100% of a value.

![image](https://user-images.githubusercontent.com/66183125/156953307-70afe4c0-d4e1-4418-9129-652e7f9870c7.png)
![image](https://user-images.githubusercontent.com/66183125/156953359-c50818aa-5319-4b50-a0d1-65927c1195c5.png)

Another problem we encountered was making the decision of whether it was best to remove the rows with missing values ​​from certain columns or whether it was best to replace those values ​​to avoid losing information, but to be honest, it is not a decision or a rule that can be applied to all cases. for example there are variables that have missing data for a reason, such as firplaces or basement, they appear to us as if certain rows had no information but this is not by mistake, it is simply that these houses do not have any of the characteristics mentioned, for the same reason all the variables that had the similar problem, all those values ​​had to be replaced with some other that would show that there is no such characteristic in the house.

![image](https://user-images.githubusercontent.com/66183125/156956068-15099e39-9894-4239-a24c-cf6062714e06.png)

