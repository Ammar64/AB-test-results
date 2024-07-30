## Analyze A/B Test Results

This project will assure you have mastered the subjects covered in the statistics lessons.  The hope is to have this project be as comprehensive of these topics as possible.  Good luck!

Corresponding with this notebook is a slide deck where you will need to update all the portions in red.  Completing the notebook will provide all the results needed for the slides.  **Correctly completing the slides is a required part of the project.**

## Table of Contents
- [Introduction](#intro)
- [Part I - Descriptive Statistics](#descriptive)
- [Part II - Probability](#probability)
- [Part III - Experimentation](#experimentation)
- [Part IV - Algorithms](#algorithms)


<a id='intro'></a>
### Introduction

A/B tests are very commonly performed by data analysts and data scientists.  For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.

**As you work through this notebook, follow along in the classroom and answer the corresponding quiz questions associated with each question.** The labels for each classroom concept are provided for each question.  This will assure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the criteria.  As a final check, assure you meet all the criteria on the [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).

<a id='descriptive'></a>
#### Part I - Descriptive Statistics

To get started, let's import our libraries.


```python
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
%matplotlib inline
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(0)
```

For each of the parts of question `1` notice links to [pandas documentation](https://pandas.pydata.org/) is provided to assist with answering the questions.  Though there are other ways you could solve the questions, the documentation is provided to assist you with one fast way to find the answer to each question.


`1.a)` Now, read in the `ab_data.csv` data. Store it in `df`. Read in the dataset and take a look at the top few rows here. **This question is completed for you**:


```python
df = pd.read_csv('ab_data.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>group</th>
      <th>converted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>UK</td>
      <td>control</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>US</td>
      <td>treatment</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>UK</td>
      <td>treatment</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>UK</td>
      <td>control</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>UK</td>
      <td>treatment</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



`b)` Use the below cell to find the number of rows in the dataset. [Helpful  Pandas Link](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shape.html#pandas.DataFrame.shape)


```python
df.shape[0]
```




    69889



`c)` The proportion of users converted.  [Helpful  Pandas Link](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.mean.html)


```python
df['converted'].mean()
```




    0.13047832992316388



`d)` Do any of the rows have missing values? [Helpful Pandas Link One](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isnull.html) and [Helpful Pandas Link Two](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sum.html)


```python
df.isnull().sum()
#No
```




    country      0
    group        0
    converted    0
    dtype: int64



`e)` How many customers are from each country? Build a bar chart to show the count of visits from each country.


```python
# number of visitors from each country - pull the necessary code from the next cell to provide just the counts
df['country'].value_counts()
```




    US    48850
    UK    17551
    CA     3488
    Name: country, dtype: int64




```python
# bar chart of results - this part is done for you
df['country'].value_counts().plot(kind='bar');
plt.title('Number of Visits From Each Country');
plt.ylabel('Count of Visits');
plt.show();
```


    
![png](images/output_12_0.png)
    


`f)` Recognize that all of your columns are of a **categorical data type** with the exception of one.  Which column is not **categorical**? [Helpful Pandas Link](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.info.html)


```python
df.info()
# It's the 'converted' column
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 69889 entries, 0 to 69888
    Data columns (total 3 columns):
     #   Column     Non-Null Count  Dtype 
    ---  ------     --------------  ----- 
     0   country    69889 non-null  object
     1   group      69889 non-null  object
     2   converted  69889 non-null  int64 
    dtypes: int64(1), object(2)
    memory usage: 1.6+ MB


`g)` What are the possible values of the `converted` column?  Does it make sense that these values are the only possible values? Why or why not? 

**Here you can use one of the functions you used in an earlier question**.


```python
df['converted'].unique()
# Only 0 and 1
# That makes sense because user is either not converted (0) or converted (1)
```




    array([0, 1])



<a id='probability'></a>
#### Part II - Probability

`1.` Now that you have had a chance to learn more about the dataset, let's look more at how different factors are related to `converting`.

`a)` What is the probability of an individual converting regardless of the page they receive or the country they are from? Simply, what is the chance of conversion in the dataset?


```python
df['converted'].mean()
# It's 13.05%
```




    0.13047832992316388



`b)` Given that an individual was in the `control` group, what is the probability they converted? **This question is completed for you**


```python
df.query('group == "control"')['converted'].mean()
```




    0.1052540515600669



`c)` Given that an individual was in the `treatment` group, what is the probability they converted?


```python
df.query('group == "treatment"')['converted'].mean()
```




    0.15532078043793132



`d)` Do you see evidence that the treatment is related to higher `converted` rates?


```python
# YES
```

`e)` What is the probability that an individual was in the `treatment`?


```python
df.query('group == "treatment"').shape[0] / df.shape[0]
```




    0.5038131894861853



`f)` What is the probability that an individual was from Canada `CA`?


```python
df[df['country'] == 'CA'].shape[0] / df.shape[0]
```




    0.04990771079855199



`g)` Given that an individual was in the `US`, what was the probability that they `converted`? **This question is completed for you**

$P(\text{converted} == 1|\text{country} ==\text{"US"})$




```python
df.query('country == "US"')['converted'].mean()
```




    0.13277379733879222



`h)` Given that an individual was in the `UK`, what was the probability that they `converted`? 

$P(\text{converted} == 1|\text{country} ==\text{"UK"})$


```python
df.query('country == "UK"')['converted'].mean()
```




    0.12512107572218106



`i)` Do you see evidence that the `converted` rate might differ from one country to the next?


```python
# No, it's so close
```

`j)` Consider the table below, fill in the conversion rates below to look at how conversion by country and treatment group vary.  The `US` column is done for you, and two methods for calculating the probabilities are shown - **COMPLETE THE REST OF THE TABLE**.  Does it appear that there could be an interaction between how country and treatment impact conversion?

These two values that are filled in can be written as:

$P(\text{converted} == 1|(\text{country} ==\text{"US" AND }\text{group} ==\text{"control"})) = 10.7\%$

$P(\text{converted} == 1|(\text{country} ==\text{"US" AND }\text{group} ==\text{"treatment"})) = 15.8\%$

|             | US          | UK          | CA          |
| ----------- | ----------- | ----------- | ----------- |
| Control     | 10.7%       |  %          |  %          |
| Treatment   | 15.8%       |  %          |  %          |


```python
# Method 1  - explicitly calculate each probability
print(df.query('country == "US" and group == "control" and converted == 1').shape[0]/df.query('country == "US" and group == "control"').shape[0]) 
print(df.query('country == "US" and group == "treatment" and converted == 1').shape[0]/df.query('country == "US" and group == "treatment"').shape[0])
```

    0.10731404958677686
    0.1577687626774848



```python
# Method 2 - quickly calculate using `groupby`
print("US\n" ,df.query('country == "US"').groupby('group')['converted'].mean(), end="\n\n")
print("UK\n" ,df.query('country == "UK"').groupby('group')['converted'].mean(), end="\n\n")
print("CA\n" ,df.query('country == "CA"').groupby('group')['converted'].mean(), end="\n\n")
```

    US
     group
    control      0.107314
    treatment    0.157769
    Name: converted, dtype: float64
    
    UK
     group
    control      0.101649
    treatment    0.148698
    Name: converted, dtype: float64
    
    CA
     group
    control      0.094474
    treatment    0.154017
    Name: converted, dtype: float64
    


|             | US          | UK          | CA          |
| ----------- | ----------- | ----------- | ----------- |
| Control     | 10.7%       |  10.16%     |  09.45%     |
| Treatment   | 15.8%       |  14.87%     |  15.40%     |


```python
# Faster method
df.groupby(["group", "country"])["converted"].mean().unstack().mul(100).round(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>country</th>
      <th>CA</th>
      <th>UK</th>
      <th>US</th>
    </tr>
    <tr>
      <th>group</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>control</th>
      <td>9.45</td>
      <td>10.16</td>
      <td>10.73</td>
    </tr>
    <tr>
      <th>treatment</th>
      <td>15.40</td>
      <td>14.87</td>
      <td>15.78</td>
    </tr>
  </tbody>
</table>
</div>



<a id='experimentation'></a>
### Part III - Experimentation

`1.` Consider you need to make the decision just based on all the data provided.  If you want to assume that the control page is better unless the treatment page proves to be definitely better at a Type I error rate of 5%, you state your null and alternative hypotheses in terms of **$p_{control}$** and **$p_{treatment}$** as:  

$H_{0}: p_{control} >= p_{treatment}$

$H_{1}: p_{control} < p_{treatment}$

Which is equivalent to:

$H_{0}: p_{treatment} - p_{control} <= 0$

$H_{1}: p_{treatment} - p_{control} > 0$


Where  
* **$p_{control}$** is the `converted` rate for the control page
* **$p_{treatment}$** `converted` rate for the treatment page

**Note for this experiment we are not looking at differences associated with country.**

Assume under the null hypothesis, $p_{treatment}$ and $p_{control}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{treatment}$ and $p_{control}$ are equal. Furthermore, assume they are equal to the **converted** rate in `df` regardless of the page. **These are set in the first cell below.**<br><br>

* Use a sample size for each page equal to the ones in `df`. **These are also set below.**  <br><br>

* Perform the sampling distribution for the difference in `converted` between the two pages over 500 iterations of calculating an estimate from the null.  <br><br>

* Use the cells below to provide the necessary parts of this simulation.  

If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 4** in the classroom to make sure you are on the right track.<br><br>

`a)` The **convert rate** for $p_{treatment}$ under the null.  The **convert rate** for $p_{control}$ under the null. The sample size for the `control` and the sample size for the `treatment` are from the original dataset. **All of these values are set below, and set the stage for the simulations you will run for the rest of this section.**


```python
p_control_treatment_null  = df['converted'].mean()
n_treatment = df.query('group == "treatment"').shape[0]
n_control = df.query('group == "control"').shape[0]

print(f'n_treatment {n_treatment}')
print(f'n_control {n_control}')
```

    n_treatment 35211
    n_control 34678


`b)` Use the results from part `a)` to simulate `n_treatment` transactions with a convert rate of `p_treatment_null`.  Store these $n_{treatment}$ 1's and 0's in a `list` of **treatment_converted**.  It should look something like the following (the 0's and and 1's **don't** need to be the same): 

`[0, 0, 1, 1, 0, ....]` 


```python
p_treatment_null = df.query('group == "treatment"')['converted'].mean()
treatment_converted = np.random.choice(2, size=n_treatment, p=[1-p_treatment_null, p_treatment_null])

```

`c)` Use the results from part `a)` to simulate `n_control` transactions with a convert rate of `p_control_null`.  Store these $n_{treatment}$ 1's and 0's in a `list` of **control_converted**.  It should look something like the following (the 0's and and 1's **don't** need to be exactly the same): 

`[0, 0, 1, 1, 0, ....]` 


```python
p_control_null = df.query('group == "control"')['converted'].mean()
control_converted = []
for i in range(n_control):
    converted = np.random.choice(2, p=[1-p_control_null, p_control_null])
    control_converted.append(converted)

```

`d)` Find the estimate for $p_{treatment}$ - $p_{control}$ under the null using the simulated values from part `(b)` and `(c)`.


```python
diff = np.mean(treatment_converted) - np.mean(control_converted)
```

`e)` Simulate 500 $p_{treatment}$ - $p_{control}$ values using this same process as `b)`- `d)` similarly to the one you calculated in parts **a. through g.** above.  Store all 500 values in an numpy array called **p_diffs**.  This array should look similar to the below **(the values will not match AND this will likely take a bit of time to run)**:

`[0.001, -0.003, 0.002, ...]`


```python
p_diffs = []
for _ in range(500):
    bootstrap = df.sample(df.shape[0], replace=True)
    control_converted = bootstrap.query('group == "control"')['converted']
    treatment_converted = bootstrap.query('group == "treatment"')['converted']
    p_control_null = control_converted.mean()
    p_treatment_null = treatment_converted.mean()
    p_diff = p_treatment_null - p_control_null
    p_diffs.append(p_diff)
p_diffs = pd.Series(p_diffs)
p_diffs -= p_diffs.mean()
```

`f)` Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.


```python
p_diffs.hist(bins=35)
```




    <AxesSubplot: >




    
![png](images/output_53_1.png)
    


`g)` What proportion of the **p_diffs** are greater than the difference observed between `treatment` and `control` in `df`?


```python
(p_diffs > diff).mean()
```




    0.0



`h)` In words, explain what you just computed in part `g)`  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages using our Type I error rate of 0.05?

**P-Value**<br>
the P-Value is 0.000 which suggests that we should reject the null (the treatment page is better) because it's less than 0.05

<a id='algorithms'></a>
### Part IV - Algorithms

`1.` In this final part, you will see that the result you acheived in the previous A/B test can also be acheived by performing regression.  All the code needed for the modeling and results of the modeling for sections `b) - f)` have been completed for you. 

**You will need to complete sections `a)` and `g)`.**  

**Then use the code from `1.` to assist with the question `2.`   You should be able to modify the code to assist in answering each of question 2's parts.**<br><br>

`a)` Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

**Linear Regression**

The goal is to use **statsmodels** to fit the regression model you specified in part `a)` to see if there is a significant difference in conversion based on which page a customer receives.  

`b)` However, you first need to create a column for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

It may be helpful to look at the [get_dummies documentation](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html) to encode the `ab_page` column.

Below you can see an example of the new columns that will need to be added (The order of columns is not important.): **This question is completed for you**

##### Example DataFrame
| intercept   | group       | ab_page     | converted   |
| ----------- | ----------- | ----------- | ----------- |
| 1           |  control    |  0          |  0          |
| 1           |  treatment  |  1          |  0          |
| 1           |  treatment  |  1          |  0          |
| 1           |  control    |  0          |  0          |
| 1           |  treatment  |  1          |  1          |
| 1           |  treatment  |  1          |  1          |
| 1           |  treatment  |  1          |  0          |
| 1           |  control    |  0          |  1          |


```python
df['intercept'] = 1
df['ab_page'] = pd.get_dummies(df['group'])['treatment']
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>group</th>
      <th>converted</th>
      <th>intercept</th>
      <th>ab_page</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>UK</td>
      <td>control</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>US</td>
      <td>treatment</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>UK</td>
      <td>treatment</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>UK</td>
      <td>control</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>UK</td>
      <td>treatment</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



`c)`  Create your `X` matrix and `y` response column that will be passed to your model, where you are testing if there is a difference in `treatment` vs. `control`. **This question is completed for you**


```python
X = df[['intercept', 'ab_page']]
y = df['converted']
```

`d)` Use **statsmodels** to import and fit your regression model on the `X` and `y` from part `c)`. 

You can find the [statsmodels documentation to assist with this exercise here](https://www.statsmodels.org/stable/discretemod.html).  **This question is completed for you**


```python
import statsmodels.api as sm

# Logit Model
logit_mod = sm.Logit(y, X)
logit_res = logit_mod.fit()
```

    Optimization terminated successfully.
             Current function value: 0.384516
             Iterations 6


`e)` Provide the summary of your model below. **This question is completed for you**


```python
logit_res.summary()
```




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>converted</td>    <th>  No. Observations:  </th>  <td> 69889</td>  
</tr>
<tr>
  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>  <td> 69887</td>  
</tr>
<tr>
  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>  <td>     1</td>  
</tr>
<tr>
  <th>Date:</th>            <td>Tue, 30 Jul 2024</td> <th>  Pseudo R-squ.:     </th> <td>0.007175</td> 
</tr>
<tr>
  <th>Time:</th>                <td>21:22:51</td>     <th>  Log-Likelihood:    </th> <td> -26873.</td> 
</tr>
<tr>
  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th> <td> -27068.</td> 
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th> <td>1.810e-86</td>
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>intercept</th> <td>   -2.1402</td> <td>    0.017</td> <td> -122.305</td> <td> 0.000</td> <td>   -2.174</td> <td>   -2.106</td>
</tr>
<tr>
  <th>ab_page</th>   <td>    0.4467</td> <td>    0.023</td> <td>   19.539</td> <td> 0.000</td> <td>    0.402</td> <td>    0.492</td>
</tr>
</table>



`f)` What is the p-value associated with **ab_page**? Does it lead you to the same conclusion you drew in the **Experiment** section.

**P-Value: 0.000**

`2. a)` Now you will want to create two new columns as dummy variables for `US` and `UK`.  Again, use `get_dummies` to add these columns.  The dataframe you create should include at least the following columns (If both columns for `US` and `UK` are `0` this represents `CA`.  The order of rows and columns is not important for you to match - it is just to illustrate how columns should connect to one another.):

##### Example DataFrame
| intercept   | group       | ab_page     | converted   | country     |  US         | UK          |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| 1           |  control    |  0          |  0          |  US         |  1          |  0          |
| 1           |  treatment  |  1          |  0          |  UK         |  0          |  1          |
| 1           |  treatment  |  1          |  0          |  US         |  1          |  0          |
| 1           |  control    |  0          |  0          |  US         |  1          |  0          |
| 1           |  treatment  |  1          |  1          |  CA         |  0          |  0          |
| 1           |  treatment  |  1          |  1          |  UK         |  0          |  1          |
| 1           |  treatment  |  1          |  0          |  US         |  1          |  0          |
| 1           |  control    |  0          |  1          |  US         |  1          |  0          |


```python
### Create the necessary dummy variables
df[['US', 'UK', 'CA']] = pd.get_dummies(df['country'])
df.drop('CA', axis=1, inplace=True)

df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>group</th>
      <th>converted</th>
      <th>intercept</th>
      <th>ab_page</th>
      <th>US</th>
      <th>UK</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>UK</td>
      <td>control</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>US</td>
      <td>treatment</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>UK</td>
      <td>treatment</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>UK</td>
      <td>control</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>UK</td>
      <td>treatment</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>69884</th>
      <td>UK</td>
      <td>treatment</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>69885</th>
      <td>UK</td>
      <td>control</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>69886</th>
      <td>UK</td>
      <td>treatment</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>69887</th>
      <td>US</td>
      <td>control</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>69888</th>
      <td>US</td>
      <td>treatment</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>69889 rows × 7 columns</p>
</div>



`b)`  Create your `X` matrix and `y` response column that will be passed to your model, where you are testing if there is 
* a difference in `converted` between `treatment` vs. `control`
* a difference in `converted` between `US`, `UK`, and `CA`


```python
X = df[['intercept', 'ab_page', 'US', 'UK']]
Y = df['converted']
```

`c)` Use **statsmodels** to import and fit your regression model on the `X` and `y` from part `b)`. 
You can find the [statsmodels documentation to assist with this exercise here](https://www.statsmodels.org/stable/discretemod.html).


```python
model = sm.Logit(Y, X).fit()
```

    Optimization terminated successfully.
             Current function value: 0.384463
             Iterations 6


`d)` Provide the summary of your model below.


```python
model.summary()
```




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>converted</td>    <th>  No. Observations:  </th>  <td> 69889</td>  
</tr>
<tr>
  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>  <td> 69885</td>  
</tr>
<tr>
  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>  <td>     3</td>  
</tr>
<tr>
  <th>Date:</th>            <td>Tue, 30 Jul 2024</td> <th>  Pseudo R-squ.:     </th> <td>0.007312</td> 
</tr>
<tr>
  <th>Time:</th>                <td>21:22:56</td>     <th>  Log-Likelihood:    </th> <td> -26870.</td> 
</tr>
<tr>
  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th> <td> -27068.</td> 
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th> <td>1.778e-85</td>
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>intercept</th> <td>   -2.1203</td> <td>    0.019</td> <td> -112.003</td> <td> 0.000</td> <td>   -2.157</td> <td>   -2.083</td>
</tr>
<tr>
  <th>ab_page</th>   <td>    0.4466</td> <td>    0.023</td> <td>   19.534</td> <td> 0.000</td> <td>    0.402</td> <td>    0.491</td>
</tr>
<tr>
  <th>US</th>        <td>   -0.0727</td> <td>    0.053</td> <td>   -1.372</td> <td> 0.170</td> <td>   -0.177</td> <td>    0.031</td>
</tr>
<tr>
  <th>UK</th>        <td>   -0.0660</td> <td>    0.026</td> <td>   -2.490</td> <td> 0.013</td> <td>   -0.118</td> <td>   -0.014</td>
</tr>
</table>



`e)` What do the `p-values` associated with `US` and `UK` suggest in relation to how they impact `converted`? 

**The conversion rate from `US` is NOT statistically significant compared to `CA`**<br>
**The conversion rate from `UK` is statistically significant compared to `CA`**

