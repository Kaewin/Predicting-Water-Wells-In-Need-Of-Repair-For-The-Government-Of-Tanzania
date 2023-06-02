# phase3_project3

[Presentation](https://docs.google.com/presentation/d/1I34XcvTqbOwh9xBtx27wmkNFPC6luxFFKyY532R_-0I/edit?usp=sharing)

[Google Document Lab Notebook](https://docs.google.com/document/d/1Spref_pjFamfD-KR-_QiNYEyXASlaG7z9inboxcsCjs/edit?usp=sharing)

[Source Of The Data](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/)


# Intro:

## Business Problem: Predicting Water Wells In Need Of Repair For The Government Of Tanzania

## Stakeholder: The Ministry Of Water in Tanzania


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
df_train_target = pd.read_csv('./Data/training_set_labels.csv')
df_train_data = pd.read_csv('./Data/training_set_values.csv')
```

# Collapsing The Target Categories

The goal is to predict wells that require repair.

The target variable is `status_group` which has three categories:

- `functional`
- `functional needs repair`
- `non functional`

I will collapse the categories `functional needs repair` and `non functional` into one category `needs repair` to create a binary target variable.


```python
# Look at df_train_target
df_train_target['status_group'].value_counts()


```




    functional                 32259
    non functional             22824
    functional needs repair     4317
    Name: status_group, dtype: int64




```python
# Change the name of the status_group in df_train_target 'functional needs repair' to 'repair'
# Also change the name of 'non functional' to 'repair'
df_train_target['status_group'] = df_train_target['status_group'].replace('functional needs repair', 'repair')
df_train_target['status_group'] = df_train_target['status_group'].replace('non functional', 'repair')

# Check
df_train_target['status_group'].value_counts()
```




    functional    32259
    repair        27141
    Name: status_group, dtype: int64




```python
# encode status_group as 0, 1
df_train_target['status_group'] = df_train_target['status_group'].astype('category')
df_train_target['status_group'] = df_train_target['status_group'].cat.codes

# Check
df_train_target['status_group'].value_counts(normalize=True)
```




    0    0.543081
    1    0.456919
    Name: status_group, dtype: float64



So now I have collapsed the target categories, and I have encoded the target variable as a binary variable.

Also note that the target variable is not imbalanced.

# Chopping out columns

Okay there is way too much here. I am going to chop out some columns.

I am going to chop out the following columns:

- id
- date recorded
- funder
- installer
- wpt_name
- recorded_by
- scheme_management
- scheme_name

public_meeting
extraction type group and class?

## the data has a high degree of collinearity

I am goign to remove some columns in an attempt to fix that.


```python
# show me the colinearlity of the data
# df_train_data.corr()
```


```python
# show me the colinearlity of the data
df_train_data.corr()

# Drop the following columns

# 'id' - not needed
# 'recorded_by' - only one value
# 'wpt_name' - not needed
# 'num_private' - not needed
# 'subvillage' - not needed
# 'region_code' - not needed
# 'district_code' - not needed
# 'lga' - not needed
# 'ward' - not needed
# 'scheme_name' - not needed
# 'extraction_type_group' - not needed
# 'extraction_type_class' - not needed
# 'management_group' - not needed
# 'payment_type' - not needed
# 'water_quality' - not needed
# 'quantity_group' - not needed
# 'source_type' - not needed
# 'source_class' - not needed
# 'waterpoint_type_group' - not needed
# management - same as management_group
# payment - same as payment_type
# quality_group - same as water_quality
# quantity - same as quantity_group
# date_recorded - not needed
# scheme_management - not needed
# funder - not needed
# public_meeting - not needed

df_train_data = df_train_data.drop(['id', 
                                    'recorded_by', 
                                    'wpt_name', 
                                    'num_private', 



                                    'lga', 
                                    'ward', 
                                    'scheme_name', 
                                    'extraction_type_group', 
                                    'extraction_type_class', 
                                     
                                    'payment_type', 
                                    'water_quality', 
                                    'quantity_group', 
                                    'source_type', 
                                    'source_class', 
                                    'waterpoint_type_group', 
                                    'management', 
                                    'payment', 
                                    'quality_group', 
                                    'quantity', 
                                    'date_recorded',

                                    
                                    'public_meeting'], axis=1)
```


```python
# # one hot encode the categorical data and show me the coliinearlity
# df_train_data = pd.get_dummies(df_train_data)
# df_train_data.corr()
```


```python
# Drop id, date recorded, funder, installer, wpt_name, recorded_by, scheme_management, scheme_name, quantity_group, payment_type, extraction_type_group, extraction_type_class, management_group, public_meeting, permit, num_private
# df_train_data = df_train_data.drop(['id', 'date_recorded', 'funder', 'installer', 'wpt_name', 'recorded_by', 'scheme_management', 'scheme_name', 'quantity_group', 'payment_type', 'extraction_type_group', 'extraction_type_class', 'management_group', 'public_meeting', 'permit', 'num_private'], axis=1)
```


```python
# remove everything but amount_tsh, gps_height, longitude, latitude, region_code, district_code, population, construction_year, lga, ward, recorded_by, scheme_management, scheme_name, management, management_group
# df_train_data = df_train_data.drop(['amount_tsh', 'wpt_name', 'id', 'date_recorded', 'num_private', 'gps_height', 'longitude', 'latitude', 'region_code', 'district_code', 'population', 'construction_year', 'lga', 'ward', 'recorded_by', 'scheme_management', 'scheme_name', 'management', 'management_group', 'public_meeting', 'extraction_type', 'extraction_type_group', 'extraction_type_class', 'payment_type', 'payment'], axis=1)
```


```python
# check what df_train_data has now
df_train_data.head()
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
      <th>amount_tsh</th>
      <th>funder</th>
      <th>gps_height</th>
      <th>installer</th>
      <th>longitude</th>
      <th>latitude</th>
      <th>basin</th>
      <th>subvillage</th>
      <th>region</th>
      <th>region_code</th>
      <th>district_code</th>
      <th>population</th>
      <th>scheme_management</th>
      <th>permit</th>
      <th>construction_year</th>
      <th>extraction_type</th>
      <th>management_group</th>
      <th>source</th>
      <th>waterpoint_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6000.0</td>
      <td>Roman</td>
      <td>1390</td>
      <td>Roman</td>
      <td>34.938093</td>
      <td>-9.856322</td>
      <td>Lake Nyasa</td>
      <td>Mnyusi B</td>
      <td>Iringa</td>
      <td>11</td>
      <td>5</td>
      <td>109</td>
      <td>VWC</td>
      <td>False</td>
      <td>1999</td>
      <td>gravity</td>
      <td>user-group</td>
      <td>spring</td>
      <td>communal standpipe</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>Grumeti</td>
      <td>1399</td>
      <td>GRUMETI</td>
      <td>34.698766</td>
      <td>-2.147466</td>
      <td>Lake Victoria</td>
      <td>Nyamara</td>
      <td>Mara</td>
      <td>20</td>
      <td>2</td>
      <td>280</td>
      <td>Other</td>
      <td>True</td>
      <td>2010</td>
      <td>gravity</td>
      <td>user-group</td>
      <td>rainwater harvesting</td>
      <td>communal standpipe</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25.0</td>
      <td>Lottery Club</td>
      <td>686</td>
      <td>World vision</td>
      <td>37.460664</td>
      <td>-3.821329</td>
      <td>Pangani</td>
      <td>Majengo</td>
      <td>Manyara</td>
      <td>21</td>
      <td>4</td>
      <td>250</td>
      <td>VWC</td>
      <td>True</td>
      <td>2009</td>
      <td>gravity</td>
      <td>user-group</td>
      <td>dam</td>
      <td>communal standpipe multiple</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>Unicef</td>
      <td>263</td>
      <td>UNICEF</td>
      <td>38.486161</td>
      <td>-11.155298</td>
      <td>Ruvuma / Southern Coast</td>
      <td>Mahakamani</td>
      <td>Mtwara</td>
      <td>90</td>
      <td>63</td>
      <td>58</td>
      <td>VWC</td>
      <td>True</td>
      <td>1986</td>
      <td>submersible</td>
      <td>user-group</td>
      <td>machine dbh</td>
      <td>communal standpipe multiple</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>Action In A</td>
      <td>0</td>
      <td>Artisan</td>
      <td>31.130847</td>
      <td>-1.825359</td>
      <td>Lake Victoria</td>
      <td>Kyanyamisa</td>
      <td>Kagera</td>
      <td>18</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>True</td>
      <td>0</td>
      <td>gravity</td>
      <td>other</td>
      <td>rainwater harvesting</td>
      <td>communal standpipe</td>
    </tr>
  </tbody>
</table>
</div>




```python
# what are the number of columns?
len(df_train_data.columns)
```




    19




```python
# df_train_data_test = pd.get_dummies(df_train_data)
# df_train_data_test.corr()
```

It might actually be smarter to do this in another notebook and see if I get any improvements. 

Let this be my long notebook?


```python
# # lowercase and replace spaces with underscores in all rows
# df_train_data.columns = df_train_data.columns.str.lower().str.replace(' ', '_')

# # Check
# df_train_data.head()

# # lowercase all values in df_train_data
# df_train_data = df_train_data.apply(lambda x: x.astype(str).str.lower())

# # replaces all spaces with underscores in all rows in df_train_data
# df_train_data = df_train_data.apply(lambda x: x.astype(str).str.replace(' ', '_'))
```

# Model with cv class


```python
class ModelWithCV():
    '''Structure to save the model and more easily see its crossvalidation'''
    
    def __init__(self, model, model_name, X, y, cv_now=True):
        self.model = model
        self.name = model_name
        self.X = X
        self.y = y
        # For CV results
        self.cv_results = None
        self.cv_mean = None
        self.cv_median = None
        self.cv_std = None
        #
        if cv_now:
            self.cross_validate()
        
    def cross_validate(self, X=None, y=None, kfolds=10):
        '''
        Perform cross-validation and return results.
        
        Args: 
          X:
            Optional; Training data to perform CV on. Otherwise use X from object
          y:
            Optional; Training data to perform CV on. Otherwise use y from object
          kfolds:
            Optional; Number of folds for CV (default is 10)  
        '''
        
        cv_X = X if X else self.X
        cv_y = y if y else self.y

        self.cv_results = cross_val_score(self.model, cv_X, cv_y, cv=kfolds)
        self.cv_mean = np.mean(self.cv_results)
        self.cv_median = np.median(self.cv_results)
        self.cv_std = np.std(self.cv_results)

        
    def print_cv_summary(self):
        cv_summary = (
        f'''CV Results for `{self.name}` model:
            {self.cv_mean:.5f} ± {self.cv_std:.5f} accuracy
        ''')
        print(cv_summary)

        
    def plot_cv(self, ax):
        '''
        Plot the cross-validation values using the array of results and given 
        Axis for plotting.
        '''
        ax.set_title(f'CV Results for `{self.name}` Model')
        # Thinner violinplot with higher bw
        sns.violinplot(y=self.cv_results, ax=ax, bw=.4)
        sns.swarmplot(
                y=self.cv_results,
                color='orange',
                size=10,
                alpha= 0.8,
                ax=ax
        )

        return ax
```

# Pipelines:

Time to set up some pipelines.

### Import Statements


```python
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer,  make_column_selector as selector
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import plot_confusion_matrix, recall_score,\
    accuracy_score, precision_score, f1_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.dummy import DummyClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImPipeline
```

### Creating The pipelines


```python
subpipe_numerics = Pipeline(steps = [
    ('num_impute', SimpleImputer(strategy='median')),
    ('ss', StandardScaler())
])


# subpipe_cat = Pipeline(steps=[
#     ('cat_impute', SimpleImputer(strategy='most_frequent')),
#     ('ohe', ExtendedOneHotEncoder(sparse=False, handle_unknown='ignore'))
# ])

# extend with get_feature_names method
class ExtendedOneHotEncoder(OneHotEncoder):
    def get_feature_names(self, input_features=None):
        return self.get_feature_names(input_features=input_features)
    
subpipe_cat = Pipeline(steps=[
    ('cat_impute', SimpleImputer(strategy='most_frequent')),
    ('ohe', ExtendedOneHotEncoder(sparse=False, handle_unknown='ignore'))
])
```

### Creating the column transformer


```python
CT = ColumnTransformer(transformers=[
    ('subpipe_num', subpipe_numerics, selector(dtype_include=np.number)),
    ('subpipe_cat', subpipe_cat, selector(dtype_include=object))
], remainder='passthrough')
```

### Dummy Model Pipeline:


```python
dummy_model_pipe = Pipeline(steps=[
    ('ct',CT),
    ('dummy', DummyClassifier(strategy='most_frequent'))
])
```

### Logreg model pipeline


```python
logreg_model_pipe = Pipeline(steps=[
    ('ct',CT),
    ('fsm', LogisticRegression(max_iter=1000))
])
```

### Decision Tree


```python
dtc_model_pipe = Pipeline(steps=[
    ('ct', CT),
    ('dtc', DecisionTreeClassifier())
])
```

# Test, Train, And Validation Split:

I will split the data into three sets:

- train 15%
- validation 15%
- test 70%


```python
# Perform a 15-15-70 split on the data
from sklearn.model_selection import train_test_split

X = df_train_data
y = df_train_target['status_group']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=77)
# holdout split. I will call this validation
X_train_both, X_val, y_train_both, y_val = train_test_split(X,y,random_state=42, test_size = .1)

# Now split again to create a validation set
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=77)
# Now, create my test and train splits for model creation, default test size
X_train, X_test, y_train, y_test = train_test_split(X_train_both, y_train_both, random_state=42)

# Check the shapes
print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
```

    (40095, 19)
    (5940, 19)
    (13365, 19)
    

# Creating The Models

### Dummy Model:


```python
# fit the dummy model
dummy_model_pipe.fit(X_train, y_train)
```




    Pipeline(steps=[('ct',
                     ColumnTransformer(remainder='passthrough',
                                       transformers=[('subpipe_num',
                                                      Pipeline(steps=[('num_impute',
                                                                       SimpleImputer(strategy='median')),
                                                                      ('ss',
                                                                       StandardScaler())]),
                                                      <sklearn.compose._column_transformer.make_column_selector object at 0x000001DC44528850>),
                                                     ('subpipe_cat',
                                                      Pipeline(steps=[('cat_impute',
                                                                       SimpleImputer(strategy='most_frequent')),
                                                                      ('ohe',
                                                                       ExtendedOneHotEncoder(handle_unknown='ignore',
                                                                                             sparse=False))]),
                                                      <sklearn.compose._column_transformer.make_column_selector object at 0x000001DC44528490>)])),
                    ('dummy', DummyClassifier(strategy='most_frequent'))])




```python
dummy_pipe = ModelWithCV(
    dummy_model_pipe,
    model_name='dummy',
    X=X_test,
    y=y_test)
fix, ax = plt.subplots(figsize=(10,10))

dummy_pipe.plot_cv(ax=ax)
```




    <AxesSubplot:title={'center':'CV Results for `dummy` Model'}>




    
![png](output_40_1.png)
    



```python
# confusion matrix
plot_confusion_matrix(dummy_model_pipe, X_train, y_train, cmap='Blues', values_format='d')
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x1dc448136a0>




    
![png](output_41_1.png)
    



```python
# what is the score on the test and train?
print(f'Train score: {dummy_model_pipe.score(X_train, y_train)}')
print(f'Test score: {dummy_model_pipe.score(X_test, y_test)}')
```

    Train score: 0.5451303155006859
    Test score: 0.5390946502057613
    


```python
dummy_pipe.print_cv_summary()
```

    CV Results for `dummy` model:
                0.53909 ± 0.00017 accuracy
            
    


```python
# plot an ROC curve
from sklearn.metrics import plot_roc_curve

plot_roc_curve(dummy_model_pipe, X_test, y_test)
```




    <sklearn.metrics._plot.roc_curve.RocCurveDisplay at 0x1dc4520ab20>




    
![png](output_44_1.png)
    


### Logreg Model:


```python
logreg_model_pipe.fit(X_train, y_train)
```




    Pipeline(steps=[('ct',
                     ColumnTransformer(remainder='passthrough',
                                       transformers=[('subpipe_num',
                                                      Pipeline(steps=[('num_impute',
                                                                       SimpleImputer(strategy='median')),
                                                                      ('ss',
                                                                       StandardScaler())]),
                                                      <sklearn.compose._column_transformer.make_column_selector object at 0x000001DC44528850>),
                                                     ('subpipe_cat',
                                                      Pipeline(steps=[('cat_impute',
                                                                       SimpleImputer(strategy='most_frequent')),
                                                                      ('ohe',
                                                                       ExtendedOneHotEncoder(handle_unknown='ignore',
                                                                                             sparse=False))]),
                                                      <sklearn.compose._column_transformer.make_column_selector object at 0x000001DC44528490>)])),
                    ('fsm', LogisticRegression(max_iter=1000))])




```python
fsm_pipe = ModelWithCV(
    logreg_model_pipe,
    model_name='FSM',
    X=X_test,
    y=y_test)
fix, ax = plt.subplots(figsize=(10,10))

fsm_pipe.plot_cv(ax=ax)
```




    <AxesSubplot:title={'center':'CV Results for `FSM` Model'}>




    
![png](output_47_1.png)
    



```python
# what is the score on the train and test data?
print(f'Train score: {fsm_pipe.model.score(X_train, y_train)}')
print(f'Test score: {fsm_pipe.model.score(X_test, y_test)}')
```

    Train score: 0.8268861454046639
    Test score: 0.7369248035914703
    


```python
# Check the confusion matrix
plot_confusion_matrix(logreg_model_pipe, X_train, y_train, cmap='Blues', values_format='d')
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x1dc444b3dc0>




    
![png](output_49_1.png)
    



```python
fsm_pipe.print_cv_summary()
```

    CV Results for `FSM` model:
                0.71463 ± 0.01352 accuracy
            
    


```python
# plot an roc curve with the dummy and FSM
plot_roc_curve(dummy_model_pipe, X_test, y_test)
plot_roc_curve(logreg_model_pipe, X_test, y_test)
```




    <sklearn.metrics._plot.roc_curve.RocCurveDisplay at 0x1dc4494a490>




    
![png](output_51_1.png)
    



    
![png](output_51_2.png)
    


### Decision Tree Model:


```python
dtc_model_pipe.fit(X_train, y_train)
```




    Pipeline(steps=[('ct',
                     ColumnTransformer(remainder='passthrough',
                                       transformers=[('subpipe_num',
                                                      Pipeline(steps=[('num_impute',
                                                                       SimpleImputer(strategy='median')),
                                                                      ('ss',
                                                                       StandardScaler())]),
                                                      <sklearn.compose._column_transformer.make_column_selector object at 0x000001DC44528850>),
                                                     ('subpipe_cat',
                                                      Pipeline(steps=[('cat_impute',
                                                                       SimpleImputer(strategy='most_frequent')),
                                                                      ('ohe',
                                                                       ExtendedOneHotEncoder(handle_unknown='ignore',
                                                                                             sparse=False))]),
                                                      <sklearn.compose._column_transformer.make_column_selector object at 0x000001DC44528490>)])),
                    ('dtc', DecisionTreeClassifier())])




```python
dtc_pipe = ModelWithCV(
    dtc_model_pipe,
    model_name='dtc',
    X=X_test,
    y=y_test)
fix, ax = plt.subplots(figsize=(10,10))

dtc_pipe.plot_cv(ax=ax)
```




    <AxesSubplot:title={'center':'CV Results for `dtc` Model'}>




    
![png](output_54_1.png)
    



```python
# What is the score on the train and test data?
print('score on training data ', dtc_model_pipe.score(X_train, y_train))
print('score on test data ', dtc_model_pipe.score(X_test, y_test))
```

    score on training data  0.9992268362638733
    score on test data  0.7583988028432472
    


```python
dtc_pipe.print_cv_summary()
```

    CV Results for `dtc` model:
                0.72451 ± 0.01107 accuracy
            
    


```python
# plot a roc curve for the decision tree
plot_roc_curve(dtc_model_pipe, X_test, y_test)
```




    <sklearn.metrics._plot.roc_curve.RocCurveDisplay at 0x1dc451d6c40>




    
![png](output_57_1.png)
    


## All models so far


```python
# plot the roc curve of the dummy, FSM, and decision tree in the same plot
plot_roc_curve(dummy_model_pipe, X_test, y_test)
plot_roc_curve(logreg_model_pipe, X_test, y_test)
plot_roc_curve(dtc_model_pipe, X_test, y_test)
```




    <sklearn.metrics._plot.roc_curve.RocCurveDisplay at 0x1dc42f1a220>




    
![png](output_59_1.png)
    



    
![png](output_59_2.png)
    



    
![png](output_59_3.png)
    


# Model evaluation

## logreg


```python
# # use my column transformer to transform the data
# X_train_transformed = CT.fit_transform(X_train)
# X_test_transformed = CT.transform(X_test)

# # check the shape
# print(X_train_transformed.shape)

# # get the feature importance from the decision tree
# dtc_model_pipe.named_steps['dtc'].feature_importances_

# # combine the names of the columns in X_train_transformed with the feature importance
# feature_importance = pd.DataFrame({'feature': CT.get_feature_names_out(), 'importance': dtc_model_pipe.named_steps['dtc'].feature_importances_})


```

## Decision Tree

# Now let's start improving the models. 

## Tuning the models with gridsearch and random gridsearch


```python
# use gridsearch on the decision tree
from sklearn.model_selection import GridSearchCV
parameters = {'dtc__criterion': ['gini', 'entropy'],
          'dtc__min_samples_leaf': [10, 15, 20]}

gridsearch = GridSearchCV(
    dtc_model_pipe,
    param_grid=parameters,
    cv=5,
    verbose=2,
    n_jobs=-1
)

gridsearch.fit(X_train, y_train)
```

    Fitting 5 folds for each of 6 candidates, totalling 30 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 24 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  15 out of  30 | elapsed:   18.7s remaining:   18.7s
    [Parallel(n_jobs=-1)]: Done  30 out of  30 | elapsed:  3.5min finished
    




    GridSearchCV(cv=5,
                 estimator=Pipeline(steps=[('ct',
                                            ColumnTransformer(remainder='passthrough',
                                                              transformers=[('subpipe_num',
                                                                             Pipeline(steps=[('num_impute',
                                                                                              SimpleImputer(strategy='median')),
                                                                                             ('ss',
                                                                                              StandardScaler())]),
                                                                             <sklearn.compose._column_transformer.make_column_selector object at 0x000001DC44528850>),
                                                                            ('subpipe_cat',
                                                                             Pipeline(steps=[('cat_impute',
                                                                                              SimpleImputer(strategy='most_frequent')),
                                                                                             ('ohe',
                                                                                              ExtendedOneHotEncoder(handle_unknown='ignore',
                                                                                                                    sparse=False))]),
                                                                             <sklearn.compose._column_transformer.make_column_selector object at 0x000001DC44528490>)])),
                                           ('dtc', DecisionTreeClassifier())]),
                 n_jobs=-1,
                 param_grid={'dtc__criterion': ['gini', 'entropy'],
                             'dtc__min_samples_leaf': [10, 15, 20]},
                 verbose=2)




```python
gridsearch.best_params_
```




    {'dtc__criterion': 'gini', 'dtc__min_samples_leaf': 10}




```python
gridsearch.best_score_
```




    nan



# Building The Final Model


```python
final_model = gridsearch.best_estimator_
```


```python
final_model
```




    Pipeline(steps=[('ct',
                     ColumnTransformer(remainder='passthrough',
                                       transformers=[('subpipe_num',
                                                      Pipeline(steps=[('num_impute',
                                                                       SimpleImputer(strategy='median')),
                                                                      ('ss',
                                                                       StandardScaler())]),
                                                      <sklearn.compose._column_transformer.make_column_selector object at 0x000001DC43CBFD90>),
                                                     ('subpipe_cat',
                                                      Pipeline(steps=[('cat_impute',
                                                                       SimpleImputer(strategy='most_frequent')),
                                                                      ('ohe',
                                                                       ExtendedOneHotEncoder(handle_unknown='ignore',
                                                                                             sparse=False))]),
                                                      <sklearn.compose._column_transformer.make_column_selector object at 0x000001DC43CBFF10>)])),
                    ('dtc', DecisionTreeClassifier(min_samples_leaf=10))])




```python
final_model.fit(X_train, y_train)
```




    Pipeline(steps=[('ct',
                     ColumnTransformer(remainder='passthrough',
                                       transformers=[('subpipe_num',
                                                      Pipeline(steps=[('num_impute',
                                                                       SimpleImputer(strategy='median')),
                                                                      ('ss',
                                                                       StandardScaler())]),
                                                      <sklearn.compose._column_transformer.make_column_selector object at 0x000001DC43CBFD90>),
                                                     ('subpipe_cat',
                                                      Pipeline(steps=[('cat_impute',
                                                                       SimpleImputer(strategy='most_frequent')),
                                                                      ('ohe',
                                                                       ExtendedOneHotEncoder(handle_unknown='ignore',
                                                                                             sparse=False))]),
                                                      <sklearn.compose._column_transformer.make_column_selector object at 0x000001DC43CBFF10>)])),
                    ('dtc', DecisionTreeClassifier(min_samples_leaf=10))])




```python
final_model_check = ModelWithCV(
    final_model,
    model_name='final_model',
    X=X_train_both,
    y=y_train_both)
fig, ax = plt.subplots(figsize=(10,10))

final_model_check.plot_cv(ax=ax);
```


    
![png](output_73_0.png)
    



```python
final_model_check.print_cv_summary()
```

    CV Results for `final_model` model:
                0.75230 ± 0.00650 accuracy
            
    


```python
# Score against validation/hold out

final_model.score(X_val, y_val)
```




    0.7540404040404041




```python
# production model
final_model.fit(X,y)
```




    Pipeline(steps=[('ct',
                     ColumnTransformer(remainder='passthrough',
                                       transformers=[('subpipe_num',
                                                      Pipeline(steps=[('num_impute',
                                                                       SimpleImputer(strategy='median')),
                                                                      ('ss',
                                                                       StandardScaler())]),
                                                      <sklearn.compose._column_transformer.make_column_selector object at 0x000001DC43CBFD90>),
                                                     ('subpipe_cat',
                                                      Pipeline(steps=[('cat_impute',
                                                                       SimpleImputer(strategy='most_frequent')),
                                                                      ('ohe',
                                                                       ExtendedOneHotEncoder(handle_unknown='ignore',
                                                                                             sparse=False))]),
                                                      <sklearn.compose._column_transformer.make_column_selector object at 0x000001DC43CBFF10>)])),
                    ('dtc', DecisionTreeClassifier(min_samples_leaf=10))])




```python
final_model.score(X,y)
```




    0.8405723905723905




```python
# get model weights
importances = final_model.named_steps['dtc'].feature_importances_


# Print feature importances
# for i, importance in enumerate(importances):
#     print(f"Feature {i}: {importance}")

# Get feature names
df_train_data_temp = pd.get_dummies(df_train_data)
feature_names = df_train_data_temp.columns
#print the shape of X.columns
print(feature_names.shape)

# Get feature importances and their indices sorted in descending order
indices = np.argsort(importances)[::-1]

# Define the number of important features you want to select
N = 20

# Print the top N features
for i in range(N):
    print(f"{i+1}. {feature_names[indices[i]]} (importance: {importances[indices[i]]})")
```

    (23421,)
    1. longitude (importance: 0.1537902957513936)
    2. latitude (importance: 0.14801190702081035)
    3. waterpoint_type_other (importance: 0.14358797765938472)
    4. gps_height (importance: 0.06648662119060796)
    5. amount_tsh (importance: 0.06474338331010528)
    6. construction_year (importance: 0.06208088196039011)
    7. waterpoint_type_communal standpipe multiple (importance: 0.045391852262995136)
    8. population (importance: 0.03880257260332795)
    9. district_code (importance: 0.023944820581879713)
    10. funder_Government Of Tanzania (importance: 0.013016506275669946)
    11. region_Iringa (importance: 0.011989390064285782)
    12. region_code (importance: 0.01012843017518051)
    13. extraction_type_nira/tanira (importance: 0.010119012692720565)
    14. extraction_type_other (importance: 0.009037259440928626)
    15. installer_DWE (importance: 0.008831090520520234)
    16. management_group_commercial (importance: 0.007832321914001723)
    17. scheme_management_VWC (importance: 0.007709506178850306)
    18. source_river (importance: 0.00684760201122662)
    19. scheme_management_Water Board (importance: 0.006512812161203489)
    20. basin_Lake Victoria (importance: 0.005880213301183874)
    


```python
# plot the feature importances
plt.figure(figsize=(10,10))
plt.title("Feature Importances")
plt.bar(range(N), importances[indices[:N]])
plt.xticks(range(N), feature_names[indices[:N]], rotation=90)
plt.show()
```


    
![png](output_79_0.png)
    



```python
# # run gridsearch on fsm_pipe
# from sklearn.model_selection import GridSearchCV

# # set up the gridsearch
# param_grid = {
#     'fsm__penalty': ['l1', 'l2', 'elasticnet', 'none'],
#     'fsm__C': [0.1, 1, 10, 100]
# }

# # set up the gridsearch
# gridsearch = GridSearchCV(
#     fsm_pipe.model,
#     param_grid=param_grid,
#     cv=5,
#     verbose=1,
#     n_jobs=-1
# )

# # fit the gridsearch
# gridsearch.fit(X_train, y_train)
```
