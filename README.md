# phase3_project3

[Presentation](https://docs.google.com/presentation/d/1I34XcvTqbOwh9xBtx27wmkNFPC6luxFFKyY532R_-0I/edit?usp=sharing)

[Google Document Lab Notebook](https://docs.google.com/document/d/1Spref_pjFamfD-KR-_QiNYEyXASlaG7z9inboxcsCjs/edit?usp=sharing)

[Source Of The Data](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/)



# %% [markdown]
# # Intro:
# 
# ## Business Problem: Predicting Water Wells In Need Of Repair For The Government Of Tanzania
# 
# ## Stakeholder: The Ministry Of Water in Tanzania

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# %%
df_train_target = pd.read_csv('./Data/training_set_labels.csv')
df_train_data = pd.read_csv('./Data/training_set_values.csv')

# %% [markdown]
# # Collapsing The Target Categories
# 
# The goal is to predict wells that require repair.
# 
# The target variable is `status_group` which has three categories:
# 
# - `functional`
# - `functional needs repair`
# - `non functional`
# 
# I will collapse the categories `functional needs repair` and `non functional` into one category `needs repair` to create a binary target variable.

# %%
# Look at df_train_target
df_train_target['status_group'].value_counts()



# %%
# Change the name of the status_group in df_train_target 'functional needs repair' to 'repair'
# Also change the name of 'non functional' to 'repair'
df_train_target['status_group'] = df_train_target['status_group'].replace('functional needs repair', 'repair')
df_train_target['status_group'] = df_train_target['status_group'].replace('non functional', 'repair')

# Check
df_train_target['status_group'].value_counts()

# %%
# encode status_group as 0, 1
df_train_target['status_group'] = df_train_target['status_group'].astype('category')
df_train_target['status_group'] = df_train_target['status_group'].cat.codes

# Check
df_train_target['status_group'].value_counts(normalize=True)

# %% [markdown]
# So now I have collapsed the target categories, and I have encoded the target variable as a binary variable.
# 
# Also note that the target variable is not imbalanced.

# %% [markdown]
# # Chopping out columns
# 
# Okay there is way too much here. I am going to chop out some columns.
# 
# I am going to chop out the following columns:
# 
# - id
# - date recorded
# - funder
# - installer
# - wpt_name
# - recorded_by
# - scheme_management
# - scheme_name
# 
# public_meeting
# extraction type group and class?

# %% [markdown]
# ## the data has a high degree of collinearity
# 
# I am goign to remove some columns in an attempt to fix that.

# %%
# show me the colinearlity of the data
# df_train_data.corr()

# %%
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

# %%
# # one hot encode the categorical data and show me the coliinearlity
# df_train_data = pd.get_dummies(df_train_data)
# df_train_data.corr()

# %%
# Drop id, date recorded, funder, installer, wpt_name, recorded_by, scheme_management, scheme_name, quantity_group, payment_type, extraction_type_group, extraction_type_class, management_group, public_meeting, permit, num_private
# df_train_data = df_train_data.drop(['id', 'date_recorded', 'funder', 'installer', 'wpt_name', 'recorded_by', 'scheme_management', 'scheme_name', 'quantity_group', 'payment_type', 'extraction_type_group', 'extraction_type_class', 'management_group', 'public_meeting', 'permit', 'num_private'], axis=1)

# %%
# remove everything but amount_tsh, gps_height, longitude, latitude, region_code, district_code, population, construction_year, lga, ward, recorded_by, scheme_management, scheme_name, management, management_group
# df_train_data = df_train_data.drop(['amount_tsh', 'wpt_name', 'id', 'date_recorded', 'num_private', 'gps_height', 'longitude', 'latitude', 'region_code', 'district_code', 'population', 'construction_year', 'lga', 'ward', 'recorded_by', 'scheme_management', 'scheme_name', 'management', 'management_group', 'public_meeting', 'extraction_type', 'extraction_type_group', 'extraction_type_class', 'payment_type', 'payment'], axis=1)

# %%
# check what df_train_data has now
df_train_data.head()

# %%
# what are the number of columns?
len(df_train_data.columns)

# %%
# df_train_data_test = pd.get_dummies(df_train_data)
# df_train_data_test.corr()

# %% [markdown]
# It might actually be smarter to do this in another notebook and see if I get any improvements. 
# 
# Let this be my long notebook?

# %%
# # lowercase and replace spaces with underscores in all rows
# df_train_data.columns = df_train_data.columns.str.lower().str.replace(' ', '_')

# # Check
# df_train_data.head()

# # lowercase all values in df_train_data
# df_train_data = df_train_data.apply(lambda x: x.astype(str).str.lower())

# # replaces all spaces with underscores in all rows in df_train_data
# df_train_data = df_train_data.apply(lambda x: x.astype(str).str.replace(' ', '_'))

# %% [markdown]
# # Model with cv class

# %%
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

# %% [markdown]
# # Pipelines:
# 
# Time to set up some pipelines.

# %% [markdown]
# ### Import Statements

# %%
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

# %% [markdown]
# ### Creating The pipelines

# %%
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

# %% [markdown]
# ### Creating the column transformer

# %%
CT = ColumnTransformer(transformers=[
    ('subpipe_num', subpipe_numerics, selector(dtype_include=np.number)),
    ('subpipe_cat', subpipe_cat, selector(dtype_include=object))
], remainder='passthrough')

# %% [markdown]
# ### Dummy Model Pipeline:

# %%
dummy_model_pipe = Pipeline(steps=[
    ('ct',CT),
    ('dummy', DummyClassifier(strategy='most_frequent'))
])

# %% [markdown]
# ### Logreg model pipeline

# %%
logreg_model_pipe = Pipeline(steps=[
    ('ct',CT),
    ('fsm', LogisticRegression(max_iter=1000))
])

# %% [markdown]
# ### Decision Tree

# %%
dtc_model_pipe = Pipeline(steps=[
    ('ct', CT),
    ('dtc', DecisionTreeClassifier())
])

# %% [markdown]
# # Test, Train, And Validation Split:
# 
# I will split the data into three sets:
# 
# - train 15%
# - validation 15%
# - test 70%

# %%
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

# %% [markdown]
# # Creating The Models

# %% [markdown]
# ### Dummy Model:

# %%
# fit the dummy model
dummy_model_pipe.fit(X_train, y_train)

# %%
dummy_pipe = ModelWithCV(
    dummy_model_pipe,
    model_name='dummy',
    X=X_test,
    y=y_test)
fix, ax = plt.subplots(figsize=(10,10))

dummy_pipe.plot_cv(ax=ax)

# %%
# confusion matrix
plot_confusion_matrix(dummy_model_pipe, X_train, y_train, cmap='Blues', values_format='d')

# %%
# what is the score on the test and train?
print(f'Train score: {dummy_model_pipe.score(X_train, y_train)}')
print(f'Test score: {dummy_model_pipe.score(X_test, y_test)}')

# %%
dummy_pipe.print_cv_summary()

# %%
# plot an ROC curve
from sklearn.metrics import plot_roc_curve

plot_roc_curve(dummy_model_pipe, X_test, y_test)

# %% [markdown]
# ### Logreg Model:

# %%
logreg_model_pipe.fit(X_train, y_train)

# %%
fsm_pipe = ModelWithCV(
    logreg_model_pipe,
    model_name='FSM',
    X=X_test,
    y=y_test)
fix, ax = plt.subplots(figsize=(10,10))

fsm_pipe.plot_cv(ax=ax)

# %%
# what is the score on the train and test data?
print(f'Train score: {fsm_pipe.model.score(X_train, y_train)}')
print(f'Test score: {fsm_pipe.model.score(X_test, y_test)}')

# %%
# Check the confusion matrix
plot_confusion_matrix(logreg_model_pipe, X_train, y_train, cmap='Blues', values_format='d')

# %%
fsm_pipe.print_cv_summary()

# %%
# plot an roc curve with the dummy and FSM
plot_roc_curve(dummy_model_pipe, X_test, y_test)
plot_roc_curve(logreg_model_pipe, X_test, y_test)

# %% [markdown]
# ### Decision Tree Model:

# %%
dtc_model_pipe.fit(X_train, y_train)

# %%
dtc_pipe = ModelWithCV(
    dtc_model_pipe,
    model_name='dtc',
    X=X_test,
    y=y_test)
fix, ax = plt.subplots(figsize=(10,10))

dtc_pipe.plot_cv(ax=ax)

# %%
# What is the score on the train and test data?
print('score on training data ', dtc_model_pipe.score(X_train, y_train))
print('score on test data ', dtc_model_pipe.score(X_test, y_test))

# %%
dtc_pipe.print_cv_summary()

# %%
# plot a roc curve for the decision tree
plot_roc_curve(dtc_model_pipe, X_test, y_test)

# %% [markdown]
# ## All models so far

# %%
# plot the roc curve of the dummy, FSM, and decision tree in the same plot
plot_roc_curve(dummy_model_pipe, X_test, y_test)
plot_roc_curve(logreg_model_pipe, X_test, y_test)
plot_roc_curve(dtc_model_pipe, X_test, y_test)

# %% [markdown]
# # Model evaluation

# %% [markdown]
# ## logreg

# %%
# # use my column transformer to transform the data
# X_train_transformed = CT.fit_transform(X_train)
# X_test_transformed = CT.transform(X_test)

# # check the shape
# print(X_train_transformed.shape)

# # get the feature importance from the decision tree
# dtc_model_pipe.named_steps['dtc'].feature_importances_

# # combine the names of the columns in X_train_transformed with the feature importance
# feature_importance = pd.DataFrame({'feature': CT.get_feature_names_out(), 'importance': dtc_model_pipe.named_steps['dtc'].feature_importances_})



# %% [markdown]
# ## Decision Tree

# %% [markdown]
# # Now let's start improving the models. 

# %% [markdown]
# ## Tuning the models with gridsearch and random gridsearch

# %%
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

# %%
gridsearch.best_params_

# %%
gridsearch.best_score_

# %% [markdown]
# # Building The Final Model

# %%
final_model = gridsearch.best_estimator_

# %%
final_model

# %%
final_model.fit(X_train, y_train)

# %%
final_model_check = ModelWithCV(
    final_model,
    model_name='final_model',
    X=X_train_both,
    y=y_train_both)
fig, ax = plt.subplots(figsize=(10,10))

final_model_check.plot_cv(ax=ax);

# %%
final_model_check.print_cv_summary()

# %%
# Score against validation/hold out

final_model.score(X_val, y_val)

# %%
# production model
final_model.fit(X,y)

# %%
final_model.score(X,y)

# %%
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

# %%
# plot the feature importances
plt.figure(figsize=(10,10))
plt.title("Feature Importances")
plt.bar(range(N), importances[indices[:N]])
plt.xticks(range(N), feature_names[indices[:N]], rotation=90)
plt.show()

# %%
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


