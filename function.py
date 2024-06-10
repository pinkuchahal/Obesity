# %% [markdown]
# ## Import necessary libraries

# %%
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# %% [markdown]
# ## metadata function

# %%
# this function returns column metadata for column name, data type, percentage of missing values, number of unique values, and basic statistics for interval columns 
def metadata(df):
  columns_list = list(df.columns.values) # get a list of column names 
  type_list = [str(item) for item in list(df.dtypes)] # get data types 
  missing_list = [round(float(num),2) for num in list((df.isnull().sum()/len(df)*100))] # find percentage of missing values 
  unique_list = [int(nunique) for nunique in list(df.nunique())] # find unique values for each column 
  # # return basic stats for interval columns (i.e. not a category or object datatype and more than 10 unique values) 
  metadata = pd.DataFrame(columns_list, columns=['column_name'])
  metadata['datatype'] = type_list
  metadata['missing_percent'] = missing_list
  metadata['unique'] = unique_list
  try:
    desc_interval = df[[item for item in columns_list if str(df[item].dtypes) != 'category' and df[item].nunique()>=10 and str(df[item].dtypes) != 'object']].describe().loc[['mean', 'std', 'min','25%', '50%', '75%', 'max']].transpose().reset_index().rename(columns = {'index':'column_name'})
    metadata = pd.merge(metadata, desc_interval, on='column_name', how='left')
  except:
    metadata
  return metadata

# %% [markdown]
# ## data_exploration function

# %%
def data_exploration (df, column): 
 ## this is same as previously to create plot unique values for categorical variables with less than 10 unique values
    if (str(df[column].dtypes) == 'object'or str(df[column].dtypes) == 'category'):
        if df[column].nunique()<10:
            count_value = df.groupby([column]).size().reset_index(name='counts')
            count_value['%count'] = [round(num/len(df)*100,2) for num in list(count_value['counts'])]
            print(count_value)
            value_list = count_value[column].tolist()
            count_list = count_value['counts'].tolist()
            fig = plt.figure(figsize=(8, 4))
            plt.bar(x=value_list, height= count_list)
            plt.xticks(fontsize=12)
            plt.show()
        else:
            print(column + ' has more than 10 unique values')
    else:
        mean = df[column].describe()['mean']
        std = df[column].describe()['std']
        outlier = df[((df[column]-mean)/std >3) | ((df[column]-mean)/std <-3)][column].tolist()
        if len(outlier) > 0:
            print('There are ' + str(len(outlier)) + ' of outliers for ' + column + '.')
            print(outlier)
        else:
            print('There is no outlier of ' + column + '.')

        ## this is to create box plot  
        print('----------------------Box plot---------------------')
        df[column].plot.box(title=column, whis =(5,95))
        plt.grid()
        plt.show()

        ## this is to plot interval column distribution by a decile 
        min_value = float(df[column].describe()['min'])
        max_value = float(df[column].describe()['max'])
        if df[column].nunique() >= 10:
            para = (max_value - min_value) / 10
            para_list = np.arange(min_value, max_value, para).round(decimals=2).tolist()
            count_table = df.loc[:, [column]]
            for num in para_list:
                count_table.loc[count_table[column] >= num, 'range'] = num
            count_table_sum = count_table.groupby(['range']).size().reset_index(name='counts')
            value_list = count_table_sum['range'].tolist()
            count_list = count_table_sum['counts'].tolist()
            print('----------------------Distribution plot---------------------')
            fig = plt.figure(figsize=(8, 4))
            plt.bar(x=value_list, height=count_list, width=para, tick_label=value_list, align='edge')
            plt.xticks(rotation=40, fontsize=12)
            plt.grid()
            plt.show()

# %% [markdown]
# ## pie_cate function

# %%
def pie_cate(df, column): 
  count_value = df.groupby([column]).size().reset_index(name='counts') # find frequency of each unique value
  count_value['%count'] = [round(num/len(column)*100,2) for num in list(count_value['counts'])] # get frequency distribution
  print(count_value)

  value_list = count_value[column].tolist()
  count_list = count_value['counts'].tolist()
  fig = plt.figure(figsize=(8, 4))

  plt.pie(count_list, labels=value_list)
  plt.show()

# %% [markdown]
# # stable_chart function

# %%
def stable_chart(lift_table):
  plt.plot(lift_table.decile.values, lift_table.resp_rate.values, marker='o', label='Model')
  plt.title('Captured Response Rate Plot', fontsize=14)
  plt.xlabel('Deciles', fontsize=10)
  plt.ylabel('Lift', fontsize=10)
  plt.legend()
  plt.grid(True)

# %% [markdown]
# ## Forward Selection 

# %%
# Forward Selection Method
def forward_selection(X, y, significance_level=0.05):
    initial_features = X.columns.tolist()
    best_features = []
    
    while len(initial_features) > 0:
        remaining_features = list(set(initial_features) - set(best_features))
        new_pval = pd.Series(index=remaining_features)
        
        for new_column in remaining_features:
            model = sm.OLS(y, sm.add_constant(X[best_features + [new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        
        min_p_value = new_pval.min()
        if min_p_value < significance_level:
            best_features.append(new_pval.idxmin())
        else:
            break
    
    return best_features

# %% [markdown]
# ## Backward Elimination

# %%

def backward_elimination(X, y, significance_level=0.05):
    features = X.columns.tolist()
    while len(features) > 0:
        # Fit the model
        X_with_const = sm.add_constant(X[features])
        model = sm.OLS(y, X_with_const).fit()
        
        # Get the p-values for the predictors and exclude the intercept
        pvalues = model.pvalues.iloc[1:]  # Assuming the first is the intercept
        
        # Find the predictor with the highest p-value
        max_p_value = pvalues.max()
        
        # If the highest p-value is above the significance level, remove the corresponding predictor
        if max_p_value > significance_level:
            worst_feature = pvalues.idxmax()
            features.remove(worst_feature)
        else:
            break  # Stop if all p-values are below the significance level
    
    return features

# %% [markdown]
# ## Stepwise Selection

# %%
def stepwise_selection(X, y, significance_level=0.05):
    initial_features = X.columns.tolist()
    best_features = []
    
    while True:
        changed = False
        
        # Forward step
        remaining_features = list(set(initial_features) - set(best_features))
        new_pval = pd.Series(index=remaining_features, dtype=float)
        for new_column in remaining_features:
            model = sm.OLS(y, sm.add_constant(X[best_features + [new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if min_p_value < significance_level:
            best_features.append(new_pval.idxmin())
            changed = True
        
        # Backward step
        model_with_best = sm.OLS(y, sm.add_constant(X[best_features])).fit()
        # Use all P-values except the intercept
        pvalues_with_best = model_with_best.pvalues.iloc[1:]
        max_p_value = pvalues_with_best.max()
        if max_p_value > significance_level:
            worst_feature = pvalues_with_best.idxmax()
            best_features.remove(worst_feature)
            changed = True
            
        if not changed:
            break
            
    return best_features

# %% [markdown]
# ## Enter Method

# %%
def enter_selection_method(X, y):
    # Adding a constant to include an intercept in the model
    X_with_const = sm.add_constant(X)
    
    # Fitting the OLS regression model with all predictors
    model = sm.OLS(y, X_with_const).fit()
    
    # Returning the model summary
    return model.summary()

# %% [markdown]
# ## Hosmer-Lemeshow Goodness-of-Fit Test

# %%
from scipy.stats import chi2
def hosmer_lemeshow_test(y_true, y_pred_probs, n_bins=10):
    data = pd.DataFrame({'y_true': y_true, 'y_pred_probs': y_pred_probs})
    data['bin'] = pd.qcut(data['y_pred_probs'], q=n_bins, duplicates='drop')
    
    obs = data.groupby('bin')['y_true'].sum()
    total = data.groupby('bin')['y_true'].count()
    exp = total * data.groupby('bin')['y_pred_probs'].mean()
    
    hl_stat = (((obs - exp) ** 2) / (exp * (1 - data.groupby('bin')['y_pred_probs'].mean()))).sum()
    p_value = 1 - chi2.cdf(hl_stat, n_bins - 2)
    
    return hl_stat, p_value


