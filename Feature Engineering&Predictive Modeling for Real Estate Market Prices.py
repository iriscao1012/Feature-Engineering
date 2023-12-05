#Load the packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sktools import GradientBoostingFeatureGenerator
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

#Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
train_df.head()


test_df.head()


#Drop NaN Arributes
train_df=train_df.drop(['Alley','PoolQC','Fence','FireplaceQu','MiscFeature'],axis=1)
train_df.drop(['Id'],axis=1,inplace=True)
train_df.info()




test_df=test_df.drop(['Id','Alley','PoolQC','Fence','FireplaceQu','MiscFeature'],axis=1)
test_df.info()



#1. Data Preprocessing
# Handling Missing values with Mean
# training set
train_df['LotFrontage']=train_df['LotFrontage'].fillna(train_df['LotFrontage'].mean())
train_df['MasVnrArea'] = train_df['MasVnrArea'].fillna(train_df['MasVnrArea'].mean())
train_df['GarageYrBlt'] = train_df['GarageYrBlt'].fillna(train_df['GarageYrBlt'].mean())
train_df['MasVnrArea'] = train_df['MasVnrArea'].fillna(train_df['MasVnrArea'].mean())



#test set
test_df['LotFrontage'] = test_df['MasVnrArea'].fillna(test_df['MasVnrArea'].mean())
test_df['MasVnrArea'] = test_df['MasVnrArea'].fillna(test_df['MasVnrArea'].mean())
test_df['BsmtFinSF1'] = test_df['BsmtFinSF1'].fillna(test_df['BsmtFinSF1'].mean())
test_df['BsmtFinSF2'] = test_df['BsmtFinSF2'].fillna(test_df['BsmtFinSF2'].mean())
test_df['BsmtUnfSF'] = test_df['BsmtUnfSF'].fillna(test_df['BsmtUnfSF'].mean())
test_df['TotalBsmtSF'] = test_df['TotalBsmtSF'].fillna(test_df['TotalBsmtSF'].mean())
test_df['BsmtFullBath'] = test_df['BsmtFullBath'].fillna(test_df['BsmtFullBath'].mean())
test_df['BsmtHalfBath'] = test_df['BsmtHalfBath'].fillna(test_df['BsmtHalfBath'].mean())
test_df['GarageCars'] = test_df['GarageCars'].fillna(test_df['GarageCars'].mean())
test_df['GarageArea'] = test_df['GarageArea'].fillna(test_df['GarageArea'].mean())
test_df['GarageYrBlt'] = test_df['GarageYrBlt'].fillna(test_df['GarageYrBlt'].mean())




# Filter Numerical Features
numerical_features  = train_df.select_dtypes(include=['int64','float64'])
numerical_features.columns




#2. Interactions between Features



#Pearson Correlation
plt.figure(figsize=(14,8))
bars=train_df.corr()['SalePrice'].sort_values(ascending=False).plot(kind='bar')



#Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(train_df.corr(), cmap="hot")
plt.title("Heatmap Correlations Between Features")
plt.show()


#3. Split into test and training data
X = train_df[['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
       'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
       'MiscVal', 'MoSold', 'YrSold']]
y = train_df['SalePrice']
train_df.describe()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)
X_train.head()



y_train.head()


#4. Fit a baseline model --XGBoost(XGBoostRegressor)
from xgboost import XGBRegressor
regressor=XGBRegressor()
model=regressor.fit(X_train,y_train)
y_pred=model.predict(X_test)


#Plot Predicted house price v.s. Actual house price(base model performance)
test=pd.DataFrame({'Predicted value':y_pred,'Actual value':y_test})
fig=plt.figure(figsize=(16,8))
test=test.reset_index()
test=test.drop(['index'],axis=1)
plt.plot(test[:50])
plt.legend(['Actual value','Predicted value'])


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


#Assess baseline model performance, get mean squared error..(aim to minimize)
print('Mean Absolute Error(MAE):',metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error(MSE):',metrics.mean_squared_error(y_test,y_pred))
print('Root Mean SquaredError(RMSE):',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


#Using Decision Tree to generate new feature


#Fit a DecisionTreeRegressor, and choose max-depth at 3
from sklearn.tree import DecisionTreeRegressor

DT=DecisionTreeRegressor(max_depth=4,random_state=11)



DT.fit(X_train,y_train)



#Extract the tree attributes
n_nodes = DT.tree_.node_count
children_left = DT.tree_.children_left
children_right = DT.tree_.children_right
feature = DT.tree_.feature



#Traverse the tree to get features in each branch
def extract_path_features(node, path_features):
    if children_left[node] != children_right[node]:  # check if it's an internal node
        left_path = path_features + [feature[node]]
        right_path = path_features + [feature[node]]

        left_branch = extract_path_features(children_left[node], left_path)
        right_branch = extract_path_features(children_right[node], right_path)

        return left_branch + right_branch

    return [path_features]

branch_features = extract_path_features(0, [])



#Convert feature indices to feature names
branch_features_named = [[X.columns[f] for f in branch] for branch in branch_features]



#Each inner list represents a branch in the tree
branch_features_named


#'OverallQual': Overall material and finish quality; 
#'GrLivArea':   Above grade (ground) living area square feet; 
#'GarageCars':  Size of garage in car capacity
#'YearBuilt': Original construction date
#'BsmtFinSF1': Type 1 finished square feet
#'1stFlrSF': First Floor square feet
#'LotFrontage': Linear feet of street connected to property
#'WoodDeckSF': Wood deck area in square feet


#Visualize this tree
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plot_tree(DT, filled=True, feature_names=X.columns, rounded=True)
plt.show()


#Find interactions by computing ratio
def create_interaction_features(data, branch_features_named):
    new_features = pd.DataFrame()
    for branch in branch_features_named:
        if len(branch) == 1:
            continue
        interaction_name = "_div_".join(branch)
        # Check if all features in branch are numerical (either int or float)
        if all(pd.api.types.is_numeric_dtype(data[col].dtype) for col in branch):
            # Compute the ratio
            denominator = data[branch[1:]].prod(axis=1)
            # Replace 0 in the denominator to avoid division by zero
            denominator = denominator.replace(0, 1)
            interaction_feature = data[branch[0]] / denominator
        else:
            # For non-numerical features, concatenate as string
            interaction_feature = data[branch].astype(str).agg("_".join, axis=1)
        new_features[interaction_name] = interaction_feature
    return new_features

new_feature_data = create_interaction_features(X, branch_features_named)


#Concatenate interaction features with original features
X_extended = pd.concat([X, new_feature_data], axis=1)
X_extended



#list of interaction feature names
new_feature_data_list = new_feature_data.columns.tolist()
print(new_feature_data_list)



#Separate each element by the ',' to get new individual features
new_feature = [name.split(',') for name in new_feature_data_list]
print(new_feature)


#    Remove the '_div_' part and duplicates from the feature names in the lists.
def clean_feature_names(new_feature):
    cleaned_lists = []
    for feature_list in new_feature:
        # Extracting the first element of each list and splitting it by '_div_'
        cleaned_features = feature_list[0].split('_div_')
        # Removing duplicates while preserving order
        cleaned_features = list(dict.fromkeys(cleaned_features))
        cleaned_lists.append(cleaned_features)
    return cleaned_lists

# Cleaning the feature names
cleaned_feature_lists = clean_feature_names(new_feature)
cleaned_feature_lists


total_number_of_lists = len(cleaned_feature_lists)
total_number_of_lists


#Rename new features:
new_feature1=cleaned_feature_lists[0]
new_feature2=cleaned_feature_lists[1]
new_feature3=cleaned_feature_lists[2]
new_feature4=cleaned_feature_lists[3]
new_feature5=cleaned_feature_lists[4]
new_feature6=cleaned_feature_lists[5]
new_feature7=cleaned_feature_lists[6]


#Hyperparameter Tuning: Optimize max-depth parameters using GridSearchCV
from sklearn.tree import DecisionTreeRegressor
param_grid={
    'max_depth':[None,2,3,4]
}
DTModel=GridSearchCV(
    DecisionTreeRegressor(random_state=11),
    cv=10,
    scoring='neg_mean_squared_error',
    param_grid=param_grid
)


#Pass to X_train, and using fit in DecisionTree model
DTModel.fit(X_train[new_feature1],y_train)
DTModel.fit(X_train[new_feature2],y_train)
DTModel.fit(X_train[new_feature3],y_train)
DTModel.fit(X_train[new_feature4],y_train)
DTModel.fit(X_train[new_feature5],y_train)
DTModel.fit(X_train[new_feature6],y_train)


#Using Predict() to predict target variable using two new features, assign back to X_train,X_test dataset

#for feature 1
X_train=X_train.assign(OverQual_Area_Year=DTModel.predict(X_train[new_feature1]))
X_test=X_test.assign(OverQual_Area_Year=DTModel.predict(X_test[new_feature1]))


#for feature 2
X_train=X_train.assign(OverQual_Area_Bsmt=DTModel.predict(X_train[new_feature2]))
X_test=X_test.assign(OverQual_Area_Bsmt=DTModel.predict(X_test[new_feature2]))


#for feature 3
X_train=X_train.assign(OverQual_Area_TBsmt=DTModel.predict(X_train[new_feature3]))
X_test=X_test.assign(OverQual_Area_TBsmt=DTModel.predict(X_test[new_feature3]))


#for feature 4
X_train=X_train.assign(OverQual_Area_Flr=DTModel.predict(X_train[new_feature4]))
X_test=X_test.assign(OverQual_Area_Flr=DTModel.predict(X_test[new_feature4]))


#for feature 5 -- only two, but DT expecting 3 features as input
#X_train=X_train.assign(OverQual_Area=DTModel.predict(X_train[new_feature5]))
#X_test=X_test.assign(OverQual_Area=DTModel.predict(X_test[new_feature5]))


#for feature 6
X_train=X_train.assign(OverQual_Car_Lot=DTModel.predict(X_train[new_feature6]))
X_test=X_test.assign(OverQual_Car_Lot=DTModel.predict(X_test[new_feature6]))



#for feature 7
X_train=X_train.assign(OverQual_Car_Wood=DTModel.predict(X_train[new_feature7]))
X_test=X_test.assign(OverQual_Car_Wood=DTModel.predict(X_test[new_feature7]))



X_train.head()


X_test.head()


y_train.head()


y_test.head()


#Re-fit baseline model using new feature to new X_train data, and predict new X_train data
regressor=XGBRegressor()
model=regressor.fit(X_train,y_train)
y_pred=model.predict(X_test)



#Print RMSE score to see if is reduced loss function, if yes, then have better performance due to new feature
print('Mean Absolute Error(MAE):',metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error(MSE):',metrics.mean_squared_error(y_test,y_pred))
print('Root Mean SquaredError(RMSE):',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)



#Plot
test=pd.DataFrame({'Predicted value':y_pred,'Actual value':y_test})
fig=plt.figure(figsize=(16,8))
test=test.reset_index()
test=test.drop(['index'],axis=1)
plt.plot(test[:50])
plt.legend('Actual value','Predicted value')





