from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import warnings
from math import sqrt
warnings.filterwarnings('ignore')

data = pd.read_csv('data.csv')
fifa = pd.DataFrame(data)


fifa = fifa.drop(['Unnamed: 0', 'ID', 'Nationality', 'Flag', 'Club', 'Club Logo', 'Special', 'Preferred Foot', 'Real Face',
                  'Position', 'Jersey Number', 'Joined', 'Loaned From', 'Contract Valid Until', 'Height', 'Weight', 'Body Type'], axis=1)

fifa = fifa.drop(['Photo'], axis=1)

test1 = fifa
test2 = fifa

# DROPPING IRRELEVANT PROBLEMS
newfifa = pd.DataFrame(data)
newfifa = newfifa.drop(['Unnamed: 0', 'ID', 'Nationality', 'Flag', 'Club', 'Club Logo', 'Special', 'Preferred Foot', 'Real Face',
                        'Position', 'Jersey Number', 'Joined', 'Loaned From', 'Contract Valid Until', 'Height', 'Weight', 'Body Type'], axis=1)
newfifa = newfifa.drop(['Photo'], axis=1)


print(fifa.shape)

newfifa.info()


# cleaning currency values
def curreny_clean(val):
    try:
        value = float(val[1:-1])
        suffix = val[-1:]

        if suffix == 'M':
            value = value*1000000

        elif suffix == 'K':
            value = value*1000

    except ValueError:
        value = 0

    except TypeError:
        value = 0

    return value


newfifa['Value'] = newfifa['Value'].apply(curreny_clean)
newfifa['Wage'] = newfifa['Wage'].apply(curreny_clean)

print(newfifa['Release Clause'])
newfifa['Release Clause'] = newfifa['Release Clause'].apply(curreny_clean)
# cleaned curriencies

newfifa.info()

# For Positions data evaluation


def evaluation(e):
    try:
        temp = str(e)
        x = int(temp[0:2])
        y = int(temp[-1:])

        expr = 'x+y'
        result = int(eval(expr))

    except ValueError:
        result = 0

    return result


newfifa['LS'] = newfifa['LS'].apply(evaluation)

newfifa['ST'] = newfifa['ST'].apply(evaluation)

newfifa['RS'] = newfifa['RS'].apply(evaluation)

newfifa['LW'] = newfifa['LW'].apply(evaluation)

newfifa['LF'] = newfifa['LF'].apply(evaluation)

newfifa['CF'] = newfifa['CF'].apply(evaluation)

newfifa['RF'] = newfifa['RF'].apply(evaluation)

newfifa['RW'] = newfifa['RW'].apply(evaluation)

newfifa['LAM'] = newfifa['LAM'].apply(evaluation)

newfifa['CAM'] = newfifa['CAM'].apply(evaluation)

newfifa['RAM'] = newfifa['RAM'].apply(evaluation)

newfifa['LM'] = newfifa['LM'].apply(evaluation)

newfifa['LCM'] = newfifa['LCM'].apply(evaluation)

newfifa['CM'] = newfifa['CM'].apply(evaluation)

newfifa['RCM'] = newfifa['RCM'].apply(evaluation)

newfifa['RM'] = newfifa['RM'].apply(evaluation)

newfifa['LWB'] = newfifa['LWB'].apply(evaluation)

newfifa['LDM'] = newfifa['LDM'].apply(evaluation)

newfifa['CDM'] = newfifa['CDM'].apply(evaluation)

newfifa['RDM'] = newfifa['RDM'].apply(evaluation)

newfifa['RWB'] = newfifa['RWB'].apply(evaluation)

newfifa['LB'] = newfifa['LB'].apply(evaluation)

newfifa['LCB'] = newfifa['LCB'].apply(evaluation)

newfifa['CB'] = newfifa['CB'].apply(evaluation)

newfifa['RCB'] = newfifa['RCB'].apply(evaluation)

newfifa['RB'] = newfifa['RB'].apply(evaluation)

newfifa.info()

newfifacopy = newfifa.copy()

newfifa.shape

clean_fifa = newfifa.fillna(newfifa.mean())

# check this filling for missing data
clean_fifa.isnull().sum()

# correlation matrix and heatmap
corrmat = clean_fifa.corr()
print(corrmat)

#DATA VISUALIZATION

sn.heatmap(corrmat, annot=True)
# plt.show()

#Value and Age
plt.bar(clean_fifa['Age'],clean_fifa['Value'])
plt.xlabel("Age")
plt.ylabel("Value")
plt.show()

#Value and Potential
plt.bar(clean_fifa['Potential'],clean_fifa['Value'])
plt.xlabel("Potential")
plt.ylabel("Value")
plt.show()














nf_x = clean_fifa.iloc[:, :-1]
copynfx = nf_x.copy()

# Split the Work Rate Column in two
tempwork = nf_x["Work Rate"].str.split("/ ", n=1, expand=True)
# Create new column for first work rate and taking the first work rate
nf_x["WorkRate1"] = tempwork[0]

nf_y=clean_fifa['Value'].values

nf_x = nf_x.drop(['Value','Work Rate', 'Name'], axis=1)


# assigning values to work rate


def work_rate(wkr):
    try:
        rate = -1
        if wkr == 'Low':
            rate = 1
        elif wkr == 'Medium':
            rate = 2
        elif wkr == 'High':
            rate = 3
    except TypeError:
        rate = 0

    return rate


# copynfx['WorkRate1']=copynfx['WorkRate1'].apply(work_rate)
nf_x['WorkRate1'] = nf_x['WorkRate1'].apply(work_rate)

nf_x.info()

nf_x = nf_x.iloc[:, :].values


# TRAIN TEST SPLIT
# random_state and test_size can be changed
X_train, X_test, y_train, y_test = train_test_split(
    nf_x, nf_y, test_size=0.3, random_state=0)

"""
#FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

y_train=sc_X.fit_transform(y_train)
"""

names = ['Random Forest', 'Gradient Boosting', 'Decision Tree','XGboost']
clfs = [
    RandomForestRegressor(),
    GradientBoostingRegressor(), DecisionTreeRegressor(),XGBRegressor()]  # linear svr doesnt fit

for name, clf in zip(names, clfs):
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(name+':')
    print('RMSE= '+str(sqrt(mean_squared_error(y_test, pred))))
    print('MAE= '+str(mean_absolute_error(y_test, pred)))
    print('R2= '+str(r2_score(y_test, pred)))

'''
# Linear Regression on Training
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predicting test
y_pred = lin_reg.predict(X_test)


# linear regression SCORE TEST
print('Linear Regression Score = ', lin_reg.score(X_test, y_test))


# Random Forest
rfr = RandomForestRegressor(n_estimators=100, max_depth=2, random_state=0)
rfr.fit(X_train, y_train)
# rfr_pred=rfr.predict(X_test)

print('Random Forest Regressor Score = ', rfr.score(X_test, y_test))
'''


"""
prediction = pd.DataFrame(y_test,pred, columns=['old value','predictions']).to_csv('prediction.csv')
"""

output=pd.DataFrame(data={"Old value":y_test,"Prediction":pred})
output.to_csv("results.csv",float_format='%.2f',index=False)













