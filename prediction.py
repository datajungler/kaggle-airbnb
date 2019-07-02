__author__ = 'Horace'

execfile("setting.py")

import numpy, pandas, random
from read import read
from visualize import plot
import datetime
from sklearn import cross_validation, preprocessing
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

# Read raw data
train_data = read.read('train_users.csv') # obs = 275547
test_input = read.read('test_users.csv')

# Sampling
train_data = train_data.sample(n=40000)

# Filter Noise & Outlier
train_data = train_data[(train_data.first_device_type <> "iPhone") | (train_data.first_browser <> "IE Mobile")]

# Transform target variable to numeric variable
train_input = train_data.drop(['country_destination'], axis=1)
train_target = label_encoder.fit_transform(train_data['country_destination'])

# Data Processing
input_data = pandas.concat((train_input, test_input), axis=0, ignore_index=True)

dac_df = pandas.to_datetime(input_data['date_account_created'])
fta_list = [datetime.datetime.strptime(str(fta)[:8], "%Y%m%d") for fta in input_data['timestamp_first_active']]

fta_df = pandas.Series(fta_list)
fta_df = pandas.to_datetime(fta_df)
input_data['diff_dac_tfa'] = (fta_df - dac_df).astype('timedelta64[h]')


# Data Transformation

# date_account_created
## the parameter: day does not significantly effect on the performance
input_data['dac_year'] = pandas.Series([dac.split('-')[0] for dac in input_data['date_account_created']])
input_data['dac_month'] = pandas.Series([dac.split('-')[1] for dac in input_data['date_account_created']])
# input_data['dac_day'] = pandas.Series([dac.split('-')[2] for dac in input_data['date_account_created']])
input_data = input_data.drop('date_account_created', axis=1)

# timestamp_first_active
## the parameter: day does not significantly effect on the performance
input_data['tfa_year'] = pandas.Series([str(tfa)[0:4] for tfa in input_data['timestamp_first_active']])
input_data['tfa_month'] = pandas.Series([str(tfa)[4:6] for tfa in input_data['timestamp_first_active']])
# input_data['tfa_day'] = pandas.Series([str(tfa)[6:8] for tfa in input_data['timestamp_first_active']])
input_data = input_data.drop('timestamp_first_active', axis=1)

# Standardization of input parameters
## age
input_data = input_data.fillna(0)
input_data['age_normal'] = preprocessing.scale(input_data['age'])
age_normal_values = input_data['age_normal'].values
age_normal_mean = numpy.mean(age_normal_values)
input_data.loc[(input_data.age < 10) & (input_data.age > 100), ['age_normal']] = age_normal_mean
input_data = input_data.drop('age', axis=1)

## difference between date_account_created and timestamp_first_active
input_data['diff_dac_tfa_normal'] = preprocessing.scale(input_data['diff_dac_tfa'])
input_data = input_data.drop('diff_dac_tfa', axis=1)

## Feature Engineering
input_data = input_data.drop(['id', 'date_first_booking'], axis=1)
discrete_var = ['signup_method', 'signup_flow', 'gender', 'signup_app', 'language', 'affiliate_channel', 'affiliate_provider',
                'first_affiliate_tracked', 'first_device_type', 'first_browser']

for var in discrete_var:
    dummy_data = pandas.get_dummies(data=input_data[var], prefix=var)
    input_data = input_data.drop(var, axis=1)
    input_data = pandas.concat((input_data, dummy_data), axis=1)

for col in input_data.columns:
    print col

# Data Partition to split the train and test data
input_list = input_data.values
train_obs = len(train_input)
exclude_test_input = input_list[:train_obs]
exclude_test_target = train_target
# train_input, validation_input, train_target, validation_target = cross_validation.train_test_split(exclude_test_input, exclude_test_target, test_size=0.4)

# Choose Model
## XGBClassifier performs the best

from model.tree import visualize
from sklearn import tree, linear_model
from sklearn import svm
# from sklearn import neural_network
from sklearn import ensemble
import xgboost
from xgboost import sklearn
# classifier = tree.DecisionTreeClassifier(criterion="entropy", max_depth=7)
# classifier = linear_model.LogisticRegression(C=1.0, solver="lbfgs", multi_class="multinomial")
# classifier = svm.SVC(C=0.1, kernel='rbf')
# classifier = neural_network.MLPClassifier()
# classifier = ensemble.GradientBoostingClassifier()
# classifier = ensemble.RandomForestClassifier(criterion="entropy", max_depth=15)
classifier = sklearn.XGBClassifier(base_score=0.5, learning_rate=0.05, gamma=1.5, max_depth=7, colsample_bytree=1, subsample=0.2, n_estimators=25, seed=0, objective="multi:softprob")

# params = {'max_depth': 6, 'colsample_bytree': 1, 'n_estimators': 25, 'objective': 'multi:softprob', 'num_class': 12}
# dtrain = xgboost.DMatrix(exclude_test_input, exclude_test_target)
# classifier = xgboost.train(params=params, dtrain=dtrain, num_boost_round=1)
# xgboost.plot_importance(classifier)

classifier.fit(exclude_test_input, exclude_test_target)

# visualize
# visualize(classifier, "tree")

print "The model has been created."

# Model Assessment

## Validation Curve
# plot.plot_validation_curve(classifier, exclude_test_input, exclude_test_target, param_name="gamma")

## Learning Curve
plot.plot_learning_curve(classifier, exclude_test_input, exclude_test_target)

# Predict testing Data
test_input_values = input_list[train_obs:]
test_target = classifier.predict_proba(test_input_values)
# test_target = classifier.predict(xgboost.DMatrix(test_input_values)) #.reshape(len(test_input_values), 12)

id_list = []
country_list = []
test_id = test_input.id.values
for i in range(len(test_id)):
    id_list += [test_id[i]] * 5
    country_list += label_encoder.inverse_transform(numpy.argsort(test_target[i])[::-1])[:5].tolist()

submission = pandas.DataFrame(numpy.column_stack((id_list, country_list)), columns=['id', 'country'])
submission.to_csv('submission_v9.csv', index=False)

print submission
