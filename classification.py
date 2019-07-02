__author__ = 'Horace'

execfile("setting.py")

import numpy, pandas
from read import read
from model import tree, lin_reg
from transform import split

# Import dataset
from sklearn.datasets import load_iris
iris = load_iris()

# Choose Model
data = pandas.DataFrame(iris.data, columns=iris.feature_names[0:4])

input_data, target_data = split.split_to_train(data, target_index=3)

train, test = split.partition(dataset=input_data)

#lin_classifier = lin_reg.fit(train_input=input_data[input_data.columns[0]], train_target=target_data)
#lin_reg.plot(lin_classifier, input=input_data[input_data.columns[0]], target=target_data)

lin_classifier = lin_reg.fit(train_input=input_data, train_target=target_data)
lin_reg.plot(lin_classifier, input=input_data, input_index=2, target=target_data)






