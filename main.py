__author__ = 'Horace'

execfile("setting.py")
import os, sys, setting
import numpy, pandas
from read import read
from transform import transform
from visualize import plot
from statistics import summary


FILE_NAME = sys.argv[1]
if sys.argv[1] is None:
    raise AttributeError("Please also input the file name first")

ext = FILE_NAME.split('.')
raw_data = read.read(FILE_NAME)
print raw_data.columns

# clean age


# Visualize and save the data

# ## gender
# plot.bar(raw_data, index=5)
#
# ## age
# plot.hist(raw_data, index=6, bins=5, min_threshold=0, max_threshold=100)
#
# ## signup_method
# plot.bar(raw_data, index=7)
#
# ## affiliate_channel
# plot.bar(raw_data, index=10)
#
# ## signup_app
# plot.bar(raw_data, index=13)
#
## first_device_type
# plot.bar(raw_data, index=14, rotation=45)
#
# ## country_destination (target variable)
# plot.bar(raw_data, index=16)


# gender_list, transform_gender_data = transform.transform_nominal_col(raw_data, column_index=5)
# print transform_gender_data.columns
# classified_target_data = transform.group_class_variable(transform_gender_data, column_index=16, target_list=["NDF", "US"])
# classified_target_data = transform_gender_data[transform_gender_data[transform_gender_data.columns[15]] != "NDF"]
# print gender_list
# plot.bar_plot(raw_data, index=17)

# # scatter plot
# clean_age_data = classified_target_data[classified_target_data.age <= 100]
# plot.scatter(clean_age_data, x_index=17 , y_index=6, color_index=16, x_class_list=gender_list) # color_index=18)

# # Only NDF
# ndf_data = raw_data[raw_data.country_destination == "NDF"]
# ndf_data.to_csv('./clean_data_sources/train_users_ndf.csv', index=False)

# # filter out NDF
# without_ndf_data = raw_data[raw_data.country_destination != "NDF"]
# without_ndf_data.to_csv('./clean_data_sources/train_users_without_ndf.csv', index=False)

# # clean Age
# clean_age_data = raw_data[(raw_data.age <= 100) & (raw_data.age >= 10)]
# clean_age_data.to_csv('./clean_data_sources/train_users_clean_age.csv', index=False)

# # Male distribution
# male_data = clean_age_data[clean_age_data[clean_age_data.columns[4]] == "MALE"]
# plot.hist(male_data, index=6, bins=5, type_name="male")
#
# # Female distribution
# male_data = clean_age_data[clean_age_data[clean_age_data.columns[4]] == "FEMALE"]
# plot.hist(male_data, index=6, bins=5, type_name="female")

# Facebook login => Clear Gender
# facebook_data = raw_data[raw_data.signup_method=="facebook"]
# print raw_data.gender.value_counts()
# print facebook_data.gender.value_counts()

# # verify first_device_type
# facebook_data = raw_data[raw_data.signup_method!="facebook"]
# plot.bar(facebook_data, index=14, type_name="basic login user", rotation=45)