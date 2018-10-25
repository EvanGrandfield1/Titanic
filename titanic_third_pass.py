from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
# import relevant functions and libraries 

train_file_path = '/Users/egrandfield/Desktop/Titanic/train.csv'
test_file_path = '/Users/egrandfield/Desktop/Titanic/test.csv'
# files paths to data


train_df = pd.read_csv(train_file_path)
#read csv into pandas Dataframe
train_df = train_df.fillna(train_df.mean())
#fill null values wth whatever the mean of the column is
test_df = pd.read_csv(test_file_path)
test_df = test_df.fillna(test_df.mean())
#same stuff, but for the test data 

train_df['Sex'] = train_df['Sex'].apply({'male':0, 'female':1}.get)
test_df['Sex'] = test_df['Sex'].apply({'male':0, 'female':1}.get)
#reduce sex to integer binary

features_train = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Child', 'Young Adult', 'Old']
features_test = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Child', 'Young Adult', 'Old']
#Relevant columns of both csvs to make decisions based on 
done = ["PassengerId", "Survived"]
#Columns in final output csv

x_train, x_test, y_train, y_test = train_test_split(train_df[features_train], train_df['Survived'], test_size=0.3)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)
#split training data into training and faux test (faux test for iterating over fitting data to minimize MAE


#establish classifier 
#n jobs has to do parallel processing, deeper dive into hardware...
#picking random state so output can be replicated


MAE_dict = {}
#establish list
i = 0 
#create iterator
while(i < 100):
	i = i + 1
	clf = RandomForestClassifier(n_estimators=i, n_jobs=2, random_state=0)

	clf.fit(x_train, y_train)
	predicted_values = clf.predict(x_test)
	#meat of the predictions
	true_list = y_val.tolist()
	pred_list = predicted_values.tolist()

	MAE = mean_absolute_error(true_list, pred_list)
	MAE_dict[i] = MAE
#find best number of n_estimators 

best = min(MAE_dict, key=MAE_dict.get)
clf = RandomForestClassifier(n_estimators=best, n_jobs=2, random_state=0)
clf.fit(x_train, y_train)
predicted_values = clf.predict(test_df[features_test])
#fit with the best number of n_estimators



test_df["Survived"] = predicted_values
done_df = test_df[done]
done_df.to_csv('/Users/egrandfield/Desktop/Titanic/done_3.csv')
