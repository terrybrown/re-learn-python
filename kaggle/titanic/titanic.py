# Source: https://www.kaggle.com/alexisbcook/titanic-tutorial
# See also: https://www.kaggle.com/c/titanic/data
#
#   Variable	    Definition	Key
#   survival	    Survival	0 = No, 1 = Yes
#   pclass	        Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
#   sex	            Sex	
#   Age	            Age in years	
#   sibsp	        # of siblings / spouses aboard the Titanic	
#   parch	        # of parents / children aboard the Titanic	
#   ticket	        Ticket number	
#   fare	        Passenger fare	
#   cabin	        Cabin number	
#   embarked	    Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton
#
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('./input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

train_data = pd.read_csv("./input/train.csv")
print(train_data.head())
print(train_data.info())
print(train_data.describe())

dataframe_numeric = train_data[['Age', 'SibSp', 'Parch', 'Fare']]
dataframe_category = train_data[['Survived', 'Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked']]

# Plot Histograms for all numeric fields in the histogram - again, to visually support the learning
# for i in dataframe_numeric.columns:
#     plt.hist(dataframe_numeric[i])
#     plt.title(i)
#     plt.show()

# Correlations
print(dataframe_numeric.corr())
sns.heatmap(dataframe_numeric.corr())

# Compare survival rate across Age, SibSp, Parch, and Fare
pd.pivot_table(train_data, index = 'Survived', values = ['Age', 'SibSp', 'Parch', 'Fare'])

# bar plot against the category data
# for i in dataframe_category.columns:
#     sns.barplot(dataframe_category[i].value_counts().index, dataframe_category[i].value_counts())\
#         .set_title(i)
#     plt.show()

print(pd.pivot_table(train_data, index = 'Survived', columns = 'Pclass', values = 'Ticket', aggfunc = 'count'))
print(pd.pivot_table(train_data, index = 'Survived', columns = 'Sex', values = 'Ticket', aggfunc = 'count'))
print(pd.pivot_table(train_data, index = 'Survived', columns = 'Embarked', values = 'Ticket', aggfunc = 'count'))

# Feature Engineering
# Take their title
#                                                              "Braund, Mr. Owen Harris"
#                                                                                "Mr. Owen Harris
#                                                                                           "Mr"
train_data['person_title'] = train_data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
# print(train_data['person_title'].value_counts())


# Data Preprocessing for Model
test_data = pd.read_csv("./input/test.csv")
# print(test_data.head())


from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# gnb = GaussianNB()
# cv = cross_val_score(gnb, train_data, test_data, cv=5)
# print(cv)


y = train_data["Survived"]

#           Passenger Class (1st, 2nd, 3rd)
#                      Gender
#                            Siblings/Spouses aboard titanic
#                                     Parents/Children aboard titanic
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)
cv = cross_val_score(model, train_data, test_data)
print(cv)

raise SystemExit(0)



# Did all women survive?
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women) # It seems not, 74.2%

# What % of men survived?
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men) # Less, 18.9%

##########################################################################
# Delving into sklearn
from sklearn.ensemble import RandomForestClassifier

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")