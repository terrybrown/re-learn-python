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

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('./input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

train_data = pd.read_csv("./input/train.csv")
print(train_data.head())

test_data = pd.read_csv("./input/test.csv")
print(test_data.head())

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

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
print(output.to_string())   # .to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")