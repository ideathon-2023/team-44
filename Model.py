import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle

data_training=pd.read_csv('C:/Users\Raghav\Documents\Ideathon\Data_Ideathon\Training.csv')


data_training.drop("Unnamed: 133", axis=1, inplace=True)
x=data_training.drop('prognosis', axis=1)
y=data_training['prognosis']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


tree=DecisionTreeClassifier()


tree.fit(x_train,y_train)


pred=tree.predict(x_test)
accuracy=tree.score(x_test,y_test)

pickle.dump(tree,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','wb'))





