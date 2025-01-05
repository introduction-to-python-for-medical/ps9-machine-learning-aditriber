import pandas as pd

df = pd.read_csv('parkinsons.csv')
df.head() 

import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df)
plt.show()

selected_features = ['PPE', 'DFA']
target = 'status'
X = df[selected_features]
y = df[target]  

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

from sklearn.svm import SVC

model = SVC()
model.fit(X_train, y_train) 

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

import joblib

joblib.dump(model, 'my_model')




