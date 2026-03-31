import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv("iris.csv")


le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])


X = df.drop("species", axis=1)
y = df["species"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

n_estimators_list = [10, 50, 100, 200]

best_accuracy = 0
best_n = 0
best_cm = None

for n in n_estimators_list:
    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1), 
        n_estimators=n,
        random_state=42
    )
   
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
   
    acc = accuracy_score(y_test, y_pred)
   
    if acc > best_accuracy:
        best_accuracy = acc
        best_n = n
        best_cm = confusion_matrix(y_test, y_pred)

print("Best Accuracy:", best_accuracy)
print("Best Number of Trees:", best_n)
print("Confusion Matrix:\n", best_cm)
