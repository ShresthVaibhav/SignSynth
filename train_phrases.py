import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('isl_landmarks.csv', low_memory=False)
df['label'] = df['label'].astype(str)

X = df.drop('label', axis=1)
y = df['label']

print(f'Total samples: {len(df)}')
print(f'Total classes: {y.nunique()}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Training...')
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Accuracy: {acc * 100:.2f}%')

with open('model_isl.pkl', 'wb') as f:
    pickle.dump(model, f)

print('Saved as model_isl.pkl')