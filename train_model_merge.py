import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

print('Loading and merging datasets...')
df_letters = pd.read_csv('landmarks.csv')
df_phrases = pd.read_csv('phrases_landmarks.csv')

df = pd.concat([df_letters, df_phrases], ignore_index=True)

print(f'Letters samples: {len(df_letters)}')
print(f'Phrases samples: {len(df_phrases)}')
print(f'Total combined: {len(df)}')
print(f'Total classes: {df["label"].nunique()}')

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Training model...')
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Accuracy: {acc * 100:.2f}%')

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print('Model saved as model.pkl')