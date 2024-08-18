# CODSOFT_T3
SPAM SMS DETECTION

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import chardet

#message = "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
message = "hello world"

with open("C:/Users/545af/Downloads/WORK/TASK_3/ML_CodSoft_Task_3/spam.csv", 'rb') as f:
  rawdata = f.read()
  result = chardet.detect(rawdata)
  encoding = result['encoding']

print("Detected encoding:", encoding)


data = pd.read_csv("C:/Users/545af/Downloads/WORK/TASK_3/ML_CodSoft_Task_3/spam.csv",engine="python",encoding=encoding)
sms = data["v2"] 
label = data["v1"]

def preprocess_text(text):
  text = text.lower()
  return text

sms = sms.apply(preprocess_text)

vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(sms)

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

model = SVC()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

processed_message = preprocess_text(message)
features = vectorizer.transform([processed_message])

prediction = model.predict(features)[0]

if prediction == 'spam':
  print("This message is likely SPAM.")
else:
  print("This message is likely NOT SPAM.")
