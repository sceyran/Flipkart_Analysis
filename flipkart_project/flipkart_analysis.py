import warnings

from sklearn.metrics import accuracy_score, confusion_matrix

warnings.filterwarnings('ignore')
import pandas as pd
import re
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


data = pd.read_csv('datasets/flipkart_data.csv')
data.head()


pd.unique(data['rating'])


# rating graphics
sns.countplot(data=data,
              x='rating',
              order=data.rating.value_counts().index)

# rating label(final)
# if the grade is less than or equal to 4, it will be negative (0), otherwise it will be positive (1)
pos_neg = []
for i in range(len(data['rating'])):
    if data['rating'][i] >= 5:
        pos_neg.append(1)
    else:
        pos_neg.append(0)

data['label'] = pos_neg


from tqdm import tqdm
def preprocess_text(text_data):
    preprocessed_text = []

    for sentence in tqdm(text_data):
        # Removing punctuations
        sentence = re.sub(r'[^\w\s]', '', sentence)

        # Converting lowercase and removing stopwords
        preprocessed_text.append(' '.join(token.lower()
                                          for token in nltk.word_tokenize(sentence)
                                          if token.lower() not in stopwords.words('english')))

    return preprocessed_text


preprocessed_review = preprocess_text(data['review'].values)
data['review'] = preprocessed_review
data.head()

data["label"].value_counts()

consolidated = ' '.join(
    word for word in data['review'][data['label'] == 1].astype(str))
wordCloud = WordCloud(width=1600, height=800,
                      random_state=21, max_font_size=110)
plt.figure(figsize=(15, 10))
plt.imshow(wordCloud.generate(consolidated), interpolation='bilinear')
plt.axis('off')
plt.show()


cv = TfidfVectorizer(max_features=2500)
x = cv.fit_transform(data['review'] ).toarray()
x


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, data['label'],
													test_size=0.33,
													stratify=data['label'],
													random_state = 42)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=0)
model.fit(X_train,y_train)

#testing the model
pred = model.predict(X_train)
print(accuracy_score(y_train,pred))

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=0)
model.fit(X_train,y_train)

#testing the model
pred = model.predict(X_train)
print(accuracy_score(y_train,pred))

from sklearn import metrics
cm = confusion_matrix(y_train,pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm,
											display_labels = [False, True])

cm_display.plot()
plt.show()
