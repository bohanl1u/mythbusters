# Mythbusters

Mythbusters is a webapp built on Anvil for detecting real or fake news from a given text input.

Visit the live app here: [MythBusters](https://mythbusters.anvil.app/)

## Installation/Dependencies

Dependencies:

pandas
sklearn
anvil-uplink

Installation
```bash
pip install pandas
```
```bash
pip install sklearn
```

We also used kaggle to find datasets used for training the ML model.
You can find the link to the libraries [here](https://www.kaggle.com/c/fake-news/data).

## Step 1: Setting up and training the model

```python
df = pd.read_csv("Tests/train.csv")
conversion_dict = {0: 'Real', 1: 'Fake'}
df['label']=df['label'].replace(conversion_dict)
df.label.value_counts()

x_train,x_test,y_train,y_test=train_test_split(df['text'], df['label'], test_size=0.25, random_state=7, shuffle=True)
tfidf_vectorizer=TfidfVectorizer(stop_words='english',max_df=0.75)

vec_train=tfidf_vectorizer.fit_transform(x_train.values.astype('U'))
vec_test=tfidf_vectorizer.transform(x_test.values.astype('U'))

pac=PassiveAggressiveClassifier(max_iter=100)
pac.fit(vec_train,y_train)

y_pred=pac.predict(vec_test)
score=accuracy_score(y_test,y_pred)
print(f'PAC Accuracy: {round(score*100,2)}%')
```

We used a Passive Aggressive Classifier to train the model.
PAC Accuracy: 96.25%

##Step 2: linking to Anvil.app

Install anvil-uplink and import and link using the uplink key obtained in the anvil.app [website](https://anvil.works/)
```bash
pip install anvil-uplink
```
Now we can link it to the app using our key and use a function to implement our model.
```python
import anvil.server
anvil.server.connect("Insert Key Here")

@anvil.server.callable
def findlabel(newtext):
    vec_newtest=tfidf_vectorizer.transform([newtext])
    y_predl=pac.predict(vec_newtest)
    return y_predl[0]
```
##Notes
You can feel free to implement and make your own changes as seem fit.
Made for HackDavis 2021
By: Jimmy Liu, Patricia Tran, Haile Bansil

##Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
