# %% read dataframe from part1
import pandas as pd

df = pd.read_pickle("sqf.pkl")


# %% Using columns of reasons for frisked and reason for stopped and converting them to boolean
rfs = [col for col in df.columns if col.startswith("rf_")]
css = [col for col in df.columns if col.startswith("cs_")]
armed = [
    "contrabn",
    "pistol",
    "riflshot",
    "asltweap",
    "knifcuti",
    "machgun",
    "othrweap",
]

x = df[rfs + css + armed]
x = x == "YES"

# create label for the dataset, then remove the weapons columns
y = (
    x["contrabn"]
    | x["pistol"]
    | x["riflshot"]
    | x["asltweap"]
    | x["knifcuti"]
    | x["machgun"]
    | x["othrweap"]
)
x = x.drop(columns=armed)

# %% Grabbing numeric and categoriccal columns
num_cols = ["age", "height", "weight"]
cat_cols = ["race", "city", "build", "haircolr", "eyecolor",]

x[num_cols] = df[num_cols]
x[cat_cols] = df[cat_cols]


# %% Making training and testing sets for modelling
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123)


# %% Formatting the columns correctly using OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

enc = OneHotEncoder(handle_unknown="ignore")
ct = ColumnTransformer([("ohe", enc, cat_cols),], remainder="drop")

x_train = ct.fit_transform(x_train)
x_test = ct.transform(x_test)



# %% Using DecisionTreeClassifyer and showing some metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import r2_score

clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
pred_train = clf.predict(x_train)
pred_test = clf.predict(x_test)

print(clf.__class__.__name__)
print(f"")
print("Training Results:")
print(f"Accuracy: {accuracy_score(y_train, pred_train)}")
print(f"Precision Score: {precision_score(y_train, pred_train)}")
print(f"Recall: {recall_score(y_train, pred_train)}")
print(f"F1 Score: {f1_score(y_train, pred_train)}")
print(f"")
print("Testing Resuls:")
print(f"Accuracy: {accuracy_score(y_test, pred_test)}")
print(f"Precision Score: {precision_score(y_test, pred_test)}")
print(f"Recall: {recall_score(y_test, pred_test)}")
print(f"F1 Score: {f1_score(y_test, pred_test)}")


# %% Visualizing DecisionTreeClassifier
from sklearn.tree import plot_tree

plot_tree(clf, filled=True, max_depth=3, feature_names=ct.get_feature_names())


# %% Using ExtraTreeClassifier and showing some metrics
from sklearn.tree import ExtraTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import r2_score

clf = ExtraTreeClassifier()
clf.fit(x_train, y_train)
pred_train = clf.predict(x_train)
pred_test = clf.predict(x_test)

print(clf.__class__.__name__)
print(f"")
print("Training Results:")
print(f"Accuracy: {accuracy_score(y_train, pred_train)}")
print(f"Precision Score: {precision_score(y_train, pred_train)}")
print(f"Recall: {recall_score(y_train, pred_train)}")
print(f"F1 Score: {f1_score(y_train, pred_train)}")
print(f"")
print("Testing Resuls:")
print(f"Accuracy: {accuracy_score(y_test, pred_test)}")
print(f"Precision Score: {precision_score(y_test, pred_test)}")
print(f"Recall: {recall_score(y_test, pred_test)}")
print(f"F1 Score: {f1_score(y_test, pred_test)}")


# %% Visualizing ExtraTreeClassifier
from sklearn.tree import plot_tree

plot_tree(clf, filled=True, max_depth=3, feature_names=ct.get_feature_names())


# %% Applying LogisticRegerssion classifier and check results
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import r2_score

clf = LogisticRegression()
clf.fit(x_train, y_train)
pred_train = clf.predict(x_train)
pred_test = clf.predict(x_test)

print(clf.__class__.__name__)
print(f"")
print("Training Results:")
print(f"Accuracy: {accuracy_score(y_train, pred_train)}")
print(f"Precision Score: {precision_score(y_train, pred_train)}")
print(f"Recall: {recall_score(y_train, pred_train)}")
print(f"F1 Score: {f1_score(y_train, pred_train)}")
print(f"")
print("Testing Resuls:")
print(f"Accuracy: {accuracy_score(y_test, pred_test)}")
print(f"Precision Score: {precision_score(y_test, pred_test)}")
print(f"Recall: {recall_score(y_test, pred_test)}")
print(f"F1 Score: {f1_score(y_test, pred_test)}")


# %% Visualizing LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
ax = sns.barplot(y=ct.get_feature_names(), x=clf.coef_[0])

ax.set(xlabel='Attribute')
ax.set(ylabel='Score')


# %% Applying MultinomialNB classifier and check results
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import r2_score


clf = MultinomialNB()
clf.fit(x_train, y_train)
pred_train = clf.predict(x_train)
pred_test = clf.predict(x_test)

print(clf.__class__.__name__)
print(f"")
print("Training Results:")
print(f"Accuracy: {accuracy_score(y_train, pred_train)}")
print(f"Precision Score: {precision_score(y_train, pred_train)}")
print(f"Recall: {recall_score(y_train, pred_train)}")
print(f"F1 Score: {f1_score(y_train, pred_train)}")
print(f"")
print("Testing Resuls:")
print(f"Accuracy: {accuracy_score(y_test, pred_test)}")
print(f"Precision Score: {precision_score(y_test, pred_test)}")
print(f"Recall: {recall_score(y_test, pred_test)}")
print(f"F1 Score: {f1_score(y_test, pred_test)}")


# %% Visualizing MultinomialNB
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
ax = sns.barplot(y=ct.get_feature_names(), x=clf.coef_[0])

ax.set(xlabel='Attribute')
ax.set(ylabel='Score')


# %% Applying SGDClassifier and check results
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import r2_score


clf = SGDClassifier()
clf.fit(x_train, y_train)
pred_train = clf.predict(x_train)
pred_test = clf.predict(x_test)

print(clf.__class__.__name__)
print(f"")
print("Training Results:")
print(f"Accuracy: {accuracy_score(y_train, pred_train)}")
print(f"Precision Score: {precision_score(y_train, pred_train)}")
print(f"Recall: {recall_score(y_train, pred_train)}")
print(f"F1 Score: {f1_score(y_train, pred_train)}")
print(f"")
print("Testing Resuls:")
print(f"Accuracy: {accuracy_score(y_test, pred_test)}")
print(f"Precision Score: {precision_score(y_test, pred_test)}")
print(f"Recall: {recall_score(y_test, pred_test)}")
print(f"F1 Score: {f1_score(y_test, pred_test)}")

# %% Visualizing SGDClassifier 
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
ax = sns.barplot(y=ct.get_feature_names(), x=clf.coef_[0])

ax.set(xlabel='Attribute')
ax.set(ylabel='Score')


# %%