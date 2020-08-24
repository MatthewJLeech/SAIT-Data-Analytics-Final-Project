# %% read dataframe from part1
# df -->500224 rows, 79 columns
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


df = pd.read_pickle("sqf.pkl")


# %% select some yes/no columns to convert into a dataframe of boolean values for apriori 
pfs = [col for col in df.columns if col.startswith("pf_")]
# creates a new list called pfs
armed = [
    "contrabn",
    "pistol",
    "riflshot",
    "asltweap",
    "knifcuti",
    "machgun",
    "othrweap",
]
# x is a new dataframe that lets us know if someone is armed within the pfs dataframe
# x --> 500224 rows, 17 columns
x = df[pfs + armed]
x = x == "YES"


# %% create a new column to represent whether a person is armed
x["armed"] = (
    x["contrabn"]
    | x["pistol"]
    | x["riflshot"]
    | x["asltweap"]
    | x["knifcuti"]
    | x["machgun"]
    | x["othrweap"]
)

# %% select some categorical columns and do one hot encoding to allow usage with apriori
# here, add more columns from df, (build, haircolour, eyecolour, other?) and import them into x
# x --> 500224 rows, 32 columns
for val in df["haircolr"].unique():
    x[f"haircolr_{val}"] = df["haircolr"] == val

for val in df["eyecolor"].unique():
    x[f"eyecolor_{val}"] = df["eyecolor"] == val

for val in df["build"].unique():
    x[f"build_{val}"] = df["build"] == val

for val in df["race"].unique():
    x[f"race_{val}"] = df["race"] == val


for val in df["city"].unique():
    x[f"city_{val}"] = df["city"] == val


for val in df["sex"].value_counts().index:
    x[f"sex_{val}"] = df["sex"] == val

for val in df["age"].value_counts().index:
    x[f"age_{val}"] = df["age"] == val

# %% applying frequent itemsets mining
frequent_itemsets = apriori(x, min_support=0.5, use_colnames=True)
frequent_itemsets


# %% apply association rules mining 
rules = association_rules(frequent_itemsets, min_threshold=0.7)
rules


# %% sort rules by confidence and select rules with "sex_MALE" in it

rules.sort_values("confidence", ascending=False)[
    rules.apply(
        lambda r: "sex_MALE" in r["antecedents"]
    or "sex_MALE" in r["consequents"],
    axis=1,
    )
]

# %% creating a table 
ax = sns.scatterplot(
x="support", y="confidence", alpha=0.5, data=rules
)
plt.show()


