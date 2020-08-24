# %% start by downloading libraries needed --> with the requirements.txt file
# pip install -r .\requirements.txt

# %% importing data from for "2012.csv" file 
# df --> 532911 rows, 112 columns
import pandas as pd

df = pd.read_csv("2012.csv")


# %% looking at the data to describe it
df.describe()


# %% this cell block is to convert columns to int and to corece the errors 
cols = [
    "perobs",
    "perstop",
    "age",
    "weight",
    "ht_feet",
    "ht_inch",
    "datestop",
    "timestop",
    "xcoord",
    "ycoord",
]

for col in cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# %% creating a dataframe that drops cells with invalid values 
# df --> 516986 rows, 112 columns
df = df.dropna()


# %% making a datetime column 
# df --> 516986 rows, 113 columns
df["datestop"] = df["datestop"].astype(str).str.zfill(8)
df["timestop"] = df["timestop"].astype(str).str.zfill(4)

from datetime import datetime

#the below section takes information from both datestop and timestop based on its index and creates a new column
def make_datetime(datestop, timestop):
    year = int(datestop[-4:])
    month = int(datestop[:2])
    day = int(datestop[2:4])
    
    hour = int(timestop[:2])
    minute = int(timestop[2:])

    return datetime(year, month, day, hour, minute)

# datetime is now the new column created with the attributes above and datatype datetime64
df["datetime"] = df.apply(
    lambda r: make_datetime(r["datestop"], r["timestop"]),
    axis=1
)



# %% converting all values to label in the dataframe and removing rows that cannot be mapped
# df --> 503074 rows, 113 columns
import numpy as np
from tqdm import tqdm

value_label = pd.read_excel(
    r"2012 SQF File Spec.xlsx",
    sheet_name="Value Labels",
    skiprows=range(4)
)
value_label["Field Name"] = value_label["Field Name"].fillna(
    method="ffill"
)
value_label["Field Name"] = value_label["Field Name"].str.lower()
value_label["Value"] = value_label["Value"].fillna(" ")
value_label = value_label.groupby("Field Name").apply(
    lambda x: dict([(row["Value"], row["Label"]) for row in x.to_dict("records")])
)

cols = [col for col in df.columns if col in value_label]

for col in tqdm(cols):
    df[col] = df[col].apply(
        lambda val: value_label[col].get(val, np.nan)
    )

df["trhsloc"] = df["trhsloc"].fillna("P (unknown)")
df = df.dropna()


# %% converting xcoord and ycoord to (lon, lat) and creating a new column called "coord"
# df --> 503074 rows, 114 columns
import pyproj

srs = "+proj=lcc +lat_1=41.03333333333333 +lat_2=40.66666666666666 +lat_0=40.16666666666666 +lon_0=-74 +x_0=300000.0000000001 +y_0=0 +ellps=GRS80 +datum=NAD83 +to_meter=0.3048006096012192 +no_defs"
p = pyproj.Proj(srs)

df["coord"] = df.apply(
    lambda r: p(r["xcoord"], r["ycoord"], inverse=True), axis=1
)


# %% converting height in feet/inch to cm and creating a new column called "height"
# df --> 503074 rows, 115 columns
df["height"] = (df["ht_feet"] * 12 + df["ht_inch"]) * 2.54


# %% removing outliers in the "age" and "weight" columns
# df --> 500224 rows, 115 columns
df = df[(df["age"] <= 80) & (df["age"] >= 18)]
df = df[(df["weight"] <= 350) & (df["weight"] >= 50)]


# %% deleting columns that are not needed
# df --> 500224 rows, 79 columns
df = df.drop(
    columns=[
        # processed columns
        "datestop",
        "timestop",
        "ht_feet",
        "ht_inch",
        "xcoord",
        "ycoord",        
        
        # not useful
        "year",
        "recstat",
        "crimsusp",
        "dob",
        "ser_num",
        "arstoffn",
        "sumoffen",
        "compyear",
        "comppct",
        "othfeatr",
        "adtlrept",
        "dettypcm",
        "linecm",
        "repcmd",
        "revcmd",

        # location of stop 
        # only use coord and city
        "addrtyp",
        "rescode",
        "premtype",
        "premname",
        "addrnum",
        "stname",
        "stinter",
        "crossst",
        "aptnum",
        "state",
        "zip",
        "addrpct",
        "sector",
        "beat",
        "post",
    ]
)


# %% creating a simple graph for month
import folium 
import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.countplot(df["datetime"].dt.month)
ax.set_xticklabels(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
)
ax.set(xlabel='Month of Year')

for p in ax.patches:
    percentage = p.get_height() * 100 / df.shape[0]
    txt = f"{percentage:.1f}%"
    x_loc = p.get_x()
    y_loc = p.get_y() + p.get_height()
    ax.text(x_loc, y_loc, txt)


# %% creating a simple graph for weekday
ax = sns.countplot(df["datetime"].dt.weekday)

ax.set_xticklabels(
    ["Mon", "Tue", "Wed", "Thur", "Fri", "Sat", "Sun"]
)
ax.set(xlabel='Day of Week')

for p in ax.patches:
    percentage = p.get_height() * 100 / df.shape[0]
    txt = f"{percentage:.1f}%"
    x_loc = p.get_x()
    y_loc = p.get_y() + p.get_height()
    ax.text(x_loc, y_loc, txt)


# %% creating a graph for hour
ax = sns.countplot(df["datetime"].dt.hour)

ax.set(xlabel='Hour of Day')

for p in ax.patches:
    percentage = p.get_height() * 100 / df.shape[0]
    txt = f"{percentage:.1f}%"
    x_loc = p.get_x()
    y_loc = p.get_y() + p.get_height()
    ax.text(x_loc, y_loc, txt)


# %% creating a graph for race
ax = sns.countplot(data=df,x="race")
ax.set_xticklabels(
    ax.get_xticklabels(), rotation=45, fontsize=5
)
for p in ax.patches:
    percentage = p.get_height() * 100 / df.shape[0]
    txt = f"{percentage:.1f}%"
    x_loc = p.get_x()
    y_loc = p.get_y() + p.get_height()
    ax.text(x_loc, y_loc, txt)


# %% creating a simple graph for age
ax = sns.countplot(data=df,x="age")
ax.set_xticklabels(
    ax.get_xticklabels(), rotation=45, fontsize=5
)


# %% creating a simple graph for build
ax = sns.countplot(data=df,x="build")
ax.set_xticklabels(
    ax.get_xticklabels(), rotation=45, fontsize=5
)
for p in ax.patches:
    percentage = p.get_height() * 100 / df.shape[0]
    txt = f"{percentage:.1f}%"
    x_loc = p.get_x()
    y_loc = p.get_y() + p.get_height()
    ax.text(x_loc, y_loc, txt)


# %% creating a graph for eye colour
ax = sns.countplot(data=df,x="eyecolor")
ax.set_xticklabels(
    ax.get_xticklabels(), rotation=45, fontsize=5
)
for p in ax.patches:
    percentage = p.get_height() * 100 / df.shape[0]
    txt = f"{percentage:.1f}%"
    x_loc = p.get_x()
    y_loc = p.get_y() + p.get_height()
    ax.text(x_loc, y_loc, txt)


# %% creating a graph for race and sex
ax = sns.countplot(df["sex"], hue=df["city"])
ax.set_xticklabels(
    ax.get_xticklabels(), rotation=45, fontsize=5
)
for p in ax.patches:
    percentage = p.get_height() * 100 / df.shape[0]
    txt = f"{percentage:.1f}%"
    x_loc = p.get_x()
    y_loc = p.get_y() + p.get_height()
    ax.text(x_loc, y_loc, txt)


# %% creating a graph for race and city
ax = sns.countplot(df["race"], hue=df["city"])
ax.set_xticklabels(
    ax.get_xticklabels(), rotation=45, fontsize=5
)
for p in ax.patches:
    percentage = p.get_height() * 100 / df.shape[0]
    txt = f"{percentage:.1f}%"
    x_loc = p.get_x()
    y_loc = p.get_y() + p.get_height()
    ax.text(x_loc, y_loc, txt)


# %% creating a scatterplot with age, weight, and sex
ax = sns.scatterplot(x="age", y="weight", hue="sex", data=df)
plt.xlabel("Age")
plt.ylabel("Weight in lbs")
plt.show()

# %% creating a lineplot showing the weight based off of age and sex
ax = sns.lineplot(x="age", y="weight", hue="sex", data=df)
plt.show()

# %% creating a lineplot shoing the weight based off og age and crime committed
ax = sns.lineplot(x="age", y="weight", hue="detailcm", data=df)
plt.show()
# %% adding a map of NYC
nyc = (40.730610, -73.935242)

m = folium.Map(location=nyc)


# %% creating circles for grand larceny auto crimes

for coord in df.loc[df["detailcm"]=="GRAND LARCENY AUTO", "coord"]:
    folium.CircleMarker(
        location=(coord[1], coord[0]), radius=0.2, color="green"
    ).add_to(m)

m


# %% creating circles for kidnapping crimes
for coord in df.loc[df["detailcm"]=="KIDNAPPING", "coord"]:
    folium.CircleMarker(
        location=(coord[1], coord[0]), radius=0.2, color="blue"
    ).add_to(m)

m



# %% creating a pickle file
df.to_pickle("sqf.pkl")
# %%