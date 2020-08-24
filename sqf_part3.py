# %% read dataframe from part1
import pandas as pd

df = pd.read_pickle("sqf.pkl")


# %% Using prostitution as a category to view clusters to attain infoamation

df_prostitution = df[df["detailcm"] == "PROSTITUTION"]

# creating coordinates readable for mapping purposes by assigning boolean values
df_prostitution["lat"] = df["coord"].apply(lambda val: val[1])
df_prostitution["lon"] = df["coord"].apply(lambda val: val[0])


# %% Using Hierarchical clustering using silhouette scores to define the number of clusters

# try kmeans clustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from tqdm import tqdm

silhouette_scores, labels = {}, {}
num_city = df["city"].nunique()
num_pct = df["pct"].nunique()
step = 3

for k in tqdm(range(num_city, num_pct, step)):
    c = AgglomerativeClustering(n_clusters=k)
    y = c.fit_predict(df_prostitution[["lat", "lon"]])
    silhouette_scores[k] = silhouette_score(df_prostitution[["lat", "lon"]], y)
    labels[k] = y



# %% plot the silhouette scores agains different numbers of clusters
import seaborn as sns

ax = sns.lineplot(x=list(silhouette_scores.keys()), y=list(silhouette_scores.values()),)


# %% visualizing heirarchal clustering
import folium

nyc = (40.730610, -73.935242)
m = folium.Map(location=nyc)

best_k = max(silhouette_scores, key=lambda k: silhouette_scores[k])
df_prostitution["label"] = labels[best_k]

colors = sns.color_palette("husl", best_k).as_hex()

for row in tqdm(df_prostitution[["lat", "lon", "label"]].to_dict("records")):
    folium.CircleMarker(
        location=(row["lat"], row["lon"]), radius=3, color=colors[row["label"]]
    ).add_to(m)

m


# %% Using KMeans clustering using silhouette scores to define the number of clusters
from sklearn.cluster import KMeans

for k in tqdm(range(num_city, num_pct, step)):
    c = KMeans(n_clusters=k)
    y = c.fit_predict(df_prostitution[["lat", "lon"]])
    silhouette_scores[k] = silhouette_score(df_prostitution[["lat", "lon"]], y)
    labels[k] = y

ax = sns.lineplot(x=list(silhouette_scores.keys()), y=list(silhouette_scores.values()),)

# %% visualizint KMeans clustering
nyc = (40.730610, -73.935242)
m = folium.Map(location=nyc)

best_k = max(silhouette_scores, key=lambda k: silhouette_scores[k])
df_prostitution["label"] = labels[best_k]

colors = sns.color_palette("husl", best_k).as_hex()

for row in tqdm(df_prostitution[["lat", "lon", "label"]].to_dict("records")):
    folium.CircleMarker(
        location=(row["lat"], row["lon"]), radius=3, color=colors[row["label"]]
    ).add_to(m)

m

# %% Creating a dataframe for columns that start with cs_ --> meaning reason the person was stopped
css = [col for col in df.columns if col.startswith("cs_")]


# %% Running DBSCAN and getting its silhouette score --> DBSCAN automatically choses the number of clusters
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from tqdm import tqdm

c = DBSCAN()
# changing it to a boolean df
x = df_prostitution[css] == "YES"
y = c.fit_predict(x)
print(silhouette_score(x, y))


# %% Visualizing DBSCAN clustering
import numpy as np

nyc = (40.730610, -73.935242)
m = folium.Map(location=nyc)
df_prostitution["label"] = y
colors = sns.color_palette("husl", len(np.unique(y))).as_hex()
for row in tqdm(df_prostitution[["lat", "lon", "label"]].to_dict("records")):
    folium.CircleMarker(
        location=(row["lat"], row["lon"]),
        radius=0.1,
        color=colors[row["label"]],
        alpha=0.3,
    ).add_to(m)
m


# %% Picking labels to see how DBSCAn decides its clusters
# changing the label means changing y[0]
# find how the y[0] is the amount of clusters made by DBSACN based on the reason for stop
# Use df_prostitution["label"].unique() to find how many clusters were made
import numpy as np

nyc = (40.730610, -73.935242)
m = folium.Map(location=nyc)
df_prostitution["label"] = y
colors = sns.color_palette("husl", len(np.unique(y))).as_hex()
for row in tqdm(
    df_prostitution.loc[df_prostitution["label"] == y[0], ["lat", "lon", "label"]].to_dict(
        "records"
    )
):
    folium.CircleMarker(
        location=(row["lat"], row["lon"]),
        radius=0.1,
        color=colors[row["label"]],
        alpha=0.3,
    ).add_to(m)
m


# %% Picking another cluster group decided by DBSCAN to understand clustering better
import numpy as np

nyc = (40.730610, -73.935242)
m = folium.Map(location=nyc)
df_prostitution["label"] = y
colors = sns.color_palette("husl", len(np.unique(y))).as_hex()
for row in tqdm(
    df_prostitution.loc[df_prostitution["label"] == y[16], ["lat", "lon", "label"]].to_dict(
        "records"
    )
):
    folium.CircleMarker(
        location=(row["lat"], row["lon"]),
        radius=0.1,
        color=colors[row["label"]],
        alpha=0.3,
    ).add_to(m)
m


# %% picking labels to find any insight into DBSCAN clustering
import numpy as np

nyc = (40.730610, -73.935242)
m = folium.Map(location=nyc)
df_prostitution["label"] = y
colors = sns.color_palette("husl", len(np.unique(y))).as_hex()
for row in tqdm(
    df_prostitution.loc[df_prostitution["label"] == y[58], ["lat", "lon", "label"]].to_dict(
        "records"
    )
):
    folium.CircleMarker(
        location=(row["lat"], row["lon"]),
        radius=0.1,
        color=colors[row["label"]],
        alpha=0.3,
    ).add_to(m)
m


# %% Creating a new dataframe for containing reasons for frisking from original ds
rf = [col for col in df.columns if col.startswith("rf_")]

c = DBSCAN()
# changing it to a boolean df
x = df_prostitution[rf] == "YES"
y = c.fit_predict(x)
print(silhouette_score(x, y))

# %% visualizing clusters created for reasons being frisked by location
nyc = (40.730610, -73.935242)
m = folium.Map(location=nyc)
df_prostitution["label"] = y
colors = sns.color_palette("husl", len(np.unique(y))).as_hex()
for row in tqdm(df_prostitution[["lat", "lon", "label"]].to_dict("records")):
    folium.CircleMarker(
        location=(row["lat"], row["lon"]),
        radius=0.1,
        color=colors[row["label"]],
        alpha=0.3,
    ).add_to(m)
m

# %%
