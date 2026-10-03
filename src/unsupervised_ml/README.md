The dataset `data/kpop_groups.csv` is actually quite suitable for teaching a **broad introductory-to-intermediate unsupervised machine learning curriculum**, because it contains continuous variables, categorical variables, correlated features, latent archetypes, and deliberately injected anomalies.

One important distinction: the `true_archetype` and `is_anomaly` columns in the answer key should be treated as **evaluation-only ground truth**. Students should not use them while fitting unsupervised models.

## 1. The overall teaching roadmap

I would structure the unsupervised ML portion like this:

```text
Dataset understanding
       ↓
Data preprocessing
       ↓
Exploratory analysis
       ↓
Feature scaling
       ↓
Dimensionality reduction
       ↓
Clustering
       ├── K-Means
       ├── Hierarchical clustering
       ├── DBSCAN
       └── Gaussian Mixture Models
       ↓
Cluster evaluation
       ├── Silhouette score
       ├── Davies-Bouldin
       └── Calinski-Harabasz
       ↓
Cluster interpretation
       ↓
Anomaly detection
       ├── Isolation Forest
       ├── Local Outlier Factor
       └── DBSCAN-based outliers
       ↓
Latent structure
       ├── PCA
       └── relationship to hidden archetypes
       ↓
Advanced topics
       ├── mixed-type clustering
       ├── feature selection
       ├── stability
       └── unsupervised model comparison
```

---

# 2. Data preprocessing for unsupervised learning

This dataset gives you a good opportunity to teach that **preprocessing is part of the ML problem**, rather than something done automatically before ML.

### Topics

Students can learn:

* identifying numerical vs categorical variables
* missing-value checking
* duplicate detection
* outlier inspection
* feature distributions
* skewness
* logarithmic transformations
* standardization
* normalization
* encoding categorical variables
* deciding which columns should be excluded

For example:

```python
df.info()
df.describe()
df.isna().sum()
df.nunique()
```

Then discuss why these columns should generally **not** be treated as ordinary numerical features:

```text
group_name
firm_name
gender
debut_year
contract_renewed
```

while variables such as:

```text
streaming_log
album_sales_log
vocal_stability
choreo_complexity
fan_cafe_log
internal_conflict
```

can be candidate features.

---

# 3. Feature scaling

This dataset is particularly useful for explaining **why scaling matters**.

You have variables on very different scales:

```text
member_count          3–12
vocal_stability       0–1
choreo_complexity     0–100
award_nominations     0–30
social_growth_pct     roughly -10–40
sentiment_polarity    -1–1
```

Without scaling, a distance-based algorithm can give disproportionately large influence to variables with larger numerical ranges.

Students can compare:

```python
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)
```

against unscaled data.

This is an excellent experiment:

> **Run K-Means before and after scaling. How much do the clusters change?**

---

# 4. Correlation and redundancy

Before clustering, students can investigate whether variables are measuring similar underlying concepts.

For example:

```text
streaming_log
mv_views_log
album_sales_log
award_nominations
music_show_wins
brand_deals
```

may contain overlapping information about commercial success.

Similarly:

```text
fan_cafe_log
fan_labor_log
merch_index
intl_fan_ratio
viral_moments
```

may contain information related to fandom/digital activity.

Students can learn:

* correlation matrices
* Pearson correlation
* redundant features
* multicollinearity
* feature selection
* why too many correlated variables can affect clustering

A heatmap is a natural visualization here.

---

# 5. Principal Component Analysis — PCA

This is probably one of the **most valuable topics** for this dataset.

The original dataset has many dimensions, but the generator actually created the data from only five hidden latent dimensions:

```text
commercial strength
digital momentum
fandom strength
instability
prestige
```

Students don't see those five variables in the public dataset.

That creates a very nice teaching problem:

> Can we recover some of the hidden structure using PCA?

Students can learn:

* dimensionality reduction
* principal components
* explained variance
* eigenvectors/eigenvalues conceptually
* loading interpretation
* 2D visualization
* 3D visualization
* information loss

Example:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)
```

Then:

```python
pca.explained_variance_ratio_
```

becomes meaningful.

---

# 6. K-Means clustering

This dataset is excellent for introducing K-Means.

Students can ask:

> Can we automatically discover different types of K-pop groups without being told what the groups are?

Example:

```python
from sklearn.cluster import KMeans

model = KMeans(
    n_clusters=5,
    random_state=42,
    n_init="auto"
)

clusters = model.fit_predict(X_scaled)
```

Then visualize:

```text
PC1
 ↑
 │       ● ●
 │    ● ●
 │
 │                    ● ●
 │
 │  ● ●
 └────────────────────────→ PC2
```

Students can investigate what each cluster represents.

---

# 7. Choosing the number of clusters

This dataset allows several important unsupervised-learning concepts to be taught together.

### Elbow method

```python
inertias = []

for k in range(2, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init="auto"
    )

    model.fit(X_scaled)
    inertias.append(model.inertia_)
```

Students plot:

```text
inertia
  │\
  │ \
  │  \
  │   \__
  │      \___
  └──────────────
       k
```

and discuss the "elbow."

### Silhouette score

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(
    X_scaled,
    clusters
)
```

This teaches an important principle:

> In unsupervised learning, we often don't have a known correct answer, so we need indirect measures of cluster quality.

---

# 8. Hierarchical clustering

This dataset also works well for teaching hierarchical clustering.

Students can learn:

* agglomerative clustering
* linkage
* Euclidean distance
* dendrograms
* choosing a cluster cut

Example:

```python
from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(
    X_scaled,
    method="ward"
)

dendrogram(Z)
```

This gives students a visual representation of how groups merge.

It is particularly useful to compare:

```text
K-Means
vs
Hierarchical clustering
```

because they approach clustering differently.

---

# 9. DBSCAN

DBSCAN is especially useful because your generator deliberately contains unusual observations.

Students can learn:

* density-based clustering
* core points
* border points
* noise points
* `eps`
* `min_samples`

Example:

```python
from sklearn.cluster import DBSCAN

model = DBSCAN(
    eps=0.8,
    min_samples=5
)

labels = model.fit_predict(X_scaled)
```

The interesting part is:

```python
labels == -1
```

which represents points classified as noise.

This creates a natural bridge:

```text
Clustering
     ↓
Density-based clustering
     ↓
Noise detection
     ↓
Anomaly detection
```

---

# 10. Gaussian Mixture Models

For a slightly more advanced course, you can introduce GMM.

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(
    n_components=5,
    random_state=42
)

labels = gmm.fit_predict(X_scaled)
```

This lets you explain the difference between:

### K-Means

```text
Hard assignment

Group A
Group B
Group C
```

and GMM:

```text
Group A: 0.70
Group B: 0.25
Group C: 0.05
```

That is useful for explaining **soft clustering**.

---

# 11. Anomaly detection

This is one of the strongest parts of your dataset because you explicitly inject anomalies.

The public dataset contains unusual cases such as:

```text
legacy_viral_contradiction
factory_fandom_misallocation
rising_conflict_failure
niche_mainstream_breakout
```

Students don't know which rows were modified.

That's perfect for an anomaly-detection exercise.

---

## 12. Isolation Forest

Students can learn:

```python
from sklearn.ensemble import IsolationForest

model = IsolationForest(
    contamination=0.05,
    random_state=42
)

predictions = model.fit_predict(X_scaled)
```

Then:

```text
-1 → anomaly
 1 → normal
```

The answer key can be used **afterward** to evaluate the detector.

For example:

```text
Actual anomaly       Detected anomaly
       1                    1
       1                    0
       0                    1
       0                    0
```

That naturally leads to:

* precision
* recall
* confusion matrix
* false positives
* false negatives

Even though the **training process is unsupervised**, evaluation can use the hidden labels afterward.

---

# 13. Local Outlier Factor

LOF provides another perspective.

```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.05
)

predictions = lof.fit_predict(X_scaled)
```

Now students can compare:

```text
Isolation Forest
       vs
LOF
       vs
DBSCAN noise
```

and ask:

> Why do different anomaly-detection algorithms identify different observations?

That's a very useful ML lesson.

---

# 14. One-Class approaches

For an advanced section, you could introduce:

```text
One-Class SVM
```

and compare it with:

```text
Isolation Forest
LOF
DBSCAN
```

This gives students exposure to different philosophies of anomaly detection.

---

# 15. Cluster profiling

This is where the dataset becomes especially interesting.

Suppose K-Means discovers:

```text
Cluster 0
Cluster 1
Cluster 2
Cluster 3
Cluster 4
```

The next question is:

> **What does each cluster actually mean?**

Students can calculate cluster averages:

```python
df.groupby("cluster")[numeric_features].mean()
```

Then they might discover something like:

```text
Cluster A
high streaming
high album sales
high awards

Cluster B
high social growth
high MV views
high choreography

Cluster C
high fandom activity
high merchandise

...
```

This teaches that:

> Clustering itself doesn't automatically give a meaningful interpretation. Humans still need to interpret the resulting groups.

---

# 16. PCA + clustering together

This is probably the most natural complete practical project.

Pipeline:

```text
Raw dataset
     ↓
Select features
     ↓
StandardScaler
     ↓
PCA
     ↓
K-Means
     ↓
Plot clusters
     ↓
Profile clusters
```

Students can visualize:

```text
             Cluster 2
          ● ● ●
       ● ●
                 ● ●
                       Cluster 4
                    ● ● ●
          Cluster 1
       ● ●
```

This gives them a complete unsupervised-learning workflow rather than isolated algorithms.

---

# 17. Compare clustering with the hidden archetypes

This is one of the most educational possibilities in your particular dataset.

Remember:

```text
The model does NOT receive:

true_archetype
```

It only receives public features.

After clustering, however, the instructor can compare:

```text
K-Means cluster
        ↓
true_archetype
```

using the answer key.

For example:

```text
                 True Archetype
Cluster       1     2     3     4     5
-----------------------------------------
0             5     2     1     0     3
1             1     6     0     0     1
2             0     0     7     1     0
...
```

This teaches an extremely important concept:

> **Unsupervised clusters do not necessarily correspond one-to-one with the hidden categories used to generate the data.**

The clustering algorithm discovers structure based on the observable feature space; it does not know the generator's conceptual labels.

You can then introduce:

* contingency tables
* Adjusted Rand Index
* Normalized Mutual Information
* cluster purity

as **post-hoc evaluation**, rather than as training labels.

---

# 18. Feature importance in unsupervised learning

Traditional supervised ML has concepts like feature importance.

For unsupervised learning, students can instead investigate:

> Which variables most distinguish the clusters?

For example:

```python
cluster_means = df.groupby("cluster")[features].mean()
```

Then compare standardized cluster means.

This can lead to:

```text
feature
   ↓
cluster separation
```

rather than:

```text
feature
   ↓
target prediction
```

It's a useful conceptual difference.

---

# 19. Visualization

You can teach quite a lot of visualization through this dataset.

### Basic

* histograms
* boxplots
* scatterplots
* correlation heatmaps

### Intermediate

* PCA plots
* cluster-colored scatterplots
* dendrograms
* pair plots

### Advanced

* 3D PCA
* t-SNE
* UMAP

For example:

```text
Raw 25-dimensional data
          ↓
         PCA
          ↓
       2 dimensions
          ↓
       scatterplot
```

---

# 20. t-SNE

For an advanced visualization module:

```python
from sklearn.manifold import TSNE

X_tsne = TSNE(
    n_components=2,
    random_state=42
).fit_transform(X_scaled)
```

Students can compare:

```text
PCA
vs
t-SNE
```

and learn that they have different purposes.

A very important lesson here is that a t-SNE visualization **is not automatically evidence that real clusters exist**.

---

# 21. UMAP

If you want to move toward modern dimensionality reduction:

```text
PCA
t-SNE
UMAP
```

can be presented as three different approaches to representing high-dimensional structure.

UMAP is particularly useful if you want students to work with larger datasets later, although with only 100 rows it isn't essential.

---

# 22. Cluster stability

This is a more advanced and very valuable topic.

Students can ask:

> If I change the random seed, do I get the same clusters?

For example:

```python
KMeans(
    n_clusters=5,
    random_state=1
)

KMeans(
    n_clusters=5,
    random_state=2
)
```

Then compare the assignments.

They learn that an unsupervised result isn't necessarily trustworthy merely because an algorithm produced it.

---

# 23. Sensitivity analysis

Your dataset also allows students to investigate:

```text
What happens if I remove:
    streaming?

What happens if I remove:
    fandom variables?

What happens if I remove:
    company variables?

What happens if I don't scale?

What happens if I include gender?

What happens if I include debut year?
```

This teaches **feature engineering for unsupervised learning**.

---

# 24. Mixed-data clustering

You have an interesting complication:

```text
numerical
categorical
binary
```

variables coexist.

For example:

```text
Numerical:
streaming_log
album_sales_log
member_count

Categorical:
firm_name

Binary:
contract_renewed
gender
```

This provides an opportunity to teach why simply converting everything to numbers isn't necessarily statistically appropriate.

For an advanced course, students could explore:

* one-hot encoding
* Gower distance
* k-prototypes
* mixed-type clustering

---

# 25. A particularly good teaching sequence

If this is for a student course, I would **not teach every algorithm at once**.

I'd structure the dataset into progressively harder exercises.

### Level 1 — Understanding the data

```text
1. Dataset structure
2. Variable types
3. Distributions
4. Correlations
5. Missing values
6. Outliers
7. Log transformations
```

### Level 2 — Preparing for ML

```text
8. Feature selection
9. Encoding
10. Standardization
11. Distance metrics
```

### Level 3 — Dimensionality reduction

```text
12. PCA
13. Explained variance
14. PCA loadings
15. 2D visualization
```

### Level 4 — Clustering

```text
16. K-Means
17. Choosing K
18. Elbow method
19. Silhouette score
20. Cluster profiling
```

### Level 5 — Other clustering methods

```text
21. Hierarchical clustering
22. Dendrograms
23. DBSCAN
24. Gaussian Mixture Models
```

### Level 6 — Anomaly detection

```text
25. What is an anomaly?
26. Isolation Forest
27. LOF
28. DBSCAN noise
29. Compare anomaly detectors
```

### Level 7 — Evaluation

```text
30. Hidden ground truth
31. Adjusted Rand Index
32. NMI
33. Precision/recall for anomaly detection
34. False positives / false negatives
```

### Level 8 — Advanced unsupervised learning

```text
35. t-SNE
36. UMAP
37. Cluster stability
38. Feature sensitivity
39. Mixed-type clustering
40. End-to-end unsupervised project
```

---

# 26. What your dataset can teach particularly well

The nice thing about your generator is that there are **three different layers of truth**:

```text
                    DATA GENERATOR
                          │
             ┌────────────┴────────────┐
             ↓                         ↓
       Hidden structure          Observable data
             │                         │
     latent variables             CSV features
             │                         │
     true archetype             students see this
     anomaly label
             │
       ANSWER KEY
```

That enables a very good educational experiment:

### Students receive

```text
kpop_groups.csv
```

They don't know:

```text
true_archetype
is_anomaly
latent dimensions
```

They perform:

```text
PCA
   ↓
K-Means
   ↓
DBSCAN
   ↓
Isolation Forest
```

Then the instructor reveals:

```text
answer_key.csv
```

and students ask:

> How well did our unsupervised methods recover the hidden structure?

That's much more interesting pedagogically than simply giving students a dataset and asking them to run `KMeans()`.

---

## Recommended core syllabus

If your goal is an **introductory unsupervised ML course**, I would make these the core topics:

| Topic                   | Fit for this dataset |
| ----------------------- | -------------------- |
| EDA                     | Excellent            |
| Feature scaling         | Excellent            |
| Correlation/redundancy  | Excellent            |
| PCA                     | Excellent            |
| K-Means                 | Excellent            |
| Elbow method            | Excellent            |
| Silhouette score        | Excellent            |
| Hierarchical clustering | Excellent            |
| DBSCAN                  | Excellent            |
| Cluster profiling       | Excellent            |
| Isolation Forest        | **Excellent**        |
| LOF                     | Very good            |
| GMM                     | Very good            |
| t-SNE                   | Good                 |
| UMAP                    | Optional             |
| Cluster stability       | Very good            |
| Mixed-type clustering   | Advanced             |
| Gower distance          | Advanced             |

The **strongest learning project** would be:

> **“Discover the hidden structure of 100 fictional K-pop groups without using any labels, then compare the discovered clusters and anomalies with the hidden ground truth.”**

That single project can take students through **preprocessing → scaling → PCA → clustering → cluster evaluation → visualization → anomaly detection → post-hoc evaluation**, which gives you a coherent unsupervised ML curriculum rather than a collection of disconnected algorithms.
