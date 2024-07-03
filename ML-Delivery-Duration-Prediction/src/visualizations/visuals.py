# let's show missing values with white color
plt.figure(figsize=(20, 10)) 
sns.heatmap(df.isnull(), cbar=False)

# numeric feature distribution
numerics = ['total_items',  'subtotal', 'num_distinct_items', 'min_item_price','max_item_price','total_onshift_dashers','total_busy_dashers','total_outstanding_orders']
fig, axes = plt.subplots(3, 3, figsize=(20, 20))
for i,n in enumerate(numerics):
    axes[i//3][(i%3)-1].hist(df[n],bins=50)
    axes[i//3][(i%3)-1].set_title(n)

# checking the scale of outliers

for i in (numerics):
    boxplt = df.boxplot(column=[i], figsize = (4, 4))
    plt.show()

boxplt = df.boxplot(column=['total_delivery_duration'])
boxplt

# masked correlation matrix
corr = train_df.corr()

# generate the mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype = bool))

# set up figure
f, ax = plt.subplots(figsize = (15,13))

# set up diverging palette
cmap = sns.diverging_palette(230, 20, as_cmap = True)

# form heatmap
sns.heatmap(corr, mask = mask, cmap = cmap, vmax= .3, center = 0,
            square = True, linewidths = .5, cbar_kws = {"shrink": .5})

# apply Random Forest

feature_names = [f"feature {i}" for i in range((X.shape[1]))]
forest = RandomForestRegressor(random_state=42)
forest.fit(X_train, y_train)

feats = {} # dict to hold feature importance
for feature, importance in zip(X.columns, forest.feature_importances_):
    feats[feature] = importance # add name/value pair
importances = pd.DataFrame.from_dict(feats, orient = 'index').rename(columns = {0: 'Gini-importance'})
importances.sort_values(by = 'Gini-importance').plot(kind = 'bar', rot = 90, figsize = (15,12))
plt.show()


# Principal Component Analysis

X_Train = X_train
X_train = np.asarray(X_Train)

# finding normalized array of X_Train
X_std = StandardScaler().fit_transform(X_Train)
pca = PCA().fit(X_std)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0, 81, 1)
plt.xlabel('# of Components')
plt.ylabel('Cumulative explained variance')
plt.show()

pred_df.plot(kind = "bar", figsize = (12, 8))

# epochs

plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()