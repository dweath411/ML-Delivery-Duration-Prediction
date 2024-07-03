# apply train_test_split

X = train_df[selected_features] # from multicollinearity checking
y = train_df["total_delivery_duration"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

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

# obtain column names as test

importances.sort_values(by = 'Gini-importance')[-20:].index.tolist()

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