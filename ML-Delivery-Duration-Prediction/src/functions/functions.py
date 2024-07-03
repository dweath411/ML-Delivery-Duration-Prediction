# function to impute values
def impute(store_id):
    """Return primary store category from the dictionary"""
    try:
        return store_id_and_category[store_id].values[0]
    except:
        return np.nan

# function to get redunant values
def get_redundant_values(df):
    """Get diagonal and lower triangular values of correlation matrix"""
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

# function to get top correlated values
def get_top_correlation(df, n = 5):
    """Sort correlations in the descending order and return n highest results"""
    au_corr = df.corr().abs().unstack() # important to take the absolute value
    labels_to_drop = get_redundant_values(df)
    au_corr = au_corr.drop(labels = labels_to_drop).sort_values(ascending = False)
    return au_corr[0:n]

# function to compute variable inflation factor
def compute_vif(features):
    """Compute VIF score"""
    vif_data = pd.DataFrame()
    vif_data["feature"] = features # column of feature names
    vif_data["VIF"] = [variance_inflation_factor(train_df[features].values, i) for i in range(len(features))] # column of VIF score
    return vif_data.sort_values(by=["VIF"]).reset_index(drop = True)

# scale function

def scale(scaler, X, y):
    """Apply the selected scaler to features and target variable"""
    X_scaler = scaler
    X_scaler.fit(X = X, y = y)
    X_scaled = X_scaler.transform(X)
    y_scaler = scaler
    y_scaler.fit(y.values.reshape(-1, 1))
    y_scaled = y_scaler.transform(y.values.reshape(-1,1))

    return X_scaled, y_scaled, X_scaler, y_scaler

# when we rename the scaler as min/max, this function will apply the min/max scaler to the features.

# inverse transform function

def rmse_with_inv_transform(scaler, y_test, y_pred_scaled, model_name):
    """Convert the scaled error to actual error"""
    y_predict = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
    # return RMSE with squared False
    rmse_error = mean_squared_error(y_test, y_predict[:,0], squared = False)
    print("Error = "'{}'.format(rmse_error)+" in " + model_name)

    return rmse_error, y_predict

# create a function that can work across multiple machine learning models
def predicting(X_train, y_train, X_test, y_test, model, model_name, verbose = True):
    """Apply selected regression model to data and measure error"""
    model.fit(X_train, y_train)
    y_predict = model.predict(X_train)
    train_error = mean_squared_error(y_train, y_predict, squared = False)
    y_predict = model.predict(X_test)
    test_error = mean_squared_error(y_test, y_predict, squared = False)
    if verbose:
        print("Train error = "'{}'.format(train_error)+" in " + model_name)
        print("Test error = "'{}'.format(test_error)+" in " + model_name)
    trained_model = model

    return trained_model, y_predict, train_error, test_error

def create_model(feature_set_size):

    # define model
    model = Sequential()
    model.add(Dense(16, input_dim = feature_set_size, activation = 'relu'))
    model.add(Dense(1, activation = 'linear'))

    # complile model
    model.compile(optimizer = 'sgd', loss = 'mse', # stochastic gradient descent, mean squared error
        metrics = [tf.keras.metrics.RootMeanSquaredError()])

    return model


