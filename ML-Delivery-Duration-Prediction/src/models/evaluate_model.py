# define 4 different dictionaries containing different parameters

pred_dict = {
    "regression_model": [],
    "feature_set": [],
    "scaler_name": [],
    "RMSE": [],
} # predictions dictionary

regression_models = {
    "Ridge": linear_model.Ridge(),
    "DecisionTree": tree.DecisionTreeRegressor(max_depth = 6),
    "RandomForest": RandomForestRegressor(),
    "XGBoost": XGBRegressor(),
    "LGBM": LGBMRegressor(),
    "MLP": MLPRegressor(),
} # fun models to use!

feature_sets = {
    "full dataset": X.columns.to_list(),
    "selected_features_40": importances.sort_values(by = 'Gini-importance')[-40:].index.tolist(),
    "selected_features_20": importances.sort_values(by = 'Gini-importance')[-20:].index.tolist(), 
    "selected_features_10": importances.sort_values(by = 'Gini-importance')[-10:].index.tolist(),
} # feature set choices

scalers = {
    "Standard Scaler": StandardScaler(),
    "MinMax scaler": MinMaxScaler(),
    "NotScale": None,
} # last but not least, scalers

# for loops to examine the error for each combination
for feature_set_name in feature_sets.keys():
    feature_set = feature_sets[feature_set_name]
    for scaler_name in scalers.keys():
        print(f"-----scaled with {scaler_name}------- included columns are {feature_set_name}")
        print("")
        for model_name in regression_models.keys():
            if scaler_name == "NotScale":
                X = train_df[feature_set]
                y = train_df["total_delivery_duration"]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
                predicting(X_train, y_train, X_test, y_test, regression_models[model_name], model_name, verbose = True)
            else:
                X_scaled,y_scaled,X_scaler,y_scaler = scale(scalers[scaler_name], X[feature_set],y)
                X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
                X_scaled, y_scaled, test_size = 0.2,random_state = 42
                )
                _, y_predict_scaled, _, _ = predicting(X_train_scaled, y_train_scaled[:,0], X_test_scaled, y_test_scaled, regression_models[model_name], model_name, verbose=True)
                rmse_error, y_predict = rmse_with_inv_transform(y_scaler, y_test, y_predict_scaled, model_name)
                pred_dict["regression_model"].append(model_name)
                pred_dict["feature_set"].append(feature_set_name)
                pred_dict["scaler_name"].append(scaler_name)
                pred_dict["RMSE"].append(rmse_error)  

# save prediction dictionary

pred_df = pd.DataFrame(pred_dict)
pred_df

# drop columns here if not already dropped

# train_df[feature_set].drop(columns=['estimated_store_to_consumer_driving_duration','estimated_order_place_duration'])

# check if columns are dropped
train_df[feature_set].columns

scalers = {
    "Standard Scaler": StandardScaler(),
}

feature_sets = {
    "selected_features_40": importances.sort_values(by = 'Gini-importance')[-40:].index.tolist(),
}

# use LGBM, the best performing model
regression_models = {
    "LGBM": LGBMRegressor(),
}

# for loop from earlier (modified)
for feature_set_name in feature_sets.keys():
    feature_set = feature_sets[feature_set_name]
    for scaler_name in scalers.keys():
        print(f"-----scaled with {scaler_name}------- included columns are {feature_set_name}")
        print("")
        for model_name in regression_models.keys():
            # modified to drop estimated_store_to_consumer_driving_duration and estimated_order_place_duration
                X = train_df[feature_set]
            
#.drop(columns=['estimated_store_to_consumer_driving_duration','estimated_order_place_duration']) --> option to drop columns here as well
                y = train_df["prep_time"]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
                # get indices
                train_indices = X_train.index
                test_indices = X_test.index

                # scale
                X_scaled, y_scaled, X_scaler, y_scaler = scale(scalers[scaler_name], X ,y)

                # apply indexing
                X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
                X_scaled, y_scaled, test_size = 0.2,random_state = 42)
            
                _, y_predict_scaled, _, _ = predicting(X_train_scaled, y_train_scaled[:,0], X_test_scaled, y_test_scaled, regression_models[model_name], model_name, verbose=True)
                rmse_error, y_predict = rmse_with_inv_transform(y_scaler, y_test, y_predict_scaled, model_name)
                pred_dict["regression_model"].append(model_name)
                pred_dict["feature_set"].append(feature_set_name)
                pred_dict["scaler_name"].append(scaler_name)
                pred_dict["RMSE"].append(rmse_error) 


# Now that we have our best model, let's extract `prep_duration` predictions.

pred_values_dict = {
    "total_delivery_duration": train_df["total_delivery_duration"][test_indices].values.tolist(),
    "prep_duration_prediction": y_predict[:,0].tolist(),
    "estimated_store_to_consumer_driving_duration": train_df["estimated_store_to_consumer_driving_duration"][test_indices].values.tolist(),
    "estimated_order_place_duration": train_df["estimated_order_place_duration"][test_indices].values.tolist(),
}   

# convert dictionary to dataframe

values_df = pd.DataFrame.from_dict(pred_values_dict) 
values_df 

# sum predictions

values_df["sum_total_delivery_duration"] = values_df["prep_duration_prediction"] + values_df["estimated_store_to_consumer_driving_duration"] + values_df["estimated_order_place_duration"]
values_df

# checking new error rate

mean_squared_error(values_df["total_delivery_duration"], values_df["sum_total_delivery_duration"], squared = False)

# Use another regression model to obtain the actual total delivery duration
X = values_df[["prep_duration_prediction", "estimated_store_to_consumer_driving_duration", "estimated_order_place_duration"]]
y = values_df["total_delivery_duration"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# define regression models dictionary, adding 6 different regression model names into it

# use a for loop and custom predicting function to select model from our regression model dictionary

regression_models = {
    "Ridge": linear_model.Ridge(),
    "DecisionTree": tree.DecisionTreeRegressor(max_depth = 6),
    "RandomForest": RandomForestRegressor(),
    "XGBoost": XGBRegressor(),
    "LGBM": LGBMRegressor(),
    "MLP": MLPRegressor(),
} # fun models to use!

for model_name in regression_models.keys():
    _, y_predict, _, _ = predicting(
        X_train, y_train, X_test, y_test, regression_models[model_name], model_name, verbose = False)
    print("RMSE of:", model_name, mean_squared_error(y_test, y_predict, squared = False))


# view predictions
values_df['prep_duration_prediction']

# use keras and tf to create a neural network

def create_model(feature_set_size):

    # define model
    model = Sequential()
    model.add(Dense(16, input_dim = feature_set_size, activation = 'relu'))
    model.add(Dense(1, activation = 'linear'))

    # complile model
    model.compile(optimizer = 'sgd', loss = 'mse', # stochastic gradient descent, mean squared error
        metrics = [tf.keras.metrics.RootMeanSquaredError()])

    return model

# call function
print(f"-----scaled with {scaler_name}------- included columns are {feature_set_name}")
print("")
# get model name, scaler name, and features ready

model_name = "ANN" 
scaler_name = "Standard Scaler"
X = values_df[["prep_duration_prediction", "estimated_store_to_consumer_driving_duration", "estimated_order_place_duration"]]
y = values_df["total_delivery_duration"]
# split data and apply scaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_scaled, y_scaled, X_scaler, y_scaler = scale(scalers[scaler_name], X ,y)
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
                X_scaled, y_scaled, test_size = 0.2,random_state = 42)
# apply custom function
print("feature_set_size:", X_train_scaled.shape[1])
model = create_model(feature_set_size = X_train_scaled.shape[1])
history = model.fit(X_train_scaled, y_train_scaled, epochs = 100, batch_size = 64, verbose = 1)
y_pred = model.predict(X_test_scaled)
rmse_error = rmse_with_inv_transform(y_scaler, y_test, y_pred, model_name) # apply rmse with inverse transform since we used a scaler
pred_dict["regression_model"].append(model_name)
pred_dict["feature_set"].append(feature_set_name)
pred_dict["scaler_name"].append(scaler_name)
pred_dict["RMSE"].append(rmse_error)  

plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()


