# create target feature
df['total_delivery_duration'] = (df['actual_delivery_time'] - df['created_at'])

# make sure target feature is in seconds
df['total_delivery_duration'] = df['total_delivery_duration'].dt.total_seconds() # datetime

# create busy dashers ratio feature
df['busy_dashers_ratio'] = df['total_busy_dashers'] / df['total_onshift_dashers']

# create non-preparation duration feature
df['estimated_non_prep_duration'] = df['estimated_store_to_consumer_driving_duration'] + df['estimated_order_place_duration'] 

# check id features to see if encoding makes sense
df['market_id'].nunique()

df['store_id'].nunique()

df['order_protocol'].nunique()

# create dummies for order_protocol 
order_protocol_dummies = pd.get_dummies(df.order_protocol)
order_protocol_dummies = order_protocol_dummies.add_prefix('order_protocol_')
order_protocol_dummies.head()

# create dummies for market_id
market_id_dummies = pd.get_dummies(df.market_id)
market_id_dummies = market_id_dummies.add_prefix('market_id_')
market_id_dummies.head()

# create reference dictionary with the most repeated categories of each store
store_id_unique = df['store_id'].unique().tolist()
store_id_and_category = {store_id: df[df.store_id == store_id].store_primary_category.mode() 
                        for store_id in store_id_unique}

# create dummies for store_primary_category
store_primary_category_dummies = pd.get_dummies(df.nan_free_store_primary_category)
store_primary_category_dummies = store_primary_category_dummies.add_prefix('category_')
store_primary_category_dummies.head()

train_df = df.drop(columns = 
['created_at', 'market_id', 'store_primary_category', 'actual_delivery_time', 'nan_free_store_primary_category'])
train_df.head()

# concat all columns together
train_df = pd.concat([train_df, order_protocol_dummies, market_id_dummies, store_primary_category_dummies], axis=1)

# convert data type to float for future models
train_df = train_df.astype("float32")
train_df.head()

train_df['busy_dashers_ratio'].describe()

# infinity values using numpy
np.where(np.any(~np.isfinite(train_df), axis = 0) == True)

# replace infinite values with NaN for easy removal
train_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# drop all NaNs
train_df.dropna(inplace=True)