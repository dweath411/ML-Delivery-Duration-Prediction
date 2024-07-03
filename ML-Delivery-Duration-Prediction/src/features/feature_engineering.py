# feature engineered features
train_df["percent_distinct_item_of_total"] = train_df["num_distinct_items"] / train_df["total_items"]
train_df["avg_price_per_item"] = train_df["subtotal"] / train_df["total_items"]
train_df.drop(columns = ["num_distinct_items", "subtotal"], inplace = True)
print("Top Absolute Correlations")
print(get_top_correlation(train_df, 20))

# new feature: price range
train_df["price_range_of_items"] = train_df["max_item_price"] - train_df["min_item_price"]
train_df.drop(columns = ["max_item_price", "min_item_price"], inplace = True)
print("Top Absolute Correlations")
print(get_top_correlation(train_df, 20))

# always room for more feature engineering!

train_df['prep_time'] = train_df["total_delivery_duration"] - train_df["estimated_store_to_consumer_driving_duration"] - train_df["estimated_order_place_duration"]