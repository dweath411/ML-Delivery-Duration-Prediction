# convert date time features
df['created_at'] = pd.to_datetime(df['created_at'])
df['actual_delivery_time'] = pd.to_datetime(df['actual_delivery_time'])

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

df['total_delivery_duration'].max() # drop the max outlier, 373879.0

df[df['total_delivery_duration'] == 373879.0]

df = df.drop([185550]) # drop delivery outlier

df['total_delivery_duration'].max() # checking the other max value now after dropping the first one

df[df['total_delivery_duration'] == 332482.0]

df = df.drop([27189]) # dropping the next value