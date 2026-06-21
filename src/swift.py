import pandas as pd
df = pd.read_csv("data/raw/Train.csv")
print(df.head())
print("Rows and Columns:", df.shape)
print(df.dtypes)
print(df.isnull().sum())
print("Duplicates:",df.duplicated().sum())
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x='Reached.on.Time_Y.N',data=df)
plt.title("Delivery Status Distribution")
plt.xlabel("0 = On Time ,1 = Late")
plt.ylabel("Count")
plt.show()
print(df['Reached.on.Time_Y.N'].value_counts())

#Univariate Analysis NumericFeatures
numeric_cols = [
    'Customer_care_calls',
    'Customer_rating',
    'Cost_of_the_Product',
    'Prior_purchases',
    'Discount_offered',
    'Weight_in_gms'
]
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()
    
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of{col}')
    plt.show()
    
    #Categorical Features
    cat_cols = [
    'Warehouse_block',
    'Mode_of_Shipment',
    'Product_importance',
    'Gender'
]

for col in cat_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(x=col, data=df)
    plt.title(f'Distribution of {col}')
    plt.show()
    sns.boxplot(
    x='Reached.on.Time_Y.N',
    y='Discount_offered',
    data=df
)
plt.show()

#Bivariate Analysis
#Discount Offered vs Delivery Status
sns.boxplot(
    x='Reached.on.Time_Y.N',
    y='Discount_offered',
    data=df
)
plt.show()

#Weight vs Delivery Status
sns.boxplot(
    x='Reached.on.Time_Y.N',
    y='Weight_in_gms',
    data=df
)
plt.show()

#Shipment Mode vs Delivery Status
pd.crosstab(
    df['Mode_of_Shipment'],
    df['Reached.on.Time_Y.N']
).plot(kind='bar')
plt.show()

#Product Importance vs Delivery Status
pd.crosstab(
    df['Product_importance'],
    df['Reached.on.Time_Y.N']
).plot(kind='bar')
plt.show()

#Correlation Heatmap
plt.figure(figsize=(10,8))

sns.heatmap(
    df[numeric_cols].corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")
plt.show()