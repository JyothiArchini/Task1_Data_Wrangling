import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load dataset
df = pd.read_csv("../dataset/sales_data_sample.csv", encoding='latin1')

# Show first 5 rows
print(df.head())

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumn Names:")
print(df.columns)

# Dataset information
print("\nDataset Information:")
print(df.info())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Unique values
print("\nUnique Values Per Column:")
print(df.nunique())
# =========================
# HANDLE MISSING VALUES
# =========================

# Fill missing STATE values
df['STATE'] = df['STATE'].fillna('Unknown')

# Fill missing TERRITORY values
df['TERRITORY'] = df['TERRITORY'].fillna('Not Specified')

# Fill missing POSTALCODE values
df['POSTALCODE'] = df['POSTALCODE'].fillna('00000')

# Fill missing ADDRESSLINE2 values
df['ADDRESSLINE2'] = df['ADDRESSLINE2'].fillna('Not Available')

print("\nMISSING VALUES AFTER CLEANING")
print(df.isnull().sum())
# =========================
# REMOVE DUPLICATES
# =========================

df = df.drop_duplicates()

print("\nDATASET SHAPE AFTER REMOVING DUPLICATES")
print(df.shape)
# =========================
# DATE CONVERSION
# =========================

df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])

print("\nORDERDATE DATA TYPE")
print(df['ORDERDATE'].dtype)
# =========================
# FEATURE ENGINEERING
# =========================

df['ORDER_MONTH'] = df['ORDERDATE'].dt.month

df['ORDER_YEAR'] = df['ORDERDATE'].dt.year

print("\nNEW FEATURES ADDED")
print(df[['ORDERDATE', 'ORDER_MONTH', 'ORDER_YEAR']].head())
# =========================
# SAVE CLEANED DATASET
# =========================

df.to_csv("../cleaned_data/cleaned_sales_data.csv", index=False)

print("\nCLEANED DATASET SAVED SUCCESSFULLY")

# =========================
# VISUALIZATION 1
# SALES DISTRIBUTION
# =========================

plt.figure(figsize=(8,5))

sns.histplot(df['SALES'], bins=30)

plt.title("Sales Distribution")

plt.xlabel("Sales")

plt.ylabel("Frequency")

plt.show()

# =========================
# VISUALIZATION 2
# COUNTRY DISTRIBUTION
# =========================

plt.figure(figsize=(10,6))

df['COUNTRY'].value_counts().head(10).plot(kind='bar')

plt.title("Top 10 Countries by Orders")

plt.xlabel("Country")

plt.ylabel("Number of Orders")

plt.show()

# =========================
# VISUALIZATION 3
# DEAL SIZE
# =========================

plt.figure(figsize=(6,4))

sns.countplot(x='DEALSIZE', data=df)

plt.title("Deal Size Distribution")

plt.show()


