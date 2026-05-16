import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/airbnb.csv")

# -----------------------------------
# REMOVE UNNECESSARY COLUMNS
# -----------------------------------

df.drop(columns=['Unnamed: 0', 'Unnamed: 23'], inplace=True)

# -----------------------------------
# CONVERT DATA TYPES
# -----------------------------------

# Convert rating to numeric
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# Convert reviews to numeric
df['reviews'] = pd.to_numeric(df['reviews'], errors='coerce')

# -----------------------------------
# HANDLE MISSING VALUES
# -----------------------------------

# Fill missing host names
df['host_name'] = df['host_name'].fillna('Unknown Host')

# Fill missing checkin/checkout
df['checkin'] = df['checkin'].fillna('Not Available')

df['checkout'] = df['checkout'].fillna('Not Available')

# -----------------------------------
# REMOVE DUPLICATES
# -----------------------------------

df.drop_duplicates(inplace=True)

# -----------------------------------
# CHECK CLEANED DATA
# -----------------------------------

print("\n CLEANED DATA INFO")
print(df.info())

print("\n MISSING VALUES AFTER CLEANING")
print(df.isnull().sum())

print("\n FIRST 5 ROWS AFTER CLEANING")
print(df.head())

df.drop_duplicates(inplace=True)

# -----------------------------------
# FEATURE ENGINEERING
# -----------------------------------

# Detect Wifi
df['has_wifi'] = df['amenities'].str.contains('Wifi', case=False, na=False)

# Detect Pool
df['has_pool'] = df['amenities'].str.contains('Pool', case=False, na=False)

# Detect Air Conditioning
df['has_ac'] = df['amenities'].str.contains('Air conditioning', case=False, na=False)

# Detect Kitchen
df['has_kitchen'] = df['amenities'].str.contains('Kitchen', case=False, na=False)

# Luxury property based on price
df['luxury_property'] = df['price'] > 15000

# Guests per bedroom ratio
df['guests_per_bedroom'] = df['guests'] / (df['bedrooms'] + 1)

# Price category
df['price_category'] = pd.cut(
    df['price'],
    bins=[0, 5000, 15000, 50000, 1000000],
    labels=['Budget', 'Standard', 'Premium', 'Luxury']
)


print("\n NEW FEATURE COLUMNS")
print(df[['has_wifi', 'has_pool', 'has_ac', 'price_category']].head())


# -----------------------------------
# EXPLORATORY DATA ANALYSIS (EDA)
# -----------------------------------

# Average price
print("\n AVERAGE PRICE")
print(df['price'].mean())

# Average rating
print("\n AVERAGE RATING")
print(df['rating'].mean())

# Top 10 countries by listings
print("\n TOP 10 COUNTRIES")
print(df['country'].value_counts().head(10))

# Average price by country
print("\n AVERAGE PRICE BY COUNTRY")
print(df.groupby('country')['price'].mean().sort_values(ascending=False).head(10))

# Price category distribution
print("\n PRICE CATEGORY DISTRIBUTION")
print(df['price_category'].value_counts())

# Wifi availability
print("\n WIFI AVAILABILITY")
print(df['has_wifi'].value_counts())

# Pool availability
print("\n POOL AVAILABILITY")
print(df['has_pool'].value_counts())

# Bedroom distribution
print("\n BEDROOM DISTRIBUTION")
print(df['bedrooms'].value_counts().head(10))


# -----------------------------------
# VISUALIZATION SECTION
# -----------------------------------

# ===== 1. TOP COUNTRIES CHART =====

top_countries = df['country'].value_counts().head(10)

plt.figure(figsize=(10,6))

top_countries.plot(kind='bar')

plt.title('Top 10 Countries by Airbnb Listings')

plt.xlabel('Country')

plt.ylabel('Number of Listings')

plt.xticks(rotation=45)

plt.savefig("images/top_countries_chart.png")

plt.close()


# ===== 2. PRICE DISTRIBUTION =====

filtered_prices = df[df['price'] < 50000]

plt.figure(figsize=(10,6))

plt.hist(filtered_prices['price'], bins=30)

plt.title('Airbnb Price Distribution (Below 50,000)')

plt.xlabel('Price')

plt.ylabel('Number of Listings')

plt.savefig("images/price_distribution.png")

plt.close()


# ===== 3. BEDROOM DISTRIBUTION =====

bedroom_counts = df['bedrooms'].value_counts().head(10)

plt.figure(figsize=(10,6))

bedroom_counts.plot(kind='bar')

plt.title('Bedroom Distribution')

plt.xlabel('Number of Bedrooms')

plt.ylabel('Number of Listings')

plt.savefig("images/bedroom_distribution.png")

plt.close()


# ===== 4. PRICE CATEGORY PIE CHART =====

price_category_counts = df['price_category'].value_counts()

plt.figure(figsize=(8,8))

plt.pie(
    price_category_counts,
    labels=price_category_counts.index,
    autopct='%1.1f%%'
)

plt.title('Price Category Distribution')

plt.savefig("images/price_category_pie.png")

plt.close()


# ===== 5. WIFI AVAILABILITY =====

wifi_counts = df['has_wifi'].value_counts()

plt.figure(figsize=(6,6))

plt.pie(
    wifi_counts,
    labels=['Wifi Available', 'No Wifi'],
    autopct='%1.1f%%'
)

plt.title('Wifi Availability')

plt.savefig("images/wifi_availability.png")

plt.close()


# ===== 6. POOL AVAILABILITY =====

pool_counts = df['has_pool'].value_counts()

plt.figure(figsize=(6,6))

plt.pie(
    pool_counts,
    labels=['No Pool', 'Pool Available'],
    autopct='%1.1f%%'
)

plt.title('Pool Availability')

plt.savefig("images/pool_availability.png")

plt.close()


# ===== 7. AVERAGE PRICE BY TOP COUNTRIES =====

avg_price_country = (
    df.groupby('country')['price']
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12,6))

avg_price_country.plot(kind='bar')

plt.title('Average Price by Country')

plt.xlabel('Country')

plt.ylabel('Average Price')

plt.xticks(rotation=45)

plt.savefig("images/avg_price_country.png")

plt.close()


# ===== FINAL MESSAGE =====

print("\n ALL CHARTS GENERATED SUCCESSFULLY!")
print("Charts saved inside IMAGES folder.")


# -----------------------------------
# EXPORT CLEANED DATASET
# -----------------------------------

df.to_csv("data/cleaned_airbnb.csv", index=False)

print("\n CLEANED DATASET EXPORTED SUCCESSFULLY!")