'''
       project: COVID-19 Data Analysis

1.Load and inspect the dataset
2.Clean the data and convert dates
3.Feature engineering
  - Daily growth rate
  - 7-day moving average
4.Exploratory analysis
 - Top 10 countries by cases/deaths
 - Case fatality rate
 - Continent comparison
5.Create 5–6+ visualizations
 - Total cases over time
 - 7-day smoothed new cases
 - Top 10 countries
 - Correlation heatmap
 - Box/violin plot by continent
'''

import pandas as pd
import numpy as np

# stage 1: Load and inspect the dataset
df = pd.read_csv(r"C:\Users\sivasankar\Downloads\Datasets\covid19_student_dataset.csv")
# to get the information about the dataset
print(df.head())
print(df.info())
print(df.describe())
print(df.columns)
print(df.dtypes)
rows, cols = df.shape
print(f'The dataset has {rows} rows and {cols} columns.')

# check the missing values in the dataset
missing_values = df.isnull().sum()
print(f'Missing values in the dataset:\n{missing_values}')

# to get the unique values in the dataset
print(f'no of countries in dataset: {df["country"].nunique()}')
print(f'list of countries in dataset: {df["country"].unique()}')

print(f'no of continents in dataset: {df["continent"].nunique()}')
print(f'list of continents in dataset: {df["continent"].unique()}')

# record count
country_counts = df['country'].value_counts()
print(f'Country-wise record count:\n{country_counts}')

print(f'Continent-wise record count:\n{df["continent"].value_counts()}')

# check the date range in the dataset
min_date = df['date'].min()
max_date = df['date'].max()
print(f'Date range in the dataset: {min_date} to {max_date}')

# check the duplicate records in the dataset
duplicate_records = df.duplicated().sum()
print(f'Number of duplicate records in the dataset: {duplicate_records}')

# ------------------------------------------------------------------------------------

# convert the date column to datetime format
df['date'] = pd.to_datetime(df['date'], errors='coerce')
print(df['date'].dtype)

# check the missing values
missing_values = df.isnull().sum()
print(f'missing value : {missing_values}')

print(missing_values[missing_values>1000])

missing_percentage = (df.isnull().sum()/len(df))*100
print(missing_percentage[missing_percentage > 0].sort_values(ascending=False))

# check the duplicate records in the dataset
duplicate_records = df.duplicated().sum()
print(f'Number of duplicate records in the dataset: {duplicate_records}')
df = df.drop_duplicates()

# sort the dataset by country and date
df = df.sort_values(by=['country', 'date']).reset_index(drop=True)
print(df.head())

# fill misssing time series values
time_series_columns = [
  "total_cases",
  "new_cases",
  "new_cases_smoothed",
  "total_cases_per_million",
  "total_deaths",
  "new_deaths",
  "new_deaths_smoothed",
  "total_deaths_per_million",
]

df[time_series_columns] = (
    df.groupby('country')[time_series_columns].ffill()
)

missing_after = df.isnull().sum()
print(f'Missing values after filling:\n{missing_after[missing_after>0]}')

print(
    df[df["total_cases"].isna()][
        ["country", "date", "total_cases", "new_cases", "total_deaths"]
    ].head(20)
)

print(
    df.groupby("country")[[
        "total_cases",
        "new_cases",
        "total_deaths"
    ]].apply(lambda x: x.isna().sum())
)

print("\nMissing values after cleaning:")

missing_after = df.isnull().sum()

print(missing_after[missing_after > 0])

static_columns = [
    "population",
    "median_age",
    "life_expectancy",
    "gdp_per_capita",
    "hospital_beds_per_thousand"
]

for col in static_columns:
    df[col] = df.groupby('country')[col].transform(lambda x: x.fillna(x.median()))

print("\nFinal shape:", df.shape)

print("\nData types:")
print(df.dtypes)

print("\nRemaining missing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\nDuplicate rows:")
print(df.duplicated().sum())

# ----------------------------------------------------------------------------
# feature engineering
# Daily Case Growth Rate

df['previous_day_cases'] = df.groupby('country')['total_cases'].shift(1)

df['daily_growth_rate'] = (df['new_cases'] / df['previous_day_cases']) * 100

print(df[['country', 'date', 'total_cases', 'new_cases',
          'previous_day_cases', 'daily_growth_rate']].head(10))

# to handle divison by zero
df['daily_growth_rate'] = np.where(
    df['previous_day_cases'] > 0,
    (df['new_cases'] / df['previous_day_cases']) * 100,
    np.nan,
)
print(df[['country', 'date', 'total_cases', 'new_cases',
          'previous_day_cases', 'daily_growth_rate']].head(100))

# calculate the 7-day moving average of new cases
df['new_cases_smoothed'] = (df.groupby('country')['new_cases'].transform(lambda x: x.rolling(7).mean()))
print(df[['country', 'date', 'new_cases', 'new_cases_smoothed']].head(20))

# create 7 days death moving average
df['new_death_7day_avg'] = df.groupby('country')['new_deaths'].transform(lambda x: x.rolling(7).mean())
print(df[['country', 'date', 'new_deaths', 'new_death_7day_avg']].head(20))

df["new_cases_7day_avg"] = (
    df.groupby("country")["new_cases"]
      .transform(lambda x: x.rolling(7).mean())
)
# case fatality rate
df['case_fatality_rate'] = np.where(df['total_cases'] > 0, (df['total_deaths'] / df['total_cases'])* 100, np.nan)
print(df[['case_fatality_rate']].head(200))
print(df["case_fatality_rate"].describe())

# to see the data
print(df[['country', 'date', 'total_cases', 'new_cases', 'total_deaths', 'previous_day_cases', 'daily_growth_rate', 'new_cases_smoothed', 'case_fatality_rate']].head(20))

print(f'daily growth rate statistics:\n{df["daily_growth_rate"].describe()}')
print(f'7-day moving average statistics:\n{df["new_cases_smoothed"].describe()}')
print(f'case fatality rate statistics:\n{df["case_fatality_rate"].describe()}')

#-----------------------------------------------------------------------------------------
# latest data 
latest_data = df.sort_values('date').groupby('country').tail(1)
print(latest_data[['country', 'date', 'total_cases', 'total_deaths', 'new_deaths', 'case_fatality_rate']])


# top 10 countries by total cases
top_10_cases = (latest_data.sort_values('total_cases', ascending=False).head(10))
print(top_10_cases[['country', 'total_cases']])
# for a clear output
print(
    top_10_cases[
        ["country", "total_cases"]
    ].to_string(index=False)
)

# top 10 countries by total deaths
top_10_deaths = (latest_data.sort_values('total_deaths', ascending=False).head(10))
print(top_10_deaths[['country', 'total_deaths']])

# case fatality rate
df['case_fatality_rate'] = np.where(df['total_cases'] > 0, (df['total_deaths'] / df['total_cases'])* 100, np.nan)

cfr_by_country = latest_data[['country', 'case_fatality_rate']].sort_values('case_fatality_rate', ascending=False).head(10)
print(cfr_by_country)

# cases per continent
continent_summary = (
    latest_data.groupby('continent', dropna=False)
    .agg(
        countries=('country', 'nunique'),
        total_cases=('total_cases', 'sum'),
        total_deaths=('total_deaths', 'sum'),
    )
    .reset_index()
)
continent_summary['average_cases_per_country'] = (
    continent_summary['total_cases'] / continent_summary['countries']
)
continent_summary['average_deaths_per_country'] = (
    continent_summary['total_deaths'] / continent_summary['countries']
)
print(continent_summary)

# per million comparison
continent_summary_per_million = (latest_data.groupby('continent').agg(
    total_cases_per_million=('total_cases_per_million', 'sum'),
    total_deaths_per_million=('total_deaths_per_million', 'sum'),
    average_cases_per_million=('total_cases_per_million', 'mean'),
    average_deaths_per_million=('total_deaths_per_million', 'mean'))
)
print(continent_summary_per_million)

# check te growth rate 
growth_rate_country = (df.groupby('country')['daily_growth_rate'].mean().sort_values(ascending=False))
print(growth_rate_country)

#-----------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
# import seaborn as sns

# sns.set_theme(style="whitegrid")

selected_countries = ['India', 'United States', 'Brazil', 'Russia', 'United Kingdom', 'Japan', 'Germany']

selected_data = df[df['country'].isin(selected_countries)]

# total cases over time
plt.figure(figsize=(14,7))
for country in selected_countries:
    country_data = selected_data[selected_data['country'] == country]
    plt.plot(country_data['date'], country_data['total_cases'], label=country, linewidth=2)
plt.title('Total COVID-19 Cases Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Total Cases', fontsize=14)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 7 day smoothed new cases
plt.figure(figsize=(14,7))
for country in selected_countries:
    country_data = selected_data[selected_data['country'] == country]
    plt.plot(country_data['date'], country_data['new_cases_smoothed'], label=country, linewidth=2)
plt.title('7-Day Smoothed New COVID-19 Cases', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('New Cases (7-Day Average)', fontsize=14)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Top 10 countries by total cases
plt.figure(figsize=(12,6))
plt.bar(top_10_cases['country'], top_10_cases['total_cases'], color='skyblue')
plt.title('Top 10 Countries by Total COVID-19 Cases', fontsize=16, fontweight='bold')
plt.xlabel('Country', fontsize=14)
plt.ylabel('Total Cases', fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='x', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

#total deaths by country
plt.figure(figsize=(12,7))
plt.bar(top_10_deaths['country'], top_10_deaths['total_deaths'], color='salmon')
plt.title('Top 10 Countries by Total COVID-19 Deaths', fontsize=16, fontweight='bold')
plt.xlabel('Country', fontsize=14)
plt.ylabel('Total Deaths', fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='x', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# correlation heatmap
health_metrics = latest_data[['total_cases', 'new_cases', 'total_deaths', 'new_deaths', 'total_cases_per_million', 'total_deaths_per_million']]
correlation_matrix = health_metrics.corr()
plt.figure(figsize=(10,8))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Correlation Coefficient')
plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation=45,ha='right')
plt.yticks(range(len(correlation_matrix)), correlation_matrix.columns)
plt.title('Correlation Heatmap of COVID-19 Metrics', fontsize=16, fontweight='bold')
for i in range(len(correlation_matrix)):
    for j in range(len(correlation_matrix)):
        plt.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}", ha='center', va='center', color='black')
plt.title('Correlation Heatmap of COVID-19 Metrics', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# cases per million by continent
continents = latest_data["continent"].dropna().unique()

data_by_continent = []

continent_names = []

for continent in continents:

    values = latest_data[
        latest_data["continent"] == continent
    ]["total_cases_per_million"].dropna()

    data_by_continent.append(values)
    continent_names.append(continent)

# create boxplot
plt.figure(figsize=(12, 7))
plt.boxplot(data_by_continent, tick_labels=continent_names, patch_artist=True)
plt.title("Distribution of Total Cases per Million by Continent", fontsize=16, fontweight='bold')
plt.xlabel("Continent", fontsize=14)
plt.ylabel("Total Cases per Million", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#---------------------------------------------------------------------------------------------------------


import matplotlib.pyplot as plt


# Create dashboard canvas
fig, axes = plt.subplots(
    2, 3,
    figsize=(22, 12)
)

fig.suptitle(
    "COVID-19 Data Analysis Dashboard",
    fontsize=24,
    fontweight="bold"
)




ax = axes[0, 0]

for country in selected_countries:

    country_data = selected_data[
        selected_data["country"] == country
    ]

    ax.plot(
        country_data["date"],
        country_data["total_cases"],
        label=country,
        linewidth=1.8
    )

ax.set_title(
    "Total COVID-19 Cases Over Time",
    fontsize=14,
    fontweight="bold"
)

ax.set_xlabel("Date")
ax.set_ylabel("Total Cases")

ax.legend(fontsize=8)

ax.tick_params(axis="x", rotation=45)

ax.grid(
    True,
    alpha=0.3
)



ax = axes[0, 1]

for country in selected_countries:

    country_data = selected_data[
        selected_data["country"] == country
    ]

    ax.plot(
        country_data["date"],
        country_data["new_cases_7day_avg"],
        label=country,
        linewidth=1.8
    )

ax.set_title(
    "7-Day Moving Average of New Cases",
    fontsize=14,
    fontweight="bold"
)

ax.set_xlabel("Date")
ax.set_ylabel("7-Day Average")

ax.legend(fontsize=8)

ax.tick_params(axis="x", rotation=45)

ax.grid(
    True,
    alpha=0.3
)



ax = axes[0, 2]

ax.barh(
    top_10_cases["country"],
    top_10_cases["total_cases"]
)

ax.set_title(
    "Top 10 Countries by Total Cases",
    fontsize=14,
    fontweight="bold"
)

ax.set_xlabel("Total Cases")
ax.set_ylabel("Country")

ax.invert_yaxis()

ax.grid(
    axis="x",
    alpha=0.3
)


ax = axes[1, 0]

ax.barh(
    top_10_deaths["country"],
    top_10_deaths["total_deaths"]
)

ax.set_title(
    "Top 10 Countries by Total Deaths",
    fontsize=14,
    fontweight="bold"
)

ax.set_xlabel("Total Deaths")
ax.set_ylabel("Country")

ax.invert_yaxis()

ax.grid(
    axis="x",
    alpha=0.3
)


ax = axes[1, 1]

heatmap_data = latest_data[
    [
        "total_cases",
        "total_deaths",
        "reproduction_rate",
        "stringency_index",
        "gdp_per_capita",
        "median_age"
    ]
]

correlation_matrix = heatmap_data.corr()

image = ax.imshow(
    correlation_matrix,
    cmap="coolwarm",
    aspect="auto"
)

fig.colorbar(
    image,
    ax=ax,
    fraction=0.046,
    pad=0.04
)

ax.set_xticks(
    range(len(correlation_matrix.columns))
)

ax.set_xticklabels(
    correlation_matrix.columns,
    rotation=45,
    ha="right",
    fontsize=8
)

ax.set_yticks(
    range(len(correlation_matrix.columns))
)

ax.set_yticklabels(
    correlation_matrix.columns,
    fontsize=8
)

# Add correlation values
for i in range(len(correlation_matrix.columns)):

    for j in range(len(correlation_matrix.columns)):

        value = correlation_matrix.iloc[i, j]

        ax.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=8
        )

ax.set_title(
    "Correlation Heatmap",
    fontsize=14,
    fontweight="bold"
)


ax = axes[1, 2]

continents = (
    latest_data["continent"]
    .dropna()
    .unique()
)

data_by_continent = []
continent_names = []

for continent in continents:

    values = latest_data[
        latest_data["continent"] == continent
    ]["total_cases_per_million"].dropna()

    data_by_continent.append(values)

    continent_names.append(continent)


ax.boxplot(
    data_by_continent,
    tick_labels=continent_names,
    patch_artist=True
)

ax.set_title(
    "Cases per Million by Continent",
    fontsize=14,
    fontweight="bold"
)

ax.set_xlabel("Continent")
ax.set_ylabel("Total Cases per Million")

ax.tick_params(
    axis="x",
    rotation=30
)

ax.grid(
    axis="y",
    alpha=0.3
)



plt.tight_layout(
    rect=[0, 0, 1, 0.95]
)

plt.show()