# COVID-19 Data Analysis

## 📌 Project Overview

This project focuses on analyzing **COVID-19 daily statistics** using Python. The analysis covers COVID-19 cases, deaths, growth rates, moving averages, case fatality rates, and country/continent-level comparisons.

The project uses **Pandas and NumPy** for data processing and **Matplotlib** for data visualization.

The dataset contains daily COVID-19 statistics for **38 countries** from **January 1, 2020 to December 31, 2023**.

---

## 🎯 Objectives

* Load and understand the COVID-19 dataset.
* Clean and preprocess the data.
* Handle missing values and duplicate records.
* Perform feature engineering.
* Calculate daily case growth rates.
* Calculate 7-day moving averages.
* Calculate Case Fatality Rate (CFR).
* Identify countries with the highest number of cases and deaths.
* Compare COVID-19 statistics across continents.
* Analyze relationships between COVID-19 and demographic/economic indicators.
* Create an interactive-looking analytical dashboard using Matplotlib.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Jupyter Notebook / VS Code**
* **Git & GitHub**

---

## 📂 Dataset

**Dataset:** `covid19_student_dataset.csv`

The dataset contains daily COVID-19 information for 38 countries.

### Important Columns

| Column                       | Description                              |
| ---------------------------- | ---------------------------------------- |
| `country`                    | Country name                             |
| `continent`                  | Continent of the country                 |
| `date`                       | Date of observation                      |
| `total_cases`                | Total confirmed COVID-19 cases           |
| `new_cases`                  | Newly reported cases                     |
| `new_cases_smoothed`         | Smoothed new cases                       |
| `total_cases_per_million`    | Total cases per million population       |
| `total_deaths`               | Total reported deaths                    |
| `new_deaths`                 | Newly reported deaths                    |
| `total_deaths_per_million`   | Total deaths per million population      |
| `reproduction_rate`          | Estimated COVID-19 reproduction rate     |
| `stringency_index`           | Government response/stringency indicator |
| `population`                 | Population                               |
| `median_age`                 | Median age of population                 |
| `gdp_per_capita`             | GDP per capita                           |
| `hospital_beds_per_thousand` | Hospital beds per thousand people        |

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Inspection
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Exploratory Data Analysis
   ↓
Data Visualization
   ↓
Dashboard
   ↓
Insights & Conclusion
```

---

## 🧹 Data Cleaning

The following preprocessing steps were performed:

* Converted the `date` column to datetime format.
* Checked for duplicate records.
* Sorted data by country and date.
* Handled missing time-series values using country-wise forward filling where appropriate.
* Handled country-level static variables using country-wise median values where appropriate.
* Checked remaining missing values before analysis.

Example:

```python
df["date"] = pd.to_datetime(df["date"])

df = df.drop_duplicates()

df = df.sort_values(["country", "date"])
```

---

## ⚙️ Feature Engineering

### 1. Previous Day Cases

```python
df["previous_day_cases"] = (
    df.groupby("country")["total_cases"].shift(1)
)
```

### 2. Daily Growth Rate

```python
df["daily_growth_rate"] = np.where(
    df["previous_day_cases"] > 0,
    (df["new_cases"] / df["previous_day_cases"]) * 100,
    np.nan
)
```

### 3. 7-Day Moving Average of New Cases

```python
df["new_cases_7day_avg"] = (
    df.groupby("country")["new_cases"]
      .transform(lambda x: x.rolling(7).mean())
)
```

### 4. 7-Day Moving Average of New Deaths

```python
df["new_deaths_7day_avg"] = (
    df.groupby("country")["new_deaths"]
      .transform(lambda x: x.rolling(7).mean())
)
```

### 5. Case Fatality Rate

```python
df["case_fatality_rate"] = np.where(
    df["total_cases"] > 0,
    (df["total_deaths"] / df["total_cases"]) * 100,
    np.nan
)
```

---

## 📊 Exploratory Data Analysis

The analysis includes:

* Top 10 countries by total COVID-19 cases.
* Top 10 countries by total COVID-19 deaths.
* Case Fatality Rate comparison.
* Continent-level case and death comparison.
* Cases per million comparison.
* Reproduction rate analysis.
* Government stringency analysis.
* Correlation analysis between selected variables.

For cumulative country-level comparisons, the **latest available observation for each country** is used instead of summing daily cumulative values.

---

# 📈 Visualizations

The project contains **6 main visualizations**.

### 1. Total COVID-19 Cases Over Time

Compares the cumulative COVID-19 cases for selected countries.

Countries included:

* India
* United States
* Brazil
* United Kingdom
* Germany
* Japan
<img width="1400" height="700" alt="Figure_1" src="https://github.com/user-attachments/assets/4aad39d8-1c19-47f8-8c24-8612caf6a334" />

---

### 2. 7-Day Moving Average of New Cases

Shows the smoothed daily COVID-19 case trends and reduces the effect of daily fluctuations.
<img width="1400" height="700" alt="Figure_2" src="https://github.com/user-attachments/assets/84a30f0c-f68d-4f0f-9d07-b85ee7408b6e" />

---

### 3. Top 10 Countries by Total Cases

A horizontal bar chart showing the countries with the highest reported cumulative COVID-19 cases.
<img width="1200" height="600" alt="Figur3" src="https://github.com/user-attachments/assets/634a2d50-987a-4cd2-8953-5897f2a7fd53" />

---

### 4. Top 10 Countries by Total Deaths

A horizontal bar chart showing the countries with the highest reported cumulative COVID-19 deaths.
<img width="1200" height="700" alt="Figure_4" src="https://github.com/user-attachments/assets/e4966a12-9b4a-479f-b50e-f67e9e4f190d" />

---

### 5. Correlation Heatmap

The correlation analysis examines relationships between selected variables such as:

* Total Cases
* Total Deaths
* Reproduction Rate
* Stringency Index
* GDP per Capita
* Median Age

The heatmap is created using **Matplotlib**, without Seaborn.
<img width="612" height="744" alt="Figure_5png" src="https://github.com/user-attachments/assets/1b6cfced-910d-4733-a8a6-2f1f3d42231b" />

---

### 6. Total Cases per Million by Continent

A box plot is used to compare the distribution of total COVID-19 cases per million population across continents.
<img width="1536" height="752" alt="Figure_6" src="https://github.com/user-attachments/assets/9595549a-4196-4a34-8fd2-ed8be6b2485c" />

---

## 📊 Dashboard

All six visualizations are combined into a single Matplotlib dashboard.

<img width="766" height="744" alt="Figure_7" src="https://github.com/user-attachments/assets/cb15d9ad-9611-4fd9-8a60-77bc46e9c2f0" />

---

## 💡 Key Insights

The analysis can be used to identify:

1. Differences in cumulative COVID-19 cases between countries.
2. Differences in reported COVID-19 deaths.
3. Changes in daily case trends using 7-day moving averages.
4. Differences in COVID-19 cases per million across continents.
5. Relationships between COVID-19 indicators and demographic/economic variables.
6. Changes in COVID-19 growth patterns over the study period.

> The exact numerical insights depend on the results generated from the provided dataset.

---


## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
```

### 2. Navigate to the project directory

```bash
cd COVID-19-Data-Analysis
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
.venv\Scripts\activate
```

### 5. Install required libraries

```bash
pip install pandas numpy matplotlib
```

### 6. Run the Python program

```bash
python covid19_analysis.py
```

---

## 📦 Requirements

```text
Python 3.x
Pandas
NumPy
Matplotlib
```

---

## 🚀 Future Improvements

Possible future enhancements include:

* Adding an interactive dashboard using Power BI or Streamlit.
* Adding country-specific filters.
* Adding date-range selection.
* Adding more advanced statistical analysis.
* Comparing vaccination data with case/death trends.
* Adding additional COVID-19 indicators.

---

## 👨‍💻 Author

**Santhosh Kumar**

### Project

**COVID-19 Data Analysis using Python**

### Skills Demonstrated

`Python` · `NumPy` · `Pandas` · `Matplotlib` · `Data Cleaning` · `Feature Engineering` · `EDA` · `Data Visualization`

---


This project is created for **educational and data-analysis purposes**.
