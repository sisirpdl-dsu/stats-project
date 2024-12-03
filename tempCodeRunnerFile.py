import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression


data = pd.read_csv("data.csv")

gdp_data = data[data["Series_name"] == "GDP_per_capita"]
unemployment_data = data[data["Series_name"] == "Unemployment_rate"]

year_columns = [str(year) for year in range(2014, 2024)]

gdp_stats_table = []
for year in year_columns:
    gdp_year = gdp_data[year]
    gdp_stats_table.append({
        "Year": year,
        "Mean GDP": gdp_year.mean(),
        "Median GDP": gdp_year.median(),
        "Std Dev GDP": gdp_year.std()
    })

gdp_stat = pd.DataFrame(gdp_stats_table)
print(gdp_stat)

mean_gdp_over_years = gdp_data[year_columns].mean()
mean_unemployment_over_years = unemployment_data[year_columns].mean()


plt.figure(figsize=(10, 6))
plt.plot(year_columns, mean_gdp_over_years, label="Average GDP per Capita", marker="o")
plt.title("Average GDP per Capita Over Time (2014–2023)")
plt.xlabel("Year")
plt.ylabel("GDP per Capita (PPP)")
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(year_columns, mean_unemployment_over_years, label="Average Unemployment Rate", marker="o", color="orange")
plt.title("Average Unemployment Rate Over Time (2014–2023)")
plt.xlabel("Year")
plt.ylabel("Unemployment Rate (%)")
plt.grid()
plt.tight_layout()
plt.show()

aligned_data = pd.concat([
    gdp_data.set_index(["Country Name"])[year_columns].stack(),
    unemployment_data.set_index(["Country Name"])[year_columns].stack()
], axis=1, keys=["GDP_per_capita", "Unemployment_rate"]).dropna()

print()
print("-----------------")
print("this is Aligned_data")
print(aligned_data)

gdp_flat = aligned_data["GDP_per_capita"].values  
unemployment_flat = aligned_data["Unemployment_rate"].values  
print("-----------------")
print("this is gdp per capita flat data")
print(gdp_flat)

print("-----------------")
print("this is unemployment flat data")
print(unemployment_flat)

print("-----------------")

overall_correlation = np.corrcoef(unemployment_flat, gdp_flat)[0, 1]
print(f"Overall Pearson Correlation between Unemployment rate and GDP per capita: {overall_correlation:.4f}\n")
X = unemployment_flat.reshape(-1, 1)  
y = gdp_flat  
model = LinearRegression()
model.fit(X, y)


overall_slope = model.coef_[0]
overall_intercept = model.intercept_
print(f"Overall R-squared: {model.score(X, y):.4f}")
print(f"Overall Regression eqn: GDP per Capita = {overall_slope:.4f} * Unemployment Rate + {overall_intercept:.4f}\n")


country_results = []

for country in data["Country Name"].unique():
    
    country_gdp = gdp_data[gdp_data["Country Name"] == country][year_columns].values.flatten()
    country_unemployment = unemployment_data[unemployment_data["Country Name"] == country][year_columns].values.flatten()
    
    
    valid_mask = ~np.isnan(country_gdp) & ~np.isnan(country_unemployment)
    country_gdp = country_gdp[valid_mask]
    country_unemployment = country_unemployment[valid_mask]
    
    if len(country_gdp) > 1:  
        
        country_correlation = np.corrcoef(country_unemployment, country_gdp)[0, 1]
        
        
        X_country = country_unemployment.reshape(-1, 1)
        y_country = country_gdp
        country_model = LinearRegression()
        country_model.fit(X_country, y_country)
        
        
        country_slope = country_model.coef_[0]
        country_intercept = country_model.intercept_
        r_squared = country_model.score(X_country, y_country)
        
        
        country_results.append({
            "Country": country,
            "Correlation": country_correlation,
            "Regression eqn": f"Y= {country_slope:.4f} * X + {country_intercept:.4f}",
            "R-squared": r_squared
        })


country_results_df = pd.DataFrame(country_results)
print("Country-Specific Statistics Table:\n")
print(country_results_df)


plt.figure(figsize=(10, 6))
sns.barplot(
    data=country_results_df,
    x="Country",
    y="Correlation",
    hue="Country", 
    dodge=False,
    palette="coolwarm"
)
plt.title("Country-Specific Correlations: Unemployment Rate vs GDP per Capita")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Correlation")
plt.xlabel("Country")
plt.legend([], [], frameon=False)  
plt.tight_layout()
plt.show()




plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5, color="blue", label="Overall Data Points")
plt.plot(X, model.predict(X), color="red", linewidth=2, label="Overall Regression Line")
plt.title("Overall Regression Plot: Unemployment Rate vs GDP per Capita")
plt.xlabel("Unemployment Rate")
plt.ylabel("GDP per Capita")
plt.legend()
plt.tight_layout()
plt.show()




for index, row in country_results_df.iterrows():
    country = row["Country"]
    country_gdp = gdp_data[gdp_data["Country Name"] == country][year_columns].values.flatten()
    country_unemployment = unemployment_data[unemployment_data["Country Name"] == country][year_columns].values.flatten()
    valid_mask = ~np.isnan(country_gdp) & ~np.isnan(country_unemployment)
    country_gdp = country_gdp[valid_mask]
    country_unemployment = country_unemployment[valid_mask]
    
    if len(country_gdp) > 1:
        X_country = country_unemployment.reshape(-1, 1)
        y_country = country_gdp
        country_model = LinearRegression()
        country_model.fit(X_country, y_country)
        
        plt.figure(figsize=(8, 5))
        plt.scatter(X_country, y_country, alpha=0.5, color="blue", label="Data Points")
        plt.plot(X_country, country_model.predict(X_country), color="red", linewidth=2, label="Regression Line")
        plt.title(f"Regression Plot for {country}")
        plt.xlabel("Unemployment Rate")
        plt.ylabel("GDP per Capita")
        plt.legend()
        plt.tight_layout()
        plt.show()
