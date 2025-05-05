# Policy Impact Analysis using PySpark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, corr, year

# Starting Spark session
spark = SparkSession.builder.appName('Policy Impact Analysis').getOrCreate()

# Loading dataset
df = spark.read.csv('/Users/Zof/Downloads/europe_covid_cleaned9.csv', header=True, inferSchema=True)

# Dropping rows with missing values in key columns
df = df.dropna(subset=['stringency_index', 'new_cases_per_million', 'new_deaths_per_million'])

# Extracting year from date column
df = df.withColumn('year', year('date'))

# Overall correlation
corr_new_cases = df.select(corr('stringency_index', 'new_cases_per_million')).collect()[0][0]
corr_new_deaths = df.select(corr('stringency_index', 'new_deaths_per_million')).collect()[0][0]

print(f'Correlation between Stringency Index and New Cases per Million: {corr_new_cases}')
print(f'Correlation between Stringency Index and New Deaths per Million: {corr_new_deaths}')

# Year and country-level correlation
yearly_country_corr = df.groupBy('location', 'year').agg(
    corr('stringency_index', 'new_cases_per_million').alias('Cases_Corr'),
    corr('stringency_index', 'new_deaths_per_million').alias('Deaths_Corr')
).orderBy('location', 'year')

# Showing results limited to first 30 rows
yearly_country_corr.show(30, truncate=False)

# Stopping Spark session
spark.stop()
