# Databricks notebook source
# MAGIC %md
# MAGIC # Project Title: Loan Approval using Pyspark

# COMMAND ----------

# MAGIC %md
# MAGIC ## Group: 4
# MAGIC ## Team Members
# MAGIC
# MAGIC - Divya Vemula
# MAGIC - Rahul Chauhan
# MAGIC - ManiKrishna Tippani
# MAGIC - Shaheryar Nadeem
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Project Description
# MAGIC
# MAGIC The objective of this project is to analyze and predict the outcomes of bank loan applications using a dataset containing 148,000 records of applicant and loan information. The dataset includes critical features such as loan amount, applicant's demographics, annual income, credit score, and the purpose of the loan, among others. The target variable, status, indicates whether a loan was approved (0) or denied (1). By studying patterns and trends within these features, the goal is to build a predictive model that can accurately forecast loan approval decisions, helping banks streamline their evaluation processes and manage risks effectively.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Project Dataset Details:
# MAGIC
# MAGIC The dataset consists of 148,671 rows with each row representing the claim details and 37 columns contains vehicle dataset - id, loan, gender, age, loan purpose, credit_score etc.
# MAGIC
# MAGIC #### Dataset Source:
# MAGIC - **Kaggle**: https://www.kaggle.com/datasets/ychope/loan-approval-dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dataset Variables Description
# MAGIC
# MAGIC ### 1. Target Variable: status
# MAGIC
# MAGIC   Description: The target variable indicates whether the loan application was approved or denied.
# MAGIC   Data Type: Integer (or Boolean)
# MAGIC
# MAGIC   Values:
# MAGIC     - 0: Loan approved.
# MAGIC     - 1: Loan denied.
# MAGIC
# MAGIC ### 2. Predictor Variables in the Dataset:
# MAGIC Demographic & Applicant Information:
# MAGIC
# MAGIC #### gender:
# MAGIC   Description: The gender of the applicant.
# MAGIC
# MAGIC   Data Type: Categorical (String)
# MAGIC
# MAGIC   Values:
# MAGIC   - Male
# MAGIC   - Female
# MAGIC   - Unknown (if not specified)
# MAGIC
# MAGIC #### age:
# MAGIC
# MAGIC   Description: Age of the applicant.
# MAGIC
# MAGIC   Data Type: Numeric (Integer)
# MAGIC
# MAGIC   Values: Age in years (e.g., 25, 35, 50)
# MAGIC
# MAGIC #### income:
# MAGIC
# MAGIC   Description: The annual income of the applicant.
# MAGIC
# MAGIC   Data Type: Numeric (Float or Integer)
# MAGIC
# MAGIC   Values: Income value in the local currency (e.g., USD, INR)
# MAGIC
# MAGIC #### credit_score:
# MAGIC
# MAGIC   Description: The applicant's credit score based on their financial history.
# MAGIC
# MAGIC   Data Type: Numeric (Integer)
# MAGIC
# MAGIC   Values: A numerical score (e.g., 650, 700, 800) reflecting the applicant’s creditworthiness.
# MAGIC
# MAGIC #### loan_amount:
# MAGIC
# MAGIC   Description: The total amount of the loan requested by the applicant.
# MAGIC
# MAGIC   Data Type: Numeric (Integer or Float)
# MAGIC
# MAGIC   Values: The loan amount in the local currency (e.g., USD, INR).
# MAGIC
# MAGIC #### loan_limit:
# MAGIC
# MAGIC   Description: The maximum allowable loan amount based on the applicant’s income, credit score, and other factors.
# MAGIC
# MAGIC   Data Type: Categorical (String)
# MAGIC
# MAGIC   Values:
# MAGIC
# MAGIC   - Low
# MAGIC   - Medium
# MAGIC   - High
# MAGIC   - Unknown
# MAGIC
# MAGIC #### loan_type:
# MAGIC
# MAGIC   Description: The type of loan being requested by the applicant (e.g., personal loan, mortgage, etc.).
# MAGIC
# MAGIC   Data Type: Categorical (String)
# MAGIC
# MAGIC   Values:
# MAGIC
# MAGIC   - Personal
# MAGIC   - Mortgage
# MAGIC   - Business
# MAGIC   - Other
# MAGIC
# MAGIC #### loan_purpose:
# MAGIC
# MAGIC   Description: The purpose for which the applicant is requesting the loan (e.g., for buying a home, starting a business, etc.).
# MAGIC
# MAGIC   Data Type: Categorical (String)
# MAGIC
# MAGIC   Values:
# MAGIC
# MAGIC   - Home
# MAGIC   - Business
# MAGIC   - Car
# MAGIC   - Personal
# MAGIC   - Education
# MAGIC   - Debt Consolidation
# MAGIC   - Other
# MAGIC
# MAGIC #### region:
# MAGIC
# MAGIC   Description: The geographic region where the applicant resides.
# MAGIC
# MAGIC   Data Type: Categorical (String)
# MAGIC
# MAGIC   Values: Examples could be Urban, Suburban, Rural, or specific geographic regions like North, South, etc.
# MAGIC
# MAGIC #### employment_status:
# MAGIC
# MAGIC   Description: The employment status of the applicant.
# MAGIC
# MAGIC   Data Type: Categorical (String)
# MAGIC
# MAGIC   Values:
# MAGIC
# MAGIC   - Employed
# MAGIC   - Unemployed
# MAGIC   - Self-Employed
# MAGIC   - Student
# MAGIC   - Retired
# MAGIC
# MAGIC #### interest_rate:
# MAGIC
# MAGIC   Description: The interest rate applied to the loan.
# MAGIC
# MAGIC   Data Type: Numeric (Float or Integer)
# MAGIC
# MAGIC   Values: The percentage rate applied to the loan (e.g., 5.0, 10.5).
# MAGIC
# MAGIC #### debt_to_income_ratio (dtir1):
# MAGIC
# MAGIC   Description: The ratio of the applicant's monthly debt payments to their monthly income.
# MAGIC
# MAGIC   Data Type: Numeric (Float)
# MAGIC
# MAGIC   Values: A ratio (e.g., 0.25 means 25% of the applicant's income goes towards debt repayment).
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Important Pyspark Libraries

# COMMAND ----------

# DBTITLE 1,Load Required Libraries
# Libraries imported for processing
import numpy as np  # Mathematical computations
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns  # Plotting
import warnings
import scipy.stats as ss  # Statistical computations
import math  # Mathematical computations
import pyspark.sql.functions as F
import pyspark.sql.types as T
import re

from itertools import product
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, Tokenizer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier, NaiveBayes, LinearSVC
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col, when, sum, sort_array, collect_list, round
from pyspark.sql.types import IntegerType, FloatType,StringType

# Filter warnings
warnings.filterwarnings("ignore")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Remove if the Loan Data Csv file already Exists in driver

# COMMAND ----------

# DBTITLE 1,Removing Loan Data Csv
rm /databricks/driver/Loan


# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Driver Content

# COMMAND ----------

# DBTITLE 1,Loading Driver Content
ls -lrt /databricks/driver

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Kaggle

# COMMAND ----------

# DBTITLE 1,Installing Kaggle
# MAGIC %sh
# MAGIC pip install kaggle

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrieve data from Kaggle

# COMMAND ----------

# DBTITLE 1,Using Kaggle API Key
# MAGIC %sh
# MAGIC export KAGGLE_USERNAME=divyavemula696 
# MAGIC export KAGGLE_KEY=e6a8bcb2cb4f105d654e17cbe054bf4c
# MAGIC kaggle datasets download -d ychope/loan-approval-dataset 
# MAGIC ls -lrt

# COMMAND ----------

# MAGIC %md
# MAGIC ## Zipped data Extraction

# COMMAND ----------

# DBTITLE 1,Extracting the Zipped data
# MAGIC %sh
# MAGIC unzip loan-approval-dataset.zip
# MAGIC ls -lrt
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the Driver Data

# COMMAND ----------

# DBTITLE 1,Loading the Driver Data
# MAGIC %sh
# MAGIC
# MAGIC ls -lrt /databricks/driver/

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read the Loan.csv file from the Driver

# COMMAND ----------

# DBTITLE 1,Reading the Loan.csv file from the Driver
# Function to load and preprocess data
def load_data(file_path):
    """
    Load the CSV data into a Spark DataFrame.
    Args:
    - file_path (str): Path to the CSV file
    
    Returns:
    - df (DataFrame): Loaded DataFrame
    """
    # Load the dataset
    df = spark.read.csv(file_path, header=True, inferSchema=True)

    return df

# Load the data
df = load_data("file:/databricks/driver/Loan.csv")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Spark Api that wasn't used in Course but used in our analysis

# COMMAND ----------

# DBTITLE 1,Special Spark api
from pyspark.pandas import Series

# Extract a column as a PySpark Pandas Series
psser = df.to_pandas_on_spark()['loan_purpose']

# Get unique values
unique_values = psser.unique()

# Display unique values
print(unique_values)


# COMMAND ----------

# MAGIC %md 
# MAGIC ### Explanation: 
# MAGIC
# MAGIC **Fetch Data**: Fetch data from the API and convert it into a Pandas DataFrame. 
# MAGIC
# MAGIC **Convert to Spark DataFrame**: Convert the Pandas DataFrame into a PySpark DataFrame. 
# MAGIC
# MAGIC **PySpark Pandas Series**: Convert a column to a PySpark Pandas Series and get unique

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dimensions of the dataset (rows, columns)

# COMMAND ----------

# DBTITLE 1,number of rows and columns
# Display number of rows and columns
num_rows = df.count()  # Get number of rows
num_columns = len(df.columns)  # Get number of columns

print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Fields Description

# COMMAND ----------

# DBTITLE 1,Data Fields
df.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ## UnCleaned or Raw Data Schema

# COMMAND ----------

# DBTITLE 1,Raw Data Schema
# This section includes the initialization of Spark session and loading of data

# Display DataFrame Schema to understand data structure
display(df.printSchema())


# COMMAND ----------

# MAGIC %md
# MAGIC ## Target Field Described and Highlighted

# COMMAND ----------

# DBTITLE 1,Target variable
# Target variable: 'status' (approved/denied)
target_field = 'status'

# Describe target field
df.select(target_field).distinct().show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Count the number of loans by status

# COMMAND ----------

# DBTITLE 1,Product line by purchases

# Product line by purchases
df.select('status').groupBy('status').count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data statistical summary on raw data

# COMMAND ----------

# DBTITLE 1,Summary of Raw Data
# Summary of Raw Data
display(df.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Raw Data table

# COMMAND ----------

# DBTITLE 1,display raw data
# display top 5 rows
display(df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check for Duplicate Rows

# COMMAND ----------

# DBTITLE 1,Display Duplicate Rows
# Duplicate rows in raw data
num_duplicates = df.count() - df.distinct().count()
print(f"Number of duplicate rows: {num_duplicates}")

# COMMAND ----------

# DBTITLE 1,Remove Duplicate Rows
# Drop duplicate rows if duplicates exist
if num_duplicates > 0:
    print("Dropping duplicate rows...")
    df = df.dropDuplicates()
    print("Duplicates dropped.")
else:
    print("No duplicate rows found. No action needed.")

# Show the updated DataFrame
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Visualizations For Raw Data

# COMMAND ----------

# MAGIC %md
# MAGIC ### Plot-1: Histogram

# COMMAND ----------

# DBTITLE 1,Missing Values Visualization
# Missing values
missing_values = df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]).toPandas()

# COMMAND ----------

# DBTITLE 1,Plot On Raw Data
# Set the figure size
plt.figure(figsize=(12, 6))

# Create the heatmap with improved aesthetics
sns.heatmap(missing_values, annot=True, cmap='viridis', cbar=False, fmt='g', linewidths=2.0, linecolor='white')

# Add a title to the heatmap
plt.title("Missing Values Heatmap", fontsize=16, fontweight='bold', color='black', pad=20)

# Display the plot
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Plot-2: Distribution of Numerical Columns

# COMMAND ----------

# DBTITLE 1,Distribution of Numerical Columns
# Calculate the grid size based on the number of numeric columns
num_cols = len(df_numeric.columns)
rows = math.ceil(num_cols / 3)  # 3 columns per row
cols = min(3, num_cols)  # At most 3 columns

# Plot histograms for each numeric column
plt.figure(figsize=(15, 5 * rows))  # Adjust figure size for better visibility

for i, col in enumerate(df_numeric.columns, 1):
    plt.subplot(rows, cols, i)  # Dynamically adjust rows and columns
    sns.histplot(df_numeric[col], kde=True, bins=20, edgecolor='black', palette='Set2')
    plt.title(f"Distribution of {col}")

plt.tight_layout()
plt.suptitle("Distribution of Numerical Features", y=1.02, fontsize=16)
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Plot-3: Boxplots for Outliers

# COMMAND ----------

# DBTITLE 1,Boxplots
# Plotting on Outliers
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols):
    plt.subplot(3, 3, i+1)  # 3x3 grid for boxplots
    sns.boxplot(x=df_numeric[col], color='lightgreen')
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Plot-4: Correlation Matrix Heatmap

# COMMAND ----------

# DBTITLE 1,correlation matrix
# Compute the correlation matrix for numerical columns
corr_matrix = df_numeric.corr()

# Plot the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0, linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Plot-5: Bar Plot for Categorical Features

# COMMAND ----------

# DBTITLE 1,Plot for categorical features
# Select categorical columns
categorical_cols = [field.name for field in df.schema.fields if isinstance(field.dataType, StringType)]

# Number of categorical columns
num_categorical_cols = len(categorical_cols)

# Set number of columns for the subplot grid
num_cols = 2  # Reduce the number of columns to give more space
num_rows = (num_categorical_cols // num_cols) + (1 if num_categorical_cols % num_cols != 0 else 0)

# Create subplots dynamically with a larger figure size
plt.figure(figsize=(35, 25))  # Increase figure size for more space

for i, col in enumerate(categorical_cols):
    plt.subplot(num_rows, num_cols, i + 1)  # Dynamic rows and columns
    # Collect category counts and plot (using PySpark)
    category_counts = df.groupBy(col).count().toPandas()  # Convert to Pandas DataFrame for plotting
    sns.barplot(x=col, y='count', data=category_counts, palette='Set2')
    plt.title(f'Bar plot of {col}')
    plt.xticks(rotation=45)  # Rotate x-axis labels if necessary for better readability

# Adjust layout to add more space between the plots
plt.subplots_adjust(hspace=1.5, wspace=0.4)  # Increase space between subplots
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Plot-6: Pairplot of Numerical Features

# COMMAND ----------

# DBTITLE 1,Plot for numerical Features
sns.pairplot(df_numeric, kind='scatter', plot_kws={'alpha': 0.7})
plt.suptitle("Pairplot of Numerical Features", y=1.02)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Cleaning and Transformation

# COMMAND ----------

# DBTITLE 1,Handling missing, null, na, empty string values
from pyspark.sql import functions as F

# Fill NA and null values with 'Unknown' for categorical columns
df_cleaned = df.fillna({
    'loan_limit': 'Unknown', 'gender': 'Unknown', 'approv_in_adv': 'Unknown', 'loan_type': 'Unknown',
    'loan_purpose': 'Unknown', 'credit_worthiness': 'Unknown', 'business_or_commercial': 'Unknown',
    'neg_ammortization': 'Unknown', 'interest_only': 'Unknown', 'lump_sum_payment': 'Unknown',
    'construction_type': 'Unknown', 'occupancy_type': 'Unknown', 'secured_by': 'Unknown',
    'credit_type': 'Unknown', 'co-applicant_credit_type': 'Unknown', 'submission_of_application': 'Unknown',
    'region': 'Unknown', 'security_type': 'Unknown', 'high_interest_rate': 'Unknown', 'senior_age': 'Unknown'
})

# List of numerical columns
numerical_columns = ["loan_amount", "rate_of_interest", "interest_rate_spread", "upfront_charges", 
                     "term", "property_value", "income", "credit_score", "ltv", "dtir1"]

# Fill NA and null values for numerical columns with the median
for column in numerical_columns:
    # Calculate median using approxQuantile
    median_value = df_cleaned.approxQuantile(column, [0.5], 0.1)[0]
    # Replace nulls with median
    df_cleaned = df_cleaned.withColumn(column, F.when(F.col(column).isNull(), median_value).otherwise(F.col(column)))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Explanation of cleaning and handling
# MAGIC
# MAGIC **Categorical Columns**: Missing values are replaced with "Unknown" to ensure that no information is lost and the dataset remains usable.
# MAGIC
# MAGIC **Numerical Columns**: Missing values are replaced with the median of each column, providing a robust central value that mitigates the effect of outliers.
# MAGIC
# MAGIC Handling missing values this way ensures that your dataset is clean, consistent, and ready for further analysis or modeling, leading to more reliable and accurate results.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Statistical summary of cleaned data

# COMMAND ----------

# DBTITLE 1,Data Summary

# Display cleaned DataFrame
display(df_cleaned.summary())

# COMMAND ----------

# DBTITLE 1,statistical summary
# Generate statistical summary of cleaned data
display(df_cleaned.describe())

# COMMAND ----------

# DBTITLE 1,Top 5 rows of cleaned data
# Display top 5 rows
display(df_cleaned.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Visualizations on cleaned data

# COMMAND ----------

# MAGIC %md
# MAGIC ### Plot1: Scatter Plot

# COMMAND ----------

# DBTITLE 1,Scatter Plot on Clean data
# Convert to Pandas DataFrame
pd_df = df_cleaned.toPandas()

# Scatter Plot of Loan Amount vs. Income
plt.figure(figsize=(10, 6))
sns.scatterplot(data=pd_df, x="income", y="loan_amount", hue="status", palette="viridis")
plt.title('Scatter Plot of Loan Amount vs. Income')
plt.xlabel('Income')
plt.ylabel('Loan Amount')
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Plot2: Strip Plot

# COMMAND ----------

# DBTITLE 1,Strip Plot
plt.figure(figsize=(10, 6))
sns.stripplot(data=pd_df, x='credit_type', y='interest_rate_spread', jitter=True, palette='Set2')
plt.title('Interest Rate Spread by Credit Type')
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Plot-3: Pie chart

# COMMAND ----------

# DBTITLE 1,Pie Chart plotting
def plot_pie_chart(ax, df, column_name):
    category_counts = df[column_name].value_counts()
    colors = sns.color_palette("Set1", len(category_counts))
    wedges, texts, autotexts = ax.pie(category_counts, autopct='%1.1f%%', colors=colors)
    ax.set_title(f'Distribution of {column_name}')
    ax.set_ylabel('')
    for text in texts + autotexts:
        text.set_fontsize(12)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
plot_pie_chart(axes[0], pd_df, 'status')
plot_pie_chart(axes[1], pd_df, 'gender')
plot_pie_chart(axes[2], pd_df, 'credit_type')
plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Plot-4: Facet Grid plot

# COMMAND ----------

# DBTITLE 1,FacetGrid
g = sns.FacetGrid(pd_df, col="region", hue="gender", palette="Set2", height=5, aspect=1)
g.map(sns.scatterplot, "loan_amount", "property_value", alpha=.7)
g.add_legend()
plt.suptitle('Loan Amount vs. Property Value by Region and Gender', y=1.03)
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Plot-5: KDE Plot

# COMMAND ----------

# DBTITLE 1,kde plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=pd_df, x='credit_score', hue='gender', fill=True, palette='Dark2')
plt.title('Credit Score Distribution by Gender')
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Partition

# COMMAND ----------

# DBTITLE 1,Data splits for training and testing

# Split the data into training and testing sets
train_df, test_df = df_cleaned.randomSplit([0.7, 0.3], seed=42)


# COMMAND ----------

# DBTITLE 1,Good to review data before starting ML
# Display training Data
display(train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Transformation and feature engineering

# COMMAND ----------

# MAGIC %md
# MAGIC #### Summary of Data Transformations:
# MAGIC
# MAGIC **Transformations**:
# MAGIC - Categorical Encoding: Converting categorical variables to numerical form.
# MAGIC - Feature Scaling: Ensuring that numerical features are on a similar scale.
# MAGIC - Feature Engineering: Creating additional features that might better represent relationships in the data.

# COMMAND ----------

# DBTITLE 1,Converting string variables types to Integer types
indexer_label = StringIndexer(inputCol="status", outputCol="label")
indexed_df = indexer_label.fit(df_cleaned).transform(df_cleaned)
# Initialize base components
indexer = StringIndexer(inputCols=["loan_limit", "gender", "approv_in_adv", "loan_type", "loan_purpose",
                                   "credit_worthiness", "business_or_commercial", "neg_ammortization",
                                   "interest_only", "lump_sum_payment", "construction_type", "occupancy_type",
                                   "secured_by", "credit_type", "submission_of_application","region", "security_type",
                                   "high_interest_rate", "senior_age"],
                        outputCols=["loan_limit_index", "gender_index", "approv_in_adv_index",
                                    "loan_type_index", "loan_purpose_index",
                                    "credit_worthiness_index", "business_or_commercial_index",
                                    "neg_ammortization_index","interest_only_index", "lump_sum_payment_index",
                                    "construction_type_index", "occupancy_type_index","secured_by_index", "credit_type_index",
                                    "submission_of_application_index","region_index", "security_type_index",
                                    "high_interest_rate_index", "senior_age_index"])

assembler = VectorAssembler(inputCols=["year", "loan_amount", "rate_of_interest",       
                                       "interest_rate_spread", "upfront_charges",
                                       "term", "property_value", "income", "credit_score", "ltv", "dtir1",
                                       "loan_limit_index", "gender_index", "approv_in_adv_index", "loan_type_index",
                                       "loan_purpose_index", "credit_worthiness_index", "business_or_commercial_index",
                                       "neg_ammortization_index", "interest_only_index", "lump_sum_payment_index",
                                       "construction_type_index", "occupancy_type_index", "secured_by_index", "credit_type_index",
                                       "submission_of_application_index","region_index", "security_type_index", 
                                       "high_interest_rate_index", "senior_age_index"],
                          outputCol="features")



# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Modeling

# COMMAND ----------

# DBTITLE 1,Multiple models from a common pipeline

# Pipeline basic to be shared across model fitting and testing
pipeline = Pipeline(stages=[])  # Must initialize with empty list!

# Initialize models
lr = LogisticRegression(maxIter=10, family='multinomial')
nb = NaiveBayes(modelType='gaussian')
svm = LinearSVC(labelCol="label", featuresCol="features")
rf = RandomForestClassifier(numTrees=50)

# Define parameter grids for each model
regParam = [0.01, 0.1]
elasticNetParam = [0.0, 0.5]
smoothing = [0.1, 1.0]
numTrees = [10, 20]
maxDepth = [5, 10]
svmRegParam = [0.01, 0.1]
svmMaxIter = [10, 20]

# Define parameter grids with pipelines
paramgrid_lr = ParamGridBuilder()\
    .baseOn({pipeline.stages: [indexer,indexer_label, assembler, lr]})\
    .addGrid(lr.regParam, regParam)\
    .addGrid(lr.elasticNetParam, elasticNetParam)\
    .build()

paramgrid_nb = ParamGridBuilder()\
    .baseOn({pipeline.stages: [indexer,indexer_label, assembler, nb]})\
    .addGrid(nb.smoothing, smoothing)\
    .build()

paramgrid_svm = ParamGridBuilder()\
    .baseOn({pipeline.stages: [indexer,indexer_label, assembler, svm]})\
    .addGrid(svm.regParam, svmRegParam)\
    .addGrid(svm.maxIter, svmMaxIter)\
    .build()

paramgrid_rf = ParamGridBuilder()\
    .baseOn({pipeline.stages: [indexer,indexer_label, assembler, rf]})\
    .addGrid(rf.numTrees, numTrees)\
    .addGrid(rf.maxDepth, maxDepth)\
    .build()

# Combine all parameter grids into one
paramGrid = paramgrid_lr + paramgrid_nb + paramgrid_svm + paramgrid_rf

# Evaluator Creation
evaluator = MulticlassClassificationEvaluator(metricName='f1')

# COMMAND ----------

# MAGIC %md
# MAGIC ## All 4 Classification Model Evaluations
# MAGIC   - Logistic Regression
# MAGIC   - Random Forest Classification
# MAGIC   - SVM Model Classification
# MAGIC   - Naive Baye's Model Classification

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model-1: Logistic Model Evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Summary

# COMMAND ----------

# DBTITLE 1,Logistic Model Summary
test_stages_lr = [indexer,indexer_label, assembler,lr] # check a model (place any model here to test it; see config below)
lr_pipe = Pipeline(stages=test_stages_lr)
lr_fitted = lr_pipe.fit(train_df)
lr_results = lr_fitted.transform(test_df)
predictions_lr = lr_results.select('label', 'prediction')

print(f"The Logistic Model Performance is:")
# Summary metrics
accuracy = evaluator.evaluate(predictions_lr, {evaluator.metricName: "accuracy"})
precision = evaluator.evaluate(predictions_lr, {evaluator.metricName: "precisionByLabel"})
recall = evaluator.evaluate(predictions_lr, {evaluator.metricName: "recallByLabel"})
f1_score = evaluator.evaluate(predictions_lr, {evaluator.metricName: "f1"})

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Logistic Model Summarization
# MAGIC   The Logistic Regression model exhibits exceptional performance across all evaluated metrics:
# MAGIC
# MAGIC   - Accuracy: 99.98%
# MAGIC   - Precision: 99.98%
# MAGIC   - Recall: 100%
# MAGIC   - F1 Score: 99.98%
# MAGIC   
# MAGIC These results highlight the model's ability to make highly accurate predictions with perfect recall and a balanced F1 Score, indicating its reliability and effectiveness in classification tasks.
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model quality visualization

# COMMAND ----------

# DBTITLE 1,Logistic model quality
from pyspark.mllib.evaluation import MulticlassMetrics

# Confusion matrix from labels and predictions
metrics = MulticlassMetrics(predictions_lr.select('label', 'prediction').rdd.map(tuple))
confusion_matrix = metrics.confusionMatrix().toArray()

# Pandas DataFrame from Spark confusion matrix
cnf_matrix = pd.DataFrame(confusion_matrix)

plt.figure(figsize = (10,7))
p = sns.heatmap(cnf_matrix/np.sum(cnf_matrix), annot=True, fmt=".2%", linewidth=0.5, annot_kws={'fontsize':10}, cmap='RdBu')
p.set(xlabel='Predicted', ylabel='Actual', title='Logistic Confusion Matrix');

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model-2: NaiveBayes Model Evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Summary

# COMMAND ----------

# DBTITLE 1,NaiveBayes Model Summary
test_stages_nb = [indexer,indexer_label, assembler,nb] # check a model (place any model here to test it; see config below)
nb_pipe = Pipeline(stages=test_stages_nb)
nb_fitted = nb_pipe.fit(train_df)
nb_results = nb_fitted.transform(test_df)
predictions_nb = nb_results.select('label', 'prediction')
# Summary metrics
accuracy = evaluator.evaluate(predictions_nb, {evaluator.metricName: "accuracy"})
precision = evaluator.evaluate(predictions_nb, {evaluator.metricName: "precisionByLabel"})
recall = evaluator.evaluate(predictions_nb, {evaluator.metricName: "recallByLabel"})
f1_score = evaluator.evaluate(predictions_nb, {evaluator.metricName: "f1"})

print(f"The NaiveBayes Model Performance is {metric_nb}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")


# COMMAND ----------

# MAGIC %md
# MAGIC ##### NaiveBayes Model Summarization
# MAGIC   The Naive Bayes model demonstrates strong performance, though slightly less than the Logistic Regression model:
# MAGIC
# MAGIC   - Accuracy: 92.20%
# MAGIC   - Precision: 99.67%
# MAGIC   - Recall: 89.88%
# MAGIC   - F1 Score: 92.49%
# MAGIC   
# MAGIC These metrics suggest the model excels in precision, accurately identifying positive cases, but has slightly lower recall, indicating room for improvement in capturing all relevant instances. The balanced F1 Score reflects overall robust performance.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Quality Visualizations

# COMMAND ----------

# DBTITLE 1,Naive Model Quality
# Confusion matrix from labels and predictions
metrics = MulticlassMetrics(predictions_nb.select('label', 'prediction').rdd.map(tuple))
confusion_matrix = metrics.confusionMatrix().toArray()

# Pandas DataFrame from Spark confusion matrix
cnf_matrix = pd.DataFrame(confusion_matrix)

plt.figure(figsize = (10,7))
p = sns.heatmap(cnf_matrix/np.sum(cnf_matrix), annot=True, fmt=".2%", linewidth=0.5, annot_kws={'fontsize':10}, cmap='RdBu')

p.set(xlabel='Predicted', ylabel='Actual', title='NaiveBayes Confusion Matrix');

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model-3: Support Vector Machine Model

# COMMAND ----------

# DBTITLE 1,LinearSVC Model
test_stages_svm = [indexer,indexer_label, assembler,svm] # check a model (place any model here to test it; see config below)
svm_pipe = Pipeline(stages=test_stages_svm)
svm_fitted = svm_pipe.fit(train_df)
svm_results = svm_fitted.transform(test_df)
predictions_svm = svm_results.select('label', 'prediction')

# Summary metrics
accuracy = evaluator.evaluate(predictions_svm, {evaluator.metricName: "accuracy"})
precision = evaluator.evaluate(predictions_svm, {evaluator.metricName: "precisionByLabel"})
recall = evaluator.evaluate(predictions_svm, {evaluator.metricName: "recallByLabel"})
f1_score = evaluator.evaluate(predictions_svm, {evaluator.metricName: "f1"})

print(f"The Support Vector Machine Model Performance is:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Support Vector Machine (SVM) Model Performance
# MAGIC
# MAGIC The Support Vector Machine (SVM) model demonstrates perfect performance across all evaluation metrics:
# MAGIC
# MAGIC - **Accuracy**: 100%  
# MAGIC - **Precision**: 100%  
# MAGIC - **Recall**: 100%  
# MAGIC - **F1 Score**: 100%  
# MAGIC
# MAGIC These results indicate that the SVM model flawlessly classifies all instances in the dataset. While this reflects exceptional performance.

# COMMAND ----------

# DBTITLE 1,LinearSVC
from pyspark.mllib.evaluation import MulticlassMetrics

# Confusion matrix from labels and predictions
metrics = MulticlassMetrics(predictions_svm.select('label', 'prediction').rdd.map(tuple))
confusion_matrix = metrics.confusionMatrix().toArray()

# Pandas DataFrame from Spark confusion matrix
cnf_matrix = pd.DataFrame(confusion_matrix)

plt.figure(figsize = (10,7))
p = sns.heatmap(cnf_matrix/np.sum(cnf_matrix), annot=True, fmt=".2%", linewidth=0.5, annot_kws={'fontsize':10}, cmap='RdBu')
p.set(xlabel='Predicted', ylabel='Actual', title='LinearSVC Confusion Matrix');

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model-4: RandomForestClassifier Model Evaluation

# COMMAND ----------

# DBTITLE 1,RandomForestClassifier Model
test_stages_rf = [indexer,indexer_label, assembler,svm] 
rf_pipe = Pipeline(stages=test_stages_rf)
rf_fitted = rf_pipe.fit(train_df)
rf_results = rf_fitted.transform(test_df)
predictions_rf = rf_results.select('label', 'prediction')

accuracy = evaluator.evaluate(predictions_rf, {evaluator.metricName: "accuracy"})
precision = evaluator.evaluate(predictions_rf, {evaluator.metricName: "precisionByLabel"})
recall = evaluator.evaluate(predictions_rf, {evaluator.metricName: "recallByLabel"})
f1_score = evaluator.evaluate(predictions_rf, {evaluator.metricName: "f1"})

print(f"The RandomForestClassifier Model Performance is:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Random Forest Classifier Model Performance
# MAGIC
# MAGIC The Random Forest Classifier model demonstrates perfect performance across all evaluation metrics:
# MAGIC
# MAGIC - **Accuracy**: 100%  
# MAGIC - **Precision**: 100%  
# MAGIC - **Recall**: 100%  
# MAGIC - **F1 Score**: 100%  
# MAGIC
# MAGIC These results indicate that the model achieves flawless classification on the dataset.
# MAGIC

# COMMAND ----------

# DBTITLE 1,RandomForestClassifier
from pyspark.mllib.evaluation import MulticlassMetrics
# Confusion matrix from labels and predictions
metrics = MulticlassMetrics(predictions_rf.select('label', 'prediction').rdd.map(tuple))
confusion_matrix = metrics.confusionMatrix().toArray()

# Pandas DataFrame from Spark confusion matrix
cnf_matrix = pd.DataFrame(confusion_matrix)

plt.figure(figsize = (10,7))
p = sns.heatmap(cnf_matrix/np.sum(cnf_matrix), annot=True, fmt=".2%", linewidth=0.5, annot_kws={'fontsize':10}, cmap='RdBu')

p.set(xlabel='Predicted', ylabel='Actual', title='RandomForestClassifier Confusion Matrix');

# COMMAND ----------

# MAGIC %md
# MAGIC ## CrossValidation on 4 Models
# MAGIC   - multi-model pipeline

# COMMAND ----------

# DBTITLE 1,Run all models in the parameterized pipelines
# The common empty pipeline
# The common evaluator
# The paramGrid, which includes models and their pipelines
cv = CrossValidator()\
      .setEstimator(pipeline)\
      .setEvaluator(evaluator)\
      .setEstimatorParamMaps(paramGrid)\
      .setNumFolds(3) \
      .setParallelism(4)


fitted_grid = cv.fit(train_df)
print(f"model averages: {fitted_grid.avgMetrics}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Simple summary of models

# COMMAND ----------

# DBTITLE 1,Simple summary of models
num_models = len(fitted_grid.getEstimatorParamMaps())
print(f"Ran {num_models} models")
print(f"Model metrics are: {fitted_grid.avgMetrics}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate models (from multi-model pipeline)

# COMMAND ----------

# DBTITLE 1,Best & Worst model
print("Best Model with metrics:")
print(fitted_grid.getEstimatorParamMaps()[ np.argmax(fitted_grid.avgMetrics) ])
print()  # This adds a blank line between the outputs
print("Worst Model with metrics:")
print (fitted_grid.getEstimatorParamMaps()[ np.argmin(fitted_grid.avgMetrics) ])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Best and Worst Model Performance
# MAGIC
# MAGIC ##### Best Model: Random Forest Classifier
# MAGIC The best-performing model is the **Random Forest Classifier** with the following configuration:
# MAGIC
# MAGIC - **Pipeline stages**: [StringIndexer, StringIndexer, VectorAssembler, RandomForestClassifier]
# MAGIC - **Number of Trees**: 10
# MAGIC - **Max Depth**: 10
# MAGIC
# MAGIC This model achieved perfect classification metrics, demonstrating strong predictive power. The random forest’s ensemble approach of combining multiple decision trees enhances generalization and reduces overfitting.
# MAGIC
# MAGIC ##### Worst Model: Naive Bayes
# MAGIC The worst-performing model is the **Naive Bayes** classifier with the following configuration:
# MAGIC
# MAGIC - **Pipeline stages**: [StringIndexer, StringIndexer, VectorAssembler, NaiveBayes]
# MAGIC - **Smoothing parameter**: 0.1
# MAGIC
# MAGIC Despite Naive Bayes being a simple and interpretable model, it struggled in this case, likely due to its assumption of feature independence and sensitivity to imbalanced datasets, leading to lower performance compared to the Random Forest.
# MAGIC
# MAGIC ##### Model Validation and Selection
# MAGIC
# MAGIC The model validation process involved **cross-validation**, where various models were compared based on performance metrics (accuracy, precision, recall, and F1 score). The **Random Forest Classifier** emerged as the best model based on these metrics.
# MAGIC
# MAGIC While **Naive Bayes** produced a suboptimal performance, it remains a useful model in certain cases where feature independence is a valid assumption or in text classification tasks where the Naive Bayes model is commonly applied.
# MAGIC
# MAGIC ##### Model Selection
# MAGIC Based on the cross-validation metrics, the **Random Forest Classifier** was selected as the optimal model for this task. This decision is supported by its superior overall performance and ability to generalize well to unseen data.
# MAGIC
# MAGIC ##### Explanation for Unusual Model
# MAGIC - Random Forest Classifier is not an unusual model. 
# MAGIC - it is often considered one of the top choices for classification tasks due to its ensemble nature, which helps improve generalization and reduce overfitting.

# COMMAND ----------

# DBTITLE 1,Model metrics for plot
from pyspark.ml.tuning import CrossValidatorModel

def paramGrid_model_name(model):
  params = [v for v in model.values() if type(v) is not list]
  name = [v[-1] for v in model.values() if type(v) is list][0]
  name = re.match(r'([a-zA-Z]*)', str(name)).groups()[0]
  return f"{name}{params}"


def cv_metrics(cv: CrossValidatorModel):
  """
  Returns metrics and model names when CrossValidator is used to run multiple models, in parameter pipelines.
  """
  # Resulting metric and model description
  # get the metric from the CrossValidator's resulting avgMetrics
  # get the model name & params from the paramGrid
  # put them together here:
  measures = zip(cv.avgMetrics, [paramGrid_model_name(m) for m in cv.getEstimatorParamMaps()])
  metrics, model_names = zip(*measures)
  return metrics, model_names

metrics, model_names = cv_metrics(fitted_grid)
metric_name = fitted_grid.getEvaluator().getMetricName()

# COMMAND ----------

# DBTITLE 1,Graph comparision of model metrics
sns.set_context('notebook')
sns.set_style('white')
sns.set_palette("bright")

def add_metric_labels(metrics):
    for i in range(len(metrics)):
        plt.text(i, metrics[i], f"{metrics[i]:.3f}", ha = 'center', fontsize=10)

plt.figure(figsize=(10, 5))

pdf = pd.DataFrame(zip(metrics, model_names), columns=['r2', 'model'])
sns.barplot(data=pdf, x='model', y='r2').set(title="Model Metrics")
plt.xticks(rotation=80)
add_metric_labels(metrics)

# COMMAND ----------

# DBTITLE 1,summary attribute
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
from pyspark.sql import DataFrame

def feature_importance(model, predictions, schema_name='features'):
  """
  Given (non-regression) model and it's predictions, 
  Returns Pandas DataFrame of features scores, sorted by importance.
  """
  idxs = [f['idx'] for f in predictions.schema[schema_name].metadata["ml_attr"]["attrs"]['numeric']]
  names = [f['name'] for f in predictions.schema[schema_name].metadata["ml_attr"]["attrs"]['numeric']]

  scores = list(model.featureImportances.toArray())
  pdf = pd.DataFrame(list(zip(idxs, names, scores)),columns = ['index','name', 'score'])
  pdf.sort_values('score', ascending=False, inplace=True)
  return pdf


def classification_metrics(predictions: DataFrame, label='label', prediction='prediction'):
  """
  Summary metrics (ROC, PR, etc.) for given predictions.
  Returned metrics dictionary, and the confusion matrix.
  Provides F-measure using beta values: https://machinelearningmastery.com/fbeta-measure-for-machine-learning
  A smaller beta value, such as 0.5, gives more weight to precision and less to recall, 
  whereas a larger beta value, such as 2.0, gives less weight to precision and more weight to recall in the calculation of the score.
  For stocks, we are interested in precision (correctly calling positive increases)
  """
  metrics_b = BinaryClassificationMetrics(predictions.select(label, prediction).rdd.map(tuple))
  metrics = {}
  metrics['PR AUC'] = metrics_b.areaUnderPR
  metrics['ROC AUC'] = metrics_b.areaUnderROC
  metrics_m = MulticlassMetrics(predictions.select(label, prediction).rdd.map(tuple))
  metrics['F0.5 Score'] = metrics_m.fMeasure(label=1.0, beta=0.5)
  metrics['F1 Score'] = metrics_m.fMeasure(label=1.0, beta=1.0)
  metrics['F2 Score'] = metrics_m.fMeasure(label=1.0, beta=2.0)
  metrics['Recall'] = metrics_m.recall(label=1)
  metrics['Precision'] = metrics_m.precision(1)
  metrics['Accuracy'] = metrics_m.accuracy
  return metrics, metrics_m.confusionMatrix().toArray()


def precision_recall(pred_df: DataFrame, col_name):
  """
  Output: precision, recall used in evaluation function
  """
  rdd_pred = pred_df.select([col_name, 'label']).rdd
  metrics_m = MulticlassMetrics(rdd_pred)
  precision = metrics_m.precision(1)
  recall = metrics_m.recall(label=1)
  f2 = metrics_m.fMeasure(1.0, 2.0)
  f1 = metrics_m.fMeasure(1.0, 1.0)
  f05 = metrics_m.fMeasure(1.0, 0.5)
  return (precision, recall, f2, f1, f05)  

def threshold_tuning(valid_df: DataFrame):
  """
  Input: a validated df
  Output: a panda df that contains thresholds from 0-1 and associated precision/recall/f2 score
  """
  pr_results = []
  preds_new = valid_df
  preds_new = preds_new.withColumn('pred_probability', firstelement('probability'))
  thresholds = np.arange(start=0.1, stop=1.1, step=0.1)
  c = ['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10']
  i=0
  for threshold in thresholds:
    preds_new = preds_new.withColumn(c[i], F.when(preds_new["pred_probability"].cast(T.DoubleType()) >= threshold , 1.0).otherwise(0.0).cast(T.DoubleType()))
    i = i+1
  for i in range(len(thresholds)-1):                       
    precision, recall, f2, f1, f05 = precision_recall(preds_new, c[i])
    pr_results.append((thresholds[i], precision, recall, f2, f1, f05))
  pr_df = pd.DataFrame(pr_results).rename(columns={0:'Threshold',1:'Precision',2:'Recall', 3:'f2-score', 4:'f1-score', 5:'f0.5-score'})
  return pr_df  

# Function to graph first position of the dense vector probability
# Used in threshold_tuning function
firstelement = udf(lambda item:float(item[1]),T.FloatType())


class CurveMetrics(BinaryClassificationMetrics):
  """
  Helper function to plot roc curve
  """
  def __init__(self, *args):
      super(CurveMetrics, self).__init__(*args)
  def _to_list(self, rdd):
      points = []
      for row in rdd.collect():
          # Results are returned as type scala.Tuple2, 
          # which doesn't appear to have a py4j mapping
          points += [(float(row._1()), float(row._2()))]
      return points
  def get_curve(self, method):
      rdd = getattr(self._java_model, method)().toJavaRDD()
      return self._to_list(rdd)
    

def evaluate(preds_train: DataFrame, preds_valid: DataFrame, model, label='label', features='features', cm_percent=True):
  """
  Input: predicted model for train and validation set, model
  Output: PySpark DataFrame of evaluation metrics for class 0, 1 
  Confusion Matrix, PR-Curve
  """
  
  print(str(model))
  tr_metrics, _ = classification_metrics(preds_train)
  ts_metrics, confusion_matrix = classification_metrics(preds_valid)
  print(f"{'Metric': <10} {'Train': >7} {'Test': >7}")
  for key in tr_metrics.keys():
    print(f"{key: <10} {tr_metrics[key]: >7,.4f} {ts_metrics[key]: >7,.4f}")      
  
  print('                                        Validation Plots')
  fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
   
  preds_valid_pr = threshold_tuning(preds_valid)
  # plot the Precision-Recall curve  
  sns.set(font_scale=1, style='whitegrid')
  sns.lineplot(x='Recall', y='Precision', data=preds_valid_pr, label='PR Curve', ax=axes[0,0])
  axes[0,0].set_title('Precision-Recall Curve')
  axes[0,0].legend()
   
  # Used by plot ROC AUC
  rdd_valid_b = preds_valid.select(label,'probability').rdd.map(lambda row: (float(row['probability'][1]), float(row[label])))  
  # plot ROC AUC
  metrics_valid = CurveMetrics(rdd_valid_b)
  points_roc = metrics_valid.get_curve('roc')
  x_val = [x[0] for x in points_roc]
  y_val = [x[1] for x in points_roc]
  sns.lineplot(x=x_val, y=y_val, color='lightsteelblue',label='ROC AUC',ax= axes[1,0])  
  # Get the xy data from the lines so that we can shade
  l1 = axes[1,0].lines[0]
  x1 = l1.get_xydata()[:,0]
  y1 = l1.get_xydata()[:,1]
  axes[1,0].fill_between(x1,y1, color="lightblue", alpha=0.3)
  axes[1,0].set_ylim([0.1, 1])
  axes[1,0].set_xlabel('FPR (1-Specificity)')
  axes[1,0].set_ylabel('TPR (Recall)')
  axes[1,0].set_title('ROC AUC curve (Validation)')
  axes[1,0].legend()
    
  # Plot confusion matrix
  cm = confusion_matrix
  confusion_matrix = pd.DataFrame(cm)
  if cm_percent:
    sns.heatmap(confusion_matrix/np.sum(confusion_matrix), annot=True, fmt=".1%", linewidth=0.5, cmap='Blues', ax=axes[1,1])
  else:
    sns.heatmap(cnf_matrix, annot=True, fmt=",.1f", linewidth=0.5, annot_kws={'fontsize':10}, cmap='RdBu', ax=axes[1,1])  
  size = int(preds_valid.count())
  size = f'{size:,}'
  axes[1,1].set_title('Confusion Matrix - N={}'.format(size))
  axes[1,1].set_ylabel('Actual Values')
  axes[1,1].set_xlabel('Predicted Values')
  plt.show()
  
  # Features
  try:
    pdf = feature_importance(model, preds_valid, schema_name=features)
    sns.catplot(data=pdf, y='name', x='score', kind='bar', orient='h', height=7, aspect=1.7).set(title='Feature Importance');
    plt.show()
  except AttributeError:
    print("Cannot display feature importance")
    pass

# COMMAND ----------

# DBTITLE 1,Run the best model
model = fitted_grid.bestModel.stages[-1]
train_predictions = fitted_grid.bestModel.transform(train_df)
test_predictions = fitted_grid.bestModel.transform(test_df)

# COMMAND ----------

# DBTITLE 1,Display Predictions
#display(test_predictions)
display(test_predictions.select('features', 'status', 'probability', 'prediction'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Best Model Evaluation

# COMMAND ----------

# DBTITLE 1,Evaluate the best model
evaluate(train_predictions, test_predictions, model, features='features')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extra visualization: Using Tableau

# COMMAND ----------

# DBTITLE 1,# Export DataFrame to CSV
## Export DataFrame to CSV
df_cleaned.write.csv("transformed_data.csv", header=True, mode="overwrite")


# COMMAND ----------

# DBTITLE 1,Save the dataframe as datatable
# Save DataFrame as a table in Databricks to use it in tableau for extra visualizations
df_cleaned.write.saveAsTable("Loan_Aproval_data_table")
# Save DataFrame as a table in Databricks
df_cleaned.write.format("delta").mode("overwrite").saveAsTable("Loan_Aproval_table")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Link to access the tableau visulizations
# MAGIC https://public.tableau.com/app/profile/divya.vemula4127/viz/Loan-Approval-Tableau-Visualizations/Loan-Approval-group4-Visualizations?publish=yes
