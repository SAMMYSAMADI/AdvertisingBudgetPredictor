from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from lightgbm import LGBMRegressor

# Load the dataset
data = pd.read_csv('Master Sheet Panthera (1).csv')

# Preprocess the Data
data = data.dropna(subset=['Result Type', 'Results'])
data = data.dropna(subset=['Conversions'])
data.fillna(0, inplace=True)
data = data[data['Marketing Channel'] != 'Unknown']
data['Category'] = data['Category'].replace('Workwear', 'Work wear')
data.columns = data.columns.str.strip()
data['Clicks_per_Impression'] = data['Clicks (all)'] / data['Impressions'].replace(0, np.nan)
data['Conversions_per_Click'] = data['Conversions'] / data['Clicks (all)'].replace(0, np.nan)
data.fillna(0, inplace=True)

# Encode categorical variables
categorical_features = ['Category', 'Marketing Channel']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Normalize numerical features
numerical_features = ['Total Sales', 'Reach', 'Impressions', 'Clicks (all)', 'Conversions', 'Clicks_per_Impression', 'Conversions_per_Click']
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Preprocess the data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split the data into training and testing sets
X = data[['Category', 'Marketing Channel', 'Total Sales', 'Reach', 'Impressions', 'Clicks (all)', 'Conversions', 'Clicks_per_Impression', 'Conversions_per_Click']]
y = data['Advertising Budget']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base models for stacking
estimators = [
    ('rf', RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=2, min_samples_leaf=1, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)),
    ('lgbm', LGBMRegressor(n_estimators=200, learning_rate=0.1, num_leaves=50, random_state=42))
]

# Define the stacking regressor with a final estimator
stacking_regressor = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0))

# Create a pipeline with preprocessing and stacking regressor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', stacking_regressor)
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Function to optimize budget for a given sales goal and category
def optimize_budget(sales_goal, category):
    category_data = data[data['Category'] == category]
    
    # Create a new DataFrame with the sales goal and other features set to their mean values for the category
    mean_values = category_data.mean()
    new_data = pd.DataFrame({
        'Category': [category],
        'Marketing Channel': mean_values['Marketing Channel'],
        'Total Sales': [sales_goal],
        'Reach': mean_values['Reach'],
        'Impressions': mean_values['Impressions'],
        'Clicks (all)': mean_values['Clicks (all)'],
        'Conversions': mean_values['Conversions'],
        'Clicks_per_Impression': mean_values['Clicks_per_Impression'],
        'Conversions_per_Click': mean_values['Conversions_per_Click']
    })
    
    # Predict the total advertising budget for the given sales goal and category
    predicted_total_budget = pipeline.predict(new_data).mean()

    # Calculate the budget ratios for each marketing channel
    total_budget = category_data['Advertising Budget'].sum()
    awareness_budget_ratio = category_data[category_data['Marketing Channel'] == 'Awareness']['Advertising Budget'].sum() / total_budget
    consideration_budget_ratio = category_data[category_data['Marketing Channel'] == 'Consideration']['Advertising Budget'].sum() / total_budget
    conversion_budget_ratio = category_data[category_data['Marketing Channel'] == 'Conversion']['Advertising Budget'].sum() / total_budget

    # Split the predicted budget into the marketing channels
    awareness_budget = awareness_budget_ratio * predicted_total_budget
    consideration_budget = consideration_budget_ratio * predicted_total_budget
    conversion_budget = conversion_budget_ratio * predicted_total_budget

    return {
        'Predicted Total Budget': predicted_total_budget,
        'Awareness Budget': awareness_budget,
        'Consideration Budget': consideration_budget,
        'Conversion Budget': conversion_budget,
        'Most Frequent Marketing Channel': category_data['Marketing Channel'].mode()[0]
    }

# Create Flask app
app = Flask(__name__)

@app.route('/')
def index():
    categories = data['Category'].unique()
    return render_template('index.html', categories=categories)

@app.route('/optimize', methods=['POST'])
def optimize():
    sales_goal = float(request.form['sales_goal'])
    category = request.form['category']
    
    optimized_budget = optimize_budget(sales_goal, category)
    
    return render_template('result.html', optimized_budget=optimized_budget)

if __name__ == '__main__':
    app.run(debug=True)