import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import NMF
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


# Step 1: Data Gathering and Preprocessing
def gather_data():
    # Extract and integrate relevant customer data from multiple sources
    data_source_1 = pd.read_csv('data_source_1.csv')
    data_source_2 = pd.read_csv('data_source_2.csv')
    data_source_3 = pd.read_csv('data_source_3.csv')

    # Merge and ensure accuracy and consistency
    merged_data = pd.merge(data_source_1, data_source_2,
                           on='customer_id', how='inner')
    merged_data = pd.merge(merged_data, data_source_3,
                           on='customer_id', how='inner')

    # Data cleaning and preprocessing steps
    merged_data = merged_data.dropna()  # Remove rows with missing values
    merged_data['purchase_date'] = pd.to_datetime(merged_data['purchase_date'])

    # Additional preprocessing steps if necessary

    return merged_data


# Step 2: Customer Segmentation using K-means clustering
def customer_segmentation(data):
    features = ['purchase_amount', 'time_spent_on_site']

    # Standardize the features
    data_scaled = (data[features] - data[features].mean()
                   ) / data[features].std()

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=5, random_state=0)
    data['segment'] = kmeans.fit_predict(data_scaled)

    return data


# Step 3: Personalized Recommendation Engine using Matrix Factorization
def recommend_products(data):
    # Create user-item matrix
    user_item_matrix = data.pivot_table(
        index='customer_id', columns='product_id', values='purchase_amount', fill_value=0)

    # Perform Non-negative Matrix Factorization (NMF)
    nmf = NMF(n_components=10)
    user_factors = nmf.fit_transform(user_item_matrix)
    item_factors = nmf.components_

    # Generate personalized product recommendations for each customer
    recommendations = defaultdict(list)
    for i, customer_id in enumerate(user_item_matrix.index):
        customer_vector = user_factors[i]
        predicted_purchase_amounts = np.dot(customer_vector, item_factors)
        top_product_indices = np.argsort(
            predicted_purchase_amounts)[-5:]  # Select top 5 products
        recommendations[customer_id] = user_item_matrix.columns[top_product_indices].tolist()

    return recommendations


# Step 4: Evaluation and Performance Metrics
def evaluate_recommendations(actual_recommendations, predicted_recommendations):
    precision = precision_score(
        actual_recommendations, predicted_recommendations)
    recall = recall_score(actual_recommendations, predicted_recommendations)
    f1 = f1_score(actual_recommendations, predicted_recommendations)
    average_precision = average_precision_score(
        actual_recommendations, predicted_recommendations)

    return precision, recall, f1, average_precision


# Step 5: Dashboard and Visualization
def create_dashboard(data, recommendations):
    # Customer Segmentation Visualization
    sns.scatterplot(x='purchase_amount', y='time_spent_on_site',
                    hue='segment', data=data)
    plt.title('Customer Segmentation')
    plt.xlabel('Purchase Amount')
    plt.ylabel('Time Spent on Site')
    plt.show()

    # Recommendation Effectiveness Visualization
    customer_id = 'A12345'
    recommended_products = recommendations[customer_id]
    customer_data = data[data['customer_id'] == customer_id]
    customer_purchases = customer_data['product_id'].tolist()

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_colors = [
        'blue' if p in customer_purchases else 'grey' for p in recommended_products]
    ax.bar(recommended_products, np.ones(
        len(recommended_products)), color=bar_colors)
    plt.title('Recommendations for Customer {}'.format(customer_id))
    plt.xlabel('Product ID')
    plt.ylabel('Purchase')
    plt.show()


# Step 6: Integration with E-commerce Platform
def integrate_with_ecommerce(data, recommendations):
    # Integrate the Python script with the e-commerce platform's database and APIs
    # Collaborate with the dev team for smooth deployment and integration
    # Set up automated data retrieval and recommendation generation

    pass


# Main function to execute the entire pipeline
def main():
    # Step 1: Data Gathering and Preprocessing
    data = gather_data()

    # Step 2: Customer Segmentation using K-means clustering
    segmented_data = customer_segmentation(data)

    # Step 3: Personalized Recommendation Engine using Matrix Factorization
    recommendations = recommend_products(segmented_data)

    # Step 4: Evaluation and Performance Metrics
    # Actual purchases as recommendations
    actual_recommendations = data[data['is_purchased'] == 1]['product_id']
    predicted_recommendations = [
        p for recommendations in recommendations.values() for p in recommendations]
    precision, recall, f1, average_precision = evaluate_recommendations(
        actual_recommendations, predicted_recommendations)

    # Print evaluation metrics
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1-score:', f1)
    print('Average Precision:', average_precision)

    # Step 5: Dashboard and Visualization
    create_dashboard(segmented_data, recommendations)

    # Step 6: Integration with E-commerce Platform
    integrate_with_ecommerce(data, recommendations)


if __name__ == '__main__':
    main()
