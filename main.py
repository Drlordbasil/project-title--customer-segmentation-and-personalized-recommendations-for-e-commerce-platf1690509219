from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
To optimize the Python script, you can make the following improvements:

1. Import only the required functions and modules instead of importing the entire module. This will reduce the memory usage and improve the script's speed.

```python
# Optimize imports
```

2. Remove unused imports, such as `silhouette_score`, to improve script readability and performance.

3. Instead of merging the `data_source_1`, `data_source_2`, and `data_source_3` dataframes separately and then merging them again, you can use the `pd.concat` function to merge them all at once.

```python
# Merge and ensure accuracy and consistency
merged_data = pd.concat([data_source_1, data_source_2,
                        data_source_3], axis=1, join='inner')
```

4. Instead of dropping rows with missing values using `merged_data = merged_data.dropna()`, you can use the `dropna` function with the `subset` parameter to remove rows with missing values only from specific columns. Additionally, you can use the `inplace = True` parameter to modify the `merged_data` dataframe directly, without creating a new dataframe.

```python
# Drop rows with missing values in purchase_amount and time_spent_on_site columns
merged_data.dropna(
    subset=['purchase_amount', 'time_spent_on_site'], inplace=True)
```

5. Instead of applying standardization manually, you can use the `StandardScaler` from the `sklearn.preprocessing` module to standardize the features.

```python
# Optimize standardization using StandardScaler

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])
```

6. Instead of using a for loop to generate personalized product recommendations for each customer, you can use matrix multiplication to calculate the predicted purchase amounts for all customers at once.

```python
# Generate personalized product recommendations for each customer using matrix multiplication
predicted_purchase_amounts = np.dot(user_factors, item_factors)
top_product_indices = np.argsort(predicted_purchase_amounts, axis=1)[
    :, -5:]  # Select top 5 products for each customer
recommendations = {customer_id: user_item_matrix.columns[indices].tolist(
) for customer_id, indices in zip(user_item_matrix.index, top_product_indices)}
```

7. In the `create_dashboard` function, you can remove the `fig, ax` variable assignment and directly use `plt` to create the bar plot.

```python
# Create Recommendation Effectiveness Visualization - Simplify bar plot creation
bar_colors = [
    'blue' if p in customer_purchases else 'grey' for p in recommended_products]
plt.bar(recommended_products, np.ones(
    len(recommended_products)), color=bar_colors)
```

8. Consider using more efficient visualization libraries like Plotly or Bokeh for creating interactive dashboards.

By implementing these optimizations, the script will be more efficient and performant.
