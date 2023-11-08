
def creation_data():
    import pandas as pd
    import numpy as np
    import random
    from faker import Faker
    fake = Faker()

    def generate_product_features(category):
        if category == "Electronics":
            return fake.random_element(["Wi-Fi Enabled", "Bluetooth Connectivity", "4K Ultra HD"])
        elif category == "Clothing":
            return fake.random_element(["100% Cotton", "Slim Fit", "UV Protection"])
        elif category == "Furniture":
            return fake.random_element(["Solid Wood", "Leather Upholstery", "Adjustable Shelves"])
        elif category == "Books":
            return fake.random_element(["Bestseller", "Award-Winning", "Hardcover"])
        elif category == "Toys":
            return fake.random_element(["Educational", "Interactive", "Battery-Powered"])
        else:
            return fake.random_element(["Color: Red", "Size: Large", "Weight: Light"])  # Default for unknown categories

    # Define the number of rows in the dataset
    num_rows = 5000

    # Generate synthetic data for the dataset
    data = {
        "Date": pd.date_range(start="2010-01-01", periods=num_rows, freq="D"),
        "Day_of_Week": [d.strftime("%A") for d in pd.date_range(start="2010-01-01", periods=num_rows, freq="D")],
        "Month": [d.strftime("%B") for d in pd.date_range(start="2010-01-01", periods=num_rows, freq="D")],
        "Product_ID": [fake.unique.random_int(min=1000, max=9999) for _ in range(num_rows)],
        "Product_Category": [fake.random_element(["Electronics", "Clothing", "Furniture", "Books", "Toys"]) for _ in range(num_rows)],
        "Product_Price": [round(random.uniform(10, 500), 2) for _ in range(num_rows)],
        "Product_Features": [],
        "Historical_Sales_Quantity": [random.randint(1, 100) for _ in range(num_rows)],
        "Historical_Sales_Revenue": [round(random.uniform(100, 5000), 2) for _ in range(num_rows)],
        "Current_Inventory_Level": [random.randint(0, 200) for _ in range(num_rows)],
        "Reorder_Point": [random.randint(10, 50) for _ in range(num_rows)],
        "Lead_Time": [random.randint(1, 10) for _ in range(num_rows)],
        "Promotion_Type": [fake.random_element(["Discount", "Bundle", "None"]) for _ in range(num_rows)],
        "Customer_Segmentation": [fake.random_element(["Segment1", "Segment2", "Segment3"]) for _ in range(num_rows)],
        "Economic_Indicator": [round(random.uniform(0.5, 2.5), 2) for _ in range(num_rows)],
        "Supplier_Performance": [round(random.uniform(0.1, 0.9), 2) for _ in range(num_rows)],
        "Customer_Rating": [round(random.uniform(1, 5), 2) for _ in range(num_rows)],
        "Stock_Available": [],    # Placeholder for Stock_Available
        "Sales_Quantity": [],     # Placeholder for Sales_Quantity
        "Promotion_Flag": [],     # Placeholder for Promotion_Flag
        "Sales_Revenue": []       # Placeholder for Sales_Revenue
    }

    # Create a proportional relationship for Stock_Available with Product_Price
    data["Stock_Available"] = [int(price * random.uniform(0.5, 1.5)) for price in data["Product_Price"]]

    # Create an inverse proportional relationship for Sales_Quantity with Stock_Available
    data["Sales_Quantity"] = [int(stock / 10) + random.randint(-10, 10) for stock in data["Stock_Available"]]

    # Create a relationship between Sales_Revenue and Historical_Sales_Revenue
    data["Sales_Revenue"] = [round(revenue * random.uniform(0.8, 1.2)) for revenue in data["Historical_Sales_Revenue"]]

    # Create a relationship between Current_Inventory_Level and Historical_Sales_Quantity
    data["Current_Inventory_Level"] = [inventory - quantity for inventory, quantity in zip(data["Current_Inventory_Level"], data["Historical_Sales_Quantity"])]

    # Create a relationship between Reorder_Point and Lead_Time
    data["Reorder_Point"] = [lead_time * random.uniform(0.5, 1.5) for lead_time in data["Lead_Time"]]

    # Create a relationship between Economic_Indicator and Product_Price
    data["Product_Price"] = [price * indicator for price, indicator in zip(data["Product_Price"], data["Economic_Indicator"])]

    # Create Promotion_Flag based on Promotion_Type
    data["Promotion_Flag"] = [1 if promotion_type != "None" else 0 for promotion_type in data["Promotion_Type"]]

    # Create a relationship between Customer_Rating and Sales_Revenue
    data["Sales_Revenue"] = [round(revenue * (rating / 5)) for revenue, rating in zip(data["Sales_Revenue"], data["Customer_Rating"])]

    # Create a relationship between Historical_Sales_Quantity and Customer_Rating
    data["Historical_Sales_Quantity"] = [int(100 * rating) + random.randint(-20, 20) for rating in data["Customer_Rating"]]

    # Create a relationship between Supplier_Performance and Historical_Sales_Revenue
    data["Historical_Sales_Revenue"] = [revenue * supplier_performance for revenue, supplier_performance in zip(data["Historical_Sales_Revenue"], data["Supplier_Performance"])]

    # Create a relationship between Product_Price and Product_Category
    price_multiplier = {
        "Electronics": 1.2,
        "Clothing": 1.0,
        "Furniture": 1.5,
        "Books": 0.8,
        "Toys": 1.1
    }
    data["Product_Price"] = [price * price_multiplier[category] for price, category in zip(data["Product_Price"], data["Product_Category"])]

    data["Product_Features"] = [generate_product_features(category) for category in data["Product_Category"]]

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Display the first few rows of the dataset
    print(df.head())

    # Save the dataset to a CSV file
    df.to_csv("synthetic_dataset_final.csv", index=False)

    import pandas as pd
    import matplotlib.pyplot as plt
    # Load the data into a pandas DataFrame
    data = pd.read_csv('synthetic_dataset_final.csv') 

    # Step 1: Handle Missing Data
    # Drop rows with missing values for simplicity
    data.dropna(inplace=True)

    # Step 2: Data Type Conversion
    data['Date'] = pd.to_datetime(data['Date'])
    day_mapping = {
    "Monday": 1,
    "Tuesday": 2,
    "Wednesday": 3,
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6,
    "Sunday": 7
}

    # Map day names to numbers
    data["Day_of_Week"] = data["Day_of_Week"].map(day_mapping)

    month_mapping = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12
    }

    # Map month names to numbers
    data["Month"] = data["Month"].map(month_mapping)
    data.head(10)
    # Now, the "Month" column contains numeric values from 1 to 12.
    # Step 5: Scale or Normalize Numerical Data
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    numerical_columns = ['Product_Price', 'Historical_Sales_Quantity', 'Historical_Sales_Revenue', 'Current_Inventory_Level', 'Reorder_Point', 'Lead_Time', 'Economic_Indicator', 'Supplier_Performance', 'Customer_Rating', 'Stock_Available', 'Sales_Quantity', 'Sales_Revenue']

    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    # Step 8: Normalize Date
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day_of_Week'] = data['Date'].dt.dayofweek

    # Step 9: Drop Unnecessary Columns
    columns_to_drop = ['Product_Features']  # Add more columns to drop as needed
    data.drop(columns=columns_to_drop, inplace=True)

    # Step 10: saving the file in a separate csv for proper file reference
    data.to_csv('preprocessed_data.csv', index=False) 
    


    df=pd.read_csv("preprocessed_data.csv")
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # plt.style.available
    # plt.style.use('ggplot')
    # numerical_columns = ["Product_Price", "Historical_Sales_Revenue"]

    # plt.figure(figsize=(18, 8))
    # for i, col in enumerate(numerical_columns, 1):
    #     plt.plot(1, len(numerical_columns), i)
    #     sns.histplot(df[col], bins=20, kde=True)

    # plt.title(f'Histogram of Historical Sales Revenue and Product Price',pad=40)
    # plt.tight_layout()

    # print()
    # plt.show()
    return df