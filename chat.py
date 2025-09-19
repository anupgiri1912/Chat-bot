import streamlit as st

st.title("Hi, Ask Me Any Qusetion?")

st.sidebar.title("Hi")
st.sidebar.selectbox("Choose Your Language",["English","Bengali","Hindi","Tamil"])


user_input = st.text_input("search")

if user_input:
    st.write(user_input)












import json
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import joblib
import pandas as pd





inventory_data = pd.DataFrame({
    'item_id' : ['M001','M002','M003'],
    'item_name' : ['Apple','Banana','Orange'],
    'stock' : ['20','50','70']
})

np.random.seed(42)
dates = pd.date_range('2023-01-01',periods=100)
rows=[]
for item in inventory_data['item_id']:
    base_usage=20 if item == 'Apple' else 10 if item == 'Banana' else 5
    for date in dates:
        usage = np.random.poisson(lam=base_usage)
        rows.append({'item_id': item, 'date': date, 'quantity_used': usage})
usage_history = pd.DataFrame(rows)

##print(usage_history)

def prepare_features(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    #print(df)
    return df

def train_models(usage_df, models_dir = "models"):
    os.makedirs(models_dir,exist_ok=True)

    usage_df['date'] = pd.to_datetime(usage_df['date'])

    daily_usage = usage_df.groupby(['item_id','date'])['quantity_used'].sum().reset_index()

    models_saved= 0 
    for item_id in daily_usage['item_id'].unique():
        item_data = daily_usage[daily_usage['item_id'] == item_id].copy()
        item_row = inventory_data[inventory_data['item_id'] == item_id]
        item_name = item_row.iloc[0]['item_name']
        item_data = prepare_features(item_data)

        X = item_data[['year','month','day','dayofweek']]
        y = item_data['quantity_used']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X,y)

        model_path = os.path.join(models_dir,f'model_{item_name}.pkl')
        joblib.dump(model,model_path)
        print(f"Model created for item {item_name} at {model_path}")
        models_saved+=1

    print(f"Trained models for {models_saved} items")

train_models(usage_history,models_dir = "models")

def get_item_id(item_name,inventory_data):
    item_row = inventory_data[inventory_data['item_name'] == item_name]
    return item_row.iloc[0]['item_id']

def predict_usage(item_name, days=7, start_date=None, model_dir='models', inventory_data = inventory_data):
    if start_date is None:
        start_date = pd.Timestamp.today()
    else:
        start_date = pd.to_datetime(start_date)
    
    model_path = os.path.join(model_dir, f'model_{item_name}.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found for {item_name}")
    
    model = joblib.load(model_path)

    days_range = pd.date_range(start=start_date, periods=days)

    features = pd.DataFrame({
        'year' : days_range.year,
        'month' : days_range.month,
        'day' : days_range.day,
        'dayofweek' : days_range.dayofweek
    })

    predictions = model.predict(features)

    total_usage = predictions.sum()
    ##print(f"Prediction for the item {item_name} over the next {days} days are {total_usage}")
    return total_usage

##predict_usage("Banana",days=7)


######query handling

def handle_query(query):
    query = query.lower()
    forecast_days = 7  # default
    item_name = None

    # Identify item name
    for name in inventory_data['item_name']:
        if name.lower() in query:
            item_name = name
            #print(item_name)
            break

    # Extract forecast days
    for word in query.split():
        if word.isdigit():
            forecast_days = int(word)
            break
    if item_name is None:
        return "Item not found in inventory."

    #index = inventory_data[inventory_data['item_name'].str.lower() == item_name.lower()].index(item_name)
    #current_stock = inventory_data['stock'][index]
    item_row = inventory_data[inventory_data['item_name'].str.lower() == item_name.lower()]
    #print(item_row)
    current_stock = item_row.iloc[0,2]
    #print(current_stock)
    
    predicted_demand = round(predict_usage(item_name, forecast_days))

    response = f"Current stock of {item_name}: {current_stock}\n"
    response += f"Predicted demand for next {forecast_days} days: {predicted_demand}\n"
    if int(current_stock) >= predicted_demand:
        response += "No restocking needed."
    else:
        restock_amount = predicted_demand - int(current_stock)
        response += f"Restocking needed: Order at least {restock_amount} more units."

    return response



# Interactive loop
print("Inventory Agent: Type your query or 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Inventory Agent: Goodbye!")
        break
    response = handle_query(user_input)
    print(f"Inventory Agent: {response}\n")


    



