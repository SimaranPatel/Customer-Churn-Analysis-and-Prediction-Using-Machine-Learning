import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load model
model_path = "data/model.pkl"
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

# Expected feature names from training (Including "Customer ID" if needed)
expected_features = [
    "Customer ID", "Product Price", "Quantity", "Total Purchase Amount", "Customer Age", "Returns",
    "Product Category_Books", "Product Category_Clothing", "Product Category_Electronics", "Product Category_Home",
    "Payment Method_Cash", "Payment Method_Credit Card", "Payment Method_PayPal",
    "Gender_Female", "Gender_Male"
]

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction_text = ""

    if request.method == "POST":
        try:
            # Get input data (Capitalize first letter)
            input_data = {
                "Customer ID": int(request.form["query1"]),  # Keeping Customer ID
                "Product Price": float(request.form["query2"]),
                "Quantity": int(request.form["query3"]),
                "Total Purchase Amount": float(request.form["query4"]),
                "Customer Age": int(request.form["query5"]),
                "Returns": int(request.form["query6"]),
                "Product Category": request.form["query7"].capitalize(),
                "Payment Method": request.form["query8"].capitalize(),
                "Gender": request.form["query9"].capitalize()
            }

            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])

            # One-hot encode categorical variables
            input_df = pd.get_dummies(input_df, columns=["Product Category", "Payment Method", "Gender"])

            # Ensure all expected columns exist (Fill missing ones with 0)
            for col in expected_features:
                if col not in input_df.columns:
                    input_df[col] = 0

            # Reorder columns to match the trained model
            input_df = input_df[expected_features]

            # Make prediction
            prediction = model.predict(input_df)[0]
            prediction_text = f"Predicted Churn: {'Yes' if prediction == 1 else 'No'}"

        except Exception as e:
            prediction_text = f"Error: {str(e)}"

    return render_template("home.html", prediction_text=prediction_text)
                            
if __name__ == "__main__":
    app.run()
