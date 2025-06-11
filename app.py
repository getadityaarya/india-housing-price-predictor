from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# --- Model and Preprocessing Loading ---
# Load the trained machine learning model
# The path is now relative, so it will work in any folder.
try:
    model = joblib.load('housing_model.pkl')
except FileNotFoundError:
    # This is a fallback for when the model isn't created yet.
    # In a real scenario, ensure the model is trained and available.
    model = None
    print("Error: 'housing_model.pkl' not found. Please train the model first using the Jupyter Notebook.")


# --- Data Definitions ---
# These lists are used to populate the dropdowns in the HTML form.
CITIES = [
    'Bangalore', 'Chennai', 'Coimbatore', 'Delhi', 'Hyderabad', 'Kochi',
    'Kolkata', 'Mumbai', 'Pune'
]
PROPERTY_TYPES = ['Apartment', 'Independent House', 'Villa']
FURNISHED_STATUS = ['Unfurnished', 'Semi-Furnished', 'Furnished']
TRANSPORT_ACCESSIBILITY = ['Low', 'Medium', 'High']
YES_NO = ['No', 'Yes']
AVAILABILITY_STATUS = ['Ready_to_Move', 'Under_Construction']


# --- Preprocessing Setup ---
# We need to encode the categorical features exactly like we did in the training notebook.
# We create encoders and fit them to our known categories.
city_encoder = LabelEncoder().fit(CITIES)
property_type_encoder = LabelEncoder().fit(PROPERTY_TYPES)
furnished_status_encoder = LabelEncoder().fit(FURNISHED_STATUS)
transport_encoder = LabelEncoder().fit(TRANSPORT_ACCESSIBILITY)
parking_encoder = LabelEncoder().fit(YES_NO)
security_encoder = LabelEncoder().fit(YES_NO)
availability_encoder = LabelEncoder().fit(AVAILABILITY_STATUS)


# --- Flask Routes ---
@app.route('/')
def home():
    """Renders the main page with the prediction form."""
    return render_template(
        'index.html',
        cities=CITIES,
        property_types=PROPERTY_TYPES,
        furnished_status=FURNISHED_STATUS,
        transport_levels=TRANSPORT_ACCESSIBILITY,
        yes_no=YES_NO,
        availability_status=AVAILABILITY_STATUS,
        prediction_text=None
    )

@app.route('/predict', methods=['POST'])
def predict():
    """Receives form data, preprocesses it, and returns the predicted price."""
    if model is None:
        return render_template('index.html', prediction_text="Error: Model not loaded.", cities=CITIES, property_types=PROPERTY_TYPES, furnished_status=FURNISHED_STATUS, transport_levels=TRANSPORT_ACCESSIBILITY, yes_no=YES_NO, availability_status=AVAILABILITY_STATUS)

    try:
        # 1. Get raw data from the form
        form_data = {
            'City': request.form['city'],
            'Property_Type': request.form['property_type'],
            'Furnished_Status': request.form['furnished_status'],
            'Public_Transport_Accessibility': request.form['transport'],
            'Parking_Space': request.form['parking'],
            'Security': request.form['security'],
            'Availability_Status': request.form['availability'],
            'BHK': int(request.form['bhk']),
            'Size_in_SqFt': float(request.form['size']),
            'Age_of_Property': int(request.form['age']),
            'Nearby_Schools': int(request.form['schools']),
            'Nearby_Hospitals': int(request.form['hospitals']),
        }

        # 2. Convert to a DataFrame
        input_df = pd.DataFrame([form_data])

        # 3. Preprocess the DataFrame using the fitted encoders
        # This transforms the string values (e.g., 'Mumbai') into numbers (e.g., 7)
        input_df['City'] = city_encoder.transform(input_df['City'])
        input_df['Property_Type'] = property_type_encoder.transform(input_df['Property_Type'])
        input_df['Furnished_Status'] = furnished_status_encoder.transform(input_df['Furnished_Status'])
        input_df['Public_Transport_Accessibility'] = transport_encoder.transform(input_df['Public_Transport_Accessibility'])
        input_df['Parking_Space'] = parking_encoder.transform(input_df['Parking_Space'])
        input_df['Security'] = security_encoder.transform(input_df['Security'])
        input_df['Availability_Status'] = availability_encoder.transform(input_df['Availability_Status'])

        # Reorder columns to match the training data exactly
        final_features = ['BHK', 'Size_in_SqFt', 'Age_of_Property', 'Nearby_Schools',
                          'Nearby_Hospitals', 'City', 'Property_Type', 'Furnished_Status',
                          'Public_Transport_Accessibility', 'Parking_Space', 'Security',
                          'Availability_Status']
        input_df = input_df[final_features]


        # 4. Make a prediction
        prediction_lakhs = model.predict(input_df)[0]
        prediction_text = f"Estimated Price: ₹{prediction_lakhs:,.2f} Lakhs"

    except Exception as e:
        # Handle potential errors during prediction
        print(f"An error occurred: {e}")
        prediction_text = "An error occurred during prediction. Please check your inputs."

    # 5. Render the page again with the prediction result
    return render_template(
        'index.html',
        prediction_text=prediction_text,
        cities=CITIES,
        property_types=PROPERTY_TYPES,
        furnished_status=FURNISHED_STATUS,
        transport_levels=TRANSPORT_ACCESSIBILITY,
        yes_no=YES_NO,
        availability_status=AVAILABILITY_STATUS,
        # To keep the form filled with the user's last inputs
        form_values=request.form
    )

if __name__ == '__main__':
    # Setting debug=True is useful for development
    app.run(debug=True)
