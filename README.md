# 🏠 Indian Housing Price Predictor

This project predicts housing prices based on user input using a trained machine learning model. It provides a web interface where users can enter property details and receive a real-time price estimate.

---

## 📁 Project Structure

```
india-housing-price-predictor/
├── app/
│   ├── app.py                # Flask backend
│   └── templates/
│       └── index.html        # HTML form UI
├── housing_model.pkl         # Trained model file
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 🔧 Tech Stack

- Python
- Pandas, NumPy, Scikit-learn
- Random Forest Regressor
- Flask
- HTML/CSS (Basic frontend)

---

## 🚀 How to Run

1. **Clone the Repository**

```bash
git clone https://github.com/getadityaarya/india-housing-price-predictor
cd india-housing-price-predictor
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Ensure the Model File Exists**

Make sure `housing_model.pkl` is in the project root. If not, train the model and save it.

4. **Run the Flask App**

```bash
cd app
python app.py
```

Open your browser at `http://127.0.0.1:5000` to use the app.

---

## ✨ Features

- User-friendly form to input property details
- Backend ML model predicts housing prices in Lakhs
- Real-time response displayed on the webpage

---

## 📌 Future Improvements

- Better styling with Bootstrap or Tailwind
- Deployment on Render or Railway
- Add charts or SHAP plots to explain predictions

---

## 👨‍💻 Author

Aditya Arya  
[GitHub](https://github.com/getadityaarya) | [LinkedIn](https://www.linkedin.com/in/aditya-arya-1b710b1a6/)
