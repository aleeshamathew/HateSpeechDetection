🧠 Sentiment Analysis Web App using Flask

This project is a Flask-based web application that performs sentiment analysis on user-provided text using a pre-trained LSTM deep learning model.
The app predicts whether a given text expresses positive, negative, or neutral sentiment.

📂 Project Structure
ABC/
│
├── templates/
│   ├── input.html          # Form page for user input
│   ├── output.html         # Result display page
│   └── welcome.html        # Landing page
│
├── app.py                  # Main Flask application
├── best_lstm_model.keras   # Trained LSTM model
├── tokenizer.pkl           # Tokenizer used for text preprocessing
├── labeled_data.csv        # Dataset used for model training
├── model.ipynb             # Jupyter notebook with model training code
├── .gitignore              # Git ignore file
└── venv/                   # Virtual environment (optional)

⚙️ Features

User-friendly web interface built with Flask and HTML.

LSTM model for text sentiment classification.

Preprocessing using saved tokenizer.pkl.

Clean architecture separating training and deployment.

Supports easy retraining and model updates.

🧩 Tech Stack

Frontend: HTML, CSS (Flask Templates)

Backend: Python (Flask)

Machine Learning: TensorFlow / Keras, NumPy, Pandas

Serialization: Pickle for tokenizer

Data: CSV dataset for labeled text samples

🚀 How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/aleeshamathew/HateSpeechDetection.git
cd HateSpeechDetection

2️⃣ Create and Activate Virtual Environment
python -m venv venv
venv\Scripts\activate        # On Windows
source venv/bin/activate     # On macOS/Linux

3️⃣ Install Dependencies
pip install -r requirements.txt


(If you don’t have a requirements.txt, you can create one using:)

pip freeze > requirements.txt

4️⃣ Run the Flask App
python app.py

5️⃣ Open in Browser

Visit → http://127.0.0.1:5000/

🧠 Model Overview

Model Type: LSTM (Long Short-Term Memory)

Framework: TensorFlow/Keras

Input: Tokenized text sequences

Output: Sentiment category (Positive / Negative / Neutral)

Files:

best_lstm_model.keras → Trained model file

tokenizer.pkl → Tokenizer for preprocessing

labeled_data.csv → Dataset used for model training

🧾 Example Workflow

Open the web app.

Enter any sentence (e.g., “I love this product!”).

The model predicts sentiment and displays the result on output.html.

🧑‍💻 Author

Aleesha Mathew

💼 BTech Graduate | Data Science & AI Enthusiast


📜 License

This project is open-source and available under the MIT License
.
