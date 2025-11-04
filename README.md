
# AI-Based Medical Injury Classification and Logging System

A web-based system for real-time injury classification using a Deep Convolutional Neural Network (MobileNetV2) and automated database logging with Flask and MySQL.

This project is a complete, end-to-end application that demonstrates the integration of a deep learning model into a practical web service with a persistent database backend. It is designed to classify medical images of injuries (e.g., burns, wounds) and log every prediction for future analysis or record-keeping.

This system was developed by Abhradeep Basu (23BCE0713)

## 🚀 Demo

<img width="556" height="507" alt="image" src="https://github.com/user-attachments/assets/efcea580-68a7-42fe-9e23-ce37f09d2158" />

<img width="734" height="969" alt="image" src="https://github.com/user-attachments/assets/8230c637-29cf-4a18-a92b-951db790af0d" />

-----

## ✨ Features

  * **Real-Time Injury Classification:** Upload an image and receive an immediate classification from the trained MobileNetV2 model.
  * **Automated Database Logging:** Every successful prediction is automatically timestamped and recorded in a MySQL database.
  * **Simple Web Interface:** A clean, user-friendly UI built with Flask and HTML for easy image uploads.
  * **Scalable Architecture:** A modular three-tier design that separates the frontend, backend (API), and database.
  * **Efficient AI Model:** Uses **MobileNetV2** with transfer learning for a lightweight, fast, and accurate classification model.

-----

## 💻 Tech Stack

  * **Backend:** **Flask** (Python Web Framework)
  * **AI/ML:** **TensorFlow** & **Keras**
  * **Database:** **MySQL**
  * **Libraries:** `mysql-connector-python` (for DB connection), `Pillow` & `OpenCV` (for image preprocessing), `NumPy`

-----

## 🏛️ System Architecture

The system follows a three-tier architecture:

1.  **Frontend (Client):** A simple HTML/CSS interface served by Flask. It provides a form for the user to upload an image.
2.  **Backend (Server):** The Flask application which:
      * Provides a web interface and an API endpoint.
      * Receives the uploaded image.
      * **Preprocesses** the image (resizing to 224x224, normalization).
      * Loads the pre-trained `model.h5` file.
      * Performs **inference** using the MobileNetV2 model to get a prediction.
      * Connects to the MySQL database.
      * **Logs** the prediction details (filename, predicted class, timestamp) into the `injury_log` table.
      * Returns the prediction to the user's browser.
3.  **Database (Data Tier):** A MySQL server that stores the `injury_db` database, which contains the log of all classifications.

\`\`

-----

## 🛠️ Installation and Setup

Follow these steps to run the project locally.

### Prerequisites

  * Python 3.8+
  * MySQL Server
  * Git
  * Datsets :
    - https://www.kaggle.com/datasets/yasinpratomo/wound-dataset
    - https://www.kaggle.com/datasets/shubhambaid/skin-burn-dataset

### 1\. Clone the Repository

```bash
git clone https://github.com/your-username/your-project-repo.git
cd your-project-repo
```

### 2\. Create a Python Virtual Environment

```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3\. Install Dependencies

Create a file named `requirements.txt` with the following content:

```txt
Flask
tensorflow
mysql-connector-python
Pillow
opencv-python-headless
numpy
```

Then, install them:

```bash
pip install -r requirements.txt
```

### 4\. Set Up the MySQL Database

1.  Log in to your MySQL server (e.g., as `root`).
2.  Create the database:
    ```sql
    CREATE DATABASE injury_db;
    ```
3.  Use the new database:
    ```sql
    USE injury_db;
    ```
4.  Create the logging table:
    ```sql
    CREATE TABLE injury_log (
        id INT AUTO_INCREMENT PRIMARY KEY,
        filename VARCHAR(255) NOT NULL,
        prediction VARCHAR(100) NOT NULL,
        confidence FLOAT,
        log_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    ```

### 5\. Configure the Application

Open the main Python file (e.g., `app.py`) and update the MySQL connection details to match your setup:

```python
# Database Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'your_mysql_username'
app.config['MYSQL_PASSWORD'] = 'your_mysql_password'
app.config['MYSQL_DB'] = 'injury_db'
```

### 6\. Add the AI Model

Place your trained Keras model file (e.g., `medical_injury_model.h5`) in the project's root directory or in a `models/` folder. Ensure the `app.py` file loads the correct model path:

```python
model = load_model('medical_injury_model.h5')
```

### 7\. Run the Application

```bash
flask run
```

The application will be available at `http://127.0.0.1:5000`.

-----

## 📈 Model Details

  * **Architecture:** **MobileNetV2** (using Transfer Learning on a base pre-trained on ImageNet).
  * **Framework:** Keras with a TensorFlow backend.
  * **Dataset:** The model was trained on a custom dataset of **3435** images, categorized into **2** classes:
      * `[Class 1: Burns]`
      * `[Class 2: Wounds]`
  * **Performance:** The model achieved an accuracy of **93-95%** on the held-out test set.

-----

## 🚀 Future Work

Potential improvements for this system include:

  * **Explainable AI (XAI):** Integrate Grad-CAM or LIME to show *why* the model made a certain prediction (a heatmap).
  * **Deployment:** Deploy the application to a cloud service like AWS, Azure, or Heroku.
  * **User Authentication:** Add a user/doctor login system to protect the prediction history.
  * **Model Expansion:** Re-train the model with a larger, more diverse dataset and more specific injury sub-types.
  * **REST API:** Develop a formal REST API for predictions, separating it from the frontend.

-----

## Acknowledgments

  The usage of Generative AI ( ChatGPT and Google Gemini ) is hereby acknowlwdged in formatting and setting up the code base and also structuring the whole project.
