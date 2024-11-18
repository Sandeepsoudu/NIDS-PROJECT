from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score, f1_score
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

def preprocess_data(df):
    """Preprocess the dataset by handling missing values, encoding, and scaling."""
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column].fillna(df[column].mode()[0], inplace=True)
            df[column] = LabelEncoder().fit_transform(df[column])
        else:
            df[column].fillna(df[column].mean(), inplace=True)
    
    target = df.columns[-1]
    features = df.drop(columns=[target])
    
    # Scale only the features
    scaler = StandardScaler()
    features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    features[target] = df[target].values  # Add target back to the DataFrame (unscaled)
    
    return features

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return redirect(url_for('select_algorithm', filename=filename))
    return redirect(request.url)

@app.route('/select_algorithm/<filename>', methods=['GET', 'POST'])
def select_algorithm(filename):
    if request.method == 'POST':
        algorithm = request.form['algorithm']
        return redirect(url_for('run_algorithm', filename=filename, algorithm=algorithm))
    return render_template('select_algorithm.html')

@app.route('/run_algorithm/<filename>/<algorithm>')
def run_algorithm(filename, algorithm):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)
    df = preprocess_data(df)

    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.3, random_state=42)
    
    if algorithm == 'knn':
        model = KNeighborsClassifier(n_neighbors=5)
    elif algorithm == 'decision_tree':
        model = DecisionTreeClassifier(random_state=42)
    elif algorithm == 'logistic_regression':
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        return "Invalid algorithm selected"
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    return render_template('results.html', 
                           accuracy=accuracy*100, 
                           report=report, 
                           recall=recall*100,
                           f1_score=f1*100,
                           precision=precision*100,
                           confusion=confusion)

if __name__ == '__main__':
    app.run(debug=True)
