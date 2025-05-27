import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

# Кэшируем данные и масштабировщик
@st.cache_data
def preprocess_data(data):
    data = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
    data['Type'] = LabelEncoder().fit_transform(data['Type'])

    numerical = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    scaler = StandardScaler()
    data[numerical] = scaler.fit_transform(data[numerical])

    # Переименуем столбцы, чтобы они были совместимы с XGBoost
    data.columns = data.columns.str.replace(r'[\[\]<>]', '', regex=True).str.replace(' ', '_')
    return data, scaler

# Оценка модели
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.text("Classification Report:\n" + classification_report(y_test, y_pred))

    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    st.pyplot(plt)

# Основная страница
def analysis_and_model_page():
    st.title("Анализ данных и обучение модели")
    uploaded_file = st.file_uploader("Загрузите CSV-файл с данными", type="csv")

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        data, scaler = preprocess_data(data)

        X = data.drop(columns=['Machine_failure']).copy()
        y = data['Machine_failure']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_choice = st.selectbox("Выберите модель", ["Logistic Regression", "Random Forest", "XGBoost", "SVM"])
        if model_choice == "Logistic Regression":
            model = LogisticRegression()
        elif model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_choice == "XGBoost":
            model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
        else:
            model = SVC(kernel='linear', probability=True, random_state=42)

        model.fit(X_train, y_train)

        st.subheader("Оценка модели")
        evaluate_model(model, X_test, y_test)

        st.subheader("Предсказание отказа по новым данным")
        with st.form("prediction_form"):
            type_ = st.selectbox("Тип продукта", ["L", "M", "H"])
            air = st.number_input("Температура воздуха [K]", value=300.0)
            process = st.number_input("Температура процесса [K]", value=310.0)
            rpm = st.number_input("Скорость вращения [rpm]", value=1500)
            torque = st.number_input("Крутящий момент [Nm]", value=40.0)
            wear = st.number_input("Износ инструмента [min]", value=0)
            submit = st.form_submit_button("Предсказать")

        if submit:
            type_map = {"L": 0, "M": 1, "H": 2}
            row = pd.DataFrame({
                'Type': [type_map[type_]],
                'Air_temperature_K': [air],
                'Process_temperature_K': [process],
                'Rotational_speed_rpm': [rpm],
                'Torque_Nm': [torque],
                'Tool_wear_min': [wear]
            })
            row_scaled = scaler.transform(row)
            pred = model.predict(row_scaled)
            prob = model.predict_proba(row_scaled)[0][1]
            st.write(f"Предсказание: {'Отказ' if pred[0] == 1 else 'Нет отказа'}")
            st.write(f"Вероятность отказа: {prob:.2f}")

# Запуск приложения
if __name__ == "__main__":
    analysis_and_model_page()