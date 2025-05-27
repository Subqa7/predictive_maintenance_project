import streamlit as st
import reveal_slides as rs

def presentation_page():
    st.title("Презентация проекта")

    presentation_markdown = """
    # Прогнозирование отказов оборудования
    ---
    ## Введение
    - Задача: предсказать отказ оборудования (Target = 1) или его отсутствие (Target = 0).
    - Используем датасет AI4I 2020 (10 000 записей, 14 признаков).
    ---
    ## Этапы работы
    1. Загрузка и предобработка данных.
    2. Обучение моделей.
    3. Оценка моделей и выбор наилучшей.
    4. Интерфейс для предсказаний.
    ---
    ## Используемые модели
    - Logistic Regression
    - Random Forest
    - XGBoost
    - SVM
    ---
    ## Streamlit-приложение
    - Основная страница: анализ, визуализация, предсказания.
    - Презентация: текущая страница с кратким обзором.
    ---
    ## Результаты
    - Accuracy до 95% (Random Forest, XGBoost).
    - ROC-AUC до 0.98.
    - Удобный веб-интерфейс для предсказания отказов.
    ---
    ## Заключение
    - Система помогает предсказывать потенциальные отказы оборудования.
    - Возможные улучшения: больше данных, продвинутая инженерия признаков.
    """

    with st.sidebar:
        st.header("Настройки")
        theme = st.selectbox("Тема", ["black", "white", "league", "beige", "sky", "night", "serif", "simple", "solarized"])
        transition = st.selectbox("Переход", ["slide", "convex", "concave", "zoom", "none"])
        height = st.number_input("Высота", value=500)

    rs.slides(
        presentation_markdown,
        height=height,
        theme=theme,
        config={"transition": transition},
    )
