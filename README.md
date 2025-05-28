# Проект: Бинарная классификация для предиктивного обслуживания оборудования

## Описание проекта
Цель проекта — разработать модель машинного обучения, которая предсказывает, произойдет ли отказ оборудования (Target = 1) или нет (Target = 0). Результаты оформлены в виде Streamlit-приложения с интерфейсом для анализа и предсказаний.

## Датасет
Используется датасет **AI4I 2020 Predictive Maintenance Dataset** (10 000 записей, 14 признаков). Подробнее: [UCI Repository](https://archive.ics.uci.edu/dataset/601/predictive+maintenance+dataset)

## Установка и запуск
```bash
git clone <ссылка на репозиторий>
cd predictive_maintenance_project
pip install -r requirements.txt
streamlit run app.py
```

## Структура репозитория
- app.py
- analysis_and_model.py
- presentation.py
- requirements.txt
- README.md
- data/
- video/

## Возможности приложения
- Загрузка и предобработка данных
- Выбор модели
- Оценка метрик
- Предсказания
- Презентация проекта

## Видео-презентация
- https://youtu.be/a0ehSDeHT-U
