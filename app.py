import streamlit as st
from streamlit_option_menu import option_menu
from analysis_and_model import analysis_and_model_page
from presentation import presentation_page

selected = option_menu(
    menu_title="Меню",
    options=["Анализ и модель", "Презентация проекта"],
    icons=["bar-chart", "slides"],
    menu_icon="cast",
    default_index=0,
    orientation="vertical"
)

if selected == "Анализ и модель":
    analysis_and_model_page()
elif selected == "Презентация проекта":
    presentation_page()