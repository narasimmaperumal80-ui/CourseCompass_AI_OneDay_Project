@echo off
python train_recommendation.py
python train_demand.py
python -m streamlit run app.py
pause
