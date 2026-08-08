from flask import Flask, request, render_template
import os
import sys

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging  # استدعاء اللوجر الاحترافي
from src.exception import CustomException

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            logging.info("Received a prediction request.")
            
            # إصلاح الخطأ البرمجي في قراءة القيم
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')), # تم الإصلاح
                writing_score=float(request.form.get('writing_score'))  # تم الإصلاح
            )
            
            pred_df = data.get_data_as_data_frame()
            logging.info(f"Data gathered successfully: \n{pred_df.to_string()}")

            predict_pipeline = PredictPipeline()
            
            logging.info("Starting Prediction...")
            results = predict_pipeline.predict(pred_df)
            logging.info(f"Prediction successful. Result: {results[0]}")
            
            # إرجاع النتيجة للمستخدم
            return render_template('home.html', results=round(results[0], 2)) # تقريب النتيجة لرقمين
            
        except Exception as e:
            # في حال حدوث خطأ (مثل إدخال نصوص بدل الأرقام)، يتم تسجيله بأمان
            logging.error(f"Error occurred during prediction: {str(e)}")
            return render_template('home.html', results="Error: Please check your inputs.")

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port) 