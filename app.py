from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.predict_pipeline import CustomData


application = Flask(__name__)
app = application


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomData(gender=request.form.get('gender'),
                          race_ethnicity=request.form.get('ethnicity'),
                          parental_level_of_education=request.form.get(
                              'parental_level_of_education'),
                          lunch=request.form.get('lunch'),
                          test_preparation_course=request.form.get(
                              'test_preparation_course'),
                          reading_score=request.form.get('reading_score'),
                          writing_score=request.form.get('writing_score'))

        data_df = data.get_data_as_df()
        print(data_df)

        predict_pipe = PredictPipeline()
        pred = predict_pipe.predict(features=data_df)
        return render_template('index.html', results=round(pred[0], 2))


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
