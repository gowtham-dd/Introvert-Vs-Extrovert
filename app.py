# app.py  – Introvert vs Extrovert
from flask import Flask, render_template, request
import os
from IntrovertVsExtrovert.pipeline.Prediction_pipeline import PersonalityPredictor

app = Flask(__name__)

# ---------------------------------------------------------------- #
# Home page                                                        #
# ---------------------------------------------------------------- #
@app.route('/', methods=['GET'])
def homePage():
    return render_template("index.html")


# ---------------------------------------------------------------- #
# Optional route to trigger full training                          #
# ---------------------------------------------------------------- #
@app.route('/train', methods=['GET'])
def training():
    os.system("python main.py")
    return "Training Successful!"


# ---------------------------------------------------------------- #
# Prediction route                                                 #
# ---------------------------------------------------------------- #
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        try:
            # Collect form values (must match <input name="..."> in index.html)
            input_data = {
                'Time_spent_Alone':           request.form['time_spent_alone'],
                'Stage_fear':                 request.form['stage_fear'],
                'Social_event_attendance':    request.form['social_event_attendance'],
                'Going_outside':              request.form['going_outside'],
                'Drained_after_socializing':  request.form['drained_after_socializing'],
                'Friends_circle_size':        request.form['friends_circle_size'],
                'Post_frequency':             request.form['post_frequency'],
                'P2':                         request.form.get('p2', 'Unknown')
            }

            predictor  = PersonalityPredictor()
            prediction = predictor.predict(input_data)       # dict with prob & label
            prediction = predictor.predict(input_data)       # same double‑call as reference

            return render_template(
                'results.html',
                prediction=prediction                       # pass whole dict to template
            )

        except Exception as e:
            print('The Exception message is:', e)
            return 'Something went wrong. Please check your input values.'

    return render_template('index.html')


# ---------------------------------------------------------------- #
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
