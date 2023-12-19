from flask import Flask, request, render_template
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
import pandas as pd

abs_path = r"C:\Users\mhasa\Google Drive\Tutorial Corner\PYTH\PyCharm " \
           r"Projects\PyPlay\naiveBayes"
df_training = pd.read_csv(abs_path + r'\X_train.csv', sep='\t', names=['sms'])
naive_bayes = joblib.load(abs_path+r'\trained_nv_model.joblib')
app = Flask(__name__)
cv = CountVectorizer()
cv.fit(df_training['sms'])


@app.route('/', methods=['GET', 'POST'])
def homepage():
    if request.method == "POST":
        text = str(request.form['ta1'])
        df_text = pd.Series(text)
        vectorised_data = cv.transform(df_text)
        prediction = naive_bayes.predict(vectorised_data)[0]

        if prediction == 1:
            message = "Definitely a SPAM!"
            class_name = "makered"

        else:
            message = "Harmless HAM."
            class_name = "makegreen"

        return render_template('app.html',
                               prediction_result=message,
                               class_name=class_name)

    return render_template('app.html')


if __name__ == '__main__':
    app.run(host='localhost', port=7557)
