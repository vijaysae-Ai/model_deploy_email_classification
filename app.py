from flask import Flask,render_template,request

import pickle

with open('models/clf.pkl','rb') as f:
    model=pickle.load(f)
with open('models/cv.pkl','rb') as f:
    tokenize=pickle.load(f)


app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction',methods=['POST'])
def predict():
    if request.method == 'POST':
        email_text=request.form.get('email-content')
        tokenizer_email=tokenize.transform([email_text])
        prediction=model.predict(tokenizer_email)
        prediction=1 if prediction==1 else -1
        return render_template('index.html',prediction=prediction,text=email_text)

if __name__=='__main__':
    app.run('0.0.0.0',debug=True)


