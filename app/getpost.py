from flask import Flask, render_template

##WSGI Application

app=Flask(__name__)

@app.route("/")
def welcome():
    return "Welcome to the best flask course. This should be an amazing course"

@app.route("/index", methods=['GET'])
def index():
    return "Welcome to the index page"

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method=='POST':
        pass
    return render_template('form.html') 

if __name__=="__main__":
    app.run(debug=True, port=5001)
