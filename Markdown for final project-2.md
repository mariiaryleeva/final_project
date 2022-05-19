## Project: "Predicting prices to rent a flat in SPb"
## Content to cover:
1. Information about source data and some statistics (maybe plots, tables, images)
2. Information about your model, choosen framework, hyperparams) 
3. How to install instructions and run your app with virtual environment
4. Information about Dockerfile and describe it’s content
5. How to open the port in your remote VM
6. How to run app using docker and which port it uses

## 1. Data source and exploratory data analysis
The data is taken from Yandex.Realty classified https://realty.yandex.ru and contains real estate listings for apartments in St. Petersburg and Leningrad Oblast from 2016 till the middle of August 2018. The data contained apartments for sale and rent. In this project I examined only rent prices.

##### Main variables in the dataset
After cleaning the data from outliers, I could start looking at main variables in detail. 
- So, the destribution of the variable rent_price is not normal as we can see from the graph. Actually, it is right skewed. As we can see, most of the renting prices do not exceed 1000 000 rubles, which is totally ok for the market.

![Histogram](https://sun9-45.userapi.com/s/v1/if2/lT3DRLGnmv-TZs3njEVfAX9o1UkOJBYscnK3ejJotWNWe4S74v_VhXIMlpUHCVZzw9-9yxQRA70aX22SqzEb2kXT.jpg?size=535x291&quality=95&type=album)
- From the list of variables that could possibly influence price I want to highlight an area of a flat. In the dataset there were also given living_area and kitchen_area, however they had missings and I decided to drop them. from the scatter plot it can be clearly seen, that the bigger the area, the greater the prices for renting the flat.

![Scatter](https://sun9-75.userapi.com/s/v1/if2/DCfhmL68fn9RXZHIetn6rLGGoyaOy0eTbOS2mlLDuMFDeqcsQ5txeSzfL0ZRNyrHNoqBuoACBIFUjOaqtH2K3Da9.jpg?size=475x302&quality=95&type=album)
- Other variables that I have decided to look in detail are: floor, open_plan, number of rooms and renpvation. All of them are categorical variables and the dependencies are better shown on a box_plot rather than scatter_plot.
![Boxplots](https://sun9-78.userapi.com/s/v1/if2/vgUy-30Jd30nb86RsMLcoycsedV6j7vBHIRqGnyusk0ObG-0ZIGx2F8u-gyCUl9_ZMHRxG5wNZ7IenW7tzd0LX38.jpg?size=790x270&quality=95&type=album)

##### Then I decided to add some new variables to the dataset:
- Firs of all, i thought that we can destinguish elite appartments from normal ones because their price shoul me higher. So I created a feature called elite appartment with an area of a flat>120 sq m:
```python
rent_df_cleaned['elite_apartment'] = rent_df_cleaned['area']>=120
```
![Boxplot of elite_apartment and last_price](https://sun9-54.userapi.com/s/v1/if2/kR6tL2wuZBwoNdqsM0-MT5S3m0wlv4oSF0jbpnb59KDvD-P56htI38WS61jzR32OsbHFuDP9Gr3gR4SLL9xKrgw3.jpg?size=421x266&quality=95&type=album)

- In the dataset there were given information about the date when a listing was first day exposed and last date exposed. I Have decided to calculate for how many days the flat was exposed. Because my initial thought was  - when the flat is exposed for many days, then the owner might decrease the price for it.
```python
rent_df_cleaned['exposition_days_start']=pd.to_datetime(rent_df_cleaned['first_day_exposition'], format='%Y-%m-%d')

rent_df_cleaned['exposition_days'] = rent_df_cleaned['exposition_days_end']-rent_df_cleaned['exposition_days_start']

rent_df_cleaned['exposition_days']=rent_df_cleaned['exposition_days'].dt.days
```
![](https://sun9-21.userapi.com/s/v1/if2/mQNKPpMkqi3XncvRq4nK_chh5ExpmQtlrpcnoc9uIDiuNenp6xXmwKqyTavOv4H2MjOO3FJ1lWZb27evveRv6cFS.jpg?size=419x272&quality=95&type=album "Text to show on mouseover").

## 2. Models
2 models were built for predicting rent price. 
At first I have decided to pick following variables: based on EDA y correlation matrix: 
###### Variables for model1: [floor, open_plan, rooms, studio, area, renovation]

Firstly, I have tried Liner Regression and results were the folowing:

![](https://sun9-87.userapi.com/s/v1/if2/gZCra4JtRZV5is16XicIo1dX6itxdaCgchBf342ZiDzxDp7cuAf6YTaEvYSdVJGJP85xPTNbMXsyKxzh5hQ6FJJZ.jpg?size=218x56&quality=95&type=album)

Then I tried Random Forest model and the results were a bit better:

![](https://sun9-16.userapi.com/s/v1/if2/JRieR60x_rAZuPrANlaav04bzGlQM_HQHe6b3LQXpmyNgDbrVzPMze4apB7NKbjPSG1OMA_aXr86rdAQ8oKrTeuA.jpg?size=229x56&quality=95&type=album)

With the help of grid search I have picked the best number of estimators and the depth of the tree:

![](https://sun9-20.userapi.com/s/v1/if2/iUNSBRSblC3h9KClCbeYSbTVAcs0XvNApyXlcsf8Q4jUi9AeyRWmQH4BaccwXKAPVVVinJwq0ln2hOr2fC8NIFtz.jpg?size=806x250&quality=95&type=album)

Then I have constructed another model, adding variables which I have created on my own. And again, used Random Forest and Grid search.
###### Variables for model1: [floor, open_plan, rooms, studio, area, renovation, exposition_days, elite_apartment]

![](https://sun9-66.userapi.com/s/v1/if2/OizjY43mmDTloTlmUaSqvp6hNrIcgfNnMj76bpdwcaerQaZmNWKMx_UwXoDAP5oozDJkdQimt6Va9QB7Am7rd1Oj.jpg?size=801x254&quality=95&type=album)

## 2. Pycharm
- Firstly you need to create a VM and create a public key with which you can easily connect to it. To connect: 
```python
ssh <your_username>@<your_virtual_machine_ip_adress>
```
- Download packages which will be needed:
```python
sudo apt install python3.8-venv
python3 -m venv env
source env/bin/activate
pip install numpy
pip install flask
pip install pandas
pip install joblib
pip install sklearn
pip list
```
- Then in your local computer create a repository, where you store the models. Open Pycharm and in preferences you have to to configure your PyCharm with remote machine via ssh. In settings, look for Python Interpreter
- Code for the app.py document:
```python
from flask import Flask, request
import joblib
import numpy

MODEL_PATH1 = 'models/model1.pkl'
SCALER_X_PATH1 = 'models/scaler_x1.pkl'
SCALER_Y_PATH1 = 'models/scaler_y1.pkl'

MODEL_PATH2 = 'models/model2.pkl'
SCALER_X_PATH2 = 'models/scaler_x2.pkl'
SCALER_Y_PATH2 = 'models/scaler_y2.pkl'

app = Flask(__name__)

@app.route('/predict_price', methods=['GET'])
def predict():
    args = request.args
    model_choice = args.get('model_version', default=-1, type=int)

    floor = args.get('floor', type=int)
    open_plan = args.get('open_plan', type=int)
    rooms = args.get('rooms', type=int)
    studio = args.get('rooms', type=int)
    area = args.get("area", type=float)
    renovation = args.get('renovation', type=int)
    exposition_days = args.get('exposition_days', type=int)
    elite_apartment = args.get('exposition_days', type=int)

    if model_choice== 1:
        model1 = joblib.load(MODEL_PATH1)
        sc_x1 = joblib.load(SCALER_X_PATH1)
        sc_y1 = joblib.load(SCALER_Y_PATH1)
        
        x1 = numpy.array([floor, open_plan, rooms, studio, area, renovation]).reshape(1, -1)
        x1 = sc_x1.transform(x1)
        result1 = model1.predict(x1)
        result1 = sc_y1.inverse_transform(result1.reshape(1, -1))

        return str(result1[0][0])

    elif model_choice == 2:
        model2 = joblib.load(MODEL_PATH2)
        sc_x2 = joblib.load(SCALER_X_PATH2)
        sc_y2 = joblib.load(SCALER_Y_PATH2)

        x2 = numpy.array([floor, open_plan, rooms, studio, area, renovation, exposition_days, elite_apartment]).reshape(1, -1)
        x2 = sc_x2.transform(x2)
        result2 = model2.predict(x2)
        result2 = sc_y2.inverse_transform(result2.reshape(1, -1))

        return str(result2[0][0])

if __name__ == '__main__':
    app.run(debug=True, port=5444, host='0.0.0.0')
```
- To check that the request is ok, I used Postman. Here is what I got in results for 2 models:
1 model:
![Results in Postman for model1](https://sun9-29.userapi.com/s/v1/if2/gQ6MCt8Oz77BDe2PhZoyzcsRYNokC_ZBX0WbMkNzXO2ukgi420hYyFrcZiOAZBsyDzop2tlK_IvPXvo3WUUKK06F.jpg?size=810x576&quality=95&type=album)
2 model:
![Results in Postman for model2](https://sun9-39.userapi.com/s/v1/if2/iIbLEZ1RziIC1Li4DILFEXKO3mFIEwZ71f-YaWO8zArLrmBxwWz7e1WXFbpP4-I4yWsp3UF_7flKq3WGPfGeWIYC.jpg?size=1072x586&quality=95&type=album)

The second model estimates the apartment higher, than the fist one. But the difference in this example is 3000 rubles.

## 4. Information about a Dockerfile
```python
cat Dockerfile

from ubuntu:20.04
MAINTAINER Mariia Ryleeva
RUN apt-get update -y
COPY . /opt/final_project
WORKDIR /opt/final_project
RUN apt install -y python3-pip
RUN pip3 install -r requirements.txt
CMD python3 app.py
```
With the help of Docker file I can create a Docker image and modify the existing one if some changes appear. Everything will be copied in the working directory. Also if we write commands such as - RUN pip3 install -r requirements.txt, the needed packages will be downloaded much faster.

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>
