![FlightDelays](https://assets.planespotters.net/files/user/profile/78/c1/78c1c06b-331e-4add-a276-1e7b1ed6166f_256.png)

# Flight Delays
Flight Delays is a data science project where I use a linear regression model to predict the target variable arrival delays from the flights dataset including records for 5,000,000+ commercial airline flights in 2015[^1].

### Deployment
 - ```git clone https://github.com/p-karmelita/Flight-Delays.git```
 - ``docker build -t flights .``
 - `docker compose up`
 - If it doesn't work on Windows change the localhost from 0.0.0.0:8000 to 127.0.0.1:8000

 ### Technologies
 - Scikit-learn
 - Pandas
 - FastAPI
 - Docker
 - NBConvert
 - Python


> Author: Piotr Karmelita

[^1]: [✈️](https://www.kaggle.com/datasets/gauravmehta13/airline-flight-delays)
