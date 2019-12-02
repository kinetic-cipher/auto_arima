"""
AutoArimaModel: 
Implements an automatic version of an Auto-Regressive Integrated Moving Average (ARIMA)
model for time series prediction. The model order (p,d,q) is automatically optimized
within limits specified by the caller.  

Author: Keith Kenemer
Note: implementation reference-
www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
"""
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm

class AutoArimaModel:

    def __init__(self,start_p=1, max_p=3, start_q=1, max_q=3):
        '''
        Initializes the ranges for p and q, which is the order
        of the AR and MA components of the model respectively. During the
        fit these will be updated to optimized values.

        Inputs:
           start_p: lowest order of p to consider (p = AR order)
           max_p: highest order of p to consider (p = AR order)
           start_q: lowest order of q to consider (p = MA order)
           max_q: highest order of q to consider (p = MA order)
        Outputs:
           n/a
        '''
        self.start_p = start_p
        self.max_p = max_p
        self.start_q = start_q
        self.max_q = max_q
        self.model = None


    def fit(self,X):
        '''
        Fit the model to the input time series data. After fitting, the
        model order (p,d,q) will be set to optimized values. Note: d = 0 implies
        the data is stationary since no differencing was required.

        Inputs:
           X: input vector of time samples (array-like)
        Outputs: 
           n/a
        ''' 
        self.model = pm.auto_arima(X, start_p=self.start_p, start_q=self.start_q,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=self.max_p, max_q=self.max_q, # maximum p and q
                      m=1,              # frequency of series (no seasonality)
                      d=None,           # let model determine 'd'
                      seasonal=False,   # no seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)


    def forecast(self,num_steps=1, conf_int=False):
        '''
        Forecast into the future using the fitted model

        Inputs:
           num_steps: number of time_steps in the future to predict (scalar)i
           conf_int:  specifes whether to return the confidence intervals 
        Outputs:
            Y: output prediction vector with length = num_steps (array-like) 
        ''' 
        if self.model == None:
            print('ERROR[forecast]: model has not been fit')
            return

        if conf_int == True:
            Y,c = self.model.predict(n_periods = num_steps, return_conf_int=True)
            return Y,c     
            
        else:
            Y = self.model.predict(n_periods = num_steps)
            return Y     


    def get_params(self):
        '''
        Returns the model params. Assumed model has been already fit.

        Inputs:
           n/a
        Outputs:
            Y: output prediction vector with length = num_steps (array-like) 
        ''' 
        if self.model == None:
            print('ERROR[get_params]: model has not been fit')
            return

        model_params = self.model.get_params()
        return model_params     

    
    def summary(self):
        '''
        Print summary of fitted model.

        Inputs:
           n/a
        Outputs:
            n/a
        ''' 
        if self.model == None:
            print('ERROR[summary]: model has not been fit')
            return

        print(self.model.summary() )


