"""
auto_arima_test:

Test auto-arima model by generating several synthetic examples including stationary
and non-stationary cases. Interested in order parameter d to infer stationary and
also qualitative review of near-term forecasting.

Author: Keith Kenemer

"""
import numpy as np
import matplotlib.pyplot as plt
import auto_arima_model as am
import argparse

#===========================

# set up command line parser 
def get_cmd_line_args():

   # create command line parser 
    parser = argparse.ArgumentParser()

    # setup positional arguments
    # n/a

    # setup optional arguments
    parser.add_argument("-f", "--forecast", 
                        help="perform forecasts and plot results", 
                        action="store_true")

    # parse command line
    args = parser.parse_args()
    return args


# process the command line
args = get_cmd_line_args()

#===========================

# initialize params
A = 1           # amplitude of sine
a = 0.35        # amplitude of noise
t_start = 0     # start of time internal
t_end = 60      # end of time interval
t_delta = 0.1   # time resolution
noise_mean = 0  # mean of generated noise
noise_std = 1   # std-deviation of noise
num_forecast_steps=100 # number of future timesteps to forecast

#===========================

print("Analyzing Noisy Sine...")

# Example#1 (stationary): generate noisy sine wave  
time = np.arange(t_start, t_end, t_delta);
y = A*np.sin(time)
noise = np.random.normal(noise_mean,noise_std,len(y))
yn = np.add(y,a*noise)

# Fit #1:  create auto-Arima model to analyze Example #1
arima_model = am.AutoArimaModel(start_p=1,max_p=3,start_q=1,max_q=3)
arima_model.fit(yn)
model_args = arima_model.get_params()
print(model_args)

# Forecast #1
if args.forecast:
   Ypred1,conf_int1 = arima_model.forecast(num_steps=num_forecast_steps,conf_int=True)
   time_axis = np.arange(len(yn)+len(Ypred1))

print("=====================") 


#===========================

print("Analyzing Up-Ramp...")

# Example #2 (non-stationary): (UP ramp + noisy sine)
y_up = np.add(time,yn)

# Fit #2:  create auto-Arima model to analyze Example #2
arima_model.fit(y_up)
model_args = arima_model.get_params()
print(model_args)

# Forecast #2
if args.forecast:
   Ypred2,conf_int2 = arima_model.forecast(num_steps=num_forecast_steps,conf_int=True)

print("=====================") 


#===========================

print("Analyzing Down-Ramp...")

# Example #3 (non-stationary): (DOWN ramp + noisy sine)
y_down = np.add(t_end-time,yn)

# Fit #3:  create auto-Arima model to analyze Example #3
arima_model.fit(y_down)
model_args = arima_model.get_params()
print(model_args)

# Forecast #3
if args.forecast:
   Ypred3,conf_int3 = arima_model.forecast(num_steps=num_forecast_steps,conf_int=True)

print("=====================") 


#===========================
print("Analyzing Quadratic-Ramp...")

# Example #4 (non-stationary): (Quadratic ramp + noisy sine)
y_quadratic = np.add(0.01*np.power(time,2),yn)

# Fit #4:  create auto-Arima model to analyze Example #4
arima_model.fit(y_quadratic)
model_args = arima_model.get_params()
print(model_args)

# Forecast #4
if args.forecast:
   Ypred4,conf_int4 = arima_model.forecast(num_steps=num_forecast_steps,conf_int=True)

print("=====================") 


#===========================
print("Analyzing Cubic-Ramp...")

# Example #5 (non-stationary): (Cubic ramp + noisy sine)
y_cubic = np.add((1/5000)*np.power(time,3),yn)

# Fit #5:  create auto-Arima model to analyze Example #5
arima_model.fit(y_cubic)
model_args = arima_model.get_params()
print(model_args)

# Forecast #5
if args.forecast:
   Ypred5,conf_int5 = arima_model.forecast(num_steps=num_forecast_steps,conf_int=True)

print("=====================") 


#===========================
print("Analyzing Fourth-power Ramp...")

# Example #6 (non-stationary): (Fourth-power ramp + noisy sine)
y_power4 = np.add( (1/200000)*np.power(time,4),yn)

# Fit #6:  create auto-Arima model to analyze Example #6
arima_model.fit(y_power4)
model_args = arima_model.get_params()
print(model_args)

# Forecast #6
if args.forecast:
   Ypred6,conf_int6 = arima_model.forecast(num_steps=num_forecast_steps,conf_int=True)

print("=====================") 


#===========================
# plot results
if args.forecast:

    fig, axs = plt.subplots(3, 2)

    # Example#1 results (Noisy sine)
    axs[0,0].plot( time_axis[:len(yn)], yn,'b')
    axs[0,0].plot( time_axis[len(yn):], Ypred1,'r')
    axs[0,0].fill_between( time_axis[len(yn):],
                           conf_int1[:, 0], conf_int1[:, 1],
                           alpha=0.3, color='b')
    axs[0,0].grid()
    axs[0,0].set_title('Noisy Sine')


    # Example#2 results (UP ramp)
    axs[0,1].plot( time_axis[:len(yn)], y_up,'b')
    axs[0,1].plot( time_axis[len(yn):], Ypred2,'r')
    axs[0,1].fill_between( time_axis[len(yn):],
                           conf_int2[:, 0], conf_int2[:, 1],
                           alpha=0.3, color='b')
    axs[0,1].grid()
    axs[0,1].set_title('Up Ramp')


    # Example#3 results (DN Ramp)
    axs[1,0].plot( time_axis[:len(yn)], y_down,'b')
    axs[1,0].plot( time_axis[len(yn):], Ypred3,'r')
    axs[1,0].fill_between( time_axis[len(yn):],
                           conf_int3[:, 0], conf_int3[:, 1],
                           alpha=0.3, color='b')
    axs[1,0].grid()
    axs[1,0].set_title('Down Ramp')
  

    # Example#4 results (Quadratic ramp)
    axs[1,1].plot( time_axis[:len(yn)], y_quadratic,'b')
    axs[1,1].plot( time_axis[len(yn):], Ypred4,'r')
    axs[1,1].fill_between( time_axis[len(yn):],
                           conf_int4[:, 0], conf_int4[:, 1],
                           alpha=0.3, color='b')
    axs[1,1].grid()
    axs[1,1].set_title('Quadratic Ramp')


    # Example#5 results (Cubic Ramp))
    axs[2,0].plot( time_axis[:len(yn)], y_cubic,'b')
    axs[2,0].plot( time_axis[len(yn):], Ypred5,'r')
    axs[2,0].fill_between( time_axis[len(yn):],
                           conf_int5[:, 0], conf_int5[:, 1],
                           alpha=0.3, color='b')
    axs[2,0].grid()
    axs[2,0].set_title('Cubic Ramp')


    # Example#6 results (Fourth-power Ramp)
    axs[2,1].plot( time_axis[:len(yn)], y_power4,'b')
    axs[2,1].plot( time_axis[len(yn):], Ypred6,'r')
    axs[2,1].fill_between( time_axis[len(yn):],
                           conf_int6[:, 0], conf_int6[:, 1],
                           alpha=0.3, color='b')
    axs[2,1].grid()
    axs[2,1].set_title('Fourth-Power Ramp')

    # set same x-labels
    for ax in axs.flat:
        ax.set(xlabel=' time sample', ylabel='amplitude')
 
    # x-labels only for bottom plots (to prevent collision w/titles)
    for ax in axs.flat:
        ax.label_outer()

    # show all subplots
    plt.show()


