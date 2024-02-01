from statsmodels.tsa.holtwinters import ExponentialSmoothing

def exp_smoothing_forecast(data):
    # create class
    model = ExponentialSmoothing(data)
    # fit model
    model_fit = model.fit()
    # make prediction
    yhat = model_fit.predict()
    return yhat