import numpy as np
import pandas as pd
pd.options.display.max_rows = 20
pd.options.display.float_format = '{:,.5f}'.format
import statsmodels.formula.api as smf
import statsmodels.api as sm

def linear_regression(x, y, ci = 0.05, zero_intercept = False):
    ''' Compute linear regression between two pandas series

        Parameters
            x: pandas series
                independent variable
            y: pandas series
                dependent variable
            ci: float, default = 0.05
                confidence interval 
            force_intercept_0: boolean, False
                If True the linear regression is computed intercepting the y-axis in zero.
        Returns
            out: pandas dataframe
                dataframe with main parameters of linear regression. Slope (m), intercept (n), r2, mse and CI values
    '''
    if zero_intercept == False:
        regr = sm.OLS(endog = y, exog = x.to_frame().assign(intercept = 1)).fit()
        dfci = regr.conf_int(alpha = ci)
        m = regr.params.values[0]
        n = regr.params.values[1]
        y_pred = m*x + n
        mse = mean_squared_error(y, y_pred)
        out = pd.DataFrame({'m': [m], 'n': [n], 
                            'r2': [regr.rsquared], 'mse': [mse],
                           f'minf_ci{ci}': [dfci.iloc[1, 0]], f'msup_ci{ci}': [dfci.iloc[1, 1]],
                           f'ninf_ci{ci}': [dfci.iloc[0, 0]], f'nsup_ci{ci}': [dfci.iloc[0, 1]]})
        
    else:
        regr = sm.OLS(endog = y, exog = x).fit()
        dfci = regr.conf_int(alpha = ci)
        m = regr.params.values[0]
        y_pred = m*x
        mse = mean_squared_error(y, y_pred)
        out = pd.DataFrame({'m': [m], 
                            'r2': [regr.rsquared], 'mse': [mse],
                           f'minf_ci{ci}': [dfci.iloc[0, 0]], f'msup_ci{ci}': [dfci.iloc[0, 1]]})        
    out = out.T
    out.columns = ['values']
    
    return out
