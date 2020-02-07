
import matplotlib.pyplot as plt
plt.style.use('seaborn')
#import matplotlib.patheffects as path_effects
import seaborn as sb
import numpy as np
import pandas as pd
from tabulate import tabulate


#measuring performance metrics quarterly, yearly, and overall

#metrics    1. return - OK
#               1.0. total return - OK
#               1.1. cagr - OK
#           2. volatility - OK               
#           3. sharpe - OK
#           4. sortino -OK
#           5. risk return ratio - OK
#           6. treynor - OK
#           6. max dd - OK
#           7. max dd length - OK
#           8. market beta - OK
#           9. alpha - OK
#               9.0 alpha raw - OK
#               9.1 alpha CAPM - OK
#           10. omega - OK
#           12. VaR - OK
#           13. max single period return - OK
#           14. min single period ret - OK
#           15. skewness - OK
#           16. kurtosis - OK
#           17. CDD (Conditional Draw Down): average of max 20% drawdowns - OK
#           18. CDD Duration - OK
#table - OK
#plots      1. compounded return
#           2. returns
#           3. underwater
#           4. heatmap
#           5. annual return
#


def performance_metrics(series_in, market=None, rf=None, target=None, freq='M', table=False, plots=False):
    
    # series_in is the pnl series of the asset or the strategy 
    
    periods = series_in.shape[0]# total length of the return series that feed in
    if freq=='Y':
        unit_period = 1
    elif freq=='M':
        unit_period = 12
    elif freq=='W':
        unit_period = 52
    elif freq=='D':
        unit_period = 252
    else:
        print('Please check freq argument!')
        return np.nan
          
    series = series_in.copy()
    idx = series_in.index
    
    if rf is None:
        print('rf is assumed as 0!')        
        series_excess = series.copy()
    elif type(rf)==int or type(rf)==float:
        print('rf converted to unit period in a non-compounding way')
        series_excess = series - rf/unit_period
    else:
        series_excess = series - rf
               
    series_compounded = (series+1).cumprod()
    series_excess_compounded = (series_excess+1).cumprod()
    
    ret_compounded = series_compounded.iloc[-1] - 1
    ret_excess_compounded = series_excess_compounded.iloc[-1] - 1
    cagr = (ret_compounded+1) ** (unit_period/periods) - 1
    
    volatility = series.std() * unit_period**.5
    series_negative = series.copy()
    series_negative[series_negative>0] = 0
    volatility_negative = series_negative.std() * unit_period**.5
    
    sharpe = cagr / volatility
    
    # sortinoe, ref: http://www.sunrisecapital.com/wp-content/uploads/2014/06/Futures_Mag_Sortino_0213.pdf
    sortino = cagr / volatility_negative
    
    # max dd
    
    max_dd_all = (series_compounded / series_compounded.cummax() )
    max_dd = max_dd_all.min()-1
    
    # max_dd duration
    
    max_dddur_all = max_dd_all.copy()
    max_dddur_all[max_dddur_all<1] = 0
    max_dddur_all_cumsum = max_dddur_all.cumsum()
    max_dddur_all = max_dddur_all_cumsum.value_counts()
    max_dddur = max_dddur_all.max() # this is in terms of unit period

    # risk return ratio [similar ratios; calmar, mar, sterling, burke... etc.]
    
    risk_return = cagr / (-max_dd)
    
    # Conditional drawdown  
    condition = .2
    n = int(np.round((max_dddur_all[max_dddur_all>1].shape[0]*condition)))
    conditional_dd = max_dddur_all_cumsum.groupby(max_dddur_all_cumsum).apply(lambda x: max_dd_all.loc[x.index].min()).sort_values().iloc[:n].mean() - 1
    #conditional_dd = 5
    # CDD duration
    
    conditional_dd_dur = max_dddur_all.iloc[:n].mean()
    
    # alpha and beta
    
    def alpha_beta(return_series, market):

        X = market.values.reshape(-1, 1)
        X = np.concatenate([np.ones_like(X), X], axis=1)
        b = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(return_series.values)
        return b[0], b[1]
    
    if market is None:
        alpha_raw = ret_compounded
        alpha = np.nan
        beta = np.nan
    else:
        alpha,beta = alpha_beta(series_excess, market)
        alpha_raw = ret_compounded -((market +1).cumprod().iloc[-1]-1)
        
    # treynor ratio
    
    if market is None:
        treynor = np.nan
    else:
        treynor = cagr / beta
    
    # max-min single
    
    max_single = series_in.max()
    min_single = series_in.min()
    
    # skewness -kurt
    
    skewness = series_in.skew()
    kurt = series_in.kurt()
    
    # Var
    
    VaR = series_in.quantile(.05)
    
    #omega ratio
    
    omega = cagr / (-series_negative.mean()) # need to be checked
    
    
    metrics_names = ['Compounded Total Return', 'Compounded Excess Return', 'CAGR',
                     'Annualized Volatility', 'Annualized Negative Volatility', 'Sharpe', 'Sortino',
                     'Treynor', 'Omega', 'Risk-Return', 'alpha Raw', 'alpha',
                     'beta', 'Max Drawdown', 'Conditional Drawdown (Highest 20%)',
                     'Max Drawdown Duration', 'Conditional Drawdown Duration (Longest 20%)',
                     'Maximum Single Period Return', 'Minimum Single Period Return', 'VaR (5%)', 
                     'Skewness', 'Kurtosis']
    
    metrics_values = [ret_compounded, ret_excess_compounded, cagr, volatility,
                      volatility_negative, sharpe, sortino, treynor, omega, 
                      risk_return, alpha_raw, alpha, beta, max_dd, conditional_dd,
                      max_dddur, conditional_dd_dur, max_single, min_single, VaR,
                      skewness, kurt]
    
    dict_table = dict(zip(metrics_names, metrics_values))
    
    
#-----------------------------------------------------------------------------------------------------    
    
    if table:
        print(tabulate(zip(metrics_names, metrics_values), headers=['Metrics', 'Value'], tablefmt="fancy_grid", floatfmt=".4f"))

#-----------------------------------------------------------------------------------------------------
    
    if plots:
        
        #-----------------------------------------------------------------------------------------------------

#        # plotting compounded returns
#        plt.figure()
#        series_compounded.plot(color='red', linewidth=1)
#        #plt.plot(series_compounded)
#        plt.fill_between(series_compounded.index,series_compounded, 1)
#        plt.ylabel("Compounded Returns")
#        plt.xlabel("Date")
#        plt.title("Portfolio in Time");
#        plt.grid(color='black', linestyle='--', linewidth=0.5)
        
        #-----------------------------------------------------------------------------------------------------
        
        # plotting raw returns
        plt.figure()
        plt.plot(series_in.index,series_in,color='blue',linewidth=0.5)
        plt.axhline(y=series_in.mean(), color='red', linewidth=1,linestyle='--')
        plt.ylabel("Return")
        plt.xlabel("Date")
        plt.title('Raw Return')
        plt.grid(color='black', linestyle='--', linewidth=0.5)
        
        #-----------------------------------------------------------------------------------------------------
        
        # plotting underwater figure
        
        plt.figure()
        plt.plot(max_dd_all.index,max_dd_all,color='red',linewidth=0.2)
        plt.fill_between(max_dd_all.index, max_dd_all,1)
        plt.ylabel("Return")
        plt.xlabel("Date")
        plt.title("Underwater graph of highest 5 drawdown");
        plt.grid(color='black', linestyle='--', linewidth=0.5)
        plt.show()
        
        #-----------------------------------------------------------------------------------------------------
        
        # plotting conditional max dd areas
    
        plt.figure()
        list_color=['red','blue','black','green','orange']
        cap_dd_toPlot = 5
        n_dd_toPlot = min(len(max_dddur_all),cap_dd_toPlot)
        
        for i in range(n_dd_toPlot):
            
           start = max_dddur_all_cumsum[(max_dddur_all_cumsum==max_dddur_all.index[i])].index[0]
           stop = max_dddur_all_cumsum[(max_dddur_all_cumsum==max_dddur_all.index[i])].index[-1]
           
           #plt.plot(series_compounded)
           plt.axvspan(start,stop, alpha=0.3, color=list_color[i])
           
        plt.plot(series_compounded)
        plt.show()
        
        
        #-----------------------------------------------------------------------------------------------------
        
        # plotting  returns
        fig, ax = plt.subplots()
        ax= sb.boxplot(saturation=5, fliersize=5,width=0.75,data=series,whis=1)
        ax = sb.swarmplot(data=series, color=".25")
        ax.set(xlabel='Date', ylabel='Return')
        plt.show()
        
        #-----------------------------------------------------------------------------------------------------
        
        # plotting heat map and annual returns
        
        if not freq=='Y':
            
            plt.figure()
            
            years = idx.year.unique()
            
            if freq=='M':
                secondary_period = idx.month.unique().sort_values()
            
            elif freq=='W':
                
                secondary_period_end = series_in.groupby(pd.Grouper(freq='A')).apply(lambda x: x.index.week.unique().shape[0]).max()#range(53)
                secondary_period = range(0,secondary_period_end)
                
            elif freq=='D':

                secondary_period_end = max(series_in.groupby(pd.Grouper(freq='A')).apply(lambda x: x.shape[0]).max(),252)#idx.day.unique().sort_values()
                secondary_period = range(0,secondary_period_end)
                
                
            series_grouped = series_in.groupby(series_in.index.year)
        
            ret_perPeriod = pd.concat([series_grouped.get_group(i).reset_index(drop=True) for i in years], axis=1).T
            ret_perPeriod.iloc[0]=ret_perPeriod.iloc[0].shift(ret_perPeriod.iloc[0].isna().sum()) #aligning the nan's as for the first year
            ret_perPeriod.index = years
            ret_perPeriod.columns = secondary_period
        
            plt.ylabel('Date')
            plt.xlabel('Month')
            plt.title('Return')
            #heat_map = 
            sb.heatmap(ret_perPeriod,cbar_kws={'label': 'Colorbar', 'orientation': 'horizontal'}) # ,annot=True,) 
            
            plt.show()
            
            # plot annualized
           
            annualized_perPeriod=(ret_perPeriod.T.replace(np.nan,0)+1).cumprod().iloc[-1,:]-1
            
            fig, ax = plt.subplots()
            y_pos = np.arange(len(annualized_perPeriod))
            
            ax.barh(y_pos,annualized_perPeriod*100, align='center',alpha=0.6)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(years)
            ax.invert_yaxis()  # labels read top-to-bottom
            ax.set_xlabel('Return  % ')
            ax.set_title('Annual Return')
            
            plt.show()        
              
        elif freq == 'Y':
            
            years = idx.year
            
            fig, ax = plt.subplots()
            y_pos = np.arange(len(series))
            
            ax.barh(y_pos,series*100, align='center',alpha=0.6)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(years)
            ax.invert_yaxis()  # labels read top-to-bottom
            ax.set_xlabel('Return  % ')
            ax.set_title('Annual Return')
            
            plt.show()
    

    return dict_table
