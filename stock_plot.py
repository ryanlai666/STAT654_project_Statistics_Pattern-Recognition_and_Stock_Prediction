
import numpy as np
import pandas as pd
import talib
import matplotlib.pyplot as plt
from talib import abstract
import mplfinance as mpf
#import goo_doc_api as goo
#import yfinance as yf
#import pandas_datareader as web
import matplotlib
matplotlib.use('ps')
from matplotlib import rc
rc('text',usetex=True)
# rc('text.latex', preamble='\usepackage{color}')
import matplotlib.pyplot as plt

import datetime
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import io



# Main Function

def color_title(labels, colors, textprops = {'size':'large'}, ax = None, y = 1.013,
               precision = 10**-2):
     
    "Creates a centered title with multiple colors. Don't change axes limits afterwards."
        
    if ax == None:
        ax = plt.gca()
        
    plt.gcf().canvas.draw()
    transform = ax.transAxes # use axes coords
    
    # initial params
    xT = 0 # where the text ends in x-axis coords
    shift = 0 # where the text starts
    
    # for text objects
    text = dict()

    while (np.abs(shift - (1-xT)) > precision) and (shift <= xT) :         
        x_pos = shift 
        
        for label, col in zip(labels, colors):

            try:
                text[label].remove()
            except KeyError:
                pass
            
            text[label] = ax.text(x_pos, y, label, 
                        transform = transform, 
                        ha = 'left',
                        color = col,
                        **textprops)
            
            x_pos = text[label].get_window_extent()\
                   .transformed(transform.inverted()).x1
            
        xT = x_pos # where all text ends
        
        shift += precision/2 # increase for next iteration
      
        if x_pos > 1: # guardrail 
            break
def MACD(df, window_slow, window_fast, window_signal):
    macd = pd.DataFrame()
    macd['ema_slow'] = df['Close'].ewm(span=window_slow).mean()
    macd['ema_fast'] = df['Close'].ewm(span=window_fast).mean()
    macd['macd'] = macd['ema_slow'] - macd['ema_fast']
    macd['signal'] = macd['macd'].ewm(span=window_signal).mean()
    macd['diff'] = macd['macd'] - macd['signal']
    macd['bar_positive'] = macd['diff'].map(lambda x: x if x > 0 else 0)
    macd['bar_negative'] = macd['diff'].map(lambda x: x if x < 0 else 0)
    return macd

def color_RG(price,old_price):
    if price>= old_price:
        return (0,225,0)
    elif price< old_price:
        return (205,0,0)
    return 
def color_RG0(price):
    if price>= 0:
        return (0,225,0)
    elif price< 0:
        return (205,0,0)
    return 

def Stochastic(df, window, smooth_window):

    stochastic = pd.DataFrame()
    stochastic['%K'] = ((df['Close'] - df['Low'].rolling(window).min()) \
                        / (df['High'].rolling(window).max() - df['Low'].rolling(window).min())) * 100
    stochastic['%D'] = stochastic['%K'].rolling(smooth_window).mean()
    stochastic['%SD'] = stochastic['%D'].rolling(smooth_window).mean()
    stochastic['UL'] = 80
    stochastic['DL'] = 20
    return stochastic


def set_image_dpi_resize(image):
    """
    Rescaling image to 300dpi while resizing
    :param image: An image
    :return: A rescaled image
    """
    length_x, width_y = image.size
    factor = min(1, float(1024.0 / length_x))
    size = int(factor * length_x), int(factor * width_y)
    image_resize = image.resize(size, Image.ANTIALIAS)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='1.png')
    temp_filename = temp_file.name
    image_resize.save(temp_filename, dpi=(300, 300))
    return temp_filename

def plot_stock(stock_list):
    image_path=[]
    print(stock_list) 
    tickers =stock_list

    #= ["AAPL", "GOOG", "TSLA", "SAB.MC"]

    names = set()
    if 0:
        market_cap_data_list=[]
        for t in tickers:
            market_cap_data = web.get_quote_yahoo(t)
            names |= set(market_cap_data.columns.to_list())
            market_cap_data_list.append(market_cap_data )

        info_pd= pd.concat(market_cap_data_list)
    #(info_pd=.loc["TSLA"]["regularMarketPrice"])
    #print(info_pd.head)
    for i in  tickers:

        # download stock price data
        Stock_Name= i
        symbol = i
        tickers = symbol
        #df = yf.download(symbol, period='6mo')
        dir_path ="" # "c/D/Ryan/TAMU_class/STAT654/Project/stock_market_data/nasdaq/csv/"

        df = pd.read_csv(dir_path+str(Stock_Name)+".csv", parse_dates=['Date'], dayfirst=True)
        
        #df= pd.read_csv(dir_path+str(Stock_Name)+".csv")  # C:\D\Ryan\TAMU_class\STAT654\Project\stock_market_data\nasdaq\csv
        # current_price = web.get_quote_yahoo(tickers)["regularMarketPrice"]
        # print(current_price)
        df.set_index('Date', inplace=True)
        df=df['20220101':'20240101']
        # Add MACD as subplot
    

        macd = MACD(df, 12, 26, 9)
        stochastic = Stochastic(df, 14, 3)

        # indicator
        buy = np.where((df['Close'] > df['Open']) & (df['Close'].shift(1) < df['Open'].shift(1)), 1, np.nan) * 0.9 * df['Low']

        plots  = [
            mpf.make_addplot((macd['macd']), color='#606060', panel=2, ylabel='MACD (12,26,9)', secondary_y=False),
            mpf.make_addplot((macd['signal']), color='#1f77b4', panel=2, secondary_y=False),
            mpf.make_addplot((macd['bar_positive']), type='bar', color='#4dc790', panel=2),
            mpf.make_addplot((macd['bar_negative']), type='bar', color='#fd6b6c', panel=2),
            
            mpf.make_addplot((stochastic[['%D', '%SD', 'UL', 'DL']]),
                            ylim=[0, 100], panel=3, ylabel='Stoch (14,3)'),

            mpf.make_addplot(buy, scatter=True, markersize=100, marker=r'$\Uparrow$', color='green')

        ]



        ## mav 10 20
        #fig1=plt.figure()
        #ax1 = fig.add_subplot(111)

    #'regularMarketChange',
    #        'regularMarketChangePercent', 'regularMarketTime', 'regularMarketPrice',
    #        'regularMarketDayHigh', 'regularMarketDayRange', 'regularMarketDayLow',
    #        'regularMarketVolume', 'regularMarketPreviousClose', 


        fig_name="stock_pic/"+str(Stock_Name)+"-"+str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))+".jpg" 

        # fig1, _ = mpf.plot(sampleH4,
        #                         type='candle',
        #                         style='yahoo',
        #                         figsize =(324/mydpi,252/mydpi), 
        #                         volume=True,
        #                         axisoff=True,
        #                         returnfig=True, 
        #                         scale_padding=0.2)
        # fig1.savefig(fname,dpi=mydpi)
        # ,scale_padding=0.1
        
        mpf.plot(df, type='candle', style='yahoo', mav=(10,20), volume=True, addplot=plots, panel_ratios=(3,1,3,3), figscale=1.5,figratio=(9,16),figsize =(9/2,16/2),
                title="  \n %s" % (Stock_Name )  )# 
                # title="  \n %s \n Last    Change   DayH    DayL   preClose \n %2.2f  %2.2f%%  %2.2f  %2.2f  %2.2f\n" % (Stock_Name 
                #                                                 , info_pd.loc[i]["regularMarketPrice"]
                #                                                 , info_pd.loc[i]["regularMarketChangePercent"]
                #                                                 ,info_pd.loc[i]["regularMarketDayHigh"]
                #                                                 ,info_pd.loc[i]["regularMarketDayLow"]
                #                                                 #,info_pd.loc[i]["regularMarketVolume"]
                #                                                 ,info_pd.loc[i]['regularMarketPreviousClose'])
                #                                                 ,tight_layout=True
                
                #)
        #fig1.savefig(fig_name,dpi=mydpi)
        fig = plt.gcf()  # gcf is "get current figure"
        fig.savefig(fig_name, dpi=300)
        #mpf.savefig(fig_name, dpi=300)
        #img=plt.imread(fig_name)
        label_list = ['The ', 'Signal', ' and the ', 'Noise']
        colors = ['black', 'C0', 'black', 'C1']
        
        path =fig_name

        img = Image.open(path)
        #img = Image.open("sample_in.jpg")
        draw = ImageDraw.Draw(img)
        # font = ImageFont.truetype(<font-file>, <font-size>)
        font = ImageFont.truetype("arial.ttf", 60)
        # draw.text((x, y),"Sample Text",(r,g,b))
        col= "Last     Change     DayH      DayL    preClose"
        # assign color 
        #print(str(info_pd.loc[i]["regularMarketPrice"]))
        startx=70*2
        draw.text((startx, 42*3),col,(0,0,0),font=font)
        starty=62*3
        inval=60*4
        #draw.text((startx, starty),"{:.2f}".format(info_pd.loc[i]["regularMarketPrice"])   ,color_RG(info_pd.loc[i]["regularMarketPrice"],info_pd.loc[i]['regularMarketPreviousClose']),font=font)
        #draw.text((startx+inval*1, starty),"{:.2f}".format(info_pd.loc[i]["regularMarketChangePercent"]) +"%"  ,color_RG0(info_pd.loc[i]["regularMarketChangePercent"]),font=font)
        #draw.text((startx+inval*2, starty),"{:.2f}".format(info_pd.loc[i]["regularMarketDayHigh"])  ,color_RG(info_pd.loc[i]["regularMarketDayHigh"],info_pd.loc[i]['regularMarketPreviousClose']),font=font)
        #draw.text((startx+inval*3, starty),"{:.2f}".format(info_pd.loc[i]["regularMarketDayLow"]) , color_RG(info_pd.loc[i]["regularMarketDayLow"],info_pd.loc[i]['regularMarketPreviousClose']),font=font)
        #draw.text((startx+inval*4, starty),"{:.2f}".format(info_pd.loc[i]["regularMarketPreviousClose"]) , (0,0,0),font=font)
        #img.save(fig_name,quality='keep')
        image_path.append("/c/D/Ryan/TAMU_class/STAT654/Project/picture"+str(Stock_Name)+'-'+str(datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))+".jpg" )
    return image_path
path=plot_stock(["AAPL"])
path