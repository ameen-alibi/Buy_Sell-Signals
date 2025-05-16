import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class Signals:
    def __init__(self, stock, start_date='2024-04-24', end_date='2025-04-24'):
        self.stockData = pd.read_csv(
            stock, parse_dates=['Date'], index_col='Date').sort_index()
        self.startDate = start_date
        self.endDate = end_date
        self.indicators = []

    def plotSignals(self, threshold=2):
        df = self.stockData

        buy_conds = [
            (df['ADX_14'] > 25) & (df['+DI'] > df['-DI']),
            (df['EMA_9'] > df['EMA_21']) & (df['EMA_21'] > df['SMA_20']),
            (df['Close'] > df['SMA_50']) & (
                df['Close'].shift(1) <= df['SMA_50'].shift(1)),
            (df['RSI_14'] > 40) & (df['RSI_14'] < 60) & (
                df['RSI_14'].diff() > 0),
            (df['Close'] <= df['LOWER_BBAND']) & (
                df['Close'].shift(1) > df['LOWER_BBAND'].shift(1))
        ]

        sell_conds = [
            (df['ADX_14'] > 25) & (df['-DI'] > df['+DI']),
            (df['EMA_9'] < df['EMA_21']) & (df['EMA_21'] < df['SMA_20']),
            (df['Close'] < df['SMA_50']) & (
                df['Close'].shift(1) >= df['SMA_50'].shift(1)),
            (df['RSI_14'] < 60) & (df['RSI_14'] > 40) & (
                df['RSI_14'].diff() < 0),
            (df['Close'] >= df['UPPER_BBAND']) & (
                df['Close'].shift(1) < df['UPPER_BBAND'].shift(1))
        ]

        df['buy_signal'] = (pd.DataFrame(buy_conds).T.sum(axis=1) >= threshold)
        df['sell_signal'] = pd.DataFrame(sell_conds).T.sum(axis=1) >= threshold

        sns.scatterplot(
            data=df[df['buy_signal']],
            x=df[df['buy_signal']].index, y='Close',
            color='lime', marker='^', s=100, label='Buy', zorder=10
        )
        sns.scatterplot(
            data=df[df['sell_signal']],
            x=df[df['sell_signal']].index, y='Close',
            color='red', marker='v', s=100, label='Sell', zorder=10
        )

    def plotIndicators(self):
        self.plotIndicator('SMA_20', label='SMA_20', color='#f41102')
        self.plotIndicator('EMA_9', label='EMA_9', color='orange')

        self.plotIndicator(
            'UPPER_BBAND', label='Higher BBand', color='purple')
        self.plotIndicator(
            'LOWER_BBAND', label='Lower BBand', color='purple')
        plt.fill_between(self.stockData.index, self.stockData['UPPER_BBAND'], self.stockData['LOWER_BBAND'],
                         color='#FFD700', alpha=0.15)

        self.plotIndicator('RSI_14', label='RSI_14', color='silver')
        self.plotIndicator('ADX_14', label='ADX_14', color='white')

    def SMA(self, span=20):
        col_name = f"SMA_{span}"
        if col_name in self.indicators:
            return -1
        self.indicators.append(col_name)
        self.stockData[col_name] = self.stockData['Close'].rolling(
            window=span).mean()
        return "SMA calculated successfully"

    def EMA(self, span=20):
        col_name = f"EMA_{span}"
        if col_name in self.indicators:
            return -1
        self.indicators.append(col_name)
        self.stockData[col_name] = self.stockData['Close'] \
            .ewm(span=span, adjust=False) \
            .mean()
        return "EMA calculated successfully"

    def BBANDS(self, alpha=2):
        if 'UPPER_BBAND' in self.indicators or 'LOWER_BBAND' in self.indicators:
            return -1
        self.indicators.append('UPPER_BBAND')
        self.SMA()
        self.STD()
        self.stockData['UPPER_BBAND'] = self.stockData['SMA_20'] + \
            self.stockData['STD']*alpha
        self.indicators.append('LOWER_BBAND')
        self.stockData['LOWER_BBAND'] = self.stockData['SMA_20'] - \
            self.stockData['STD']*alpha

    def STD(self, window=20):
        if 'STD' in self.indicators:
            return -1
        self.indicators.append('STD')
        self.stockData['STD'] = self.stockData['Close'].rolling(
            window=window).std()

    def ADX(self, period=14):
        if 'ADX' in self.indicators:
            return -1

        col_name = f"ADX_{period}"
        self.indicators.append(col_name)

        df = self.stockData

        df['h-l'] = df['High'] - df['Low']
        df['h-pc'] = (df['High'] - df['Close'].shift(1)).abs()
        df['l-pc'] = (df['Low'] - df['Close'].shift(1)).abs()

        df['TR'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        df['up_move'] = df['High'] - df['High'].shift(1)
        df['down_move'] = df['Low'].shift(1) - df['Low']

        df['+DM'] = np.where((df['up_move'] > df['down_move']) &
                             (df['up_move'] > 0),
                             df['up_move'], 0)
        df['-DM'] = np.where((df['down_move'] > df['up_move']) &
                             (df['down_move'] > 0),
                             df['down_move'], 0)

        df['ATR'] = df['TR'].ewm(com=period-1, adjust=False).mean()
        df['+DM_smooth'] = df['+DM'].ewm(com=period-1, adjust=False).mean()
        df['-DM_smooth'] = df['-DM'].ewm(com=period-1, adjust=False).mean()

        df['+DI'] = 100 * df['+DM_smooth'] / df['ATR']
        df['-DI'] = 100 * df['-DM_smooth'] / df['ATR']

        df['DX'] = 100 * (df['+DI'] - df['-DI']).abs() / \
            (df['+DI'] + df['-DI'])
        df[col_name] = df['DX'].ewm(com=period-1, adjust=False).mean()
        df[col_name].iloc[:period] = np.nan

    def RSI(self, period=14):

        col_name = f"RSI_{period}"
        self.indicators.append(col_name)

        df = self.stockData

        delta = df['Close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = pd.Series(gain, index=df.index).ewm(
            alpha=1/period, adjust=False).mean()
        avg_loss = pd.Series(loss, index=df.index).ewm(
            alpha=1/period, adjust=False).mean()

        rs = avg_gain/avg_loss
        rs.iloc[:period] = np.nan

        df[col_name] = 100-(100/(1+rs))

    def plotIndicator(self, indicator, label, color):
        if indicator not in self.indicators:
            return -1
        sns.lineplot(
            data=self.stockData[self.startDate:self.endDate], x='Date', y=indicator, label=label, color=color)

    def plotPrice(self):
        ax = sns.lineplot(
            data=self.stockData[self.startDate:self.endDate],
            x='Date',
            y='Close',
            color='deepskyblue',
            label='Closing Price'
        )
        return ax

    def get_stock_data(self):
        return self.stockData.head()

    def get_indicators(self):
        return self.indicators


if __name__ == '__main__':
    plt.style.use('dark_background')
    sns.set_theme(style="darkgrid", rc={
        'axes.facecolor': 'black',
        'figure.facecolor': 'black',
        'grid.color': '0.3',
        'grid.linestyle': '--'
    })

    signal = Signals('tesla_price.csv')
    signal.ADX()
    signal.RSI()
    signal.BBANDS()
    signal.SMA(span=50)
    signal.SMA(span=200)
    signal.EMA(span=9)
    signal.EMA(span=21)
    signal.EMA(span=50)
    signal.plotIndicators()
    signal.plotSignals()
    ax = signal.plotPrice()
    plt.fill_between(signal.stockData.index, 100, 0,
                     color='white', alpha=0.15)
    plt.xticks(rotation=45, color='white')
    plt.yticks(rotation=45, color='white')
    plt.title('TSLA inc.', color='white', fontweight='bold')
    plt.xlabel('Date', color='silver')
    plt.ylabel('Price', color='silver')
    plt.legend()
    for legend in ax.legend().get_texts():
        legend.set_color('white')
    plt.show()
