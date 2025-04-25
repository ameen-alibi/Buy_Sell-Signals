import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Signals:
    def __init__(self, stock, start_date='2024-04-24', end_date='2025-04-24'):
        self.stockData = pd.read_csv(
            stock, parse_dates=['Date'], index_col='Date').sort_index()
        self.startDate = start_date
        self.endDate = end_date
        self.indicators = []

    def SMA(self, window=20):
        col_name = f"SMA_{window}"
        if col_name not in self.indicators:
            self.indicators.append(col_name)
            self.stockData[col_name] = self.stockData['Close'].rolling(
                window=window).mean()
            self.plotIndicator(col_name, label=col_name, color='orange')
            return "SMA calculated successfully"
        else:
            return -1

    def BBANDS(self, alpha=2):
        if 'UPPER_BBAND' not in self.indicators:
            self.indicators.append('UPPER_BBAND')
            self.SMA()
            self.STD()
            self.stockData['UPPER_BBAND'] = self.stockData['SMA_20'] + \
                self.stockData['STD']*alpha
            self.plotIndicator(
                'UPPER_BBAND', label='Higher BBand', color='purple')
        if 'LOWER_BBAND' not in self.indicators:
            self.indicators.append('LOWER_BBAND')
            self.stockData['LOWER_BBAND'] = self.stockData['SMA_20'] - \
                self.stockData['STD']*alpha
            self.plotIndicator(
                'LOWER_BBAND', label='Lower BBand', color='purple')

        plt.fill_between(self.stockData.index, self.stockData['UPPER_BBAND'], self.stockData['LOWER_BBAND'],
                         color='#FFD700', alpha=0.15)

    def STD(self, window=20):
        if 'STD' not in self.indicators:
            self.indicators.append('STD')
            self.stockData['STD'] = self.stockData['Close'].rolling(
                window=window).std()

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


plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={
    'axes.facecolor': 'black',
    'figure.facecolor': 'black',
    'grid.color': '0.3',
    'grid.linestyle': '--'
})

signal = Signals('tesla_price.csv')
signal.SMA()
signal.BBANDS()
ax = signal.plotPrice()
plt.xticks(rotation=45, color='white')
plt.yticks(rotation=45, color='white')
plt.title('TSLA inc.', color='white', fontweight='bold')
plt.xlabel('Date', color='silver')
plt.ylabel('Price', color='silver')
plt.legend()
for legend in ax.legend().get_texts():
    legend.set_color('white')
plt.show()

# print(signal.get_indicators())
# print(signal.get_stock_data())
