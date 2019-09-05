import pandas as pd
from cpd import CPD

if __name__ == "__main__":
	df = pd.read_csv("bitcoinity_data.csv",
                 sep=',',
                 header=0,
                 parse_dates=["Date"],
                 names=["Date", "MarketCap"],
                 skiprows=400)
	y = df["MarketCap"].values
	cpd = CPD()
	cpd.detect(y, derivative=True)
	cpd.plot(derivative=True, detrended=True)
