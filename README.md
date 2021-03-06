# Principal Protected Floating Income Strucured Product - Linked CSI 300 Index 
- Designed/Customized two principal protected CSI 300 Index linked floating income structured products (product 1 and 2), and use the actual distirbution to do Monte Carlo Simulation, then esimate the two products' distributions of the Total return at maturity. So financial advisors and clients can see what the profitability will be like. 
- Data Source for CSI 300 Daily Price: Refinitiv, from 2005-01-04 to 2021-12-30. Attached EXCEL file.

## Product 1: 
- The product maturity is 180 natural days (N = 128) 
	- Strike price 1: (price_{T0}) * 95%
	- Strike price 2: (price_{T0}) * 105%
- Target Index: .CSI300 (China Shanghai Shenzhen 300 Index)
- Calculation Rule of income at maturity: 
	- Fixed rate of income: 0.1%
	- Floating rate of income: (0% * M1 + 4.4% * M2 + 0% * M3) / N
		- M1: From T0 to Tn, the number of the trading day as the CSI300's price-closed lower or equal to Strike price 1
		- M2: From T0 to Tn, the number of the trading day as the CSI300's price-closed higher the Strike price 1 and lower the Strike price 2
		- M3: From T0 to Tn, the number of the trading day as the CSI300's price-closed higher the  Strike price 1
		- N: From T0 to Tn, the number of trading day
- Total return at matrity = Fixed rate of income + Floating rate of income
    
## Product 2: 
- The product maturity is 180 natural days (N = 128) 
	- Strike price 1: (price_{T0}) * 95%
	- Strike price 2: (price_{T0}) * 98%
	- Strike price 3: (price_{T0}) * 102%
	- Strike price 4: (price_{T0}) * 105%
- Target Index: .CSI300 (China Shanghai Shenzhen 300 Index)
- Calculation Rule of income at maturity: 
	- Fixed rate of income: 0.1%
	- Floating rate of income: (0% * M1 + 3% * M2 + 5.9% * M3 + 3% * M4 + 0% * M5) / N
		- M1: From T0 to Tn, the number of the trading day as the CSI300's price-closed lower or equal to Strike price 1
		- M2: From T0 to Tn, the number of the trading day as the CSI300's price-closed higher the  Strike price 1 and lower the Strike price 2
		- M2: From T0 to Tn, the number of the trading day as the CSI300's price-closed higher the  Strike price 2 and lower the Strike price 3
		- M3: From T0 to Tn, the number of the trading day as the CSI300's price-closed higher the  Strike price 4
		- N: From T0 to Tn, the number of trading day
- Total return at matrity = Fixed rate of income + Floating rate of income
      
# CSI Index Overview 
![alt text](https://github.com/tomZpeng/CSI300-Index-Floating-Income-Structured-Product/blob/main/Pictures/CSI300_hist2.png?raw=ture) 
## 

![alt text](https://github.com/tomZpeng/CSI300-Index-Floating-Income-Structured-Product/blob/main/Pictures/CSI300_return_dist2.png?raw=ture)
![alt text](https://github.com/tomZpeng/CSI300-Index-Floating-Income-Structured-Product/blob/main/Pictures/CSI300_return_vol_dist2.png?raw=ture)

## 
# Products' Simulated Performance (Based on the random numbers from the Metalog distirbution)
![alt text](https://github.com/tomZpeng/CSI300-Index-Floating-Income-Structured-Product/blob/main/Pictures/CSI300_random_walk12.png?raw=ture)

## 

<br /> Product 1 total return distribution, simulated 100,000 times: 
![alt text](https://github.com/tomZpeng/CSI300-Index-Floating-Income-Structured-Product/blob/main/Pictures/product1_12.png?raw=ture)
<br />
<br /> Product 2 total return distribution, simulated 100,000 times: 
![alt text](https://github.com/tomZpeng/CSI300-Index-Floating-Income-Structured-Product/blob/main/Pictures/product2_12.png?raw=ture)

## Product 1 is better and simpler than Product 2. Total expected return for both are really low, not recommend for clients.
