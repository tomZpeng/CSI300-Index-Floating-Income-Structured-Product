# CSI300-Index-Floating-Income-Structured-Product

# Designed two structured product: Product 1 and Product 2, 
##  and use the actual distirbution to do Monte Carlo Simulation and esimate the distribution of the Total return at matrity, which the client really care 
  <br>
### Product 1:     <br>
    The product maturity is 6 months (N = 120)  <br>
      Executive price 1: (price_{T0}) * 95%  <br>
      Executive price 2: (price_{T0}) * 105%
    Target Index: .CSI300 (China Shanghai Shenzhen 300 Index)
    Calculation Rule of income at maturity:   <br>
      Fixed rate of income: 0.1%  <br>
      Floating rate of income: (0% * M1 + 4.4% * M2 + 0% * M3) / N  <br>
          M1: From T_0 to T_n, the number of the trading day as the CSI300's price-closed lower or equal to Executive price 1  <br>
          M2: From T_0 to T_n, the number of the trading day as the CSI300's price-closed over to Executive price 1 and lower the Executive price 2  <br>
          M3: From T_0 to T_n, the number of the trading day as the CSI300's price-closed over to Executive price 1  <br>
          N: From T_0 to T_n, the number of trading day  <br>
      Total return at matrity = Fixed rate of income + Floating rate of income  <br>
      <br>
### Product 2:   br>
    The product maturity is 6 months (N = 120)   <br>
      Executive price 1: (price_{T0}) * 95%  <br>
      Executive price 2: (price_{T0}) * 98%  <br>
      Executive price 2: (price_{T0}) * 102%  <br>
      Executive price 2: (price_{T0}) * 105%  <br>
    Target Index: .CSI300 (China Shanghai Shenzhen 300 Index)  <br>
    Calculation Rule of income at maturity:   <br>
      Fixed rate of income: 0.1%  <br>
      Floating rate of income: (0% * M1 + 3% * M2 + 5.9% * M3 + 3% * M4 + 0% * M5) / N  <br>
          M1: From T_0 to T_n, the number of the trading day as the CSI300's price-closed lower or equal to Executive price 1  <br>
          M2: From T_0 to T_n, the number of the trading day as the CSI300's price-closed over to Executive price 1 and lower the Executive price 2  <br>
          M2: From T_0 to T_n, the number of the trading day as the CSI300's price-closed over to Executive price 2 and lower the Executive price 3  <br>
          M3: From T_0 to T_n, the number of the trading day as the CSI300's price-closed over to Executive price 4  <br>
          N: From T_0 to T_n, the number of trading day  <br>
      Total return at matrity = Fixed rate of income + Floating rate of income  <br>
