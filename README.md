# Stock Price Prediction

## Overview

This project predicts Google's (GOOGL) stock price with a unique, industry-aware approach. Instead of relying solely on historical price data for a single company, the model first identifies stocks within and across related industries that are most correlated with Google. This selective, multivariate method improves forecasting by incorporating the market dynamics of peer companies.

## Novelty

- **Industry-Aware Filtering:**  
  Identifies and ranks companies not only in the same industry but also in connected sectors based on Pearson correlation. This informed feature selection enhances predictive power by leveraging cross-industry relationships.
  
- **Focused Data Integration:**  
  Only the most relevant, highly correlated stocks are used as inputs to the model, resulting in a leaner yet more robust prediction framework.

## Technical Summary

- **Data Source:** Yahoo Finance via the `yfinance` library.  
- **Preprocessing:** Data normalization using MinMaxScaler; sequences built with a sliding window (14 weeks context).  
- **Model:** A stacked, bidirectional LSTM in PyTorch.
  
  ```python
  class LSTMModel(nn.Module):
      def __init__(self, num_company):
          super().__init__()
          self.lstm1 = nn.LSTM(input_size=num_company, hidden_size=128, num_layers=1,
                               batch_first=True, bidirectional=True)
          self.lstm2 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1,
                               batch_first=True, bidirectional=True)
          self.ll = nn.Linear(256, 1)

      def forward(self, x):
          x, _ = self.lstm1(x)
          x, _ = self.lstm2(x)
          x = x[:, -1, :]
          return self.ll(x).squeeze(1)
  ```

## Usage

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/stock-price-prediction-lstm.git
   cd stock-price-prediction-lstm
   ```
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Project:**
   ```bash
   python stock_predictor.py
   ```

## Results

The model’s performance is evaluated using Mean Squared Error (MSE) and visual plots that compare the actual vs. predicted stock prices over time. The strategic selection of correlated stocks leads to improved accuracy relative to models using unfiltered inputs.

## License

This project is licensed under the Apache License. See the [LICENSE](LICENSE) file for details.

