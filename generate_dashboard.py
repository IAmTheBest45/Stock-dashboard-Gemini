import yfinance as yf
import pandas as pd
from ollama import Client
import json
import os
from datetime import datetime, timedelta
import re # Import the regex module

# --- Configuration ---
OLLAMA_HOST = 'http://localhost:11434'  # Default Ollama host
LLM_MODEL = 'llama3'                   # Model to use (ensure it's downloaded: ollama pull llama3)
PORTFOLIO_FILE = 'portfolio.csv'       # Or 'portfolio.json' if you chose that format
HISTORICAL_DAYS = 90                   # Number of days of historical data for LLM context and technicals

# Percentage targets for calculated entry/sell prices (adjust as per your strategy and risk tolerance)
TAKE_PROFIT_PERCENT_BUY = 7.0
STOP_LOSS_PERCENT_BUY = 3.0
TAKE_PROFIT_PERCENT_SELL = 10.0
STOP_LOSS_FROM_AVG_PRICE_PERCENT = 8.0

DASHBOARD_OUTPUT_DIR = 'docs' # Directory where the HTML dashboard will be generated

# --- Helper Functions ---

def load_portfolio(file_path):
    """Loads portfolio data from a CSV or JSON file."""
    if file_path.endswith('.csv'):
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"Error: Portfolio file '{file_path}' not found. Please create it.")
            return pd.DataFrame()
        except pd.errors.EmptyDataError:
            print(f"Error: Portfolio file '{file_path}' is empty. Please add data.")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error reading CSV file '{file_path}': {e}")
            print("Please ensure your CSV is correctly formatted (e.g., no extra commas, correct headers).")
            return pd.DataFrame()
    elif file_path.endswith('.json'):
        try:
            with open(file_path, 'r') as f:
                return pd.DataFrame(json.load(f))
        except FileNotFoundError:
            print(f"Error: Portfolio file '{file_path}' not found. Please create it.")
            return pd.DataFrame()
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in '{file_path}'. Please check file content.")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error reading JSON file '{file_path}': {e}")
            return pd.DataFrame()
    else:
        print("Error: Unsupported portfolio file format. Use .csv or .json.")
        return pd.DataFrame()

def get_stock_data(ticker, days=HISTORICAL_DAYS):
    """Fetches real-time price and historical data for a given ticker."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    try:
        stock = yf.Ticker(ticker)
        hist_data = stock.history(start=start_date, end=end_date)

        if hist_data.empty:
            print(f"Warning: No historical data found for {ticker} in the last {days} days.")
            current_price_info = stock.info.get('regularMarketPrice')
            if current_price_info:
                return current_price_info, {"full_history": pd.DataFrame(), "historical_summary": "No recent historical price summary available.", "average_volume_last_month": "N/A"}
            else:
                return None, None
        
        current_price = hist_data['Close'].iloc[-1]
        
        recent_prices_str = "Recent closing prices (last 5 days):\n"
        for i in range(min(5, len(hist_data))):
            date = hist_data.index[-1 - i].strftime('%Y-%m-%d')
            price = hist_data['Close'].iloc[-1 - i]
            recent_prices_str += f"- {date}: {price:.2f}\n"

        avg_volume_last_month = hist_data['Volume'].mean()

        return current_price, {
            "full_history": hist_data,
            "historical_summary": recent_prices_str,
            "average_volume_last_month": avg_volume_last_month
        }
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None, None

def get_news_sentiment(ticker):
    """Placeholder for fetching news sentiment."""
    return "No live news sentiment available (offline mode)."

def get_technical_indicators(hist_df):
    """Calculates simple technical indicators using pandas for a given DataFrame."""
    if hist_df is None or hist_df.empty:
        return "Not enough historical data for robust technical indicators."

    indicators_info = []

    if len(hist_df) >= 20:
        sma_20 = hist_df['Close'].iloc[-20:].mean()
        indicators_info.append(f"20-day SMA: {sma_20:.2f}")
    if len(hist_df) >= 50:
        sma_50 = hist_df['Close'].iloc[-50:].mean()
        indicators_info.append(f"50-day SMA: {sma_50:.2f}")

    if len(hist_df) >= 14:
        delta = hist_df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))
        if not pd.isna(rsi.iloc[-1]):
            indicators_info.append(f"14-day RSI (approx): {rsi.iloc[-1]:.2f}")
        else:
            indicators_info.append("14-day RSI (approx): Not enough data yet or calculation error")

    if len(hist_df) > 0:
        recent_low = hist_df['Low'].iloc[-min(30, len(hist_df)):].min()
        recent_high = hist_df['High'].iloc[-min(30, len(hist_df)):].max()
        indicators_info.append(f"Recent (30-day) Low: {recent_low:.2f}")
        indicators_info.append(f"Recent (30-day) High: {recent_high:.2f}")

    if not indicators_info:
        return "No sufficient historical data for technical indicators."
    return "\n".join(indicators_info)

def calculate_price_targets(current_price, average_price, is_new_buy=False):
    """Calculates suggested take-profit and stop-loss prices."""
    targets = {}

    if is_new_buy:
        targets['Buy Price'] = current_price
        targets['Suggested Take Profit'] = current_price * (1 + TAKE_PROFIT_PERCENT_BUY / 100)
        targets['Suggested Stop Loss'] = current_price * (1 - STOP_LOSS_PERCENT_BUY / 100)
        targets['Type'] = 'New Buy'
    else:
        targets['Current Price'] = current_price
        targets['Average Purchase Price'] = average_price
        
        targets['Suggested Take Profit'] = max(current_price, average_price) * (1 + TAKE_PROFIT_PERCENT_SELL / 100)
        targets['Suggested Stop Loss'] = average_price * (1 - STOP_LOSS_FROM_AVG_PRICE_PERCENT / 100)
        targets['Type'] = 'Existing Holding'

    return targets

def analyze_stock_with_llm(ticker, portfolio_info_series, current_price, historical_data_summary, news_sentiment, technical_indicators, price_targets):
    """Sends stock data to local LLM for analysis and recommendation."""
    client = Client(host=OLLAMA_HOST)

    has_portfolio_info = portfolio_info_series is not None and not portfolio_info_series.empty

    current_value = current_price * portfolio_info_series['Shares'] if has_portfolio_info else current_price
    profit_loss = (current_price - portfolio_info_series['Average_Price']) * portfolio_info_series['Shares'] if has_portfolio_info else 0
    profit_loss_percent = (profit_loss / (portfolio_info_series['Average_Price'] * portfolio_info_series['Shares'])) * 100 if has_portfolio_info and portfolio_info_series['Shares'] > 0 else 0

    portfolio_data_str = ""
    if has_portfolio_info:
        portfolio_data_str = f"""
**Your Portfolio Data:**
- Shares Owned: {portfolio_info_series['Shares']}
- Average Purchase Price: {portfolio_info_series['Average_Price']:.2f}
- Unrealized Profit/Loss: {profit_loss:.2f} ({profit_loss_percent:.2f}%)
"""
    else:
        portfolio_data_str = "**Analyzing for a hypothetical new investment (not in your current portfolio).**"

    llm_prompt = f"""
You are an expert AI financial analyst. Your task is to provide a concise "Buy", "Sell", or "Hold" recommendation for a stock, along with a brief, clear justification.
If you recommend "BUY" or "SELL", also suggest a specific entry or exit price/range based on the provided data and your financial expertise.

Here is the data for the stock:
---
**Company Name/Ticker:** {ticker}

- Current Market Price: {current_price:.2f}
- Current Value of your holding: {current_value:.2f}
{portfolio_data_str}

**Recent Price Action:**
{historical_data_summary['historical_summary'] if historical_data_summary else 'No recent historical price summary available.'}
- Average Daily Volume (last {HISTORICAL_DAYS} days): {historical_data_summary['average_volume_last_month']:.2f} (if available)

**Key Technical Indicators:**
{technical_indicators}

**Market Sentiment/News:**
{news_sentiment}

**Calculated Price Targets (based on your configuration):**
- Suggested Take Profit: {price_targets.get('Suggested Take Profit', 'N/A'):.2f}
- Suggested Stop Loss: {price_targets.get('Suggested Stop Loss', 'N/A'):.2f}
---

**Based on this comprehensive information, provide:**
1.  **Recommendation (BUY, SELL, or HOLD):**
2.  **Justification (1-3 sentences):**
3.  **If BUY or SELL, provide a specific Suggested Entry Price or Suggested Exit Price/Range:** (e.g., "Entry: $XXX.XX", "Exit: $YYY.YY - $ZZZ.ZZ")
"""

    try:
        print(f"\n--- Sending data for {ticker} to LLM for analysis... ---")
        response = client.generate(model=LLM_MODEL, prompt=llm_prompt, stream=False, options={'temperature': 0.3, 'num_predict': 500})
        return response['response'].strip()
    except Exception as e:
        return f"Error communicating with local LLM for {ticker}: {e}. Ensure Ollama is running and '{LLM_MODEL}' model is downloaded."

def parse_llm_analysis(llm_analysis_text):
    """
    Parses the raw LLM analysis text to extract recommendation, justification, and suggested price.
    Uses regex for more robust parsing.
    """
    recommendation = "N/A"
    justification = "No analysis provided."
    suggested_price = "N/A"

    # --- Debugging: Print raw LLM output ---
    print(f"\n--- Raw LLM Response for Parsing ---\n{llm_analysis_text}\n--- End Raw LLM Response ---\n")
    # --- End Debugging ---

    # Regex to find Recommendation
    # Looks for "Recommendation:" followed by BUY, SELL, HOLD (case-insensitive)
    rec_match = re.search(r'(?:Recommendation:|1\.\s*Recommendation:)\s*(BUY|SELL|HOLD|N\/A)(?![a-zA-Z])', llm_analysis_text, re.IGNORECASE)
    if rec_match:
        recommendation = rec_match.group(1).upper() # Ensure consistent casing

    # Regex to find Justification
    # Captures text after "Justification:" up to the next numbered point or "Suggested Price" or end of string
    just_match = re.search(r'(?:Justification:|2\.\s*Justification:)\s*(.*?)(?=\n(?:3\.\s*If BUY or SELL|Suggested Price|Suggested Entry Price|Suggested Exit Price|\Z))', llm_analysis_text, re.IGNORECASE | re.DOTALL)
    if just_match:
        justification = just_match.group(1).strip()
        # Clean up any trailing "---" or "```" if LLM generates markdown
        justification = re.sub(r'```.*|---.*', '', justification, flags=re.DOTALL).strip()


    # Regex to find Suggested Price/Range
    # Looks for various forms of suggested price headers
    price_match = re.search(r'(?:Suggested (?:Entry|Exit) Price|Suggested Price|3\.\s*If BUY or SELL, provide a specific Suggested Entry Price or Suggested Exit Price\/Range):?\s*([^\n]+)', llm_analysis_text, re.IGNORECASE)
    if price_match:
        suggested_price = price_match.group(1).strip()
        # Clean up markdown if LLM includes it
        suggested_price = suggested_price.replace('**', '')


    # Fallback if specific sections aren't found, try to infer or assign whole text to justification
    if recommendation == "N/A" and justification == "No analysis provided." and llm_analysis_text.strip():
        # If no clear recommendation/justification headings were found,
        # but there is content, assume the whole response is the justification
        justification = llm_analysis_text.strip()
        # Try to infer recommendation from the justification
        if re.search(r'\b(?:buy|accumulate|add)\b', justification, re.IGNORECASE) and not re.search(r'\b(?:not buy|avoid)\b', justification, re.IGNORECASE):
            recommendation = "BUY"
        elif re.search(r'\b(?:sell|reduce|exit)\b', justification, re.IGNORECASE) and not re.search(r'\b(?:not sell|hold)\b', justification, re.IGNORECASE):
            recommendation = "SELL"
        elif re.search(r'\b(?:hold|monitor|neutral)\b', justification, re.IGNORECASE):
            recommendation = "HOLD"

    return recommendation, justification, suggested_price


def generate_html_dashboard(results):
    """Generates the HTML content for the dashboard."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Portfolio Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Inter', sans-serif;
            background-color: #f3f4f6;
        }}
        .recommendation-buy {{
            color: #10B981; /* Green */
        }}
        .recommendation-sell {{
            color: #EF4444; /* Red */
        }}
        .recommendation-hold {{
            color: #3B82F6; /* Blue */
        }}
        .card {{
            background-color: #ffffff;
            border-radius: 0.75rem; /* rounded-xl */
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); /* shadow-md */
            padding: 1.5rem; /* p-6 */
            margin-bottom: 1.5rem; /* mb-6 */
        }}
    </style>
</head>
<body class="bg-gray-100 text-gray-800 p-4 sm:p-6 md:p-8">
    <div class="max-w-4xl mx-auto">
        <h1 class="text-4xl font-bold text-center text-gray-900 mb-4">Stock Portfolio Dashboard</h1>
        <p class="text-center text-gray-600 mb-8">Last Updated: {current_time} (Local Time)</p>

        <div class="grid grid-cols-1 gap-6">
    """

    for result in results:
        ticker = result['Ticker']
        current_price = result['Current Price']
        unrealized_pl_percent = result['Unrealized P/L (%)']
        llm_analysis_raw = result['LLM Analysis'] # Get the raw LLM response

        recommendation, justification, suggested_price = parse_llm_analysis(llm_analysis_raw) # Parse it here
        calc_targets = result['Calculated Targets']

        # Determine color for recommendation
        rec_class = ""
        if "buy" in recommendation.lower():
            rec_class = "recommendation-buy"
        elif "sell" in recommendation.lower():
            rec_class = "recommendation-sell"
        elif "hold" in recommendation.lower():
            rec_class = "recommendation-hold"
        else:
            rec_class = "text-gray-500" # Default if not parsed or N/A

        # Determine color for P/L
        pl_color_class = "text-gray-600"
        try:
            pl_value = float(unrealized_pl_percent.replace('%',''))
            if pl_value > 0:
                pl_color_class = "text-green-600"
            elif pl_value < 0:
                pl_color_class = "text-red-600"
        except ValueError:
            pass # Keep gray if N/A or not a number


        html_content += f"""
            <div class="card">
                <h2 class="text-2xl font-semibold text-gray-900 mb-2">{ticker}</h2>
                <div class="mb-4 text-lg">
                    <p><span class="font-medium">Current Price:</span> ${current_price:.2f}</p>
                    <p><span class="font-medium">Unrealized P/L:</span> <span class="{pl_color_class}">{unrealized_pl_percent}</span></p>
                    <p><span class="font-medium">Calculated Take Profit:</span> ${calc_targets.get('Suggested Take Profit', 'N/A'):.2f}</p>
                    <p><span class="font-medium">Calculated Stop Loss:</span> ${calc_targets.get('Suggested Stop Loss', 'N/A'):.2f}</p>
                </div>
                
                <div class="border-t border-gray-200 pt-4">
                    <h3 class="text-xl font-semibold mb-2">AI Recommendation:</h3>
                    <p class="text-xl font-bold {rec_class} mb-2">{recommendation}</p>
                    <p class="text-gray-700 mb-2"><span class="font-medium">Justification:</span> {justification}</p>
                    <p class="text-gray-700"><span class="font-medium">Suggested Price:</span> {suggested_price}</p>
                </div>
            </div>
        """
    
    html_content += """
        </div>
        <p class="text-center text-gray-500 text-sm mt-8">
            Disclaimer: AI recommendations are for informational purposes only and do not constitute financial advice. Always conduct your own research and consult with a financial professional before making investment decisions.
        </p>
    </div>
</body>
</html>
    """
    return html_content

# --- Main Execution ---

def main():
    print("🚀 Starting Portfolio Analysis with Local AI 🚀")
    print("--------------------------------------------------")

    # Load portfolio
    portfolio_df = load_portfolio(PORTFOLIO_FILE)
    if portfolio_df.empty:
        print("Portfolio is empty. Exiting.")
        return

    print(f"Loaded {len(portfolio_df)} holdings from {PORTFOLIO_FILE}.")
    print("\n--- Portfolio Overview ---")
    for index, row in portfolio_df.iterrows():
        print(f"Ticker: {row['Ticker']}, Shares: {row['Shares']}, Avg Price: {row['Average_Price']:.2f}")
    print("--------------------------")

    results = []

    for index, row_series in portfolio_df.iterrows():
        ticker = row_series['Ticker']
        print(f"\n--- Analyzing {ticker} ---")

        current_price, historical_data = get_stock_data(ticker)

        if current_price is None:
            llm_analysis = f"Could not fetch data for {ticker}. Cannot provide recommendation."
            calculated_targets = {}
        else:
            calculated_targets = calculate_price_targets(current_price, row_series['Average_Price'], is_new_buy=False)
            
            tech_indicators_info = get_technical_indicators(historical_data['full_history'] if historical_data and 'full_history' in historical_data else pd.DataFrame())
            
            news_info = get_news_sentiment(ticker)
            
            llm_analysis = analyze_stock_with_llm(
                ticker,
                row_series,
                current_price,
                historical_data,
                news_info,
                tech_indicators_info,
                calculated_targets
            )
        
        results.append({
            'Ticker': ticker,
            'Current Price': current_price if current_price is not None else "N/A",
            'Unrealized P/L (%)': f"{((current_price - row_series['Average_Price']) / row_series['Average_Price'] * 100):.2f}%" if current_price is not None else "N/A",
            'Calculated Targets': calculated_targets,
            'LLM Analysis': llm_analysis # Raw LLM analysis will be parsed in HTML generation
        })

    print("\nGenerating HTML dashboard...")
    html_dashboard_content = generate_html_dashboard(results)

    os.makedirs(DASHBOARD_OUTPUT_DIR, exist_ok=True)
    dashboard_file_path = os.path.join(DASHBOARD_OUTPUT_DIR, 'index.html')

    with open(dashboard_file_path, 'w', encoding='utf-8') as f:
        f.write(html_dashboard_content)

    print(f"Dashboard generated successfully at: {dashboard_file_path}")
    print("Analysis complete. Remember to always do your own research before making investment decisions.")

if __name__ == "__main__":
    main()
