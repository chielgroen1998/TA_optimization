# ================================================================================
# COPY-PASTEABLE CELLS FOR COMPREHENSIVE GRID SEARCH SYSTEM
# ================================================================================

# ============================ CELL 1: INDICATOR PARAMETERS AND BUY SIGNAL LOGIC ============================

# Define parameter ranges for each indicator
INDICATOR_PARAMS = {
    'RSI': {
        'period': list(range(7, 91)),  # 7 to 90 days
        'threshold': [30, 50, 70]  # Buy when RSI < threshold, Sell when RSI > threshold
    },
    'ADX': {
        'period': list(range(7, 91)),  # 7 to 90 days
        'threshold': [20, 25, 30, 35]  # Buy when ADX > threshold (strong trend)
    },
    'KAMA': {
        'er_period': [8, 10, 12],  # Efficiency ratio periods
        'fast_period': [5, 7, 9, 12],  # Fast EMA periods
        'slow_period': [19, 21, 25, 30]  # Slow EMA periods
    },
    'ATR': {
        'period': list(range(7, 91)),  # 7 to 90 days
        'multiplier': [1.0, 1.5, 2.0, 2.5]  # ATR multiplier for volatility signals
    },
    'MFI': {
        'period': list(range(7, 91)),  # 7 to 90 days
        'overbought': [70, 75, 80],  # Sell when MFI > overbought
        'oversold': [20, 25, 30]  # Buy when MFI < oversold
    },
    'Entropy': {
        'period': list(range(7, 91)),  # 7 to 90 days
        'threshold': [0.5, 0.6, 0.7, 0.8]  # Buy when entropy < threshold (low randomness)
    }
}

# Define buy signal logic for each indicator
def get_buy_signal(indicator_name: str, data: pd.Series, params: dict) -> pd.Series:
    """
    Generate buy signals for different indicators.
    Returns: 1 (buy), -1 (sell), 0 (hold), NaN (no signal)
    """
    if indicator_name == 'RSI':
        threshold = params.get('threshold', 50)
        # Buy when oversold (RSI < threshold), Sell when overbought (RSI > threshold)
        return pd.Series(np.where(data < threshold, 1,
                                np.where(data > threshold, -1, 0)), index=data.index)

    elif indicator_name == 'ADX':
        threshold = params.get('threshold', 25)
        # Buy when strong trend (ADX > threshold), otherwise hold
        return pd.Series(np.where(data > threshold, 1, 0), index=data.index)

    elif indicator_name == 'KAMA':
        # Buy when price > KAMA (momentum up), Sell when price < KAMA (momentum down)
        close = params.get('close', pd.Series(index=data.index))
        return pd.Series(np.where(close > data, 1,
                                np.where(close < data, -1, 0)), index=data.index)

    elif indicator_name == 'ATR':
        # Volatility-based signals - buy when volatility is low, sell when high
        multiplier = params.get('multiplier', 2.0)
        mean_atr = data.rolling(20).mean()  # 20-day average ATR
        return pd.Series(np.where(data < (mean_atr * multiplier), 1,
                                np.where(data > (mean_atr * multiplier), -1, 0)), index=data.index)

    elif indicator_name == 'MFI':
        overbought = params.get('overbought', 80)
        oversold = params.get('oversold', 20)
        # Buy when oversold, Sell when overbought
        return pd.Series(np.where(data < oversold, 1,
                                np.where(data > overbought, -1, 0)), index=data.index)

    elif indicator_name == 'Entropy':
        threshold = params.get('threshold', 0.7)
        # Buy when randomness is low (predictable market)
        return pd.Series(np.where(data < threshold, 1, 0), index=data.index)

    return pd.Series([0] * len(data), index=data.index)  # Default: no signal

# Generate all possible parameter combinations for an indicator
def generate_param_combinations(indicator_name: str) -> list:
    """Generate all possible parameter combinations for a given indicator."""
    if indicator_name not in INDICATOR_PARAMS:
        return []

    params = INDICATOR_PARAMS[indicator_name]
    param_names = list(params.keys())
    param_values = list(params.values())

    # Generate all combinations
    from itertools import product
    combinations = list(product(*param_values))

    # Convert to list of dictionaries
    return [{param_names[i]: combo[i] for i in range(len(param_names))} for combo in combinations]

# Calculate distance between two parameter sets (for optimization)
def calculate_param_distance(params1: dict, params2: dict) -> float:
    """Calculate Euclidean distance between two parameter sets."""
    if not params1 or not params2:
        return float('inf')

    common_keys = set(params1.keys()) & set(params2.keys())
    if not common_keys:
        return float('inf')

    distance = 0
    for key in common_keys:
        distance += (params1[key] - params2[key]) ** 2

    return distance ** 0.5

print("✅ Indicator parameter ranges and signal logic defined!")
print(f"Available indicators: {list(INDICATOR_PARAMS.keys())}")
print(f"Example RSI combinations: {len(generate_param_combinations('RSI'))}")
print(f"Example ADX combinations: {len(generate_param_combinations('ADX'))}")
print(f"Example KAMA combinations: {len(generate_param_combinations('KAMA'))}")

# ============================ CELL 2: MAIN GRID SEARCH FUNCTION ============================

import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def run_comprehensive_grid_search(data: pd.DataFrame, indicator_params: dict,
                                indicator_class: Indicator, max_combinations_per_indicator: int = 50) -> dict:
    """
    Run comprehensive grid search across all indicators and all tickers.

    Args:
        data: DataFrame with OHLCV data and Ticker column
        indicator_params: Dictionary of parameter ranges for each indicator
        indicator_class: Instance of Indicator class
        max_combinations_per_indicator: Limit combinations per indicator to avoid explosion

    Returns:
        Dictionary with results organized by ticker and indicator
    """
    print("🚀 Starting comprehensive grid search...")
    print(f"Available tickers: {data['Ticker'].unique().tolist()}")
    print(f"Available indicators: {list(indicator_params.keys())}")

    # Results container - organized by ticker, then indicator
    all_results = {}

    # Get unique tickers
    tickers = data['Ticker'].unique()

    for ticker in tickers:
        print(f"\n📊 Processing {ticker}...")
        all_results[ticker] = {}

        # Get data for this ticker
        ticker_data = data[data['Ticker'] == ticker].copy()
        if len(ticker_data) < 100:  # Skip if too little data
            print(f"  ⚠️  Skipping {ticker} - insufficient data ({len(ticker_data)} days)")
            continue

        # Create indicator instance for this ticker
        ticker_indicator = Indicator(ticker_data)

        for indicator_name in indicator_params.keys():
            print(f"  🔍 Testing {indicator_name} on {ticker}...")

            # Generate parameter combinations
            param_combinations = generate_param_combinations(indicator_name)

            # Limit combinations if specified
            if len(param_combinations) > max_combinations_per_indicator:
                import random
                param_combinations = random.sample(param_combinations, max_combinations_per_indicator)
                print(f"    📉 Limited to {len(param_combinations)} random combinations out of {len(generate_param_combinations(indicator_name))}")

            indicator_results = []

            for i, params in enumerate(tqdm(param_combinations, desc=f"    {indicator_name} combinations")):
                try:
                    # Calculate indicator values based on parameters
                    if indicator_name == 'RSI':
                        indicator_values = ticker_indicator.rsi(params['period'])
                    elif indicator_name == 'ADX':
                        indicator_values = ticker_indicator.adx(params['period'])
                    elif indicator_name == 'KAMA':
                        indicator_values = ticker_indicator.kama(
                            er_period=params['er_period'],
                            fast_period=params['fast_period'],
                            slow_period=params['slow_period']
                        )
                    elif indicator_name == 'ATR':
                        indicator_values = ticker_indicator.atr(params['period'])
                    elif indicator_name == 'MFI':
                        indicator_values = ticker_indicator.mfi(params['period'])
                    elif indicator_name == 'Entropy':
                        indicator_values = ticker_indicator.entropy(params['period'])

                    # Generate signals
                    if indicator_name == 'KAMA':
                        # KAMA needs close prices for comparison
                        params['close'] = ticker_data['Close']
                    signals = get_buy_signal(indicator_name, indicator_values, params)

                    # Create signals DataFrame
                    signals_df = pd.DataFrame({
                        'Date': ticker_data.index,
                        'Ticker': ticker,
                        'Signal': signals,
                        'Indicator_Value': indicator_values
                    }).set_index('Date')

                    # Calculate returns
                    signals_df['Daily_Return'] = ticker_data['Close'].pct_change()
                    signals_df['Position'] = signals_df['Signal'].shift(1)  # Use previous day's signal
                    signals_df['Strategy_Return'] = signals_df['Position'] * signals_df['Daily_Return']

                    # Calculate performance metrics
                    valid_returns = signals_df['Strategy_Return'].dropna()
                    if len(valid_returns) > 0:
                        total_return = (1 + valid_returns).prod() - 1
                        sharpe = (valid_returns.mean() / valid_returns.std()) * np.sqrt(252) if valid_returns.std() > 0 else 0
                        max_drawdown = calculate_max_drawdown((1 + valid_returns).cumprod())
                        win_rate = (valid_returns > 0).mean()
                    else:
                        total_return = sharpe = max_drawdown = win_rate = np.nan

                    # Store results
                    result = {
                        'Indicator': indicator_name,
                        'Parameters': params,
                        'Total_Return': total_return,
                        'Sharpe_Ratio': sharpe,
                        'Max_Drawdown': max_drawdown,
                        'Win_Rate': win_rate,
                        'Total_Signals': len(signals.dropna()),
                        'Buy_Signals': (signals == 1).sum(),
                        'Sell_Signals': (signals == -1).sum()
                    }

                    indicator_results.append(result)

                except Exception as e:
                    print(f"    ❌ Error with {indicator_name} params {params}: {str(e)}")
                    continue

            # Store results for this indicator
            all_results[ticker][indicator_name] = pd.DataFrame(indicator_results)

            # Show best result for this indicator
            if len(indicator_results) > 0:
                best_result = max(indicator_results, key=lambda x: x['Sharpe_Ratio'] if not np.isnan(x['Sharpe_Ratio']) else -999)
                print(f"    ✅ Best {indicator_name}: Sharpe={best_result['Sharpe_Ratio']:.3f}, Return={best_result['Total_Return']:.3f}")

    return all_results

def calculate_max_drawdown(equity: pd.Series) -> float:
    """Calculate maximum drawdown from equity curve."""
    if len(equity) == 0:
        return 0.0

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return float(drawdown.min())

print("🎯 Grid search function defined!")
print("Ready to run: run_comprehensive_grid_search(data, INDICATOR_PARAMS, Indicator())")

# ============================ CELL 3: RESULTS ANALYSIS FUNCTIONS ============================

def analyze_grid_search_results(all_results: dict) -> dict:
    """
    Analyze grid search results and extract key insights.

    Args:
        all_results: Results dictionary from run_comprehensive_grid_search

    Returns:
        Dictionary with analysis results
    """
    analysis = {
        'best_per_ticker': {},
        'best_indicators': {},
        'parameter_distances': {},
        'summary_stats': {}
    }

    # Best indicator per ticker
    for ticker, ticker_results in all_results.items():
        best_sharpe = -999
        best_indicator = None
        best_params = None

        for indicator_name, results_df in ticker_results.items():
            if len(results_df) == 0:
                continue

            max_sharpe_idx = results_df['Sharpe_Ratio'].idxmax()
            if pd.notna(max_sharpe_idx):
                current_sharpe = results_df.loc[max_sharpe_idx, 'Sharpe_Ratio']
                if current_sharpe > best_sharpe:
                    best_sharpe = current_sharpe
                    best_indicator = indicator_name
                    best_params = results_df.loc[max_sharpe_idx, 'Parameters']

        analysis['best_per_ticker'][ticker] = {
            'Best_Indicator': best_indicator,
            'Best_Sharpe': best_sharpe,
            'Best_Params': best_params
        }

    # Best performing indicators overall
    indicator_performance = {}
    for ticker, ticker_results in all_results.items():
        for indicator_name, results_df in ticker_results.items():
            if indicator_name not in indicator_performance:
                indicator_performance[indicator_name] = []

            valid_results = results_df[results_df['Sharpe_Ratio'].notna()]
            if len(valid_results) > 0:
                indicator_performance[indicator_name].extend(valid_results['Sharpe_Ratio'].tolist())

    # Calculate average performance per indicator
    for indicator, sharpes in indicator_performance.items():
        if sharpes:
            analysis['best_indicators'][indicator] = {
                'Avg_Sharpe': np.mean(sharpes),
                'Max_Sharpe': np.max(sharpes),
                'Min_Sharpe': np.min(sharpes),
                'Test_Count': len(sharpes)
            }

    return analysis

def create_results_summary_table(all_results: dict, output_file: str = None) -> pd.DataFrame:
    """
    Create a summary table of all results.

    Args:
        all_results: Results dictionary from run_comprehensive_grid_search
        output_file: Optional CSV file to save results

    Returns:
        DataFrame with summary of all results
    """
    summary_rows = []

    for ticker, ticker_results in all_results.items():
        for indicator_name, results_df in ticker_results.items():
            for _, row in results_df.iterrows():
                summary_rows.append({
                    'Ticker': ticker,
                    'Indicator': indicator_name,
                    'Parameters': str(row['Parameters']),
                    'Total_Return': row['Total_Return'],
                    'Sharpe_Ratio': row['Sharpe_Ratio'],
                    'Max_Drawdown': row['Max_Drawdown'],
                    'Win_Rate': row['Win_Rate'],
                    'Total_Signals': row['Total_Signals'],
                    'Buy_Signals': row['Buy_Signals'],
                    'Sell_Signals': row['Sell_Signals']
                })

    summary_df = pd.DataFrame(summary_rows)

    if output_file:
        summary_df.to_csv(output_file, index=False)
        print(f"📊 Results saved to {output_file}")

    return summary_df

def find_similar_parameter_combinations(results_df: pd.DataFrame, target_params: dict,
                                      max_distance: float = 5.0) -> pd.DataFrame:
    """
    Find parameter combinations similar to a target set.

    Args:
        results_df: Results DataFrame for a specific indicator
        target_params: Target parameter set to compare against
        max_distance: Maximum parameter distance to include

    Returns:
        DataFrame with similar parameter combinations
    """
    if 'Parameters' not in results_df.columns:
        return pd.DataFrame()

    similar_results = []

    for _, row in results_df.iterrows():
        try:
            distance = calculate_param_distance(target_params, row['Parameters'])
            if distance <= max_distance:
                row_copy = row.copy()
                row_copy['Parameter_Distance'] = distance
                similar_results.append(row_copy)
        except:
            continue

    return pd.DataFrame(similar_results).sort_values('Parameter_Distance')

def plot_indicator_performance_comparison(all_results: dict, metric: str = 'Sharpe_Ratio'):
    """
    Create comparison plots of indicator performance.

    Args:
        all_results: Results dictionary from run_comprehensive_grid_search
        metric: Performance metric to plot ('Sharpe_Ratio', 'Total_Return', 'Win_Rate')
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Prepare data
    plot_data = []
    for ticker, ticker_results in all_results.items():
        for indicator_name, results_df in ticker_results.items():
            valid_results = results_df[results_df[metric].notna()]
            if len(valid_results) > 0:
                for _, row in valid_results.iterrows():
                    plot_data.append({
                        'Ticker': ticker,
                        'Indicator': indicator_name,
                        metric: row[metric]
                    })

    if not plot_data:
        print(f"⚠️ No valid {metric} data found")
        return

    plot_df = pd.DataFrame(plot_data)

    # Create box plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=plot_df, x='Indicator', y=metric)
    plt.title(f'{metric} Distribution by Indicator')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Create heatmap of average performance
    pivot_table = plot_df.pivot_table(values=metric, index='Ticker', columns='Indicator', aggfunc='mean')
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdYlGn')
    plt.title(f'Average {metric} by Ticker and Indicator')
    plt.tight_layout()
    plt.show()

def save_grid_search_results(all_results: dict, base_filename: str = 'grid_search_results'):
    """
    Save grid search results to multiple CSV files.

    Args:
        all_results: Results dictionary from run_comprehensive_grid_search
        base_filename: Base filename for saving results
    """
    import os
    os.makedirs('grid_search_results', exist_ok=True)

    # Save individual ticker results
    for ticker, ticker_results in all_results.items():
        ticker_filename = f'grid_search_results/{base_filename}_{ticker}.csv'
        ticker_dfs = []
        for indicator_name, results_df in ticker_results.items():
            df_copy = results_df.copy()
            df_copy['Indicator'] = indicator_name
            ticker_dfs.append(df_copy)

        if ticker_dfs:
            ticker_combined = pd.concat(ticker_dfs, ignore_index=True)
            ticker_combined.to_csv(ticker_filename, index=False)

    # Save summary
    summary_df = create_results_summary_table(all_results)
    summary_filename = f'grid_search_results/{base_filename}_summary.csv'
    summary_df.to_csv(summary_filename, index=False)

    print(f"💾 Individual ticker results saved to grid_search_results/{base_filename}_[TICKER].csv")
    print(f"💾 Summary saved to {summary_filename}")

print("📊 Results analysis functions defined!")

# ============================ CELL 4: DEMONSTRATION AND USAGE EXAMPLES ============================

def demonstrate_grid_search():
    """Demonstrate how to use the grid search system."""
    print("=" * 80)
    print("🔬 COMPREHENSIVE GRID SEARCH DEMONSTRATION")
    print("=" * 80)

    print("\n1️⃣ Available Indicators and Parameters:")
    for indicator, params in INDICATOR_PARAMS.items():
        print(f"\n   {indicator}:")
        for param, values in params.items():
            print(f"   - {param}: {values}")

    print("\n2️⃣ How to run the grid search:")
    print("   # For all tickers (may take a while)")
    print("   results = run_comprehensive_grid_search(data, INDICATOR_PARAMS, Indicator())")

    print("\n   # For limited combinations per indicator")
    print("   results = run_comprehensive_grid_search(data, INDICATOR_PARAMS, Indicator(), max_combinations_per_indicator=20)")

    print("\n3️⃣ How to analyze results:")
    print("   analysis = analyze_grid_search_results(results)")
    print("   summary_df = create_results_summary_table(results)")
    print("   plot_indicator_performance_comparison(results)")

    print("\n4️⃣ How to find similar parameter combinations:")
    print("   similar = find_similar_parameter_combinations(results_df, target_params={'period': 14, 'threshold': 30})")

    print("\n5️⃣ How to save results:")
    print("   save_grid_search_results(results)")

    print("\n✅ Grid search system is ready to use!")

# Quick start examples
print("\n🚀 QUICK START - Limited Grid Search:")
print("results = run_comprehensive_grid_search(data, INDICATOR_PARAMS, Indicator(), max_combinations_per_indicator=20)")

print("\n🔬 FULL GRID SEARCH:")
print("full_results = run_comprehensive_grid_search(data, INDICATOR_PARAMS, Indicator())")

print("\n🎯 CUSTOM INDICATORS:")
print("custom_params = {k: INDICATOR_PARAMS[k] for k in ['RSI', 'ADX', 'MFI']}")
print("custom_results = run_comprehensive_grid_search(data, custom_params, Indicator(), max_combinations_per_indicator=30)")

print("\n📊 SINGLE TICKER TEST:")
print("single_ticker_data = data[data['Ticker'] == 'QQQ']")
print("single_results = run_comprehensive_grid_search(single_ticker_data, INDICATOR_PARAMS, Indicator())")

print("\n✨ Your comprehensive grid search system is ready!")
print("   Copy each cell above into your Jupyter notebook and run them in order.")

# Run demonstration
demonstrate_grid_search()



