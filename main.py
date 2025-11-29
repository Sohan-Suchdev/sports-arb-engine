import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from api_manager import OddsAPIManager

INITIAL_CAPITAL = 1000.0
EXCHANGE_FEE = 0.02 

DATA_SOURCE = "SIMULATION" 

class PortfolioManager:
    def __init__(self, initial_capital):
        self.balance = initial_capital
        self.history = []

    def execute_trade(self, stake, profit, description):
        if stake > self.balance:
            print(f"EXECUTION FAILED: Insufficient funds for {description}")
            print(f"   Required: £{stake:.2f} | Available: £{self.balance:.2f}")
            return
        
      # Simulation: We assume instantaneous execution and immediate payout
        self.balance += profit
        roi = (profit / stake) * 100

        self.history.append({
            'event': description,
            'stake': stake,        
            'profit': profit,
            'roi': roi,
            'running_balance': self.balance
        })
        
        print(f"       EXECUTED. Stake: £{stake:.2f} | Profit: +£{profit:.2f} | New Balance: £{self.balance:.2f}")

class IndexArbitrageEngine:
    @staticmethod
    def calculate_synthetic_dnb(home_odds, draw_odds):
        # The math to calculate synthetic DNB odds:
        return (home_odds * (draw_odds - 1)) / draw_odds

    @staticmethod
    def get_signal(match_data):
        syn_price = IndexArbitrageEngine.calculate_synthetic_dnb(match_data['home_odds'], match_data['draw_odds'])
        market_price = match_data['market_dnb_odds']
        
        edge = syn_price - market_price
        
        if edge > 0.05: # Filter out small noise
            stake = 100.0
            profit = stake * edge
            
            return {
                'strategy': 'Index Arb',
                'id': match_data['id'],
                'stake': stake,
                'profit': profit,
                'description': f"Index Arb: {match_data['id']} (Edge: {edge:.2f})"
            }
        return None

class StandardArbitrageEngine:
    @staticmethod
    def calculate_stakes(capital, odds1, odds2):
        """Delta Neutral Staking Formula"""
        p1 = 1/odds1
        p2 = 1/odds2
        market_sum = p1 + p2
        
        s1 = (capital * p1) / market_sum
        s2 = (capital * p2) / market_sum
        return s1, s2

    @staticmethod
    def get_signal(odds_a, odds_b, available_capital, match_id="Unknown"):
        # 1. Apply Fees
        eff_o1 = (odds_a - 1) * (1 - EXCHANGE_FEE) + 1
        eff_o2 = (odds_b - 1) * (1 - EXCHANGE_FEE) + 1
        
        # 2. Check Efficiency
        imp_prob = (1/eff_o1) + (1/eff_o2)
        
        if imp_prob < 1.0:
            margin = (1.0 - imp_prob) / imp_prob * 100
            s1, s2 = StandardArbitrageEngine.calculate_stakes(available_capital, eff_o1, eff_o2)
            total_stake = s1 + s2
            profit = (s1 * eff_o1) - available_capital
            
            return {
                'strategy': 'Standard Arb',
                'id': match_id,
                'stake': total_stake,
                'profit': profit,
                'description': f"Std Arb: {match_id} (Margin: {margin:.2f}%)"
            }
        return None

class MarketVisualizer:
    @staticmethod
    def plot_efficiency_window(timestamps, odds_a, odds_b, fee):
        print("\nGENERATING VISUALIZATION...")
        market_sums = []
        
        # Re-calculate the market efficiency for plotting
        for o1, o2 in zip(odds_a, odds_b):
            eff_o1 = (o1 - 1) * (1 - fee) + 1
            eff_o2 = (o2 - 1) * (1 - fee) + 1
            prob_sum = (1/eff_o1) + (1/eff_o2)
            market_sums.append(prob_sum)
        
        plt.figure(figsize=(12, 6))
        
        # Plot the Market Sum
        plt.plot(timestamps, market_sums, label='Total Market Implied Prob', color='blue', linewidth=1.5)
        
        # Draw the 'Efficient Market' line at 1.0
        plt.axhline(y=1.0, color='black', linestyle='--', label='Efficiency Threshold (1.0)', linewidth=2)
        
        # Highlight the Arbitrage Window (Area below 1.0)
        # This is the 'Profit Zone'
        plt.fill_between(timestamps, 1.0, market_sums, where=(np.array(market_sums) < 1.0), 
                         color='green', alpha=0.3, label='Arbitrage Window (Profit Zone)')
        
        plt.title('Real-Time Arbitrage Detection: Market Efficiency Monitor')
        plt.xlabel('Time (t)')
        plt.ylabel('Total Implied Probability (Inverse Price)')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        filename = 'arbitrage_simulation.png'
        plt.savefig(filename)
        print(f"Graph saved as '{filename}'. Check your file explorer!")
        plt.close()

class LiveMarketAdapter:
    def __init__(self):
        self.api = OddsAPIManager()
    
    def get_latest_tick(self):
        raw_data = self.api.fetch_live_odds(sport='americanfootball_nfl', region='us')
        if not raw_data:
            return []

        clean_feed = []
        t = int(time.time()) # Current Timestamp
        
        # We loop through the first 3 matches to keep the log clean
        for i, match in enumerate(raw_data[:3]):
            match_name = f"{match['home_team']} vs {match['away_team']}"
            bookmakers = match['bookmakers']
            
            if len(bookmakers) < 2:
                continue

            # STRATEGY: Pick the first two bookies to compare 
            bookie_a = bookmakers[0]
            bookie_b = bookmakers[1]
            
            # Extract Home Odds for A and Away Odds for B
            try:
                odds_a = bookie_a['markets'][0]['outcomes'][0]['price'] # Home Win
                odds_b = bookie_b['markets'][0]['outcomes'][1]['price'] # Away Win (Simplified)
            except:
                continue

            tick_data = {
                't': t,
                'std_market': {
                    'odds_a': odds_a, 
                    'odds_b': odds_b, 
                    'id': match_name
                },
                'index_market': None 
            }
            clean_feed.append(tick_data)
            
        return clean_feed

# --- DATA GENERATORS ---
def generate_merged_market_feed():
    # 1. Generate Standard Time Series
    t_steps = np.arange(0, 50, 1)
    odds_a = 2.0 + np.sin(t_steps / 5) * 0.2 + np.random.normal(0, 0.05, 50)
    odds_b = 2.0 - np.sin(t_steps / 5) * 0.2 + np.random.normal(0, 0.05, 50)
    
    # 2. Define Index Events (inserted at specific timestamps)
    index_events = {
        10: {'id': 'Arsenal vs Chelsea', 'home_odds': 2.10, 'draw_odds': 3.50, 'market_dnb_odds': 1.40},
        25: {'id': 'Man Utd vs Liverpool', 'home_odds': 2.50, 'draw_odds': 3.20, 'market_dnb_odds': 1.65},
        40: {'id': 'Real vs Barca', 'home_odds': 2.80, 'draw_odds': 3.10, 'market_dnb_odds': 1.95}
    }
    
    # 3. Create the Master Feed
    feed = []
    for t, oa, ob in zip(t_steps, odds_a, odds_b):
        tick_data = {
            't': t,
            'std_market': {'odds_a': oa, 'odds_b': ob},
            'index_market': index_events.get(t) # Returns None if no event at this time
        }
        feed.append(tick_data)
        
    return feed, t_steps, odds_a, odds_b

# --- THE MASTER LOOP ---
def run_event_driven_simulation():
    print(f"\nSTARTING ENGINE | MODE: {DATA_SOURCE}")
    print(f"Initial Capital: £{INITIAL_CAPITAL:.2f}")
    print("-" * 60)
    
    portfolio = PortfolioManager(INITIAL_CAPITAL)
    
    # 1. Select Data Source
    if DATA_SOURCE == "SIMULATION":
        market_feed, t_raw, oa_raw, ob_raw = generate_merged_market_feed()
    else:
        # LIVE MODE
        adapter = LiveMarketAdapter()
        market_feed = adapter.get_latest_tick()
        if not market_feed:
            print("No live data received. Check API quota or internet.")
            return

    for tick in market_feed:
        t = tick['t']
        potential_trades = []
        
        # --- SCAN STANDARD MARKET ---
        if tick.get('std_market'):
            std_signal = StandardArbitrageEngine.get_signal(
                tick['std_market']['odds_a'], 
                tick['std_market']['odds_b'],
                portfolio.balance,
                match_id=tick['std_market'].get('id', 'Unknown')
            )
            if std_signal:
                potential_trades.append(std_signal)
            
        # --- SCAN INDEX MARKET ---
        if tick.get('index_market'):
            idx_signal = IndexArbitrageEngine.get_signal(tick['index_market'])
            if idx_signal:
                potential_trades.append(idx_signal)
        
        # --- RISK MANAGER ---
        if not potential_trades:
            continue
            
        print(f"[t={t}] Found {len(potential_trades)} Opportunity(s)...")
        best_trade = sorted(potential_trades, key=lambda x: x['profit'], reverse=True)[0]
        
        for trade in potential_trades:
            if trade != best_trade:
                print(f"       REJECTED: {trade['description']} (Profit: £{trade['profit']:.2f})")
        
        print(f"       SELECTED: {best_trade['description']} (Profit: £{best_trade['profit']:.2f})")
        portfolio.execute_trade(best_trade['stake'], best_trade['profit'], best_trade['description'])
        print("-" * 60)

    print(f"\n RUN COMPLETE. Final Balance: £{portfolio.balance:.2f}")
    
    # Only visualize if we have the raw arrays (Simulation only)
    if DATA_SOURCE == "SIMULATION":
        MarketVisualizer.plot_efficiency_window(t_raw, oa_raw, ob_raw, EXCHANGE_FEE)

if __name__ == "__main__":
    run_event_driven_simulation()