# Quantitative Sports Betting Arbitrage Engine

A Python-based high-frequency trading simulation that identifies and exploits inefficiencies in the sports betting market.

## Project Overview

This project simulates a quantitative trading environment where "Sports Odds" serve as a proxy for asset prices. This engine utilises an Event-Driven Architecture to process varied market data streams (Standard Match Odds vs. Synthetic Derivatives) in chronological order.


## Key Features

### 1. Event-Driven Architecture
Instead of batch processing, the engine utilizes a Master Event Loop that processes market "ticks" sequentially. This allows the system to:
- Handle competing signals from different strategies simultaneously.
- Simulate real-time capital constraints.
- Prioritise high-yield trades over low-yield ones via a Risk Manager.

### 2. Arbitrage Strategies
- **Standard Arb:** Scans implied probabilities across two competing bookmakers to identify when $$\sum P < 1.0$$.
- **Index/Synthetic Arb:** Detects mispricing between complex packaged bets (like "Draw No Bet") and their underlying constituent odds using a spread capture strategy.

### 3. Execution & Risk Management
- **Delta-Neutral Staking:** Implements an inverse-proportional staking formula to hedge directional risk.
- **Liquidity Checks:** The PortfolioManager prevents overdrafts and tracks running P&L.
- **Signal Filtering:** Implements a strict "Alpha Threshold" ($$> 5\%$$ edge) for Index Arbs to filter out market noise.
- **Friction Modeling:** Applies a $$2\%$$ Exchange Fee to all trades to simulate real-world execution costs.

## Quantitative Theory

This project is built on four fundamental concepts in Quantitative Finance:

### 1. The Law of One Price
In an efficient market, identical assets must trade at the same price. Divergences in bookmaker odds represent a violation of this law.

### 2. Implied Probability & The "Vig"
Decimal odds are treated as the inverse of probability: $$P = \frac{1}{\text{Odds}}$$.  
A market is inefficient when the sum of implied probabilities across all outcomes is $$< 1.0$$.

### 3. Synthetic Derivatives
We replicate complex products using basic constituents. For example, a "Draw No Bet" (DNB) product is synthetically created by betting on Home and hedging on the Draw:

$$
\text{Synthetic DNB} = \frac{H \times (D - 1)}{D}
$$

If the synthetic price deviates significantly from the market price, we execute an Index Arbitrage strategy.

### 4. Opportunity Cost
In a capital-constrained environment, the engine must decide which trade to take. The Risk Manager sorts simultaneous signals by ROI and executes only the most profitable one per tick.

## Assumptions & Limitations

- **Instant Execution:** Trades are assumed to fill instantly. 
- **Infinite Liquidity:** The simulation assumes we can bet any amount without moving the odds.
- **Idealized Data:** While the Standard Arb data uses stochastic volatility (sin waves + Gaussian noise), the Index Arb data currently uses injected "known positives" for unit testing.

## Live Testing & Verification
The engine runs in two modes: **Simulation** (for demonstration) and **Live** (for validation).

### The "False Positive" Discovery (3-Way vs 2-Way Markets)
During initial live testing on English Premier League (EPL) data, the engine detected abnormally high margins (>30%). 
* **Root Cause:** The arbitrage formula $\frac{1}{Home} + \frac{1}{Away}$ is valid only for 2-way markets (e.g., Tennis, NFL). In Soccer, the missing **Draw** outcome caused the sum of probabilities to artificially drop below 1.0.
* **Resolution:** The engine was re-configured to target 2-way markets (NFL/NBA) where the "No-Draw" condition holds true.

## Technical Stack

- **Python 3.10+**
- **NumPy:** For generating stochastic market feeds and handling vector operations.
- **Matplotlib:** For visualizing the "Arbitrage Window" and market efficiency over time.
- **Object-Oriented Design:** Modular architecture separating ArbitrageEngine (Logic), PortfolioManager (State), and RiskManager (Decision).

## Installation & Usage

Clone the repository:
```bash
git clone https://github.com/yourusername/sports-arb-engine.git

cd sports-arb-engine
```

Set up the environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Run the simulation:
```bash
python main.py
```

