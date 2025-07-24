# Fast Options Generator - Enhancement Tasks

This document outlines comprehensive improvements to make our fast options generator more realistic and complete compared to the LEAN RandomDataGenerator.

## Phase 1: Critical Data Correlation & Realism Issues 🚨

### P1.1: Underlying Price Continuity & Synchronization
**Priority: CRITICAL** - These issues could cause unrealistic backtesting results

- [ ] **Fix price gaps between trading days**: Currently each day starts from base_price, causing unrealistic jumps
- [ ] **Implement realistic weekend/holiday gap handling**: Add proper gap distributions for non-trading periods  
- [ ] **Add intraday mean reversion**: Prevent excessive price drift within a single trading day
- [ ] **Ensure option prices properly track underlying movements**: Validate Black-Scholes inputs update correctly
- [ ] **Fix time-to-expiration calculations**: Account for intraday theta decay and business days only
- [ ] **Implement proper business day calculations**: Handle weekends, holidays, half-days

### P1.2: Option-Underlying Price Correlation
**Priority: CRITICAL** - Core to realistic option behavior

- [ ] **Validate underlying price feeds to option pricing**: Ensure same underlying price used across all options
- [ ] **Fix volatility consistency**: Ensure volatility estimates reflect actual underlying price movements
- [ ] **Implement Greeks consistency**: Validate delta, gamma behavior matches underlying moves
- [ ] **Add price arbitrage checks**: Detect and prevent unrealistic option pricing scenarios

### P1.3: Universe File Accuracy & Option Chain Management
**Priority: HIGH** - Affects strategy availability and realism

- [ ] **Validate option chains properly represent available contracts**: Check strike spacing and availability
- [ ] **Ensure strike prices are realistic for underlying price levels**: Use proper rounding based on price
- [ ] **Fix expiration date generation**: Ensure proper weekly/monthly/daily expiration schedules
- [ ] **Add strike filtering based on liquidity**: Remove strikes that would be illiquid (deep ITM/OTM)

## Phase 2: Advanced Pricing Models & Market Microstructure

### P2.1: Dynamic Volatility Models
**Priority: HIGH** - Static volatility is unrealistic

- [ ] **Replace static volatility with volatility surface**: Vary by strike and time to expiration
- [ ] **Add volatility smile/skew effects**: Higher IV for OTM puts, smile for equity options
- [ ] **Implement time-varying volatility**: Volatility that changes over time and correlates with price moves
- [ ] **Add implied volatility clustering**: Periods of high/low volatility
- [ ] **Implement volatility term structure**: Different IV for different expirations

### P2.2: Market Microstructure Realism
**Priority: MEDIUM** - Improves execution realism

- [ ] **Implement realistic bid-ask spreads**: Based on option liquidity, moneyness, time to expiration
- [ ] **Add proper volume patterns by moneyness**: Higher volume for ATM options
- [ ] **Fix open interest calculations**: More realistic OI based on actual option trading patterns
- [ ] **Add liquidity modeling**: Some options should have low/no volume on certain days
- [ ] **Implement time-of-day effects**: Different behavior near open/close, lunch hour

## Phase 3: Missing LEAN RandomDataGenerator Features

### P3.1: Data Density Control
**Priority: MEDIUM** - Currently always generates dense data

- [ ] **Implement Dense/Sparse/VerySparse modes**: Control tick frequency
- [ ] **Add configurable tick frequency**: Not every minute for illiquid options
- [ ] **Add data density based on option characteristics**: ATM options more dense than OTM
- [ ] **Implement realistic "no trading" periods**: Some options don't trade every minute

### P3.2: Corporate Actions Support
**Priority: MEDIUM** - Missing completely from current implementation

- [ ] **Add basic stock splits generation**: 2:1, 3:2 splits with proper option adjustments
- [ ] **Implement dividend payments**: Regular quarterly dividends affecting option pricing
- [ ] **Add stock renames/symbol changes**: Handle ticker symbol changes over time
- [ ] **Implement IPO simulation**: New stocks entering the market
- [ ] **Add spin-off events**: Corporate restructuring events

### P3.3: Advanced Configuration & Market Integration
**Priority: LOW** - Nice to have features

- [ ] **Add quote/trade ratio control**: Configure relative frequency of quotes vs trades
- [ ] **Integrate market hours database**: Handle early close days, holidays, extended hours
- [ ] **Implement advanced strike rounding logic**: Price-dependent strike increments ($0.50, $1, $2.50, $5)
- [ ] **Add multiple underlying support**: Generate options for multiple stocks simultaneously
- [ ] **Implement realistic expiration cycles**: Monthly, weekly, quarterly options

## Phase 4: Validation, Testing & Output Enhancement

### P4.1: Data Validation Suite
**Priority: HIGH** - Essential for production use

- [ ] **Check for arbitrage opportunities**: Call-put parity, calendar spread arbitrage
- [ ] **Validate price relationships**: Calls > puts for same strike when stock > strike
- [ ] **Add statistical realism tests**: Price movements follow reasonable distributions
- [ ] **Implement Greeks validation**: Ensure Greeks behave as expected
- [ ] **Add benchmark comparison**: Compare generated data patterns with real market data

### P4.2: Greeks Calculation & Advanced Output
**Priority: MEDIUM** - Useful for advanced strategies

- [ ] **Calculate and output Greeks**: Delta, gamma, theta, vega, rho for each option
- [ ] **Implement Greeks-based validation**: Use Greeks to validate option price consistency
- [ ] **Add portfolio-level risk metrics**: For multi-leg strategies
- [ ] **Output volatility surface data**: Implied volatility by strike/expiration
- [ ] **Add performance analytics**: Track hit rates, PnL attribution

## Phase 5: Performance & Scalability
**Priority: LOW** - Current performance is already excellent

- [ ] **Optimize memory usage for large datasets**: Handle year+ of data generation
- [ ] **Add resumption capability enhancement**: Better checkpoint/restart functionality
- [ ] **Implement distributed generation**: Multi-machine scaling for very large datasets
- [ ] **Add streaming output**: Generate and immediately stream to target directory

---

## Phase 6: LEAN Compatibility & QuantConnect Integration 🚨

**Priority: BLOCKER** - Based on comprehensive analysis of LEAN CLI random data generator output, these components are REQUIRED for QuantConnect backtests to work.

### P6.1: BLOCKER LEVEL (Must Fix First - Prevents Any Backtest)

#### Task 6.1: Coarse Fundamental Data Generation
**Priority: BLOCKER** | **Effort: ~20 lines** | **Files:** `equity/usa/fundamental/coarse/YYYYMMDD.csv`

- [x] **Generate daily universe files for QC security selection**
- [x] **Required Format:** `SecurityIdentifier,Symbol,Price,Volume,DollarVolume,HasFundamentalData,PriceFactor,VolumeFactor`
- [x] **Sample:** `AAPL R735QTJ8XC9X,AAPL,645.57,12116671,7822159297,False,0.9011818,0.0357143`

**Critical:** QC uses this for universe selection BEFORE loading any price data. Without this, backtests fail at initialization with "no securities found".

#### Task 6.2: Security Database Integration  
**Priority: BLOCKER** | **Effort: ~10 lines** | **Files:** `symbol-properties/security-database.csv`

- [x] **Add entries to security database for symbol identity resolution**
- [x] **Required Format:** `SecurityIdentifier,CIK,BloombergTicker,CompositeFigi,ISIN,PrimarySymbol`
- [x] **Sample:** `AAPL R735QTJ8XC9X,03783310,BBG000B9XRY4,2046251,US0378331005,320193`

**Critical:** Links SecurityIdentifier to actual symbols. Missing entries cause "security not found" errors.

#### Task 6.3: Equity Daily Data Generation
**Priority: BLOCKER** | **Effort: ~20 lines** | **Files:** `equity/usa/daily/{symbol}.zip` containing `{symbol}.csv`

- [x] **Create underlying equity price data files**
- [x] **Required Format:** `DateTime,Open,High,Low,Close,Volume` (prices scaled by 10,000)
- [x] **Sample:** `19980102 00:00,136300,162500,135000,162500,6315000`

**Critical:** QC needs underlying price data to calculate option theoretical values. Without this, options contracts cannot be priced.

### P6.2: HIGH PRIORITY (Breaks QC Backtests)

#### Task 6.4: Map Files Generation
**Priority: HIGH** | **Effort: ~15 lines** | **Files:** `equity/usa/map_files/{symbol}.csv`

- [ ] **Create ticker mapping files with exchange designation**
- [ ] **Required Format:** `Date,Symbol,Exchange` (e.g., `19980102,aapl,Q`)
- [ ] **Add exchange codes (Q=NASDAQ, P=NYSE, etc.)**

#### Task 6.5: Factor Files Generation  
**Priority: HIGH** | **Effort: ~10 lines** | **Files:** `equity/usa/factor_files/{symbol}.csv`

- [ ] **Create split/dividend adjustment factor files**
- [ ] **Required Format:** `Date,PriceFactor,VolumeFactor,LastPrice`
- [ ] **Simple Implementation:** `19980102,1,1,1` for no-split scenario

#### Task 6.6: Options File Naming Convention Fix
**Priority: HIGH** | **Effort: ~15 lines** | **Files:** Individual contract CSV files within ZIP archives

- [ ] **Update file naming to LEAN convention**
- [ ] **Current:** `{date}_{symbol}_{type}.zip`
- [ ] **Required:** `YYYYMMDD_symbol_minute_type_style_optiontype_strike_expiry.csv`
- [ ] **Example:** `20140606_aapl_minute_trade_american_call_5900000_20140621.csv`

### P6.3: MEDIUM PRIORITY (Improves Accuracy)

#### Task 6.7: Symbol Properties Integration
**Priority: MEDIUM** | **Effort: ~15 lines** | **Files:** `symbol-properties/symbol-properties-database.csv`

- [ ] **Add contract specifications and multipliers**
- [ ] **Standard equity option properties with timezone, currency, market designation**

#### Task 6.8: Market Hours Compliance
**Priority: MEDIUM** | **Effort: ~10 lines** | **Files:** `market-hours/market-hours-database.json`

- [ ] **Add trading session definitions with timezone support**  
- [ ] **Include standard market hours (9:30-16:00 EST) and early close days**

#### Task 6.9: Shortable Securities Data
**Priority: MEDIUM** | **Effort: ~5 lines** | **Files:** `equity/usa/shortable/testbrokerage/symbols/{symbol}.csv`

- [ ] **Add short selling availability data**
- [ ] **Required Format:** `Date,AvailableShares` (e.g., `20140325,400`)

### P6.4: Integration Configuration

#### Task 6.10: Generator Config Updates
**Priority: HIGH** | **Effort: ~10 lines**

- [ ] **Add LEAN compatibility flags to GeneratorConfig:**
  ```python
  generate_coarse_data: bool = True
  generate_security_database: bool = True  
  generate_equity_daily: bool = True
  generate_map_files: bool = True
  generate_factor_files: bool = True
  exchange_code: str = "Q"  # NASDAQ default
  ```

#### Task 6.11: Main Generator Integration
**Priority: HIGH** | **Effort: ~15 lines**

- [ ] **Update FastOptionsGenerator.generate() method to call LEAN compatibility functions**
- [ ] **Add conditional generation based on config flags**
- [ ] **Maintain existing options generation workflow**

### P6.5: Testing & Validation

#### Task 6.12: QC Backtest Validation
**Priority: BLOCKER** | **Effort: Testing**

- [ ] **Phase 1:** Test backtest initialization with Tasks 6.1-6.3 
- [ ] **Phase 2:** Test complete backtest execution with Tasks 6.4-6.6
- [ ] **Phase 3:** Test advanced features with Tasks 6.7-6.9
- [ ] **Verify:** 100% QC compatibility while maintaining 36x speed advantage

---

## Updated Implementation Priority

### 🚨 CRITICAL PATH (Essential for QC Compatibility)
1. **Phase 6.1 (BLOCKER)** - Tasks 6.1-6.3: Coarse data, security database, equity data
2. **Phase 6.2 (HIGH)** - Tasks 6.4-6.6: Map files, factor files, options naming
3. **Phase 6.3-6.5 (MEDIUM)** - Tasks 6.7-6.12: Properties, hours, testing

### 📈 ENHANCEMENT PATH (Original Tasks)  
4. **Phase 1** - Critical data correlation issues that affect backtesting accuracy
5. **Phase 2** - Advanced pricing models for realism  
6. **Phase 3** - Missing LEAN features for completeness
7. **Phase 4** - Validation and testing
8. **Phase 5** - Performance optimizations

## Updated Success Metrics

### QC Compatibility (Phase 6)
- [ ] **QC backtests initialize successfully** (BLOCKER level complete)
- [ ] **QC backtests execute without errors** (HIGH priority complete)  
- [ ] **Full feature parity with LEAN CLI output** (MEDIUM priority complete)
- [ ] **Generated files pass LEAN format validation**
- [ ] **Performance maintained: <5 minutes for full month** (vs 3 hours LEAN CLI)

### Data Quality (Original Phases)
- [ ] Generated option data passes arbitrage checks
- [ ] Options properly track underlying price movements
- [ ] Backtesting results are realistic and actionable
- [ ] Data quality matches or exceeds LEAN RandomDataGenerator
- [ ] Performance remains 20x+ faster than original LEAN implementation

## Key Implementation Insight

**Total LEAN Compatibility Implementation: ~120 lines of code across 9 new functions**

After comprehensive analysis of 5,863 AAPL-related files in LEAN CLI output, the primary blocker for QC backtests is **missing coarse fundamental data** - QC cannot find securities to trade without daily universe files. The fast generator is 60% compatible with LEAN format but needs these critical supporting files for full QC integration.