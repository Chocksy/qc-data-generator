# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a high-performance Python-based options data generator that creates LEAN-compatible data files for QuantConnect backtesting. It provides 36x faster generation compared to the original LEAN RandomDataGenerator.

## Core Architecture

### Main Components
- `fast_options_generator.py` - Core generator with vectorized Black-Scholes pricing and parallel processing
- `test_generator.py` - Performance testing and validation suite  
- `data_analyzer.py` - Data format analysis and validation tools
- `run.sh` - Universal runner script for all operations

### Key Classes
- `FastOptionsGenerator` - Main generator class with vectorized calculations
- `GeneratorConfig` - Configuration dataclass for all generation parameters
- `DataAnalyzer` - Validates generated data format and structure

## Development Commands

### Environment Setup
```bash
./run.sh setup           # Set up Python environment with uv and install dependencies
```

### Testing
```bash
./run.sh test            # Run performance tests and validation
uv run python test_generator.py --test-type small  # Quick 3-day test
uv run python test_generator.py --test-type full   # Full month test
```

### Code Quality
```bash
# Linting and formatting (configured in pyproject.toml)
uv run black . --line-length 100    # Format code
uv run flake8 .                     # Lint code
```

### Data Generation
```bash
./run.sh generate [args]  # Generate options data
# Example: ./run.sh generate --start-date 2024-01-01 --end-date 2024-01-31 --underlying SPY
```

### Analysis
```bash
./run.sh analyze         # Analyze generated data format and quality
```

## Technology Stack

- **Python 3.9+** with uv package manager
- **NumPy/SciPy** for vectorized Black-Scholes calculations
- **Pandas** for data manipulation
- **yfinance** for real underlying price data
- **Threading/Concurrent.futures** for parallel processing

## Output Structure

The generator creates LEAN-compatible data in this structure:
```
generated_data/
├── option/usa/minute/{symbol}/     # ZIP files with quote/trade/openinterest data
└── option/usa/universes/{symbol}/  # CSV universe files for QuantConnect
```

## Performance Targets

- Small dataset (3 days): <10 seconds
- Full month: <5 minutes (vs 3 hours for original LEAN generator)
- Memory efficient through streaming I/O

## Key Configuration Options

Major parameters in `GeneratorConfig`:
- `strikes_per_expiration` - Number of strikes per expiration (default: 40)
- `min_dte/max_dte` - Days to expiration range
- `max_workers` - Parallel processing threads
- `copy_target_dir` - Optional directory to copy generated data
- `fetch_underlying` - Whether to fetch real price data via yfinance