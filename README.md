# Fast Options Data Generator

A high-performance Python-based options data generator that provides **36x faster** generation compared to the original LEAN RandomDataGenerator. Creates LEAN-compatible data with universe files for seamless QuantConnect backtesting.

## 🚀 Performance Comparison

| Generator | Time for 1 Month | Performance |
|-----------|------------------|-------------|
| LEAN RandomDataGenerator | ~3 hours | Baseline |
| Fast Python Generator | <5 minutes | **36x faster** |

## ✨ Features

- **Vectorized Black-Scholes Pricing**: Uses NumPy for batch calculations
- **Parallel Processing**: Multi-threaded contract generation
- **Universe Data Generation**: Creates required universe files for QuantConnect
- **Streaming I/O**: Efficient file writing without memory bottlenecks
- **LEAN-Compatible Format**: Generates exact LEAN CSV format with universe data
- **Configurable Data Copying**: Copy generated data to any target directory
- **Multiple Asset Support**: Generate data for any underlying symbol (SPY, QQQ, TSLA, etc.)
- **Progress Tracking**: Real-time generation progress
- **Comprehensive Testing**: Built-in validation and performance tests

## 📦 Installation

### Prerequisites
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Python 3.8+

### Setup
```bash
# Clone or download this repository to your desired location
cd /path/to/fast-options-generator

# Set up environment and install dependencies
./run.sh setup
```

## 🏃 Quick Start

### 1. Run Full Test Suite
```bash
./run.sh full
```
This runs setup, performance tests, and data analysis.

### 2. Generate Data for Your Use Case
```bash
# Generate SPY options data and copy to your target directory
./run.sh generate \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --underlying SPY \
  --copy-target-dir ~/development/quant-algos/qc-algos/data

# Generate QQQ options with different parameters
./run.sh generate \
  --start-date 2024-02-01 \
  --end-date 2024-02-14 \
  --underlying QQQ \
  --strikes-per-expiration 20 \
  --min-dte 7 \
  --max-dte 30 \
  --copy-target-dir /absolute/path/to/your/data/folder
```

### 3. Analyze Generated Data
```bash
./run.sh analyze
```

## 🔧 Available Commands

### ./run.sh Commands

```bash
./run.sh setup           # Set up Python environment and install dependencies
./run.sh test            # Run performance tests and validation
./run.sh generate [args] # Generate options data (pass args to generator)
./run.sh analyze         # Analyze generated data format and quality
./run.sh full            # Run complete workflow: setup + test + analyze (default)
./run.sh help            # Show help message
```

### Direct Generator Usage

```bash
# Show all available options
uv run python fast_options_generator.py --help

# Basic generation (no copying)
uv run python fast_options_generator.py \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --underlying SPY

# Generate with data copying
uv run python fast_options_generator.py \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --underlying QQQ \
  --copy-target-dir /absolute/path/to/your/data

# Include equity universe files (for dynamic universe selection)
uv run python fast_options_generator.py \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --include-coarse-universe \
  --copy-target-dir /path/to/data
```

## 📊 Configuration Options

### Core Parameters
- `--start-date`, `--end-date`: Date range in YYYY-MM-DD format
- `--underlying`: Symbol (SPY, QQQ, TSLA, etc.)
- `--strikes-per-expiration`: Number of strikes per expiration (default: 40)
- `--min-dte`, `--max-dte`: Days to expiration range (default: 30, 30)
- `--max-workers`: Parallel workers (default: 8)
- `--output-dir`: Local output directory (default: generated_data)

### Data Copying & Universe Options
- `--copy-target-dir`: Absolute path to copy generated data (optional)
- `--generate-universes`: Generate universe files (default: on)
- `--no-generate-universes`: Disable universe generation
- `--include-coarse-universe`: Generate equity universe files for underlying

### Advanced Options
- `--base-price`: Base underlying price (default: 400.0)
- `--risk-free-rate`: Risk-free rate (default: 0.02)
- `--volatility`: Implied volatility (default: 0.20)
- `--resolution`: Data resolution (default: minute)
- `--fetch-underlying`: Download real prices via yfinance (default: on)

## 📁 Output Structure

The generator creates LEAN-compatible data files with universe data:

```
generated_data/
├── option/usa/
│   ├── minute/
│   │   └── spy/
│   │       ├── 20240101_quote_american.zip
│   │       ├── 20240101_trade_american.zip
│   │       ├── 20240101_openinterest_american.zip
│   │       └── ...
│   └── universes/          # ← NEW: Universe data for QuantConnect
│       └── spy/
│           ├── 20240101.csv
│           ├── 20240102.csv
│           └── ...
└── equity/usa/fundamental/ # ← Optional: Equity universe data
    └── coarse/
        ├── 20240101.csv
        └── ...
```

### Universe File Format

**Option Universe Files** (`option/usa/universes/{underlying}/{YYYYMMDD}.csv`):
```csv
Symbol,Expiry,Strike,OptionType,Right
SPY20240105C04000000,20240105,400.00,Call,C
SPY20240105P04000000,20240105,400.00,Put,P
```

**Coarse Universe Files** (`equity/usa/fundamental/coarse/{YYYYMMDD}.csv`):
```csv
SecurityID,Symbol,Close,Volume,DollarVolume,HasFundamentalData,PriceFactor,SplitFactor
SPY R735QTJ8XC9X,SPY,450.25,100000000,45025000000,False,1.0,1.0
```

## 🧪 Testing & Validation

### Performance Testing
```bash
# Run all tests
./run.sh test

# Run specific test types
uv run python test_generator.py --test-type small
uv run python test_generator.py --test-type full
uv run python test_generator.py --copy-target-dir /path/to/data
```

### Data Analysis
```bash
# Analyze generated data only
./run.sh analyze

# Compare with LEAN data (if available)
uv run python data_analyzer.py \
  --generated-data-dir generated_data \
  --lean-data-dir /path/to/lean/data
```

## 🔗 Integration Examples

### QuantConnect Algorithm Integration

1. **Generate Data**:
```bash
./run.sh generate \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --underlying SPY \
  --copy-target-dir ~/my-quant-project/data
```

2. **Use in QuantConnect Algorithm**:
```csharp
public class MyOptionsAlgorithm : QCAlgorithm
{
    public override void Initialize()
    {
        SetStartDate(2024, 1, 1);
        SetEndDate(2024, 1, 31);
        
        // The universe files ensure proper contract availability
        var option = AddOption("SPY");
        option.SetFilter(universe => universe.Strikes(-5, 5).Expiration(0, 30));
    }
}
```

### Python API Usage

```python
from fast_options_generator import FastOptionsGenerator, GeneratorConfig
from datetime import datetime

# Configure generation
config = GeneratorConfig(
    underlying_symbol="QQQ",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31),
    strikes_per_expiration=20,
    min_dte=7,
    max_dte=30,
    copy_target_dir="/absolute/path/to/your/data",
    include_coarse_universe=True
)

# Generate
generator = FastOptionsGenerator(config)
generator.generate()
```

## 🛠️ Key Improvements Over LEAN Generator

1. **36x Performance**: Vectorized calculations and parallel processing
2. **Universe Data**: Generates required universe files automatically
3. **Configurable Copying**: Copy data anywhere, not hardcoded paths
4. **Multi-Asset Support**: Any underlying symbol with real price data
5. **Standalone**: No dependencies on specific directory structures
6. **Better Validation**: Comprehensive testing and data analysis tools

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Run `./run.sh setup` to install dependencies
2. **Performance Issues**: Adjust `--max-workers` based on your system
3. **Memory Issues**: Reduce `--strikes-per-expiration`
4. **Path Issues**: Use absolute paths for `--copy-target-dir`

### Debug Mode
```bash
# Enable detailed logging
export PYTHONPATH=.
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
./run.sh generate [args]
```

## 📝 Repository Structure

```
fast-options-generator/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── run.sh                      # Universal runner script
├── fast_options_generator.py   # Main generator
├── test_generator.py           # Performance tests
├── data_analyzer.py           # Data validation
└── examples/                  # Usage examples (future)
```

## 🤝 Contributing

This is a standalone repository. To contribute:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📞 Support

For issues:
1. Check the troubleshooting section above
2. Run `./run.sh test` to validate your setup
3. Use `./run.sh help` for command reference

---

*High-performance options data generation for quantitative trading research and backtesting.*