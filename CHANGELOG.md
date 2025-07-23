# Changelog

All notable changes to the Fast Options Generator will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-23

### Added
- 36x performance improvement over LEAN RandomDataGenerator
- Vectorized Black-Scholes pricing using NumPy
- Parallel processing for multiple contracts
- Universe data generation for QuantConnect compatibility
- Configurable data copying to any target directory
- Multi-asset support (SPY, QQQ, TSLA, etc.)
- Real underlying price data via yfinance
- Comprehensive testing and validation tools
- Standalone repository structure
- Enhanced run.sh script with multiple modes
- CLI arguments for all configuration options
- Data analysis and validation tools

### Features
- **Core Generator**: High-performance options data generation
- **Universe Data**: Generates required option and equity universe files
- **Flexible Copying**: Copy generated data to any absolute path
- **Multi-Asset**: Support for any underlying symbol with real price data
- **Testing Suite**: Performance tests and data validation
- **Analysis Tools**: Data format analysis and LEAN comparison
- **CLI Interface**: Complete command-line interface
- **Standalone**: No dependencies on specific directory structures

### Performance
- **Generation Speed**: <5 minutes for 1 month of data vs 3+ hours for LEAN
- **Memory Efficiency**: Streaming I/O prevents memory bottlenecks
- **Parallel Processing**: Configurable worker threads for optimal performance
- **Vectorized Calculations**: Batch processing using NumPy

### Data Compatibility
- **LEAN Format**: Exact CSV format compatibility
- **Universe Files**: Required universe data for proper backtesting
- **ZIP Compression**: Efficient storage using ZIP files
- **Naming Conventions**: LEAN-compatible file and directory naming

### Technical Details
- Python 3.8+ compatibility
- uv package manager support
- Comprehensive error handling
- Detailed logging and progress tracking
- Configurable parameters for all aspects of generation