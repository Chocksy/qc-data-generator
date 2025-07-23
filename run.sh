#!/bin/bash
# Universal Options Data Generator Runner
# Usage: ./run.sh [setup|test|generate|analyze|full] [args...]

echo "🚀 Fast Options Data Generator"
echo "=============================="

# Check if we're in the right directory
if [ ! -f "fast_options_generator.py" ]; then
    echo "❌ Error: fast_options_generator.py not found. Please run from project directory."
    exit 1
fi

# Function to setup environment
setup_environment() {
    echo "📦 Setting up Python environment..."
    uv venv
    source .venv/bin/activate
    uv pip install -r requirements.txt
    
    if [ $? -eq 0 ]; then
        echo "✅ Environment setup completed successfully"
        return 0
    else
        echo "❌ Failed to setup environment"
        return 1
    fi
}

# Function to run performance tests
run_tests() {
    echo "🧪 Running performance tests..."
    uv run python test_generator.py
    
    if [ $? -eq 0 ]; then
        echo "✅ Tests completed successfully"
        return 0
    else
        echo "❌ Tests failed"
        return 1
    fi
}

# Function to analyze generated data
analyze_data() {
    echo "🔍 Analyzing generated data..."
    
    # Prepare generated_data directory for analysis if it doesn't exist
    if [ ! -d "generated_data" ] && [ -d "test_output_full" ]; then
        echo "🔧 Preparing generated data for analysis..."
        rm -rf generated_data
        cp -R test_output_full generated_data
    fi
    
    uv run python data_analyzer.py --generated-data-dir generated_data
    
    if [ $? -eq 0 ]; then
        echo "✅ Analysis completed successfully"
        return 0
    else
        echo "❌ Analysis failed"
        return 1
    fi
}

# Function to generate data
generate_data() {
    echo "🔧 Generating options data with args: $@"
    uv run python fast_options_generator.py "$@"
    return $?
}

# Main script logic
case "$1" in
    "setup")
        setup_environment
        exit $?
        ;;
    
    "test")
        echo "Ensuring environment is ready..."
        if [ ! -d ".venv" ]; then
            setup_environment || exit 1
        fi
        run_tests
        exit $?
        ;;
    
    "generate")
        shift
        echo "Ensuring environment is ready..."
        if [ ! -d ".venv" ]; then
            setup_environment || exit 1
        fi
        generate_data "$@"
        exit $?
        ;;
    
    "analyze")
        echo "Ensuring environment is ready..."
        if [ ! -d ".venv" ]; then
            setup_environment || exit 1
        fi
        analyze_data
        exit $?
        ;;
    
    "full"|"")
        echo "🚀 Running full setup, test, and analysis workflow..."
        
        # Setup environment
        setup_environment || exit 1
        
        # Run tests
        echo ""
        run_tests || exit 1
        
        # Analyze data
        echo ""
        analyze_data
        
        echo ""
        echo "🎉 Full workflow completed!"
        echo ""
        echo "Next steps:"
        echo "1. Review test results and analysis above"
        echo "2. Check generated data in test_output_small/ and test_output_full/"
        echo "3. Generate data for your use case:"
        echo "   ./run.sh generate --start-date YYYY-MM-DD --end-date YYYY-MM-DD --copy-target-dir /path/to/target"
        echo "4. Use generated data in your trading algorithms"
        ;;
    
    "help"|"-h"|"--help")
        echo ""
        echo "Usage: ./run.sh [command] [options]"
        echo ""
        echo "Commands:"
        echo "  setup           Set up Python environment and install dependencies"
        echo "  test            Run performance tests and validation"
        echo "  generate [args] Generate options data (pass args to fast_options_generator.py)"
        echo "  analyze         Analyze generated data format and quality"
        echo "  full            Run complete workflow: setup + test + analyze (default)"
        echo "  help            Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./run.sh setup"
        echo "  ./run.sh test"
        echo "  ./run.sh generate --start-date 2024-01-01 --end-date 2024-01-31"
        echo "  ./run.sh generate --underlying QQQ --copy-target-dir ~/data"
        echo "  ./run.sh analyze"
        echo "  ./run.sh full"
        echo ""
        ;;
    
    *)
        echo "❌ Unknown command: $1"
        echo "Run './run.sh help' for usage information"
        exit 1
        ;;
esac