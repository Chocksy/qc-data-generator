#!/usr/bin/env python3
"""
Data Format Analyzer
==================

Analyzes and validates generated options data formats.
Can optionally compare with LEAN RandomDataGenerator output if available.
"""

import os
import zipfile
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd


class DataAnalyzer:
    """Analyzes option data formats and structure"""
    
    def __init__(self, generated_data_dir: str = "generated_data", lean_data_dir: Optional[str] = None):
        self.generated_data_path = Path(generated_data_dir)
        self.lean_data_path = Path(lean_data_dir) if lean_data_dir else None
    
    def analyze_lean_data(self) -> Dict:
        """Analyze LEAN RandomDataGenerator output (if available)"""
        if not self.lean_data_path:
            print("ℹ️  No LEAN data directory specified, skipping LEAN analysis")
            return {}
            
        print("🔍 Analyzing LEAN RandomDataGenerator output...")
        
        if not self.lean_data_path.exists():
            print(f"❌ LEAN data directory not found: {self.lean_data_path}")
            return {}
        
        # Look for spy data first, then any subdirectory
        spy_path = self.lean_data_path / "option" / "usa" / "minute" / "spy"
        if spy_path.exists():
            search_path = spy_path
        else:
            # Try direct path
            search_path = self.lean_data_path
        
        zip_files = list(search_path.glob("*.zip"))
        if not zip_files:
            print(f"❌ No LEAN ZIP files found in {search_path}")
            return {}
        
        analysis = {
            'total_zip_files': len(zip_files),
            'file_types': set(),
            'csv_files': [],
            'sample_data': {}
        }
        
        # Analyze first ZIP file
        sample_zip = zip_files[0]
        print(f"📁 Analyzing {sample_zip.name}...")
        
        with zipfile.ZipFile(sample_zip, 'r') as zf:
            csv_files = zf.namelist()
            analysis['csv_files'] = csv_files
            
            # Extract file types
            for csv_file in csv_files:
                parts = csv_file.split('_')
                if len(parts) >= 6:
                    file_type = parts[5]  # quote, trade, openinterest
                    analysis['file_types'].add(file_type)
            
            # Sample data from first CSV
            if csv_files:
                sample_csv = csv_files[0]
                print(f"📄 Reading sample data from {sample_csv}")
                try:
                    with zf.open(sample_csv) as csv_file:
                        df = pd.read_csv(csv_file, header=None)
                        analysis['sample_data'][sample_csv] = {
                            'shape': df.shape,
                            'columns': df.columns.tolist(),
                            'first_rows': df.head().to_dict('records')
                        }
                except Exception as e:
                    print(f"⚠️  Failed to read sample data from {sample_csv}: {e}")
        
        return analysis
    
    def analyze_generated_data(self) -> Dict:
        """Analyze Fast Options Generator output"""
        print("🔍 Analyzing Fast Options Generator output...")
        
        if not self.generated_data_path.exists():
            print(f"❌ Generated data directory not found: {self.generated_data_path}")
            return {}
        
        # Look for option data in the typical structure
        spy_path = self.generated_data_path / "option" / "usa" / "minute" / "spy"
        if spy_path.exists():
            search_path = spy_path
        else:
            # Try to find any option data
            option_dirs = list(self.generated_data_path.rglob("option/usa/minute/*"))
            if option_dirs:
                search_path = option_dirs[0]  # Use first found
            else:
                search_path = self.generated_data_path
        
        zip_files = list(search_path.glob("*.zip"))
        if not zip_files:
            print(f"❌ No generated ZIP files found in {search_path}")
            return {}
        
        analysis = {
            'total_zip_files': len(zip_files),
            'file_types': set(),
            'csv_files': [],
            'sample_data': {}
        }
        
        # Analyze first ZIP file
        sample_zip = zip_files[0]
        print(f"📁 Analyzing {sample_zip.name}...")
        
        with zipfile.ZipFile(sample_zip, 'r') as zf:
            csv_files = zf.namelist()
            analysis['csv_files'] = csv_files
            
            # Extract file types
            for csv_file in csv_files:
                parts = csv_file.split('_')
                if len(parts) >= 6:
                    file_type = parts[5]  # quote, trade, openinterest
                    analysis['file_types'].add(file_type)
            
            # Sample data from first CSV
            if csv_files:
                sample_csv = csv_files[0]
                print(f"📄 Reading sample data from {sample_csv}")
                
                with zf.open(sample_csv) as csv_file:
                    df = pd.read_csv(csv_file, header=None)
                    analysis['sample_data'][sample_csv] = {
                        'shape': df.shape,
                        'columns': df.columns.tolist(),
                        'first_rows': df.head().to_dict('records')
                    }
        
        return analysis
    
    def compare_formats(self, lean_analysis: Dict, generated_analysis: Dict) -> None:
        """Compare data formats between LEAN and generated data"""
        print("\n📊 Format Comparison:")
        print("=" * 60)
        
        # Compare file counts
        print(f"LEAN ZIP files:      {lean_analysis.get('total_zip_files', 0)}")
        print(f"Generated ZIP files: {generated_analysis.get('total_zip_files', 0)}")
        
        # Compare file types
        print(f"\nLEAN file types:      {sorted(lean_analysis.get('file_types', set()))}")
        print(f"Generated file types: {sorted(generated_analysis.get('file_types', set()))}")
        
        # Compare sample data structures
        lean_sample = lean_analysis.get('sample_data', {})
        gen_sample = generated_analysis.get('sample_data', {})
        
        if lean_sample and gen_sample:
            lean_key = list(lean_sample.keys())[0]
            gen_key = list(gen_sample.keys())[0]
            
            print(f"\n📋 Sample Data Comparison:")
            print(f"LEAN shape:      {lean_sample[lean_key]['shape']}")
            print(f"Generated shape: {gen_sample[gen_key]['shape']}")
            
            print(f"\n📋 Column Count Comparison:")
            print(f"LEAN columns:    {len(lean_sample[lean_key]['columns'])}")
            print(f"Generated columns: {len(gen_sample[gen_key]['columns'])}")
    
    def validate_naming_conventions(self) -> None:
        """Validate file naming conventions"""
        print("\n🏷️  Validating naming conventions...")
        
        # Check LEAN naming (if available)
        if self.lean_data_path and self.lean_data_path.exists():
            zip_files = list(self.lean_data_path.glob("*.zip"))
            if zip_files:
                print(f"LEAN ZIP naming: {zip_files[0].name}")
                
                with zipfile.ZipFile(zip_files[0], 'r') as zf:
                    csv_files = zf.namelist()
                    if csv_files:
                        print(f"LEAN CSV naming: {csv_files[0]}")
        
        # Check generated naming
        if self.generated_data_path.exists():
            # Look for option data in the typical structure first
            search_paths = [
                self.generated_data_path / "option" / "usa" / "minute" / "spy",
                self.generated_data_path
            ]
            
            for search_path in search_paths:
                if search_path.exists():
                    zip_files = list(search_path.glob("*.zip"))
                    if zip_files:
                        print(f"Generated ZIP naming: {zip_files[0].name}")
                        
                        with zipfile.ZipFile(zip_files[0], 'r') as zf:
                            csv_files = zf.namelist()
                            if csv_files:
                                print(f"Generated CSV naming: {csv_files[0]}")
                        break
    
    def analyze_expiration_dates(self) -> None:
        """Analyze expiration date distribution"""
        print("\n📅 Analyzing expiration date distribution...")
        
        def extract_expirations(data_path: Path, name: str) -> List[str]:
            if not data_path.exists():
                return []
            
            expirations = set()
            zip_files = list(data_path.glob("*.zip"))
            
            for zip_file in zip_files:
                try:
                    with zipfile.ZipFile(zip_file, 'r') as zf:
                        csv_files = zf.namelist()
                        for csv_file in csv_files:
                            parts = csv_file.split('_')
                            if len(parts) >= 7:
                                expiration = parts[-1].replace('.csv', '')
                                expirations.add(expiration)
                except Exception as e:
                    print(f"⚠️  Skipping invalid zip file {zip_file.name}: {e}")
            
            return sorted(list(expirations))
        
        lean_expirations = extract_expirations(self.lean_data_path, "LEAN") if self.lean_data_path else []
        gen_expirations = extract_expirations(self.generated_data_path, "Generated")
        
        print(f"LEAN expirations: {lean_expirations}")
        print(f"Generated expirations: {gen_expirations}")
        
        # Check if we fixed the single expiration issue
        if len(lean_expirations) == 1:
            print("⚠️  LEAN data has single expiration issue")
        else:
            print("✅ LEAN data has multiple expirations")
            
        if len(gen_expirations) > 1:
            print("✅ Generated data has multiple expirations")
        else:
            print("❌ Generated data has single expiration issue")
    
    def run_full_analysis(self) -> None:
        """Run complete analysis and comparison"""
        print("🔬 Data Format Analysis and Comparison")
        print("=" * 60)
        
        # Analyze both datasets
        lean_analysis = self.analyze_lean_data()
        generated_analysis = self.analyze_generated_data()
        
        # Compare formats
        if lean_analysis and generated_analysis:
            self.compare_formats(lean_analysis, generated_analysis)
        
        # Validate naming conventions
        self.validate_naming_conventions()
        
        # Analyze expiration dates
        self.analyze_expiration_dates()
        
        print("\n✅ Analysis completed!")


def main():
    """Main analysis runner with CLI arguments"""
    parser = argparse.ArgumentParser(description="Analyze options data format and quality")
    parser.add_argument("--generated-data-dir", type=str, default="generated_data", 
                       help="Directory containing generated options data (default: generated_data)")
    parser.add_argument("--lean-data-dir", type=str, default=None,
                       help="Directory containing LEAN data for comparison (optional)")
    args = parser.parse_args()
    
    analyzer = DataAnalyzer(
        generated_data_dir=args.generated_data_dir,
        lean_data_dir=args.lean_data_dir
    )
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()