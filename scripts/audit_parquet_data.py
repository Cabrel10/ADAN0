#!/usr/bin/env python3
"""
Audit script for Parquet market data files to check for:
1. Volume anomalies (zeros, constant values, outliers)
2. Mismatch between configured and actual features
3. Data quality issues
"""
import os
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('parquet_audit.log')
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

class ParquetAuditor:
    def __init__(self, config_path: str = None):
        """Initialize with config."""
        if config_path is None:
            config_path = PROJECT_ROOT / 'bot' / 'config' / 'config.yaml'
        
        self.config = self._load_config(config_path)
        self.data_dir = Path(self.config['paths']['processed_data_dir']) / 'indicators'
        self.timeframes = self.config['data']['timeframes']
        self.assets = self.config['environment']['assets']
        self.features_config = self.config['data']['features_per_timeframe']
        
        # Results storage
        self.volume_issues = []
        self.feature_issues = []
        self.data_quality_issues = []
    
    def _load_config(self, config_path: str) -> dict:
        """Load YAML config file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Forcer la structure attendue pour les features
            if 'data' not in config:
                config['data'] = {}
                
            if 'features_per_timeframe' not in config.get('data', {}):
                # Valeurs par défaut basées sur les colonnes trouvées
                config['data']['features_per_timeframe'] = {
                    '5m': {
                        'indicators': [
                            'RSI_14', 'CCI_20', 'ROC_9', 'MFI_14', 'EMA_5', 'EMA_20',
                            'SUPERTREND_14_2.0', 'PSAR_0.02_0.2', 'ATR_14', 'BB_UPPER',
                            'BB_MIDDLE', 'BB_LOWER', 'VWAP', 'OBV', 'ATR_PCT',
                            'MACD_12_26_9', 'MACD_SIGNAL_12_26_9', 'MACD_HIST_12_26_9',
                            'ADX_14', 'STOCH_K_14_3_3', 'STOCH_D_14_3_3', 'EMA_200',
                            'EMA_RATIO_FAST_SLOW', 'EMA_12', 'EMA_26'
                        ]
                    },
                    '1h': {
                        'indicators': [
                            'RSI_14', 'CCI_20', 'ROC_9', 'MFI_14', 'EMA_5', 'EMA_20',
                            'SUPERTREND_14_2.0', 'PSAR_0.02_0.2', 'ATR_14', 'BB_UPPER',
                            'BB_MIDDLE', 'BB_LOWER', 'VWAP', 'OBV', 'ATR_PCT',
                            'MACD_12_26_9', 'MACD_SIGNAL_12_26_9', 'MACD_HIST_12_26_9',
                            'ADX_14', 'STOCH_K_14_3_3', 'STOCH_D_14_3_3', 'EMA_200',
                            'EMA_RATIO_FAST_SLOW', 'EMA_12', 'EMA_26'
                        ]
                    },
                    '4h': {
                        'indicators': [
                            'RSI_14', 'CCI_20', 'ROC_9', 'MFI_14', 'EMA_5', 'EMA_20',
                            'SUPERTREND_14_2.0', 'PSAR_0.02_0.2', 'ATR_14', 'BB_UPPER',
                            'BB_MIDDLE', 'BB_LOWER', 'VWAP', 'OBV', 'ATR_PCT',
                            'MACD_12_26_9', 'MACD_SIGNAL_12_26_9', 'MACD_HIST_12_26_9',
                            'ADX_14', 'STOCH_K_14_3_3', 'STOCH_D_14_3_3', 'EMA_200',
                            'EMA_RATIO_FAST_SLOW', 'EMA_12', 'EMA_26'
                        ]
                    }
                }
                
            return config
            
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def find_parquet_files(self) -> Dict[str, Dict[str, Dict[str, Path]]]:
        """Find all parquet files in the processed data directory."""
        files = {}
        base_dir = Path('/home/morningstar/Documents/trading/bot/data/processed/indicators')
        
        for split in ['train', 'val', 'test']:
            split_dir = base_dir / split
            if not split_dir.exists():
                logger.warning(f"Split directory not found: {split_dir}")
                continue
                
            # Parcourir les sous-répertoires d'actifs
            for asset_dir in split_dir.iterdir():
                if not asset_dir.is_dir():
                    continue
                    
                asset = asset_dir.name.upper()
                if asset not in files:
                    files[asset] = {}
                    
                # Vérifier les fichiers par timeframe
                for tf in ['5m', '1h', '4h']:
                    file_path = asset_dir / f"{tf}.parquet"
                    if file_path.exists():
                        if tf not in files[asset]:
                            files[asset][tf] = {}
                        files[asset][tf][split] = file_path
        
        return files
    
    def check_volume(self, df: pd.DataFrame, asset: str, timeframe: str) -> List[dict]:
        """Check for volume anomalies."""
        issues = []
        
        if 'VOLUME' not in df.columns:
            issues.append({
                'type': 'missing_volume',
                'message': f"VOLUME column missing in {asset} {timeframe}"
            })
            return issues
            
        volume = df['VOLUME']
        
        # Check for all zeros
        if (volume == 0).all():
            issues.append({
                'type': 'all_zero_volume',
                'message': f"All volume values are zero in {asset} {timeframe}"
            })
        
        # Check for constant volume
        elif volume.nunique() == 1:
            issues.append({
                'type': 'constant_volume',
                'message': f"Volume is constant ({volume.iloc[0]}) in {asset} {timeframe}"
            })
            
        # Check for extreme values (beyond 6 standard deviations)
        if len(volume) > 1:  # Need at least 2 points for std
            z_scores = (volume - volume.mean()) / volume.std()
            extreme_mask = z_scores.abs() > 6
            if extreme_mask.any():
                num_extreme = extreme_mask.sum()
                pct_extreme = (num_extreme / len(volume)) * 100
                issues.append({
                    'type': 'extreme_volume',
                    'message': (
                        f"Found {num_extreme} extreme volume values "
                        f"({pct_extreme:.2f}%) in {asset} {timeframe}"
                    )
                })
        
        return issues
    
    def check_features(self, df: pd.DataFrame, asset: str, timeframe: str) -> List[dict]:
        """Check for feature mismatches between config and actual data."""
        issues = []
        
        # Colonnes de base attendues
        base_columns = {'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'}
        
        # Vérifier les colonnes de base
        missing_base = base_columns - set(col.upper() for col in df.columns)
        if missing_base:
            issues.append({
                'type': 'missing_base_columns',
                'message': f"Missing base columns in {asset} {timeframe}: {', '.join(missing_base)}"
            })
        
        # Vérifier les indicateurs techniques
        expected_indicators = {
            'RSI_14', 'CCI_20', 'ROC_9', 'MFI_14', 'EMA_5', 'EMA_20',
            'SUPERTREND_14_2.0', 'PSAR_0.02_0.2', 'ATR_14', 'BB_UPPER',
            'BB_MIDDLE', 'BB_LOWER', 'VWAP', 'OBV', 'ATR_PCT',
            'MACD_12_26_9', 'MACD_SIGNAL_12_26_9', 'MACD_HIST_12_26_9',
            'ADX_14', 'STOCH_K_14_3_3', 'STOCH_D_14_3_3', 'EMA_200',
            'EMA_RATIO_FAST_SLOW', 'EMA_12', 'EMA_26'
        }
        
        actual_columns = set(col.upper() for col in df.columns)
        
        # Vérifier les indicateurs manquants
        missing_indicators = expected_indicators - actual_columns
        if missing_indicators:
            issues.append({
                'type': 'missing_indicators',
                'message': (
                    f"Missing {len(missing_indicators)} indicators in {asset} {timeframe}: "
                    f"{', '.join(sorted(missing_indicators))}"
                )
            })
        
        # Vérifier les colonnes inattendues
        extra_columns = actual_columns - expected_indicators - base_columns
        if extra_columns:
            issues.append({
                'type': 'extra_columns',
                'message': (
                    f"Found {len(extra_columns)} unexpected columns in {asset} {timeframe}: "
                    f"{', '.join(sorted(extra_columns))}"
                )
            })
            
        return issues
    
    def check_data_quality(self, df: pd.DataFrame, asset: str, timeframe: str) -> List[dict]:
        """Check for general data quality issues."""
        issues = []
        
        # Check for missing values
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0]
        
        for col, count in null_cols.items():
            pct = (count / len(df)) * 100
            issues.append({
                'type': 'missing_values',
                'message': (
                    f"Column {col} in {asset} {timeframe} has {count} "
                    f"({pct:.2f}%) missing values"
                )
            })
        
        # Check for constant columns (excluding datetime index)
        for col in df.columns:
            if df[col].nunique() == 1 and col != 'datetime':
                issues.append({
                    'type': 'constant_column',
                    'message': f"Column {col} in {asset} {timeframe} is constant"
                })
                
        return issues
    
    def process_file(self, file_paths: dict, asset: str, timeframe: str):
        """Process parquet files for train/val/test splits."""
        for split, file_path in file_paths.items():
            try:
                logger.info(f"Processing {asset} {timeframe} - {split}")
                df = pd.read_parquet(file_path)
                if df.empty:
                    logger.warning(f"Empty DataFrame: {file_path}")
                    continue
                    
                # Convert column names to uppercase for consistency
                df.columns = [col.upper() for col in df.columns]
                
                # Run checks
                self.volume_issues.extend(self.check_volume(df, f"{asset} ({split})", timeframe))
                self.feature_issues.extend(self.check_features(df, f"{asset} ({split})", timeframe))
                self.data_quality_issues.extend(self.check_data_quality(df, f"{asset} ({split})", timeframe))
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
    
    def run_audit(self):
        """Run the full audit."""
        logger.info("Starting Parquet data audit...")
        
        # Find all parquet files
        files = self.find_parquet_files()
        if not files:
            logger.error("No parquet files found!")
            return
            
        logger.info(f"Found {len(files)} assets with parquet files")
        
        # Process each file
        for asset in tqdm(sorted(files.keys()), desc="Processing assets"):
            timeframes = files[asset]
            for tf in sorted(timeframes.keys()):
                file_paths = timeframes[tf]
                self.process_file(file_paths, asset, tf)
        
        # Print summary
        self.print_summary()
        
        # Save detailed report
        self.save_report()
    
    def print_summary(self):
        """Print audit summary."""
        print("\n" + "="*50)
        print("PARQUET DATA AUDIT SUMMARY")
        print("="*50)
        
        # Volume issues
        print(f"\nVOLUME ISSUES ({len(self.volume_issues)}):")
        for issue in self.volume_issues[:5]:  # Show first 5
            print(f"- {issue['message']}")
        if len(self.volume_issues) > 5:
            print(f"... and {len(self.volume_issues) - 5} more")
        
        # Feature issues
        print(f"\nFEATURE ISSUES ({len(self.feature_issues)}):")
        for issue in self.feature_issues[:5]:
            print(f"- {issue['message']}")
        if len(self.feature_issues) > 5:
            print(f"... and {len(self.feature_issues) - 5} more")
        
        # Data quality issues
        print(f"\nDATA QUALITY ISSUES ({len(self.data_quality_issues)}):")
        for issue in self.data_quality_issues[:5]:
            print(f"- {issue['message']}")
        if len(self.data_quality_issues) > 5:
            print(f"... and {len(self.data_quality_issues) - 5} more")
        
        print("\n" + "="*50)
        print("AUDIT COMPLETE")
        print("="*50)
    
    def save_report(self, output_file: str = "parquet_audit_report.txt"):
        """Save detailed report to file."""
        with open(output_file, 'w') as f:
            f.write("PARQUET DATA AUDIT REPORT\n")
            f.write("="*50 + "\n\n")
            
            # Volume issues
            f.write("VOLUME ISSUES:\n")
            f.write("-"*50 + "\n")
            for issue in self.volume_issues:
                f.write(f"{issue['message']}\n")
            
            # Feature issues
            f.write("\nFEATURE ISSUES:\n")
            f.write("-"*50 + "\n")
            for issue in self.feature_issues:
                f.write(f"{issue['message']}\n")
            
            # Data quality issues
            f.write("\nDATA QUALITY ISSUES:\n")
            f.write("-"*50 + "\n")
            for issue in self.data_quality_issues:
                f.write(f"{issue['message']}\n")
            
            # Summary
            f.write("\n" + "="*50 + "\n")
            f.write("SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total volume issues: {len(self.volume_issues)}\n")
            f.write(f"Total feature issues: {len(self.feature_issues)}\n")
            f.write(f"Total data quality issues: {len(self.data_quality_issues)}\n")
            
        logger.info(f"Detailed audit report saved to {output_file}")

if __name__ == "__main__":
    # Run the auditor
    auditor = ParquetAuditor()
    auditor.run_audit()
