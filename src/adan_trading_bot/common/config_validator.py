#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration validation utilities for the ADAN trading bot.

This module provides comprehensive validation for all configuration files
to ensure they meet the requirements specified in the design document.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

class ConfigValidator:
    """
    Validates configuration dictionaries against expected schemas.
    
    This validator ensures that all required configuration parameters are present
    and have valid values according to the ADAN trading bot specifications.
    """
    
    def __init__(self):
        """Initialize the configuration validator."""
        self.validation_errors = []
        self.validation_warnings = []
    
    def validate_all_configs(self, config_dir: Union[str, Path]) -> bool:
        """
        Validate all configuration sections in a single configuration file.
        
        Args:
            config_dir: Path to the configuration directory
            
        Returns:
            True if all configurations are valid, False otherwise
        """
        config_dir = Path(config_dir)
        self.validation_errors.clear()
        self.validation_warnings.clear()
        
        logger.info(f"Validating all configurations in {config_dir}")
        
        # Validate single configuration file
        config_path = config_dir / 'config.yaml'
        if not config_path.exists():
            self.validation_errors.append(f"Missing required configuration file: config.yaml")
            return False
            
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate each section
            validations = [
                self.validate_main_config(config, config_path.name),
                self.validate_data_config(config, config_path.name),
                self.validate_environment_config(config, config_path.name),
                self.validate_train_config(config, config_path.name),
                self.validate_dbe_config(config, config_path.name),
                self.validate_ppo_config(config, config_path.name),
                self.validate_trading_config(config, config_path.name),
                self.validate_memory_config(config, config_path.name),
                self.validate_risk_config(config, config_path.name),
                self.validate_agent_config(config, config_path.name)
            ]
            
            # Validate worker consistency
            if not self.validate_worker_config_consistency(config, config_path.name):
                validations.append(False)
                
            all_valid = all(validations)
            
        except Exception as e:
            self.validation_errors.append(f"Error loading configuration: {str(e)}")
            all_valid = False
        
        # Log validation results
        if self.validation_errors:
            logger.error(f"Configuration validation failed with {len(self.validation_errors)} errors:")
            for error in self.validation_errors:
                logger.error(f"  - {error}")
        
        if self.validation_warnings:
            logger.warning(f"Configuration validation completed with {len(self.validation_warnings)} warnings:")
            for warning in self.validation_warnings:
                logger.warning(f"  - {warning}")
        
        if all_valid:
            logger.info("All configuration files validated successfully")
        
        return all_valid
    
    def validate_main_config(self, config: Dict[str, Any], filename: str) -> bool:
        """Validate main configuration."""
        logger.debug(f"Validating {filename}")
        valid = True
        
        # Required sections
        required_sections = ['data', 'agent', 'logging', 'training', 'environment', 'paths']
        for section in required_sections:
            if section not in config:
                self.validation_errors.append(f"{filename}: Missing required section '{section}'")
                valid = False
        
        # Validate data section
        if 'data' in config:
            data_config = config['data']
            required_data_keys = ['data_dir']
            for key in required_data_keys:
                if key not in data_config:
                    self.validation_errors.append(f"{filename}: Missing required data key '{key}'")
                    valid = False
        
        return valid
    
    
    
    def validate_environment_config(self, config: Dict[str, Any], filename: str) -> bool:
        """Validate environment configuration."""
        logger.debug(f"Validating {filename} - Environment section")
        valid = True

        # Check if environment configuration exists
        if 'environment' not in config:
            self.validation_errors.append(f"Missing required environment configuration in {filename}")
            return False

        # Get environment configuration
        env_config = config['environment']
        
        # Validate assets
        if 'assets' not in env_config:
            self.validation_errors.append(f"{filename}: Missing required assets key in environment section")
            valid = False
        else:
            expected_assets = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA']
            if not all(asset in expected_assets for asset in env_config['assets']):
                self.validation_warnings.append(
                    f"{filename}: Some assets not in expected list {expected_assets}"
                )

        # Validate observation configuration
        if 'observation' in env_config:
            obs_config = env_config['observation']
            
            # Validate timeframes
            if 'timeframes' not in obs_config:
                self.validation_errors.append(f"{filename}: Missing required timeframes key in observation section")
                valid = False
            else:
                expected_timeframes = ['5m', '1h', '4h']
                if obs_config['timeframes'] != expected_timeframes:
                    self.validation_warnings.append(
                        f"{filename}: Timeframes {obs_config['timeframes']} differ from expected {expected_timeframes}"
                    )

            # Validate observation space
            if 'shape' not in obs_config:
                self.validation_errors.append(f"Missing required key 'shape' in observation space")
                valid = False

            # Validate observation features
            if 'features' in obs_config:
                required_features = ['base', 'indicators']
                missing_features = [f for f in required_features if f not in obs_config['features']]
                if missing_features:
                    self.validation_errors.append(f"Missing required observation features sections: {missing_features}")
                    valid = False
        else:
            self.validation_errors.append("Missing observation section in environment configuration")
            valid = False

        # Validate memory configuration
        if 'memory' in env_config:
            mem_config = env_config['memory']
            
            required_keys = ['chunk_size', 'max_chunks_in_memory']
            for key in required_keys:
                if key not in mem_config:
                    self.validation_errors.append(f"Missing required key '{key}' in memory configuration")
                    valid = False
        else:
            self.validation_errors.append("Missing memory section in environment configuration")
            valid = False
        
        return valid
    
    def validate_train_config(self, config: Dict[str, Any], filename: str) -> bool:
        """Validate training configuration."""
        logger.debug(f"Validating {filename} - Training section")
        valid = True
        
        if 'training' not in config:
            self.validation_errors.append(f"Missing required training section in {filename}")
            return False
            
        train_config = config['training']
        
        # Required parameters
        required_params = ['num_instances', 'timesteps_per_instance', 'batch_size', 'save_freq']
        for param in required_params:
            if param not in train_config:
                self.validation_errors.append(f"{filename}: Missing required training parameter '{param}'")
                valid = False
        
        if 'agent' in config and 'n_envs' in config['agent'] and config['agent']['n_envs'] != train_config.get('num_instances'):
            self.validation_warnings.append(
                f"{filename}: n_envs in agent config should match num_instances in training config"
            )
        
        return valid
    
    def validate_dbe_config(self, config: Dict[str, Any], filename: str) -> bool:
        """Validate Dynamic Behavior Engine configuration."""
        logger.debug(f"Validating {filename} - DBE section")
        valid = True
        
        if 'dbe' not in config:
            self.validation_errors.append(f"Missing required dbe section in {filename}")
            return False

        dbe_config = config['dbe']
        
        # Validate workers configuration
        if 'workers' not in dbe_config:
            self.validation_errors.append(f"Missing workers configuration in DBE section")
            return False

        # Validate each worker
        required_workers = ['w1', 'w2', 'w3', 'w4']
        for worker_id in required_workers:
            if worker_id not in dbe_config['workers']:
                self.validation_errors.append(f"Missing required worker '{worker_id}' in DBE section")
                valid = False
                continue

            worker_config = dbe_config['workers'][worker_id]
            
            # Required fields for each worker
            required_fields = ['regime', 'bias', 'indicators', 'filters']
            if 'adaptive' in worker_config and worker_config['adaptive']:
                required_fields.remove('bias') # Bias is not required for adaptive workers

            for field in required_fields:
                if field not in worker_config:
                    self.validation_errors.append(f"Missing required field '{field}' in worker {worker_id}")
                    valid = False

            # Validate regime
            valid_regimes = ['bull', 'volatile', 'sideways', 'dynamic']
            if 'regime' in worker_config and worker_config['regime'] not in valid_regimes:
                self.validation_errors.append(f"Invalid regime '{worker_config['regime']}' for worker {worker_id}")
                valid = False

            # Validate bias
            if 'bias' in worker_config:
                try:
                    float(worker_config['bias'])
                except (ValueError, TypeError):
                    self.validation_errors.append(f"Invalid bias format for worker {worker_id}")
                    valid = False

            # Validate indicators
            if 'indicators' in worker_config and (not isinstance(worker_config['indicators'], list) or len(worker_config['indicators']) == 0):
                self.validation_errors.append(f"Invalid indicators list for worker {worker_id}")
                valid = False
        
        return valid
    
    def validate_agent_config(self, config: Dict[str, Any], filename: str) -> bool:
        """Validate agent configuration."""
        logger.debug(f"Validating {filename} - Agent section")
        valid = True

        if 'agent' not in config:
            self.validation_errors.append(f"Missing required agent section in {filename}")
            return False

        agent_config = config['agent']
        
        # Required fields
        required_fields = ['algorithm', 'policy', 'seed', 'verbose', 'deterministic_inference',
                         'custom_log_freq_rollouts', 'eval_freq', 'checkpoint_freq', 'n_envs',
                         'batch_size', 'buffer_size', 'window_size', 'features_extractor_kwargs',
                         'policy_kwargs']
        for field in required_fields:
            if field not in agent_config:
                self.validation_errors.append(f"Missing required agent key '{field}'")
                valid = False

        # Validate policy_kwargs
        if 'policy_kwargs' in agent_config:
            policy_kwargs = agent_config['policy_kwargs']
            if 'net_arch' not in policy_kwargs:
                self.validation_errors.append("Missing net_arch in policy_kwargs")
                valid = False
            else:
                net_arch = policy_kwargs['net_arch']
                required_arch_parts = ['shared', 'pi', 'vf']
                for part in required_arch_parts:
                    if part not in net_arch:
                        self.validation_errors.append(f"Missing {part} in net_arch")
                        valid = False
                    else:
                        try:
                            layers = net_arch[part]
                            if not isinstance(layers, list):
                                self.validation_errors.append(f"{part} must be a list of integers")
                                valid = False
                            for layer in layers:
                                if not isinstance(layer, int) or layer <= 0:
                                    self.validation_errors.append(f"Invalid layer size in {part}")
                                    valid = False
                        except (ValueError, TypeError):
                            self.validation_errors.append(f"Invalid layer configuration in {part}")
                            valid = False

            # Validate additional policy parameters
            required_params = ['activation_fn', 'gamma', 'gae_lambda', 'ent_coef', 'learning_rate',
                             'learning_rate_schedule', 'n_steps']
            for param in required_params:
                if param not in policy_kwargs:
                    self.validation_errors.append(f"Missing required policy parameter '{param}'")
                    valid = False
        return valid

    def validate_trading_config(self, config: Dict[str, Any], filename: str) -> bool:
        """Validate trading configuration."""
        logger.debug(f"Validating {filename} - Trading section")
        valid = True

        # Check if trading configuration exists
        if 'trading' not in config:
            self.validation_errors.append(f"Missing required trading configuration in {filename}")
            return False

        # Get trading configuration
        trading_config = config['trading']
        
        # Check if workers configuration exists
        if 'workers' not in trading_config:
            self.validation_errors.append(f"Missing required workers in trading configuration")
            return False

        # Validate each worker
        required_workers = ['w1', 'w2', 'w3', 'w4']
        for worker in required_workers:
            if worker not in trading_config['workers']:
                self.validation_errors.append(f"Missing required worker {worker} in trading configuration")
                valid = False
                continue

            worker_config = trading_config['workers'][worker]
            
            # Check required fields
            required_fields = ['stop_loss_pct', 'take_profit_pct', 'position_size_pct']
            for field in required_fields:
                if field not in worker_config:
                    self.validation_errors.append(f"Missing required field '{field}' in trading worker {worker}")
                    valid = False
                    continue

                value = worker_config[field]
                
                # Validate stop_loss_pct/take_profit_pct format
                if field in ['stop_loss_pct', 'take_profit_pct']:
                    if isinstance(value, str):
                        if 'ATR' in value:
                            try:
                                multiplier = float(value.split('*')[1])
                            except (ValueError, IndexError):
                                self.validation_errors.append(f"Invalid {field} format for worker {worker}: {value}")
                                valid = False
                        else:
                            self.validation_errors.append(f"Invalid {field} format for worker {worker}: {value}")
                            valid = False
                    else:
                        try:
                            float(value)
                        except ValueError:
                            self.validation_errors.append(f"Invalid {field} format for worker {worker}: {value}")
                            valid = False
                
                # Validate position_size_pct
                elif field == 'position_size_pct':
                    try:
                        float(value)
                    except ValueError:
                        self.validation_errors.append(f"Invalid position_size_pct format for worker {worker}: {value}")
                        valid = False

        return valid

    def validate_ppo_config(self, config: Dict[str, Any], filename: str) -> bool:
        """Validate PPO configuration."""
        logger.debug(f"Validating {filename} - PPO section")
        valid = True

        if 'ppo' not in config:
            self.validation_errors.append(f"Missing required PPO configuration in {filename}")
            return False

        ppo_config = config['ppo']
        if 'workers' not in ppo_config:
            self.validation_errors.append(f"Missing required workers in PPO configuration")
            return False

        required_workers = ['w1', 'w2', 'w3', 'w4']
        for worker in required_workers:
            if worker not in ppo_config['workers']:
                self.validation_errors.append(f"Missing required worker {worker} in PPO configuration")
                valid = False
                continue

            worker_config = ppo_config['workers'][worker]
            required_fields = ['learning_rate', 'ent_coef', 'batch_size']
            for field in required_fields:
                if field not in worker_config:
                    self.validation_errors.append(f"Missing required field '{field}' in PPO worker {worker}")
                    valid = False
                    continue

                # Validate learning_rate format
                if field == 'learning_rate':
                    lr = worker_config[field]
                    if isinstance(lr, str) and '->' in lr:
                        try:
                            start_lr, end_lr = lr.split('->')
                            float(start_lr)
                            float(end_lr)
                        except ValueError:
                            self.validation_errors.append(f"Invalid learning_rate format for worker {worker}: {lr}")
                            valid = False
                    else:
                        try:
                            float(lr)
                        except ValueError:
                            self.validation_errors.append(f"Invalid learning_rate format for worker {worker}: {lr}")
                            valid = False
                else:
                    # Validate numerical values
                    try:
                        if field == 'ent_coef':
                            float(worker_config[field])
                        elif field == 'batch_size':
                            int(worker_config[field])
                    except ValueError:
                        self.validation_errors.append(f"Invalid {field} format for worker {worker}: {worker_config[field]}")
                        valid = False
        return valid

    def validate_worker_config_consistency(self, config: Dict[str, Any], filename: str) -> bool:
        """Validate worker configuration consistency across dbe, ppo, and trading sections."""
        logger.debug(f"Validating {filename} - Worker consistency")
        valid = True

        # Check if all sections exist
        if not all(section in config for section in ['dbe', 'ppo', 'trading']):
            self.validation_errors.append("Missing one or more required worker configurations (dbe, ppo, trading)")
            return False

        # Check worker count consistency
        worker_counts = {
            'dbe': len(config['dbe']['workers']),
            'ppo': len(config['ppo']['workers']),
            'trading': len(config['trading']['workers'])
        }

        if len(set(worker_counts.values())) > 1:
            self.validation_errors.append(f"Inconsistent number of workers across configurations: {worker_counts}")
            valid = False

        # Check worker ID consistency
        worker_ids = {
            'dbe': set(config['dbe']['workers'].keys()),
            'ppo': set(config['ppo']['workers'].keys()),
            'trading': set(config['trading']['workers'].keys())
        }

        expected_ids = {'w1', 'w2', 'w3', 'w4'}
        if worker_ids['dbe'] != expected_ids or worker_ids['ppo'] != expected_ids or worker_ids['trading'] != expected_ids:
            self.validation_errors.append(f"Inconsistent worker IDs across configurations: {worker_ids}")
            valid = False

        return valid

    def validate_memory_config(self, config: Dict[str, Any], filename: str) -> bool:
        """Validate memory configuration."""
        logger.debug(f"Validating {filename} - Memory section")
        valid = True

        if 'environment' not in config or 'memory' not in config['environment']:
            self.validation_errors.append(f"Missing required memory configuration in {filename}")
            return False
        
        mem_config = config['environment']['memory']
        
        required_keys = [
            'chunk_size', 'max_chunks_in_memory', 'aggressive_cleanup', 
            'force_gc_after_chunk', 'memory_warning_threshold_mb', 
            'memory_critical_threshold_mb', 'num_workers', 'pin_memory', 
            'batch_size', 'prefetch_factor', 'shuffle', 'drop_last', 
            'include_portfolio_state'
        ]
        for key in required_keys:
            if key not in mem_config:
                self.validation_errors.append(f"Missing required memory key '{key}'")
                valid = False
        
        return valid

    def validate_risk_config(self, config: Dict[str, Any], filename: str) -> bool:
        """Validate risk configuration."""
        logger.debug(f"Validating {filename} - Risk section")
        valid = True

        if 'environment' not in config or 'risk_management' not in config['environment']:
            self.validation_errors.append(f"Missing required risk_management configuration in {filename}")
            return False

        risk_config = config['environment']['risk_management']

        # Validate position_sizing
        if 'position_sizing' not in risk_config:
            self.validation_errors.append(f"Missing required position_sizing in risk_management section")
            valid = False
        else:
            ps_config = risk_config['position_sizing']
            required_ps_fields = ['max_risk_per_trade_pct', 'max_asset_allocation_pct', 'concentration_limits', 'take_profit']
            for field in required_ps_fields:
                if field not in ps_config:
                    self.validation_errors.append(f"Missing required position sizing key '{field}'")
                    valid = False
            
            if 'concentration_limits' in ps_config:
                limits = ps_config['concentration_limits']
                if not isinstance(limits, dict):
                    self.validation_errors.append("Concentration limits must be a dictionary")
                    valid = False
                else:
                    for asset, limit in limits.items():
                        if not isinstance(limit, (int, float)) or limit <= 0 or limit > 100:
                            self.validation_errors.append(f"Invalid concentration limit for {asset}")
                            valid = False
            
            if 'take_profit' in ps_config:
                tp_config = ps_config['take_profit']
                required_tp_fields = ['enabled', 'risk_reward_ratio', 'trailing_enabled', 'trailing_deviation_pct']
                for field in required_tp_fields:
                    if field not in tp_config:
                        self.validation_errors.append(f"Missing required take profit key '{field}'")
                        valid = False

        # Validate penalties
        if 'penalties' in config['environment']:
            pen_config = config['environment']['penalties']
            required_pen_fields = [
                'invalid_action', 'order_rejection', 'inaction_freq_threshold', 
                'inaction_pnl_threshold', 'base_inaction_penalty'
            ]
            for field in required_pen_fields:
                if field not in pen_config:
                    self.validation_errors.append(f"Missing required penalty key '{field}'")
                    valid = False
        else:
            self.validation_errors.append("Missing required penalties section in environment")
            valid = False

        # Validate reward shaping
        if 'reward_shaping' in config['environment']:
            rs_config = config['environment']['reward_shaping']
            required_rs_fields = [
                'realized_pnl_multiplier', 'unrealized_pnl_multiplier', 
                'reward_clipping_range', 'optimal_trade_bonus', 'performance_threshold'
            ]
            for field in required_rs_fields:
                if field not in rs_config:
                    self.validation_errors.append(f"Missing required reward shaping key '{field}'")
                    valid = False
        else:
            self.validation_errors.append("Missing required reward_shaping section in environment")
            valid = False

        return valid

    def validate_data_config(self, config: Dict[str, Any], filename: str) -> bool:
        """Validate data configuration."""
        logger.debug(f"Validating {filename} - Data section")
        valid = True

        if 'data' not in config:
            self.validation_errors.append(f"Missing required data section in {filename}")
            return False

        data_config = config['data']
        
        # Validate features_per_timeframe
        if 'features_per_timeframe' not in data_config:
            self.validation_errors.append(f"Missing required features_per_timeframe in data section")
            valid = False
        else:
            features = data_config['features_per_timeframe']
            if not isinstance(features, dict):
                self.validation_errors.append("Features must be a dictionary")
                valid = False
            else:
                required_timeframes = ['5m', '1h', '4h']
                for tf in required_timeframes:
                    if tf not in features:
                        self.validation_errors.append(f"Missing features for timeframe '{tf}'")
                        valid = False
                    else:
                        if not isinstance(features[tf], list):
                            self.validation_errors.append(f"Features for timeframe '{tf}' must be a list")
                            valid = False
                        if len(features[tf]) == 0:
                            self.validation_errors.append(f"Features list for timeframe '{tf}' cannot be empty")
                            valid = False

        # Validate data_dir
        if 'data_dir' not in data_config:
            self.validation_errors.append(f"Missing required data_dir in data section")
            valid = False
        else:
            data_dir = data_config['data_dir']
            if not isinstance(data_dir, str):
                self.validation_errors.append("data_dir must be a string")
                valid = False

        return valid

    def validate_single_config(self, config_path: Union[str, Path], config_type: str) -> bool:
        """
        Validate a single configuration file.
        
        Args:
            config_path: Path to the configuration file
            config_type: Type of configuration ('main', 'data', 'environment', etc.)
            
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate based on config_type
            if config_type == 'main':
                return self.validate_main_config(config, config_path)
            elif config_type == 'data':
                return self.validate_data_config(config, config_path)
            elif config_type == 'environment':
                return self.validate_environment_config(config, config_path)
            elif config_type == 'train':
                return self.validate_train_config(config, config_path)
            elif config_type == 'dbe':
                return self.validate_dbe_config(config, config_path)
            elif config_type == 'ppo':
                return self.validate_ppo_config(config, config_path)
            elif config_type == 'trading':
                return self.validate_trading_config(config, config_path)
            elif config_type == 'memory':
                return self.validate_memory_config(config, config_path)
            elif config_type == 'risk':
                return self.validate_risk_config(config, config_path)
            else:
                self.validation_errors.append(f"Unknown configuration type: {config_type}")
                return False
        except Exception as e:
            self.validation_errors.append(f"Error reading configuration file: {str(e)}")
            return False

    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors."""
        return self.validation_errors.copy()

    def get_validation_warnings(self) -> List[str]:
        """Get list of validation warnings."""
        return self.validation_warnings.copy()

    def has_errors(self) -> bool:
        """Check if there are validation errors."""
        return len(self.validation_errors) > 0

    def has_warnings(self) -> bool:
        """Check if there are validation warnings."""
        return len(self.validation_warnings) > 0

# ------------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------------

def validate_config_directory(config_dir: Union[str, Path]) -> bool:
    """
    Convenience function to validate all configurations in a directory.
    
    Args:
        config_dir: Path to the configuration directory
        
    Returns:
        True if all configurations are valid, False otherwise
    """
    validator = ConfigValidator()
    return validator.validate_all_configs(config_dir)

def validate_single_config(config_path: Union[str, Path], config_type: str) -> bool:
    """
    Validate a single configuration file.
    
    Args:
        config_path: Path to the configuration file
        config_type: Type of configuration ('main', 'data', 'environment', etc.)
        
    Returns:
        True if configuration is valid, False otherwise
    """
    validator = ConfigValidator()
    return validator.validate_single_config(config_path, config_type)