"""
Utility to load and resolve configuration with environment variables.
"""
import os
from typing import Dict, Any, Optional
import re
import yaml

class ConfigLoader:
    """Load and resolve configuration with environment variables."""

    @classmethod
    def resolve_env_vars(cls, config: Dict[str, Any], parent_key: str = '', processed: Optional[set] = None, root_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Recursively resolve environment variables and path references in the configuration.

        Args:
            config: The configuration dictionary to process
            parent_key: The parent key for nested dictionaries (used for error messages)
            processed: Set of already processed keys to detect circular references
            root_config: The root configuration dict (always use for dotted path resolution)
        Returns:
            Dict with resolved environment variables and paths
        Raises:
            ValueError: If there are undefined variables or circular references
        """
        if root_config is None:
            root_config = config
        if processed is None:
            processed = set()

        # Make a deep copy to avoid modifying the original
        config_copy = {k: v for k, v in config.items()}

        # Detect circular references
        current_key = f"{parent_key}" if parent_key else "root"
        if current_key in processed:
            raise ValueError(f"Circular reference detected in config at {current_key}")
        processed.add(current_key)

        # Processing config section silently

        # First pass: resolve all environment variables (${ENV_VAR} syntax)
        for key, value in list(config_copy.items()):
            if isinstance(value, str):
                # Handle environment variables like ${ENV_VAR}
                def replace_env_var(match):
                    env_var = match.group(1)
                    if env_var in os.environ:
                        return os.environ[env_var]
                    return match.group(0)  # Return original if not found

                new_value = re.sub(r'\${([A-Z0-9_]+)}', replace_env_var, value)
                if new_value != value:
                    config_copy[key] = new_value

        # Second pass: resolve simple variables (non-nested)
        changed = True
        while changed:
            changed = False
            for key, value in list(config_copy.items()):
                if isinstance(value, dict):
                    # Process nested dictionaries recursively
                    config_copy[key] = cls.resolve_env_vars(value, f"{parent_key}.{key}" if parent_key else key, set(processed), root_config)
                elif isinstance(value, str) and '${' in value:
                    # Handle simple variable references (no dots)
                    def replace_simple_var(match):
                        var_name = match.group(1)
                        if '.' not in var_name:  # Only handle simple variables in this pass
                            try:
                                return str(config_copy[var_name])
                            except KeyError:
                                # Check if it's an environment variable that wasn't resolved earlier
                                if var_name in os.environ:
                                    return os.environ[var_name]
                                return match.group(0)  # Return original if not found
                        return match.group(0)

                    new_value = re.sub(r'\${([^}]+)}', replace_simple_var, value)
                    if new_value != value:
                        config_copy[key] = new_value
                        changed = True

        # Third pass: resolve nested paths (with dots)
        for key, value in list(config_copy.items()):
            if isinstance(value, str) and '${' in value:

                def replace_nested_var(match):
                    var_path = match.group(1)

                    # First try to resolve as a direct environment variable
                    if var_path in os.environ:
                        resolved = os.environ[var_path]
                        return resolved

                    # Then try to resolve as a nested path in the config
                    parts = var_path.split('.')
                    current = root_config

                    try:
                        # Try to resolve the full path first
                        full_path = var_path
                        for part in parts:
                            if isinstance(current, dict) and part in current:
                                current = current[part]
                                full_path += f".{part}"
                            else:
                                # If we can't find the path, try to resolve each part
                                resolved_part = cls._resolve_part(part, root_config, full_path, root_config)
                                if resolved_part != part:
                                    current = resolved_part
                                else:
                                    raise KeyError(part)

                        result = str(current)
                        return result

                    except (KeyError, TypeError) as e:
                        # If we can't resolve the full path, try to resolve each part
                        resolved_parts = []
                        for part in parts:
                            resolved = cls._resolve_part(part, root_config, f"{full_path}.{part}", root_config)

                            if resolved == part and part not in os.environ:
                                raise ValueError(
                                    f"Undefined variable path in config: {var_path} "
                                    f"(at {parent_key}.{key if parent_key else ''})"
                                )

                            final_value = resolved if resolved != part else os.environ.get(part, part)
                            resolved_parts.append(str(final_value))

                        result = '.'.join(resolved_parts)
                        return result

                try:
                    config_copy[key] = re.sub(r'\${([^}]+)}', replace_nested_var, value)
                except RecursionError:
                    raise ValueError(f"Circular reference detected in config at {parent_key}.{key}")

        return config_copy

    @classmethod
    def _get_nested_value(cls, d: Dict[str, Any], path: str):
        """
        Get a value from a nested dictionary using a dot notation path.

        Args:
            d: The dictionary to search in
            path: The dot notation path (e.g., 'paths.base_dir')

        Returns:
            The value if found, None otherwise
        """
        keys = path.split('.')
        current = d

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    @classmethod
    def _resolve_part(
        cls,
        part: str,
        config: Dict[str, Any],
        full_path: str,
        root_config: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Resolve a single part of a path against the config.

        Args:
            part: The part to resolve (e.g., 'paths' or 'base_dir')
            config: The current config level
            full_path: The full path being resolved (for error messages)
            root_config: The root config for cross-references

        Returns:
            The resolved value

        Raises:
            ValueError: If the part cannot be resolved
        """

        # 1. Try resolving as environment variable with default
        resolved = cls._resolve_env_or_default(part, root_config or config)
        if resolved is not None:
            return resolved

        # 2. Check if part is a direct environment variable
        if part in os.environ:
            value = os.environ[part]
            return value

        # 3. Try direct lookup in current config
        if part in config:
            value = config[part]
            return str(value) if not isinstance(value, (dict, list)) else value

        # 4. Handle nested paths (with dots)
        if '.' in part:
            return cls._resolve_nested_path(part, config, full_path, root_config)

        # If we get here, we couldn't resolve the part
        raise ValueError(f"Undefined variable path in config: {part} (at {full_path})")

    @classmethod
    def _resolve_nested_path(
        cls,
        path: str,
        config: Dict[str, Any],
        full_path: str,
        root_config: Optional[Dict[str, Any]]
    ) -> Any:
        """Resolve a nested path (with dots) by resolving each part."""
        root = root_config if root_config is not None else config

        # First try direct nested lookup
        try:
            nested_value = cls._get_nested_value(root, path)
            if nested_value is not None:
                if debug:
                    msg = f"[DEBUG] Resolved nested path '{path}': {nested_value}"
                    print(msg)
                if isinstance(nested_value, (dict, list)):
                    return nested_value
                return str(nested_value)
        except (KeyError, AttributeError):
            pass

        # If direct lookup failed, resolve each part
        parts = path.split('.')
        resolved_parts = []

        for i, part in enumerate(parts):
            current_path = '.'.join(parts[:i+1])
            try:
                # Try to resolve each part
                resolved = cls._resolve_part(
                    part,
                    root,
                    f"{full_path}.{current_path}",
                    root
                )

                # If resolution failed, try to get default value
                if resolved == part and part not in os.environ:
                    resolved = cls._resolve_env_or_default(part, root) or part

                resolved_parts.append(str(resolved))
            except Exception as e:
                print(f"[ERROR] Error resolving part '{part}': {e}")
                raise

        result = '.'.join(resolved_parts)
        return result

    @classmethod
    def _resolve_env_or_default(cls, part, root_config):
        """Resolve environment variable with default value in format VAR:-default.

        Args:
            part: The part to resolve (e.g., 'VAR:-default' or 'VAR:-${OTHER_VAR:-default}')
            root_config: The root config for cross-references

        Returns:
            The resolved value or None if no match found
        """
        import os
        import re

        # Check if part matches VAR:-default pattern
        m = re.match(r'^([A-Za-z0-9_]+):-((?:.|\s)*)$', part)
        if m:
            var = m.group(1)
            default = m.group(2).strip("'\"")  # Remove quotes if present

            # First try to get from environment variables
            env_val = os.environ.get(var)
            if env_val is not None:
                return env_val

            # If not in environment, try to resolve as config path
            try:
                config_val = cls._get_nested_value(root_config, var)
                if config_val is not None:
                    return config_val
            except (KeyError, AttributeError):
                pass  # Continue to use default value

            # Process default value (which might contain nested variables)
            if default:
                if debug:
                    print(f"[DEBUG] Processing default value: {default}")

                # If default contains another variable, try to resolve it
                if '${' in default or ':-' in default:
                    try:
                        # Create a temporary config with the default value
                        temp_config = {'temp': default}
                        resolved = cls.resolve_env_vars(
                            temp_config,
                            root_config=root_config
                        )
                        result = resolved['temp']
                        if debug:
                            print(f"[DEBUG] Resolved default value: {default} -> {result}")
                        return result
                    except Exception as e:
                        msg = f"Could not resolve default value '{default}': {e}"
                        print(f"[WARNING] {msg}")
                        return default
                return default
            return ""  # Return empty string if no default value

        # Check for nested variables in the part (e.g., ${VAR:-default} or ${VAR})
        if '${' in part and '}' in part:
            if debug:
                print(f"[DEBUG] Found nested variable in part: {part}")
            try:
                # Handle the case where part is already a variable reference
                if part.startswith('${') and part.endswith('}'):
                    # Extract the inner part (without ${ and })
                    inner = part[2:-1]
                    # Try to resolve it as a simple variable first
                    if inner in os.environ:
                        return os.environ[inner]
                    # Then try to resolve with defaults
                    return cls._resolve_env_or_default(inner, root_config) or part
                else:
                    # For more complex strings with embedded variables
                    temp_config = {'temp': part}
                    resolved = cls.resolve_env_vars(
                        temp_config,
                        root_config=root_config
                    )
                    result = str(resolved['temp'])
                    if debug:
                        print(f"[DEBUG] Resolved nested variable: {part} -> {result}")
                    return result
            except Exception as e:
                msg = f"Could not resolve nested variable '{part}': {e}"
                print(f"[WARNING] {msg}")
                return part  # Return original part if resolution fails

        # If no match, return None to let normal processing continue
        if debug:
            print(f"[DEBUG] No match found for part: {part}")
        return None

    @classmethod
    def _resolve_part(cls, part: str, config: Dict[str, Any], full_path: str, root_config: Optional[Dict[str, Any]] = None):
        """
        Resolve a single part of a path against the config.

        Args:
            part: The part to resolve (e.g., 'paths' or 'base_dir')
            config: The current config level
            full_path: The full path being resolved (for error messages)
            root_config: The root config for cross-references

        Returns:
            The resolved value

        Raises:
            ValueError: If the part cannot be resolved
        """
        debug = True  # Set to True to enable debug output

        if debug:
            print(f"[DEBUG] Resolving part: '{part}' at path: '{full_path}'")
            print(f"[DEBUG] Available keys: {list(config.keys())}")

        # Try to resolve as environment variable with default value first
        env_or_default = cls._resolve_env_or_default(part, root_config or config)
        if env_or_default is not None:
            if debug:
                print(f"[DEBUG] Resolved environment variable with default '{part}': {env_or_default}")
            return env_or_default

        # First try direct lookup
        if part in config:
            if debug:
                print(f"[DEBUG] Found direct match for '{part}': {config[part]}")
            return config[part]

        # Try to resolve as environment variable
        env_val = os.environ.get(part)
        if env_val is not None:
            if debug:
                print(f"[DEBUG] Found environment variable '{part}': {env_val}")
            return env_val

        # Try to resolve as a nested path (e.g., 'paths.base_dir')
        if '.' in part:
            parts = part.split('.')
            current = config
            try:
                for p in parts:
                    if isinstance(current, dict) and p in current:
                        current = current[p]
                    else:
                        # If we can't find the path, try to resolve each part
                        resolved_part = cls._resolve_part(p, root_config or config, f"{full_path}.{p}", root_config)
                        if resolved_part != p:  # Only update if we actually resolved something
                            current = resolved_part
                        else:
                            raise KeyError(p)

                # If we get here, we successfully resolved the full path
                if debug:
                    print(f"[DEBUG] Resolved full path from context: '{full_path}': {current}")
                return str(current) if not isinstance(current, (dict, list)) else current
            except (KeyError, TypeError) as e:
                if debug:
                    print(f"[DEBUG] Error resolving from context: {e}")
                pass

        # If we get here, we couldn't resolve the part
        if debug:
            print(f"[ERROR] Failed to resolve part: '{part}' in path: '{full_path}'")
            print(f"[ERROR] Available keys at this level: {list(config.keys())}")
        raise ValueError(f"Undefined variable path in config: {part} (at {full_path})")

    @classmethod
    def _find_in_dict(cls, d: Dict[str, Any], target_key: str):
        """
        Recursively search for a key in a nested dictionary.

        Args:
            d: The dictionary to search in
            target_key: The key to find

        Returns:
            The value if found, None otherwise
        """
        if target_key in d:
            return d[target_key]

        for k, v in d.items():
            if isinstance(v, dict):
                result = cls._find_in_dict(v, target_key)
                if result is not None:
                    return result

        return None

    @classmethod
    def load_config(cls, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file and resolve all variables.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            Resolved configuration dictionary
        """
        print(f"[DEBUG] Loading config from: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        print("[DEBUG] Raw config loaded:")
        print(yaml.dump(config, default_flow_style=False))

        # First pass: resolve all variable references
        print("[DEBUG] Resolving environment variables...")
        resolved_config = cls.resolve_env_vars(config)

        print("[DEBUG] Resolved config:")
        print(yaml.dump(resolved_config, default_flow_style=False))

        return resolved_config
