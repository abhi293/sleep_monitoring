#!/usr/bin/env python3
"""
CLI_LOGIC_DEMONSTRATION.py
──────────────────────────────────────────────────────────────
Quick test to verify the three operating modes work correctly.
This demonstrates the config resolution logic without needing to run full training.
"""

class MockArgs:
    """Mock argparse Namespace for testing"""
    def __init__(self, mopso=False, use_config_model=False, window_size=None, 
                 stride=None, smoke_test=False):
        self.mopso = mopso
        self.use_config_model = use_config_model
        self.window_size = window_size
        self.stride = stride
        self.smoke_test = smoke_test

def resolve_config_mode(args):
    """
    Mimics the config resolution logic from train.py
    Returns: (config_mode_name, window_size, stride)
    """
    use_yaml_config = (not args.mopso) and args.use_config_model
    use_ablation_defaults = (not args.mopso) and (not args.use_config_model)
    
    # Set sensible defaults based on mode
    if use_ablation_defaults:
        default_window_size = 30
        default_stride = 10
        config_mode = "Ablation defaults"
    elif use_yaml_config:
        default_window_size = 20
        default_stride = 3
        config_mode = "YAML config"
    else:
        default_window_size = 20
        default_stride = 3
        config_mode = "MOPSO optimization"
    
    # Apply CLI overrides (CLI always wins)
    final_window_size = args.window_size if args.window_size is not None else default_window_size
    final_stride = args.stride if args.stride is not None else default_stride
    
    return config_mode, final_window_size, final_stride

# ═══════════════════════════════════════════════════════════════
# TEST CASES
# ═══════════════════════════════════════════════════════════════

test_cases = [
    # (description, args_dict, expected_mode, expected_window, expected_stride)
    ("Default ablation", 
     {},
     "Ablation defaults", 30, 10),
    
    ("Ablation with custom window_size",
     {"window_size": 25},
     "Ablation defaults", 25, 10),
    
    ("Ablation with custom window_size and stride",
     {"window_size": 25, "stride": 5},
     "Ablation defaults", 25, 5),
    
    ("YAML config mode",
     {"use_config_model": True},
     "YAML config", 20, 3),
    
    ("YAML config with CLI override",
     {"use_config_model": True, "window_size": 30},
     "YAML config", 30, 3),
    
    ("MOPSO mode",
     {"mopso": True},
     "MOPSO optimization", 20, 3),
    
    ("MOPSO mode with --use_config_model (should be ignored)",
     {"mopso": True, "use_config_model": True},
     "MOPSO optimization", 20, 3),
    
    ("MOPSO with custom window (CLI override)",
     {"mopso": True, "window_size": 25},
     "MOPSO optimization", 25, 3),
]

print("\n" + "="*80)
print("CLI CONFIG RESOLUTION LOGIC TEST")
print("="*80 + "\n")

passed = 0
failed = 0

for desc, args_dict, expected_mode, expected_window, expected_stride in test_cases:
    args = MockArgs(**args_dict)
    mode, window, stride = resolve_config_mode(args)
    
    success = (mode == expected_mode and 
               window == expected_window and 
               stride == expected_stride)
    
    status = "✓ PASS" if success else "✗ FAIL"
    
    if success:
        passed += 1
    else:
        failed += 1
    
    print(f"{status}: {desc}")
    print(f"  Args: {args_dict if args_dict else '{default}'}")
    print(f"  Result: mode={mode}, window_size={window}, stride={stride}")
    
    if not success:
        print(f"  EXPECTED: mode={expected_mode}, window_size={expected_window}, stride={expected_stride}")
    print()

print("="*80)
print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} tests")
print("="*80 + "\n")

if failed == 0:
    print("✓ All tests passed! CLI logic is working correctly.\n")
else:
    print(f"✗ {failed} test(s) failed. Check the logic above.\n")
