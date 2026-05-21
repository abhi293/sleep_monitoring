# Documentation Index

## Start Here
- **IMPLEMENTATION_COMPLETE.md** ← Start here for overview
- **QUICK_REFERENCE.md** ← Copy-paste commands

## Detailed Documentation

- **CLI_CONTROL_GUIDE.md** ← Comprehensive usage guide  

## Verification
- **CLI_LOGIC_DEMONSTRATION.py** ← Run this to verify: `python CLI_LOGIC_DEMONSTRATION.py`



## Key Files at a Glance

```
Sleep/
├── train.py                          ← Modified (lines 112-335)
├── IMPLEMENTATION_COMPLETE.md        ← Overview
├── QUICK_REFERENCE.md               ← Quick commands
├── CLI_CONTROL_GUIDE.md             ← Detailed guide
├── SOLUTION_SUMMARY.md              ← Technical details
├── BEFORE_AND_AFTER.md              ← Visual comparison
├── CLI_LOGIC_DEMONSTRATION.py       ← Test script
└── README.md (existing)
```

---

## The Three Modes (TL;DR)

| Mode | Command | Window | Stride | Use Case |
|------|---------|--------|--------|----------|
| **Ablation** | `python train.py` | 30 | 10 | Clean comparisons ✓ |
| **YAML Config** | `python train.py --use_config_model` | 20 | 3 | Custom hyperparams |
| **MOPSO** | `python train.py --mopso` | 20 | 3 | Find best params |

---


