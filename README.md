# Novelty Game Simulator

Python implementation of the novelty (p,k,1) game that computes outputs avoiding certain digit patterns across ancestor levels.

## Quick Start

```bash
python novelty_game.py -p 3 -k 2 -i "10,20"
```

## Key Parameters

- `-p, --num_players`: Number of ancestor levels (default: 3)
- `-k, --num_inputs`: Number of inputs per call, 2-20 (default: 2)
- `-i, --inputs`: Comma-separated integers
- `-o, --opt_level`: Optimization level 0-6 (default: 0)
- `--debug`: Show intermediate calculations
- `--print_bound`: Display combinatorial bound

## Input Methods

```bash
# Integers
python novelty_game.py -i "10,20,30"

# Digit arrays (colon-separated groups)
python novelty_game.py -I "1,2,3:4,5,6"

# File (one integer per line)
python novelty_game.py --input_file inputs.txt
```

## Optimization Levels

- **0**: Baseline - avoid all ancestor and last-level digits
- **1**: Input avoidance - select unused digits when possible
- **2**: Cycle avoidance - substring checks
- **3**: Cycle avoidance - multiset checks with sorting
- **4**: Merge equivalent ancestors (subset + size constraint)
- **5**: Merge neighboring ancestors (subset only)
- **6**: Enhanced merge

## Examples

```bash
# Basic usage
python novelty_game.py -p 3 -k 2 -i "5,10"

# With optimization and bound
python novelty_game.py -p 4 -k 2 -i "100,200" -o 3 --print_bound

# Debug mode
python novelty_game.py -p 3 -k 2 -i "10,20" --debug
```

## Output

```
Base: 15
Inputs: [10, 20]
Output: 12345
```
