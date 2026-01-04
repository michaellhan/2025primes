import argparse
import sys
import os
import math
from collections import Counter


def parse_arguments():
    """
    Parse command-line arguments for the novelty game application.

    Returns:
        Namespace with parsed arguments, including --debug, --sort_order, --print_bound.
    """
    parser = argparse.ArgumentParser(
        description="Compute novelty (p,k,1) game outputs with various optimizations.")
    parser.add_argument('-p', '--num_players', type=int, default=3,
                        help="Number of ancestor levels (p)")
    parser.add_argument('-k', '--num_inputs', type=int, default=2,
                        help="Number of inputs per call (k)")
    parser.add_argument('-i', '--inputs', type=str,
                        help="Comma-separated list of non-negative integer inputs")
    parser.add_argument('-I', '--inputs_array', type=str,
                        help=": separated list of base-t arrays (for direct digit input)")
    parser.add_argument('-o', '--opt_level', type=int, default=0,
                        help="Optimization level (0-6)")
    parser.add_argument('--sort_order', choices=['asc', 'desc'], default='desc',
                        help="Sorting order for ancestor subarrays (asc or desc)")
    parser.add_argument('--print_bound', action='store_true',
                        help="Compute and print the combinatorial bound for the chosen opt level")
    parser.add_argument('--input_file', type=str,
                        help="File containing one integer input per line")
    parser.add_argument('--output_file', type=str,
                        help="File to write result to")
    parser.add_argument('--base', type=int, default=None,
                        help="Base for interpreting inputs and constructing outputs; if omitted, computed via digit_range")
    parser.add_argument('--normalize', action='store_true',
                        help="Normalize inputs/outputs to [0, base-1] for each digit")
    parser.add_argument('-d', '--debug', action='store_true',
                        help="Enable debug prints of intermediate values")

    args = parser.parse_args()
    if not (2 <= args.num_inputs <= 20):
        parser.error("--num_inputs must be between 2 and 20")
    if not (0 <= args.opt_level <= 6):
        parser.error("--opt_level must be between 0 and 6")
    print(f"\n\nComputing output for {args.num_players} players with {args.num_inputs} inputs per player.")
    return args

# ----------------- novelty game logic -----------------

def to_base_t_array(x, t, length):
    """
    Convert integer x into its base-t representation array
    (least significant digit first), padded to the given length.
    """
    digits = []
    while x > 0:
        digits.append(x % t)
        x //= t
    digits.extend([0] * (length - len(digits)))
    return digits


def base_t_array_to_number(digits, base):
    """
    Convert a base-'base' digit array (most significant first)
    into its integer value.
    """
    x = 0
    for d in digits:
        x = x * base + d
    return x


def get_digit_levels(digits, k, p, start_level=0):
    """
    Partition a flat list of digits into ancestor levels A_start...A_{p-1}.
    Level i has size k^i.
    Returns a list of p-start_level subarrays.
    """
    levels = []
    idx = 0
    for level in range(start_level, p):
        size = k ** level
        levels.append(digits[idx:idx + size])
        idx += size
    return levels


def ancestor_bookkeeping(inputs, p, k, t=None, debug=False):
    """
    Perform ancestor-bookkeeping to derive ancestor_digits and determine base t.

    Returns:
        ancestor_digits: flattened list of concatenated ancestor levels
        t: digit range base
    """
    if t is None:
        t = (k ** (p + 1) - 1) // (k - 1)
    L = (k ** p - 1) // (k - 1)
    base_t_inputs = [to_base_t_array(x, t, L) for x in inputs]
    level_lists = [get_digit_levels(arr, k, p) for arr in base_t_inputs]

    ancestor_levels = []
    for lvl in range(p - 1):
        concat = []
        for levels in level_lists:
            concat.extend(levels[lvl])
        ancestor_levels.append(concat)
    ancestor_digits = [d for level in ancestor_levels for d in level]

    if debug:
        print("[DEBUG] ancestor-bookkeeping level_lists:", level_lists)
        print("[DEBUG] ancestor-bookkeeping ancestor_levels:", ancestor_levels)
        print("[DEBUG] ancestor-bookkeeping ancestor_digits:", ancestor_digits)
    return ancestor_digits, t


def select_last_digit_opt0(inputs, ancestor_digits, p, k, t):
    """
    Baseline (opt0): Avoid any digit in ancestors or last input level.
    """
    L = (k ** p - 1) // (k - 1)
    arrs = [to_base_t_array(x, t, L) for x in inputs]
    used = set(ancestor_digits)
    for arr in arrs:
        used.update(get_digit_levels(arr, k, p)[p - 1])
    for v in range(t):
        if v not in used:
            return v
    raise ValueError("No valid digit found for opt0")


def select_last_digit_opt1(inputs, ancestor_digits, p, k, t):
    """
    Opt1: Input Avoidance. If too few unique digits across inputs,
    select unused; otherwise pick an input last-digit not in ancestors.
    """
    L = (k ** p - 1) // (k - 1)
    arrs = [to_base_t_array(x, t, L) for x in inputs]
    unique_all = set(d for arr in arrs for d in arr)
    unique_anc = set(d for arr in arrs for d in arr[1:])
    last_digits = [arr[0] for arr in arrs]

    if len(unique_all) < t:
        for v in range(t):
            if v not in unique_all:
                return v
    else:
        for v in last_digits:
            if v not in unique_anc:
                return v
    raise ValueError("No valid digit found for opt1")


def select_last_digit_opt2(inputs, ancestor_digits, p, k, t, use_counter=False):
    """
    Opt2/3: Cycle Avoidance using substring or multiset inclusion checks.
    """
    def array_succ(Ajz, Aix):
        sublen = len(Ajz)
        return any(Aix[i:i + sublen] == Ajz for i in range(len(Aix) - sublen + 1))
    def multiset_succ(Ajz, Aix):
        c1, c2 = Counter(Ajz), Counter(Aix)
        return all(c1[d] <= c2[d] for d in c1)

    L = (k ** p - 1) // (k - 1)
    arrs = [to_base_t_array(x, t, L) for x in inputs]
    levels = [get_digit_levels(arr, k, p) for arr in arrs]
    anc_lvls = get_digit_levels(ancestor_digits, k, p, start_level=1)

    used = set(d for lvl in levels for d in lvl[p - 1])
    for i in range(1, math.ceil(p / 2)):
        for lvl in levels:
            f = False
            for m in range(1, i + 1):
                Ajz = anc_lvls[m - 1]
                Aix = lvl[p - i - m + 1]
                check = multiset_succ if use_counter else array_succ
                if not check(Ajz, Aix):
                    f = True
                    break
            if not f:
                used.update(lvl[p - i - 1])
    for v in range(t):
        if v not in used:
            return v
    raise ValueError("No valid digit found for opt2/3")


def select_last_digit_opt4(inputs, ancestor_digits, p, k, t):
    """
    Opt4: Merge Equivalent Ancestors.
    Criterion: A_j ⊆ A_i and |A_i|-|A_j| ≤ k^i - k^j.
    """
    def succ4(Ajz, Aix, j, i):
        return set(Ajz).issubset(Aix) and (len(Aix) - len(Ajz) <= k**i - k**j)

    L = (k ** p - 1) // (k - 1)
    arrs = [to_base_t_array(x, t, L) for x in inputs]
    levels = [get_digit_levels(arr, k, p) for arr in arrs]
    anc_lvls = get_digit_levels(ancestor_digits, k, p, start_level=1)

    used = set(d for lvl in levels for d in lvl[p - 1])
    for i in range(1, math.ceil(p / 2)):
        for lvl in levels:
            f = False
            for m in range(1, i + 1):
                Ajz = anc_lvls[m - 1]
                Aix = lvl[p - i - m + 1]
                if not succ4(Ajz, Aix, m, p - i - m + 1):
                    f = True
                    break
            if not f:
                used.update(lvl[p - i - 1])
    for v in range(t):
        if v not in used:
            return v
    raise ValueError("No valid digit found for opt4")


def select_last_digit_opt5(inputs, ancestor_digits, p, k, t):
    """
    Opt5: Merge Neighboring Ancestors.
    Criterion: A_j ⊆ A_i.
    """
    L = (k ** p - 1) // (k - 1)
    arrs = [to_base_t_array(x, t, L) for x in inputs]
    levels = [get_digit_levels(arr, k, p) for arr in arrs]
    anc_lvls = get_digit_levels(ancestor_digits, k, p, start_level=1)

    used = set(d for lvl in levels for d in lvl[p - 1])
    for i in range(1, math.ceil(p / 2)):
        for lvl in levels:
            f = False
            for m in range(1, i + 1):
                Ajz = anc_lvls[m - 1]
                Aix = lvl[p - i - m + 1]
                if not set(Ajz).issubset(Aix):
                    f = True
                    break
            if not f:
                used.update(lvl[p - i - 1])
    for v in range(t):
        if v not in used:
            return v
    raise ValueError("No valid digit found for opt5")


def opt_level_output(inputs, p, k, t, level, debug=False, sort_order='asc'):
    """
    Dispatch to the specified optimization and construct output z.
    """
    ancestor_digits, t = ancestor_bookkeeping(inputs, p, k, t, debug=debug)
    if debug:
        print("[DEBUG] ancestor_digits:", ancestor_digits)

    reverse = (sort_order == 'desc')
    result = None

    if level == 3:
        anc_lvls = get_digit_levels(ancestor_digits, k, p, start_level=1)
        for idx in range(len(anc_lvls)):
            anc_lvls[idx] = sorted(anc_lvls[idx], reverse=reverse)
        if debug:
            print("[DEBUG][opt3] reordered ancestor_levels:", anc_lvls)
        flat = [d for lvl in anc_lvls for d in lvl]
        last = select_last_digit_opt2(inputs, flat, p, k, t, use_counter=True)
        result = [last] + flat
        if debug:
            print("[DEBUG][opt3] result   digits:", result)
            print("[DEBUG][opt3] reversed digits:", result[::-1])

    elif level == 4:
        anc_lvls = get_digit_levels(ancestor_digits, k, p, start_level=1)
        if debug:
            print("[DEBUG][opt4] before merge equivalent:", anc_lvls)
        for idx, lvl in enumerate(anc_lvls):
            U = set(lvl)
            dmin = min(U)
            need = k**(idx + 1) - len(U)
            new = [dmin]*need + sorted(U, reverse=reverse)
            anc_lvls[idx] = new
        if debug:
            print("[DEBUG][opt4] merged ancestor_levels:", anc_lvls)
        flat = [d for lvl in anc_lvls for d in lvl]
        if debug:
            print("[DEBUG][opt4] new_ancestors:", flat)
        last = select_last_digit_opt4(inputs, flat, p, k, t)
        result = [last] + flat
        if debug:
            print("[DEBUG][opt4] result   digits:", result)
            print("[DEBUG][opt4] reversed digits:", result[::-1])

    elif level == 5 or level == 6:
        anc_lvls = get_digit_levels(ancestor_digits, k, p, start_level=1)
        if debug:
            print(f"[DEBUG][opt{level}] before merge neighboring:", anc_lvls)
        for idx, lvl in enumerate(anc_lvls):
            U = set(lvl)
            need = k**(idx + 1) - len(U)
            W = [d for d in range(t) if d not in U]
            F = sorted(W)[:need]
            new = sorted(list(U)+F, reverse=reverse)
            anc_lvls[idx] = new
        if debug:
            print(f"[DEBUG][opt{level}] merged ancestor_levels:", anc_lvls)
        flat = [d for lvl in anc_lvls for d in lvl]
        if debug:
            print(f"[DEBUG][opt{level}] new_ancestors:", flat)
        last = select_last_digit_opt5(inputs, flat, p, k, t)
        result = [last] + flat
        if debug:
            print(f"[DEBUG][opt{level}] result   digits:", result)
            print(f"[DEBUG][opt{level}] reversed digits:", result[::-1])

    else:
        flat = ancestor_digits
        if level == 0:
            last = select_last_digit_opt0(inputs, flat, p, k, t)
        elif level == 1:
            last = select_last_digit_opt1(inputs, flat, p, k, t)
        else:
            last = select_last_digit_opt2(inputs, flat, p, k, t)
        result = [last] + flat
        if debug:
            print(f"[DEBUG][opt{level}] result   digits:", result)
            print(f"[DEBUG][opt{level}] reversed digits:", result[::-1])

    z = 0
    for d in reversed(result):
        z = z * t + d
    return z

# ------------- combinatorial bound logic -------------

def binom(n, r):
    """Safe binomial coefficient (returns 0 if invalid)."""
    try:
        return math.comb(n, r)
    except ValueError:
        return 0


def f(k, n):
    """Sum of geometric series 1 + k + ... + k^(n-1)."""
    return sum(k**i for i in range(n))


def get_s(p):
    """Compute s = ceil(p/2) - 1."""
    return (p + 1)//2 - 1


def extra_term(p, k):
    """Extra term for odd p when j = s."""
    s = get_s(p)
    return k**(p - s - 1) - k if p % 2 == 1 else 0


def cycle_contribution(p, k, L):
    """Contribution of cycle length L to digit range."""
    j = p - L
    s = get_s(p)
    if j == s:
        return k**(p - s) + k**s + k - k*f(k, s+1) + extra_term(p, k)
    if j == 0:
        return k**p
    return k**(p - j) + k**j + k - k*f(k, j+1)


def digit_range(opt_level, p, k, debug=False):
    """
    Compute digit range D(p,k) via cycle contributions.
    Debug prints each cycle's contribution.
    """
    q = sum(k**i for i in range(p+1))
    if opt_level == 0:
        return q
    elif opt_level == 1:
        return q - k

    # Compute via cycle contributions for all other optimization levels
    max_i = get_s(p)
    total = 0
    for i in range(max_i + 1):
        contrib = cycle_contribution(p, k, p - i)
        total += contrib
        if debug:
            print(f"[DEBUG][digit_range] cycle {i}: contrib={contrib}, total={total}")
    return total + 1

def compute_bound(p, k, level, base=0, debug=False):
    """
    Compute combinatorial bound for given p, k, and opt level.

    Args:
        p (int): number of ancestor levels.
        k (int): number of inputs per call.
        level (int): optimization level (0-6).
        base (int): if >0, use as digit range; otherwise compute via digit_range.
        debug (bool): enable debug prints.

    Returns:
        int: computed bound.
    """
    q = sum(k**i for i in range(p+1))
    r = sum(k**i for i in range(p))
    dr = base if base else digit_range(level, p, k, debug=debug)

    if debug:
        print(f"[DEBUG][compute_bound] q={q}, r={r}, dr={dr}")

    if level == 0:
        return q**r
    if level == 1:
        return (q - k)**r
    if level == 2:
        return dr**r
    if level == 3:
        result = 1
        for i in range(p):
            term = binom(dr + k**i - 1, k**i)
            if debug:
                print(f"[DEBUG][compute_bound] level3 i={i}, term={term}")
            result *= term
        return result
    if level == 4:
        result = 1
        for i in range(p):
            sum_term = sum(binom(dr, kj) for kj in range(1, k**i + 1))
            if debug:
                print(f"[DEBUG][compute_bound] level4 i={i}, sum_term={sum_term}")
            result *= sum_term
        return result
    if level == 5:
        result = 1
        for i in range(p):
            term = binom(dr, k**i)
            if debug:
                print(f"[DEBUG][compute_bound] level5 i={i}, term={term}")
            result *= term
        return result
    if level == 6:
        opt5 = 1
        for i in range(p):
            opt5 *= binom(dr, k**i)
        reduction = binom(dr - 1, k - 1)*(k**(p-1) - k**2 + k)
        if debug:
            print(f"[DEBUG][compute_bound] level6 initial opt5={opt5}, reduction start={reduction}")
        for i in range(2, p):
            rterm = binom(dr, k**i)
            if debug:
                print(f"[DEBUG][compute_bound] level6 i={i}, reduction_term={rterm}")
            reduction *= rterm
        return opt5 - reduction
    return None

# ----------------------- main ------------------------

def main():
    args = parse_arguments()
    p, k = args.num_players, args.num_inputs
    base = args.base if args.base is not None else digit_range(args.opt_level, p, k, debug=args.debug)

    inputs = []
    if args.inputs_array:
        try:
            in_array = []
            for grp in args.inputs_array.split(':'):
                digits = list(map(int, grp.split(',')))
                if args.debug:
                    print(f"[DEBUG] got one input array: {digits}")
                if args.normalize:
                    digits = [v % base for v in digits]
                    if args.debug:
                        print(f"[DEBUG] normalized input array: {digits}")
                inputs.append(base_t_array_to_number(digits, base))
                in_array.append(digits)
            if args.debug:
                print(f"[DEBUG] got input arrays {in_array}")
        except Exception as e:
            sys.exit(f"--inputs_array parse error: {e}")
    elif args.input_file:
        if not os.path.isfile(args.input_file):
            sys.exit(f"File not found: {args.input_file}")
        with open(args.input_file) as f:
            for line in f:
                val = line.strip()
                if val:
                    inputs.append(int(val, base=base))
    else:
        if not args.inputs:
            sys.exit("Provide --inputs or --inputs_array or --input_file")
        inputs = [int(x, 0) for x in args.inputs.split(',')]

    if len(inputs) != k:
        sys.exit(f"Expected {k} inputs, got {len(inputs)}")

    if args.debug:
        print(f"[DEBUG] got inputs: {inputs}")

    output = opt_level_output(
        inputs, p, k, base,
        level=args.opt_level,
        debug=args.debug,
        sort_order=args.sort_order
    )

    print("Base:", base)  # base of input array digits
    print("Inputs:", inputs)
    print("Output:", output, "\n\n")

    if args.print_bound:
        bound = compute_bound(p, k, args.opt_level, base=base, debug=args.debug)
        print(f"Optimization Level {args.opt_level} Bound: {bound}")

    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(f"Inputs: {inputs}\nBase: {base}\nOutput: {output}\n")
            if args.print_bound:
                f.write(f"OptLevel {args.opt_level} Bound: {bound}\n") # type: ignore

if __name__ == '__main__':
    main()
