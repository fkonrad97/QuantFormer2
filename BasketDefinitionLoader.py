import itertools
import json
import random
from typing import Dict, List, Literal, Optional
import copy

def load_basket_definitions(
    tickers: List[str],
    sectors: Dict[str, str],
    mode: Literal["manual", "random", "exhaustive", "mixed"] = "manual",
    manual_baskets_path: Optional[str] = None,
    basket_size: int = 2,
    num_random_baskets: int = 5,
    aggregated_loss: bool = False,
    expand_aggregated_variants: bool = False,
    num_variants: int = 3,
) -> List[Dict]:
    """
    Returns a list of basket definitions with optional aggregation metadata.

    Each basket dict may include:
    - name
    - tickers
    - sector_vector
    - aggregate_output (bool)
    - weights (optional, for aggregate_output=True)
    """
    unique_sectors = sorted(set(sectors.values()))
    sector_to_idx = {s: i for i, s in enumerate(unique_sectors)}

    def one_hot_sector_vector(basket_tickers):
        vec = [0] * len(sector_to_idx)
        for t in basket_tickers:
            s = sectors.get(t)
            if s and s in sector_to_idx:
                vec[sector_to_idx[s]] = 1
        return vec

    def generate_weights(size):
        raw = [random.random() for _ in range(size)]
        total = sum(raw)
        return [round(w / total, 3) for w in raw]

    def expand_variants(base_entry):
        variants = []
        for v in range(num_variants):
            new_weights = generate_weights(len(base_entry["tickers"]))
            variants.append({
                **base_entry,
                "name": f'{base_entry["name"]}_v{v+1}',
                "weights": new_weights,
                "aggregate_output": True
            })
        return variants
    
    def generate_basket_name(tickers: List[str]) -> str:
        return "_".join(sorted(tickers))  # Alphabetical to ensure consistency

    baskets = []

    if mode == "manual":
        if manual_baskets_path is None:
            raise ValueError("Manual mode requires manual_baskets_path.")
        with open(manual_baskets_path, "r") as f:
            data = json.load(f)
        for b in data:
            entry = {
                "name": b["name"],
                "tickers": b["tickers"],
                "sector_vector": one_hot_sector_vector(b["tickers"])
            }
            if aggregated_loss and b.get("aggregate_output", False):
                entry["aggregate_output"] = True
                entry["weights"] = generate_weights(len(b["tickers"]))
            if expand_aggregated_variants and entry.get("aggregate_output"):
                baskets.extend(expand_variants(entry))
            else:
                baskets.append(entry)

    elif mode == "mixed":
        sizes = list(range(1, min(basket_size, len(tickers)) + 1))
        for i in range(num_random_baskets):
            size = random.choice(sizes)
            sample = random.sample(tickers, size)
            entry = {
                "name": "_".join(sorted(sample)),  # <-- updated name here
                "tickers": sample,
                "sector_vector": one_hot_sector_vector(sample)
            }
            if aggregated_loss and size > 1 and random.random() < 0.5:
                entry["aggregate_output"] = True
                entry["weights"] = generate_weights(size)
            if expand_aggregated_variants and entry.get("aggregate_output"):
                baskets.extend(expand_variants(entry))
            else:
                baskets.append(entry)

    elif mode == "exhaustive":
        combos = list(itertools.combinations(tickers, basket_size))
        for i, combo in enumerate(combos):
            entry = {
                "name": f"Basket_{i+1}",
                "tickers": list(combo),
                "sector_vector": one_hot_sector_vector(combo)
            }
            if aggregated_loss and random.random() < 0.5:
                entry["aggregate_output"] = True
                entry["weights"] = generate_weights(len(combo))
            if expand_aggregated_variants and entry.get("aggregate_output"):
                baskets.extend(expand_variants(entry))
            else:
                baskets.append(entry)

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return baskets

def expand_baskets_with_random_weights(
    baskets: List[Dict],
    num_variants_per_basket: int = 3,
    max_unique_weights: int = 5
) -> List[Dict]:
    """
    Expands each multi-asset basket (with aggregation) into multiple versions
    by sampling new random weights for aggregation.

    - Keeps single-asset or non-aggregated baskets untouched.
    - Expands only those with aggregate_output == True.

    Returns:
        List of enriched baskets with added variants.
    """
    new_baskets = []

    for b in baskets:
        # Single-asset or no aggregation → leave as is
        if len(b["tickers"]) == 1 or not b.get("aggregate_output", False):
            new_baskets.append(b)
            continue

        # Base: Add original as-is
        new_baskets.append(b)

        # Add N more with different weights
        for i in range(num_variants_per_basket):
            variant = copy.deepcopy(b)
            variant["name"] = f'{b["name"]}_w{i+1}'

            weights = [random.random() for _ in variant["tickers"]]
            total = sum(weights)
            variant["weights"] = [round(w / total, 3) for w in weights]

            new_baskets.append(variant)

    return new_baskets
