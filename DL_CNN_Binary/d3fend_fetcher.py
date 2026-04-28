"""
D3FEND Technique Fetcher.

Automatically queries d3fend.mitre.org to retrieve:
    - Category / Tactic (Detect, Isolate, Evict, Restore, Model, Harden)
    - Definition / Description
    - Digital Artifact relationships

Results are cached locally (d3fend_cache.json) so the API is only queried once
per action ID. On subsequent runs, data is loaded from cache instantly.

This module is designed to work with ANY set of D3FEND actions — if you change
to a different action set in the future, it will auto-fetch the new ones.
"""

import os
import re
import json
import time

import requests

import config
from logger import get_logger


# =============================================================================
# CONFIGURATION
# =============================================================================

D3FEND_TECHNIQUE_API = "https://d3fend.mitre.org/api/technique/d3f:{name}.json"
D3FEND_ONTOLOGY_API = "https://d3fend.mitre.org/api/ontology/inference/d3fend-full-mappings.json"
D3FEND_TIMEOUT = 15
D3FEND_DELAY = 0.5  # seconds between API calls (be polite to the server)

CACHE_FILENAME = "d3fend_cache.json"


# =============================================================================
# LABEL → CAMELCASE CONVERSION
# =============================================================================

def label_to_camel_case(label):
    """
    Convert D3FEND label to CamelCase for API URL.

    Examples:
        "Process Termination"        → "ProcessTermination"
        "DNS Traffic Analysis"       → "DNSTrafficAnalysis"
        "Executable Denylisting"     → "ExecutableDenylisting"
        "Network Isolation"          → "NetworkIsolation"
        "File Eviction"              → "FileEviction"
        "Forward Resolution Domain Denylisting" → "ForwardResolutionDomainDenylisting"
    """
    # Remove leading/trailing whitespace
    label = label.strip()

    # Split by spaces and join without spaces (capitalize each word)
    words = label.split()
    result = ""
    for word in words:
        # Keep acronyms/abbreviations uppercase (DNS, HTTP, IP, etc.)
        if word.isupper() and len(word) > 1:
            result += word
        else:
            result += word[0].upper() + word[1:]
    return result


# =============================================================================
# KNOWN TACTIC KEYWORDS (fallback classification)
# =============================================================================

# If the API is unreachable, we classify based on the action name itself
TACTIC_KEYWORDS = {
    "Detect": [
        "Analysis", "Detection", "Verification", "Monitoring",
        "Filtering",  # some filtering is detect
    ],
    "Isolate": [
        "Isolation", "Denylisting", "Filtering", "Allowlisting",
        "Quarantine", "Sandboxing",
    ],
    "Evict": [
        "Termination", "Eviction", "Deletion", "Removal",
        "Revocation",
    ],
    "Restore": [
        "Restore", "Recovery", "Backup",
    ],
    "Harden": [
        "Hardening", "Encryption", "Authentication",
        "Credential",
    ],
    "Model": [
        "Modeling", "Inventory", "Mapping", "Baseline",
    ],
}


def guess_tactic_from_label(label):
    """
    Fallback: guess tactic from action label keywords.
    Used only when API is unreachable.
    """
    label_lower = label.lower()

    # Special cases (order matters — check specific before generic)
    # "Filtering" can be Detect or Isolate depending on context
    if "traffic filtering" in label_lower:
        return "Isolate"
    if "traffic analysis" in label_lower:
        return "Detect"

    for tactic, keywords in TACTIC_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in label_lower:
                return tactic

    return "Unknown"


# =============================================================================
# API FETCHING
# =============================================================================

def fetch_technique_info(d3fend_id, label):
    """
    Fetch technique info from D3FEND API.

    Tries the technique-specific API endpoint first.
    Returns dict with category, description, or None on failure.

    Args:
        d3fend_id: e.g. "D3-PT"
        label: e.g. "Process Termination"

    Returns:
        dict with keys: category, description, definition, artifacts
        or None if API call fails
    """
    log = get_logger()
    camel_name = label_to_camel_case(label)
    url = D3FEND_TECHNIQUE_API.format(name=camel_name)

    try:
        response = requests.get(url, timeout=D3FEND_TIMEOUT)
        if response.status_code != 200:
            log.debug(f"  API returned {response.status_code} for {d3fend_id} ({camel_name})")
            return None

        data = response.json()
        return _parse_technique_response(data, d3fend_id, camel_name)

    except requests.RequestException as e:
        log.debug(f"  API error for {d3fend_id}: {e}")
        return None
    except (json.JSONDecodeError, KeyError) as e:
        log.debug(f"  Parse error for {d3fend_id}: {e}")
        return None


def _parse_technique_response(data, d3fend_id, camel_name):
    """
    Parse the D3FEND technique API response.

    The response structure varies but generally contains:
    - A description/definition nested under the technique URI key
    - Tactic/category information

    This parser handles multiple known response formats.
    """
    info = {
        "category": "Unknown",
        "description": "",
        "definition": "",
        "artifacts": [],
    }

    # Strategy 1: Look for the technique under its d3fend.owl URI
    owl_key = f"http://d3fend.mitre.org/ontologies/d3fend.owl#{camel_name}"
    tech_data = data.get(owl_key, None)

    if not tech_data and isinstance(data, dict):
        # Strategy 2: Search all keys for the CamelCase name
        for key in data:
            if camel_name in key:
                tech_data = data[key]
                break

    if not tech_data and isinstance(data, dict):
        # Strategy 3: Look in nested "results" → "bindings" (SPARQL format)
        bindings = (data.get("results", {}).get("bindings", [])
                    or data.get("def_to_off", {}).get("results", {}).get("bindings", []))
        if bindings and isinstance(bindings, list) and len(bindings) > 0:
            b = bindings[0]
            info["category"] = b.get("def_tactic_label", {}).get("value", "Unknown")
            info["description"] = b.get("def_tech_desc", {}).get("value", "")
            return info

    if tech_data and isinstance(tech_data, dict):
        # Extract definition / description
        definition = tech_data.get("d3fend:definition", "")
        if isinstance(definition, list):
            definition = definition[0] if definition else ""
        elif isinstance(definition, dict):
            definition = definition.get("@value", str(definition))
        info["definition"] = str(definition).strip()
        info["description"] = info["definition"]

        # Extract d3fend:d3fend-id to verify
        fetched_id = tech_data.get("d3fend:d3fend-id", "")
        if isinstance(fetched_id, list):
            fetched_id = fetched_id[0] if fetched_id else ""
        elif isinstance(fetched_id, dict):
            fetched_id = fetched_id.get("@value", "")

        # Extract tactic/category from rdfs:subClassOf
        sub_classes = tech_data.get("rdfs:subClassOf", [])
        if isinstance(sub_classes, dict):
            sub_classes = [sub_classes]
        if isinstance(sub_classes, list):
            for sc in sub_classes:
                sc_id = ""
                if isinstance(sc, dict):
                    sc_id = sc.get("@id", "")
                elif isinstance(sc, str):
                    sc_id = sc

                sc_lower = sc_id.lower()
                if "detect" in sc_lower:
                    info["category"] = "Detect"
                elif "isolat" in sc_lower:
                    info["category"] = "Isolate"
                elif "evict" in sc_lower:
                    info["category"] = "Evict"
                elif "restor" in sc_lower:
                    info["category"] = "Restore"
                elif "harden" in sc_lower:
                    info["category"] = "Harden"
                elif "model" in sc_lower:
                    info["category"] = "Model"

        # Extract related digital artifacts
        kb_refs = tech_data.get("d3fend:kb-reference", [])
        if isinstance(kb_refs, dict):
            kb_refs = [kb_refs]

    return info


# =============================================================================
# CACHE MANAGEMENT
# =============================================================================

def _get_cache_path():
    """Get path to D3FEND cache file."""
    return os.path.join(config.BASE_DIR, CACHE_FILENAME)


def load_cache():
    """Load cached D3FEND technique info."""
    cache_path = _get_cache_path()
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_cache(cache):
    """Save D3FEND technique info to cache."""
    cache_path = _get_cache_path()
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def enrich_action_info(action_info, force_refresh=False):
    """
    Enrich action_info dict with category + description from D3FEND API.

    This is the main function called by action_loader.py.

    Args:
        action_info: Dict {d3fend_id: {"label": "..."}} from action_loader
        force_refresh: If True, re-fetch even if cached

    Returns:
        Updated action_info with added keys:
            - "category": Detect/Isolate/Evict/Restore/Harden/Model
            - "description": Full definition from D3FEND ontology
            - "source": "api" or "fallback" (for transparency in logs)

    Modifies action_info in-place AND returns it.
    """
    log = get_logger()
    log.info(f"Enriching {len(action_info)} D3FEND actions with category + description...")

    # Load cache
    cache = load_cache()
    fetched_count = 0
    cached_count = 0
    fallback_count = 0
    api_available = True  # assume yes until first failure

    from tqdm import tqdm
    for aid, info in tqdm(action_info.items(), desc="Fetching D3FEND info"):
        label = info.get("label", "")

        # Check cache first
        if not force_refresh and aid in cache:
            cached_data = cache[aid]
            info["category"] = cached_data.get("category", "Unknown")
            info["description"] = cached_data.get("description", "")
            info["source"] = "cache"
            cached_count += 1
            continue

        # Try API fetch (only if API seems available)
        fetched = None
        if api_available and label:
            fetched = fetch_technique_info(aid, label)
            if fetched is None:
                # After 2 consecutive failures, assume API is down
                api_available = False
                log.warning("D3FEND API appears unreachable. Using fallback classification.")

        if fetched and fetched.get("category", "Unknown") != "Unknown":
            info["category"] = fetched["category"]
            info["description"] = fetched.get("description", "") or fetched.get("definition", "")
            info["source"] = "api"
            fetched_count += 1

            # Save to cache
            cache[aid] = {
                "label": label,
                "category": info["category"],
                "description": info["description"],
            }

            time.sleep(D3FEND_DELAY)  # polite delay
        else:
            # Fallback: guess category from label keywords
            info["category"] = guess_tactic_from_label(label)
            info["description"] = ""
            info["source"] = "fallback"
            fallback_count += 1

            # Cache the fallback too (can be overwritten later with force_refresh)
            cache[aid] = {
                "label": label,
                "category": info["category"],
                "description": "",
                "fallback": True,
            }

    # Save updated cache
    save_cache(cache)

    log.info(f"D3FEND enrichment complete:")
    log.info(f"  From cache: {cached_count}")
    log.info(f"  From API:   {fetched_count}")
    log.info(f"  Fallback:   {fallback_count}")

    # Log the category distribution
    cat_counts = {}
    for aid, info in action_info.items():
        cat = info.get("category", "Unknown")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    log.info(f"  Category distribution: {cat_counts}")

    return action_info
