import json
import os
import re
import time
import hashlib
import pickle
from typing import List, Dict, Optional

from utils.llm_client import LLMClient
from utils.data_structures import Attraction
from utils.logging_utils import log_step


class LocationScoutAgent:
    """
    ⚡ FAST Location Scout Agent
    - Single LLM call
    - No retries
    - No fallbacks
    - Strict JSON
    - Cache-first
    - Fail fast
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
        self.cache_dir = "cache/attractions"
        os.makedirs(self.cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------
    def _cache_key(self, city: str, profile: str, constraints: Dict) -> str:
        payload = {
            "city": city.lower().strip(),
            "profile": profile[:250],  # keep small
            "constraints": {
                "budget": constraints.get("budget"),
                "people": constraints.get("people"),
                "with_children": constraints.get("with_children"),
                "with_disabled": constraints.get("with_disabled"),
            },
        }
        return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _load_cache(self, key: str) -> Optional[List[Attraction]]:
        path = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(path):
            # valid for 24h
            if time.time() - os.path.getmtime(path) < 86400:
                try:
                    with open(path, "rb") as f:
                        return pickle.load(f)
                except Exception:
                    return None
        return None

    def _save_cache(self, key: str, data: List[Attraction]) -> None:
        try:
            with open(os.path.join(self.cache_dir, f"{key}.pkl"), "wb") as f:
                pickle.dump(data, f)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def generate_attractions(
        self,
        city: str,
        refined_profile: str,
        constraints: Dict,
    ) -> List[Attraction]:

        start = time.time()
        log_step("LOCATION_SCOUT", f"⚡ FAST attraction generation for {city}")

        if not city or not refined_profile:
            return []

        # Cache first
        cache_key = self._cache_key(city, refined_profile, constraints)
        cached = self._load_cache(cache_key)
        if cached:
            log_step("LOCATION_SCOUT", f"✅ Loaded {len(cached)} attractions from cache")
            return cached

        # ------------------------------------------------------------------
        # VERY SHORT PROMPT (this is critical for speed)
        # ------------------------------------------------------------------
        prompt = f"""
Return EXACTLY 10 attractions in {city} as JSON.

User preferences:
{refined_profile}

Rules:
- Output ONLY valid JSON
- Array of 10 objects
- Each object MUST have:
  name
  short_description
  approx_price_per_person (number)
  tags (array)
  reason_for_user

JSON ONLY.
"""

        # ------------------------------------------------------------------
        # SINGLE LLM CALL — NO RETRIES
        # ------------------------------------------------------------------
        try:
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=0.25,
            )
        except Exception:
            return []

        # ------------------------------------------------------------------
        # FAST JSON EXTRACTION
        # ------------------------------------------------------------------
        match = re.search(r"\[\s*{.*}\s*\]", response, re.DOTALL)
        if not match:
            return []

        try:
            raw_items = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

        if not isinstance(raw_items, list) or len(raw_items) != 10:
            return []

        attractions: List[Attraction] = []

        # ------------------------------------------------------------------
        # STRICT OBJECT CREATION (NO DEFAULTS)
        # ------------------------------------------------------------------
        for item in raw_items:
            try:
                attractions.append(
                    Attraction(
                        name=str(item["name"])[:100],
                        short_description=str(item["short_description"])[:200],
                        approx_price_per_person=float(item["approx_price_per_person"]),
                        tags=list(item["tags"])[:5],
                        reason_for_user=str(item["reason_for_user"])[:200],
                    )
                )
            except Exception:
                # Fail fast: if ANY item is invalid, abort everything
                return []

        # Must be exactly 10
        if len(attractions) != 10:
            return []

        self._save_cache(cache_key, attractions)

        elapsed = time.time() - start
        log_step("LOCATION_SCOUT", f"✅ Generated 10 attractions in {elapsed:.2f}s")

        return attractions
