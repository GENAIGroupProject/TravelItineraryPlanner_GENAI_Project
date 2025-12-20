import json
import os
import re
import time
from datetime import datetime
from typing import List, Dict, Optional
import hashlib
import pickle

from utils.llm_client import LLMClient
from utils.json_parser import JSONParser
from utils.data_structures import Attraction
from config import Config
from utils.logging_utils import log_agent_communication, log_step

class LocationScoutAgent:
    """Optimized agent for generating attractions in a chosen city."""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
        self.json_parser = JSONParser()
        self.cache_dir = "cache/attractions"
        self._ensure_cache_dir()
        self.timeout = 30  # seconds
        self.max_retries = 3
        self.enable_cache = True
    
    def _ensure_cache_dir(self):
        """Ensure cache directory exists."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_key(self, city: str, profile: str, constraints: Dict) -> str:
        """Generate cache key from inputs."""
        key_data = {
            "city": city.lower().strip(),
            "profile": profile[:200],
            "constraints": {
                "with_children": constraints.get("with_children", False),
                "with_disabled": constraints.get("with_disabled", False),
                "budget": constraints.get("budget", 0),
                "people": constraints.get("people", 1)
            }
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[List[Attraction]]:
        """Load attractions from cache if available."""
        if not self.enable_cache:
            return None
        
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                # Check if cache is still valid (24 hours)
                if time.time() - os.path.getmtime(cache_file) < 86400:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                    log_step("LOCATION_SCOUT", f"Loaded {len(cached_data)} attractions from cache")
                    return cached_data
            except Exception as e:
                log_step("LOCATION_SCOUT", f"Cache load error: {e}", level="debug")
        return None
    
    def _save_to_cache(self, cache_key: str, attractions: List[Attraction]):
        """Save attractions to cache."""
        if not self.enable_cache:
            return
        
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump(attractions, f)
            log_step("LOCATION_SCOUT", f"Cached {len(attractions)} attractions", level="debug")
        except Exception as e:
            log_step("LOCATION_SCOUT", f"Cache save error: {e}", level="debug")
    
    def _create_optimized_prompt(self, city: str, user_wants: str, 
                                constraints: Dict) -> str:
        """Create an optimized prompt that's faster to process."""
        with_children = constraints.get("with_children", False)
        with_disabled = constraints.get("with_disabled", False)
        budget = constraints.get("budget", Config.DEFAULT_BUDGET)
        people = constraints.get("people", Config.DEFAULT_PEOPLE)
        
        # Simple, clear prompt
        prompt = f"""Generate exactly 10 tourist attractions in {city}.

User preferences: {user_wants}

Important constraints:
- Traveling with children: {with_children}
- Accessibility needs: {with_disabled}
- Total budget: {budget}€ for {people} people

For each attraction, return a JSON object with these exact fields:
1. "name": Specific attraction name
2. "short_description": One sentence description
3. "approx_price_per_person": Number (0-50 EUR)
4. "tags": Array of 2-3 tags like ["hiking", "nature", "outdoor"]
5. "reason_for_user": Why this matches their preferences

Return ONLY a valid JSON array with 10 objects. Do not include any other text.

Example format:
[
  {{
    "name": "Lake Walk",
    "short_description": "Beautiful walk around the lake with mountain views",
    "approx_price_per_person": 0,
    "tags": ["hiking", "lake", "nature"],
    "reason_for_user": "Perfect for nature lovers who enjoy scenic walks"
  }}
]"""
        
        return prompt
    
    def _parse_and_clean_json(self, text: str) -> List[Dict]:
        """Parse JSON from LLM response with robust error handling."""
        print(f"📄 Raw response length: {len(text)} chars")
        print(f"📄 Response preview: {text[:200]}...")
        
        # First, try to find JSON array
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        
        if not json_match:
            print("❌ No JSON array pattern found in response")
            return []
        
        json_str = json_match.group(0)
        print(f"📄 Found JSON string, length: {len(json_str)} chars")
        
        try:
            # Try direct parse first
            data = json.loads(json_str)
            if isinstance(data, list):
                print(f"✅ Direct JSON parse successful, found {len(data)} items")
                return data
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parse error: {e}")
            print(f"📄 JSON preview: {json_str[:200]}...")
            
            # Try to fix common JSON issues
            fixed_json = self._fix_json_common_issues(json_str)
            try:
                data = json.loads(fixed_json)
                if isinstance(data, list):
                    print(f"✅ Fixed JSON parse successful, found {len(data)} items")
                    return data
            except json.JSONDecodeError as e2:
                print(f"❌ Fixed JSON also failed: {e2}")
                print(f"📄 Fixed JSON preview: {fixed_json[:200]}...")
        
        return []
    
    def _fix_json_common_issues(self, json_str: str) -> str:
        """Fix common JSON formatting issues."""
        # Remove trailing commas before ] or }
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        # Fix missing quotes on keys
        json_str = re.sub(r'(\{|\,\s*)(\w+)(\s*:)', r'\1"\2"\3', json_str)
        
        # Fix single quotes to double quotes
        json_str = json_str.replace("'", '"')
        
        # Ensure proper string escaping
        json_str = re.sub(r'(?<!\\)"([^"]*?)(?<!\\)"', r'"\1"', json_str)
        
        return json_str
    
    def _create_attraction_objects(self, raw_data: List[Dict], city: str) -> List[Attraction]:
        """Create Attraction objects from raw data."""
        attractions = []
        
        for i, item in enumerate(raw_data):
            try:
                # Ensure required fields
                if not isinstance(item, dict) or 'name' not in item:
                    print(f"  ⚠️ Item {i}: Not a dict or missing 'name' field")
                    continue
                
                # Clean and validate fields
                name = str(item.get('name', f'Attraction {i+1}')).strip()
                if not name or len(name) < 2:
                    print(f"  ⚠️ Item {i}: Invalid name '{name}'")
                    continue
                
                description = str(item.get('short_description', f'Popular attraction in {city}')).strip()
                
                # Validate price
                price = 15.0  # default
                price_raw = item.get('approx_price_per_person')
                if price_raw is not None:
                    try:
                        if isinstance(price_raw, (int, float)):
                            price = float(price_raw)
                        elif isinstance(price_raw, str):
                            # Extract numbers from string
                            numbers = re.findall(r'\d+\.?\d*', price_raw)
                            if numbers:
                                price = float(numbers[0])
                    except (ValueError, TypeError):
                        pass
                
                if price < 0 or price > 1000:
                    price = 15.0
                
                # Validate tags
                tags = item.get('tags', [])
                if not isinstance(tags, list):
                    tags = ['sightseeing']
                tags = [str(tag).lower().strip() for tag in tags if tag and str(tag).strip()]
                if not tags:
                    tags = ['sightseeing']
                
                reason = str(item.get('reason_for_user', f'Recommended attraction in {city}')).strip()
                
                # Create attraction
                attraction = Attraction(
                    name=name[:100],
                    short_description=description[:200],
                    approx_price_per_person=price,
                    tags=tags[:5],
                    reason_for_user=reason[:200]
                )
                
                attractions.append(attraction)
                print(f"  ✅ Created: {name[:30]}... (price: {price}€, tags: {tags})")
                
            except Exception as e:
                print(f"  ❌ Error creating attraction {i}: {e}")
                continue
        
        return attractions

    def generate_attractions(self, city: str, refined_profile: str, 
                           constraints: Dict) -> List[Attraction]:
        """Generate attractions for the chosen city."""
        print(f"\n🔍 Starting attraction generation for {city}")
        start_time = time.time()
        
        # Check cache first
        cache_key = self._get_cache_key(city, refined_profile, constraints)
        cached_result = self._load_from_cache(cache_key)
        if cached_result:
            print(f"✅ Using cached attractions for {city}")
            return cached_result
        
        print(f"🔄 Generating new attractions for {city}")
        
        # Analyze preferences
        user_wants = self._analyze_user_preferences(refined_profile)
        print(f"📋 User preferences: {user_wants}")
        
        # Create prompt
        prompt = self._create_optimized_prompt(city, user_wants, constraints)
        
        # Call LLM with retries - WITHOUT max_tokens parameter
        raw_response = None
        for attempt in range(self.max_retries):
            try:
                print(f"📤 Attempt {attempt + 1}/{self.max_retries} calling LLM...")
                
                # Call LLM WITHOUT max_tokens parameter
                raw_response = self.llm_client.generate(
                    prompt, 
                    temperature=0.7
                    # Remove max_tokens parameter
                )
                
                if raw_response and len(raw_response) > 100:
                    print(f"📥 Received LLM response ({len(raw_response)} chars)")
                    break
                else:
                    print(f"⚠️ Short response received: {len(raw_response) if raw_response else 0} chars")
                    
            except Exception as e:
                print(f"❌ LLM attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    print("❌ All LLM attempts failed")
                    return []
                time.sleep(2)  # Wait before retry
        
        if not raw_response or len(raw_response) < 100:
            print("❌ No valid response from LLM")
            return []
        
        # Parse response
        print("🔍 Parsing LLM response...")
        raw_data = self._parse_and_clean_json(raw_response)
        
        if not raw_data:
            print("❌ Could not parse any attractions from response")
            return []
        
        print(f"✅ Parsed {len(raw_data)} raw attraction items")
        
        # Create Attraction objects
        attractions = self._create_attraction_objects(raw_data, city)
        
        if not attractions:
            print("❌ Could not create any valid attraction objects")
            return []
        
        print(f"✅ Created {len(attractions)} valid attractions")
        
        # Cache the result
        self._save_to_cache(cache_key, attractions)
        
        elapsed = time.time() - start_time
        print(f"⏱️  Generated {len(attractions)} attractions in {elapsed:.1f}s")
        
        return attractions[:10]  # Ensure max 10
    
    def _analyze_user_preferences(self, refined_profile: str) -> str:
        """Analyze user preferences to guide attraction selection."""
        profile_lower = refined_profile.lower()
        
        requirements = []
        
        # Check for specific preferences
        if any(word in profile_lower for word in ["hiking", "forest", "park", "nature", "outdoor"]):
            requirements.append("Include hiking trails, forests, parks, outdoor nature areas")
        
        if any(word in profile_lower for word in ["not cultural", "no cultural", "not historical", "no museums"]):
            requirements.append("Avoid museums, historical sites, cultural attractions")
        elif any(word in profile_lower for word in ["cultural", "museum", "historical", "art", "history"]):
            requirements.append("Include cultural and historical attractions")
        
        if any(word in profile_lower for word in ["beach", "coast", "sea", "lake", "water"]):
            requirements.append("Include beach, lake or water activities")
        
        if any(word in profile_lower for word in ["food", "cuisine", "restaurant", "eating", "dining"]):
            requirements.append("Include local food experiences, restaurants, markets")
        
        if any(word in profile_lower for word in ["relaxed", "slow", "chill", "leisurely"]):
            requirements.append("Prefer relaxed pace activities")
        elif any(word in profile_lower for word in ["fast", "busy", "active", "energetic"]):
            requirements.append("Prefer active, fast-paced experiences")
        
        if any(word in profile_lower for word in ["shopping", "shop", "mall", "market"]):
            requirements.append("Include shopping opportunities")
        
        if not requirements:
            return "Looking for popular attractions and experiences suitable for tourists"
        
        return "; ".join(requirements)