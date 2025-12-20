import os
import re
from typing import List, Dict, Optional

import numpy as np

from utils.data_structures import PreferenceState, PreferenceSnippet
from config import Config
from utils.logging_utils import log_step

# IMPORTANT: avoid weird GPU/meta behaviors in Streamlit reruns
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Global singleton to avoid re-loading the embedding model on every Streamlit rerun
_EMB_MODEL = None


def _load_sentence_transformer():
    """
    Load SentenceTransformer safely to avoid meta-tensor/device_map issues.
    """
    global _EMB_MODEL
    if _EMB_MODEL is not None:
        return _EMB_MODEL

    from sentence_transformers import SentenceTransformer

    model_name = getattr(Config, "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Force CPU, and prevent device_map/low_cpu_mem_usage meta init issues
    # SentenceTransformer passes model_kwargs to Transformers AutoModel internally.
    try:
        _EMB_MODEL = SentenceTransformer(
            model_name,
            device="cpu",
            model_kwargs={
                "device_map": None,
                "low_cpu_mem_usage": False,
            },
        )
        log_step("SEMANTIC", f"Loaded embedding model on CPU: {model_name}")
        return _EMB_MODEL
    except TypeError:
        # Some SentenceTransformer versions don't accept model_kwargs.
        _EMB_MODEL = SentenceTransformer(model_name, device="cpu")
        log_step("SEMANTIC", f"Loaded embedding model on CPU (no model_kwargs support): {model_name}")
        return _EMB_MODEL
    except Exception as e:
        # Last resort: try without extra args
        log_step("SEMANTIC", f"Embedding model load fallback due to error: {e}", level="warning")
        _EMB_MODEL = SentenceTransformer(model_name)
        return _EMB_MODEL


class SemanticAgent:
    """Agent for semantic understanding of user preferences."""

    def __init__(self):
        log_step("SEMANTIC", "Initializing semantic agent")

        self.emb_model = _load_sentence_transformer()
        self.slot_labels = self._initialize_slot_labels()
        self.slot_embeddings = self._initialize_slot_embeddings()

        log_step("SEMANTIC", "Semantic agent initialized")

    def _initialize_slot_labels(self) -> Dict[str, str]:
        slots = {
            "activities": "Preferred activities or attractions like hiking, parks, nightlife, beaches, museums.",
            "pace": "Preferred travel pace like relaxed, slow, or packed schedule.",
            "budget": "Mentions budget, cheap/expensive, price range, cost.",
            "constraints": "Mentions children, accessibility, mobility limitations, disabilities.",
            "food": "Mentions restaurants, food, cuisine preferences.",
            "other": "Other preferences not covered above.",
        }
        return slots

    def _initialize_slot_embeddings(self) -> Dict[str, np.ndarray]:
        log_step("SEMANTIC", "Computing slot embeddings")
        embeddings = {}
        for k, v in self.slot_labels.items():
            # normalize_embeddings=True makes cosine similarity = dot product
            embeddings[k] = self.emb_model.encode([v], normalize_embeddings=True)[0]
        return embeddings

    def split_into_sentences(self, text: str) -> List[str]:
        parts = re.split(r"[.!?]\s+", (text or "").strip())
        return [p.strip() for p in parts if p.strip()]

    def classify_slot(self, sentence_emb: np.ndarray) -> str:
        best_slot, best_score = "other", -1.0
        for slot, label_emb in self.slot_embeddings.items():
            score = float(sentence_emb @ label_emb)
            if score > best_score:
                best_score, best_slot = score, slot
        return best_slot

    def update_state(self, state: PreferenceState, user_msg: str) -> PreferenceState:
        log_step("SEMANTIC", f"Updating state with message: '{(user_msg or '')[:60]}...'")

        sentences = self.split_into_sentences(user_msg)
        if not sentences:
            return state

        new_snippets: List[PreferenceSnippet] = []

        for sentence in sentences:
            emb = self.emb_model.encode([sentence], normalize_embeddings=True)[0]
            slot = self.classify_slot(emb)

            same_slot = [sn for sn in state.snippets if sn.slot == slot]
            best_sn, best_sim = None, -1.0

            for snippet in same_slot:
                sim = float(emb @ snippet.embedding)
                if sim > best_sim:
                    best_sim, best_sn = sim, snippet

            if best_sn is not None and best_sim > getattr(Config, "SIM_UPDATE_THRESHOLD", 0.78):
                best_sn.text = sentence
                best_sn.embedding = emb
            else:
                new_snippets.append(PreferenceSnippet(text=sentence, embedding=emb, slot=slot))

        state.snippets.extend(new_snippets)

        # rebuild slot texts
        state.slots = {k: "" for k in state.slots}
        for snippet in state.snippets:
            sep = " " if state.slots[snippet.slot] else ""
            state.slots[snippet.slot] += sep + snippet.text

        # global embedding
        all_embs = [sn.embedding for sn in state.snippets]
        state.global_embedding = np.mean(all_embs, axis=0) if all_embs else None

        state.turns += 1
        return state

    def build_profile_summary(self, state: PreferenceState) -> str:
        lines = []
        for slot_name, label in [
            ("activities", "Activities"),
            ("pace", "Pace"),
            ("food", "Food"),
            ("constraints", "Constraints"),
            ("budget", "Budget"),
        ]:
            value = state.slots.get(slot_name) or "not specified yet"
            lines.append(f"{label}: {value}")
        return "\n".join(lines)
