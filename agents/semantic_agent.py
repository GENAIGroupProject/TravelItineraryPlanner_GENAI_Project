import re
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer

from utils.data_structures import PreferenceState, PreferenceSnippet
from config import Config

class SemanticAgent:
    """Agent for semantic understanding of user preferences."""
    
    def __init__(self):
        self.emb_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.slot_labels = self._initialize_slot_labels()
        self.slot_embeddings = self._initialize_slot_embeddings()
    
    def _initialize_slot_labels(self) -> Dict[str, str]:
        """Initialize slot definitions."""
        return {
            "activities": "The user talks about preferred activities or attractions like museums, parks, nightlife, beaches.",
            "pace": "The user describes how fast they like to travel, for example slow, relaxed, or packed schedule.",
            "budget": "The user mentions budget, price range, cheap, expensive or how much money to spend.",
            "constraints": "The user mentions children, kids, disabled travelers, mobility limitations or accessibility.",
            "food": "The user talks about restaurants, food or cuisine preferences.",
            "other": "Other preferences not covered above."
        }
    
    def _initialize_slot_embeddings(self) -> Dict[str, np.ndarray]:
        """Pre-compute embeddings for slot labels."""
        return {
            k: self.emb_model.encode([v], normalize_embeddings=True)[0]
            for k, v in self.slot_labels.items()
        }
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        parts = re.split(r"[.!?]\s+", text.strip())
        return [p.strip() for p in parts if p.strip()]
    
    def classify_slot(self, sentence_emb: np.ndarray) -> str:
        """Classify sentence into the most relevant slot."""
        best_slot, best_score = None, -1.0
        for slot, label_emb in self.slot_embeddings.items():
            score = float(sentence_emb @ label_emb)
            if score > best_score:
                best_score, best_slot = score, slot
        return best_slot
    
    def update_state(self, state: PreferenceState, user_msg: str) -> PreferenceState:
        """Update preference state with new user message."""
        sentences = self.split_into_sentences(user_msg)
        if not sentences:
            return state
        
        new_snippets = []
        for sentence in sentences:
            emb = self.emb_model.encode([sentence], normalize_embeddings=True)[0]
            slot = self.classify_slot(emb)
            
            # Find similar snippets in same slot
            same_slot = [sn for sn in state.snippets if sn.slot == slot]
            best_sn, best_sim = None, -1.0
            
            for snippet in same_slot:
                sim = float(emb @ snippet.embedding)
                if sim > best_sim:
                    best_sim, best_sn = sim, snippet
            
            # Update or add new snippet
            if best_sn is not None and best_sim > Config.SIM_UPDATE_THRESHOLD:
                best_sn.text = sentence
                best_sn.embedding = emb
            else:
                new_snippets.append(PreferenceSnippet(
                    text=sentence, embedding=emb, slot=slot
                ))
        
        # Add new snippets
        state.snippets.extend(new_snippets)
        
        # Rebuild slot texts
        state.slots = {k: "" for k in state.slots}
        for snippet in state.snippets:
            sep = " " if state.slots[snippet.slot] else ""
            state.slots[snippet.slot] += sep + snippet.text
        
        # Update global embedding
        all_embs = [sn.embedding for sn in state.snippets]
        state.global_embedding = np.mean(all_embs, axis=0) if all_embs else None
        
        state.turns += 1
        return state
    
    def build_profile_summary(self, state: PreferenceState) -> str:
        """Create summary of current preferences."""
        summary_lines = []
        for slot_name, slot_display in [
            ("activities", "Activities"),
            ("pace", "Pace"),
            ("food", "Food"),
            ("constraints", "Constraints"),
            ("budget", "Budget")
        ]:
            value = state.slots[slot_name] or "not specified yet"
            summary_lines.append(f"{slot_display}: {value}")
        
        return "\n".join(summary_lines)