from dataclasses import dataclass, field
from typing import Optional
from src.agents.retriever import HybridRetriever
from src.agents.redacteur import Redacteur
from src.agents.verificateur import Verificateur
from src.core.sensitivity import SensitivityClassifier

# Result dataclass
@dataclass
class AgentResult:
    response: str
    sensitivity: int                        # 1, 2 ou 3
    escalated: bool                         # True if level 3: redirected to HR
    attempts: int = 1                       # how many redactor passes were needed
    grounding_score: float = 0.0            # 0.0-1.0 from verificateur
    grounded: bool = True                   # False if verificateur flagged issues
    issues: list[str] = field(default_factory=list)    # specific issues found
    sources: list[dict] = field(default_factory=list)  # [{title, section, score}]

class Orchestrator:

    MAX_REVISION_ATTEMPTS = 2       # Maximum of 2 revision attempt
    GROUNDING_THRESHOLD   = 0.75    # below this -> trigger revision pass

    ESCALATION_MESSAGE = (
        "Cette question nécessite un accompagnement humain." \
        "Veuillez contacter directement l'équipe RH à rh@talsom.com" \
        "ou appeler le poste 1234."
    )

    def __init__(self):
        # Each component is instantiated once and reused across calls
        # LLM singleton inside Redacteur and Verificateur is loaded only on first import
        self.classifier   = SensitivityClassifier()
        self.retriever    = HybridRetriever(chroma_path="./data/chroma_db")
        self.redacteur    = Redacteur()
        self.verificateur = Verificateur()

    # Public entry point
    def run(self, query: str) -> AgentResult:
        """
        Full pipeline for ONE user query
          1. Classify sensitivity (keyword-based, à revoir)
            1.1 If level 3 -> escalate immediately, no LLM call
          2. Retrieve relevant chunks (hybrid BM25 + dense): retriever agent
          3. [Loop max 2 times]
             a. Redacteur generates a grounded response
             b. Verificateur scores the response against the chunks
             c. If score >= threshold then it's done
             d. If score < threshold then pass feedback to Redacteur and retry
           4. Return AgentResult
        """

        # Step 1
        sensitivity_result = self.classifier.classify(query)
        level = sensitivity_result.level

        if level == 3:
            return AgentResult(
                response=self.ESCALATION_MESSAGE,
                sensitivity=3,
                escalated=True,
            )
        
        # Step 2 
        chunks = self.retriever.search(query, top_k=5)     # This call happens once, before the revision loop

        # If chunks is empty, it returns a honest message rather than sending
        # an empty context to the LLM (would cause Llama to hallucinate)
        if not chunks:
            return AgentResult(
                response=(
                "Je n'ai pas trouvé d'information pertinente dans les politiques RH" \
                "disponibles. Veuillez contacter l'équipe RH directement."
            ),
            sensitivity=level,
            escalated=False,
            grouding_score=0.0,
            grounded=False,
            issues=["Aucun document source trouvé pour cette requête."]

            )

        # Step 3
        feedback     = ""
        verification = {}

        for attempt in range(1, self.MAX_REVISION_ATTEMPTS+1):
            response = self.redacteur.generate(
                query=query,
                chunks=chunks,
                sensitivity=level,
                feedback=feedback,      # empty on first pass
            )

            verification = self.verificateur.verify(
                query=query,
                response=response,
                chunks=chunks,
            )

            # Exit loop early if grounding is good enough
            if verification["score"] >= self.GROUNDING_THRESHOLD:
                break

            # Prepare feedback for next pass
            feedback = self._build_feedback(verification)

        # Step 4
        return AgentResult(
            response=response,
            sensitivity=level,
            escalated=False,
            attempts=attempt,
            grounding_score=round(verification.get("score", 0.0), 2),
            grounded=verification.get("grounded", True),
            issues=verification.get("issues", []),
            sources=self._extract_sources(chunks),
        )


    # PRIVATE HELPERS 

    def _build_feedback(self, verification:dict) -> str:
        """
        Converts the verificateur structured output into plain-text feedback
        string that the redacteur can act on in the next pass.
        """
        issues = verification.get("issues", [])
        if not issues:
            return "La réponse manque d'ancrage dans les sources. Cite explicitement les documents."
        
        formatted = " ; ".join(issues)
        return f"Problèmes détectés : {formatted}. Corrige ces points dans ta révision."
    
    def _extract_sources(self, chunks: list[dict]) -> list[dict]:
        """
        Returns a clean, deduplicated list of source references from the top
        retrieved chunks, suitable for display in the UI
        """
        seen    = set()
        sources = []

        for chunk in chunks[:3]:    # top 3 only (beyond 3 it gets noisy)
            meta    = chunk.get("metadata", {})
            title   = meta.get("title", "Document inconnu")
            section = meta.get("section", "-")
            key     = (title, section) 

            if key not in seen:
                seen.add(key)
                sources.append({
                    "title":   title,
                    "section": section,
                    "score":   chunk.get("score", 0.0),
                })

        return sources