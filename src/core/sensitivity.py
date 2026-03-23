from dataclasses import dataclass

"""
KNOWN LIMITATIONS: 
Classifier is purely lexical, it matches substrings with no understanding of context or negation
e.g. "Je n'ai pas de problème de harcèlement mais..."   # → level 3 (wrong)
it contains "harcèlement" in a negated context but still triggers level 3
so may have false positives
"""


# KEYBOARD ANCHORS

"""
Each level contains keywords that strongly signal that sensitivity tier
Order of evaluation in classify() is always 3 -> 2 -> 1, so a query
containing both a level-2 and level-3 keyword always resolves to level 3
"""
LEVEL_3_KEYWORDS = [
    # Harassment and violence
    "harcèlement", "harcelement", "harceler",
    "violence", "agression", "intimidation",
    "menace", "abus",

    # Discrimination
    "discrimination", "racisme", "sexisme",
    "homophobie", "transophobie",

    # Serious personal distress
    "burnout", "épuisement professionnel",
    "dépression", "santé mentale",
    "suicide", "crise",

    # Formal complaints and legal
    "plainte formelle", "recours",
    "syndicat", "arbitrage", "tribunal",
    "avocat", "juridique", "légal",
]

LEVEL_2_KEYWORDS = [
    # Medical leave
    "congé maladie", "arrêt maladie",
    "invalidité", "accident de travail",
    "blessure", "médecin", "diagnostic",

    # Performance and conflict
    "performance", "évaluation",
    "avertissement", "mesure disciplinaire",
    "congédiement", "licenciement", "mise à pied",
    "conflit", "désaccord", "plainte",

    # Sensitive leave types
    "congé parental", "maternité", "paternité",
    "deuil", "décès", "famille",
    "accommodement", "handicap",
]

LEVEL_1_KEYWORDS = [
    # Vacation and time off
    "vacances", "congé annuel", "jours fériés",
    "temps libre", "absence planifiée",

    # Benefits
    "avantages sociaux", "assurance", "remboursement",
    "régime", "cotisation", "retraite", "rrsp", "reer",

    # Schedules and logistics
    "horaire", "télétravail", "bureau",
    "stationnement", "transport",

    # Training and development
    "formation", "développement", "certification",
    "conférence", "cours",

    # Payroll
    "salaire", "paie", "rémunération",
    "prime", "bonus", "augmentation",
]

# RESULT DATACLASS
@dataclass
class SensitivityResult:
    level: int                  # 1, 2 or 3
    matched_keyword: str       # keyboard that triggered the level
    reason: str                 # human-readable explanation for logging

# Classifier
class SensitivityClassifier:

    def classify(self, query: str) -> SensitivityResult:
        """
        Classifies a query into sensitivity level 1, 2, or 3

        Evaluation order is strictly 3 → 2 → 1. The first match wins.
        If no keyword matches, defaults to level 1 — the safest assumption
        for an unanticipated question is to attempt an answer rather than
        escalate unnecessarily.

        Returns a SensitivityResult dataclass, so the orchestrator and UI can 
        log which keyboard triggered the level without re-running the classification
        """
        normalized = self._normalize(query)

        for keyword in LEVEL_3_KEYWORDS:
            if keyword in normalized:
                return SensitivityResult(
                    level=3,
                    matched_keyword=keyword,
                    reason=f"Mot-clé de niveau 3 détecté : '{keyword}'. Redirection vers RH."
                )

        for keyword in LEVEL_2_KEYWORDS:
            if keyword in normalized:
                return SensitivityResult(
                    level=2,
                    matched_keyword=keyword,
                    reason=f"Mot-clé de niveau 2 détecté : '{keyword}'. Ton empathique activé."
                )

        for keyword in LEVEL_1_KEYWORDS:
            if keyword in normalized:
                return SensitivityResult(
                    level=1,
                    matched_keyword=keyword,
                    reason=f"Mot-clé de niveau 1 détecté : '{keyword}'."
                )  

        return SensitivityResult(
            level=1,
            matched_keyword="",
            reason="Aucun mot-clé détecté. Niveau 1 par défaut."
        )
    # PRIVATE HELPER
    def _normalize(self, text: str) -> str:
        """
        Lowercases and strips accents from common French characters
        Two reasons for this:
          1. "harcelement" vs "harcèlement" vs. "Harcèlement" should all trigger level 3
          2. Avoids importing heavy NLP libraries (spacy, nltk)

        Only covers the French accented characters
        """
        text = text.lower()

        replacements = {
            "é": "e", "è": "e", "ê": "e", "ë": "e",
            "à": "a", "â": "a", "ä": "a",
            "ù": "u", "û": "u", "ü": "u",
            "î": "i", "ï": "i",
            "ô": "o", "ö": "o",
            "ç": "c",
        }

        for accented, plain in replacements.items():
            text = text.replace(accented, plain)

        return text
    

