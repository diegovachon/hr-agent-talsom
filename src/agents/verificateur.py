import re 
import json
from src.core.llm import get_llm

# PROMPT TEMPLATES

# SYSTEM_PROMPT: tells the model what it is, what to produce, and how to reason
SYSTEM_PROMPT = """\
Tu es un vérificateur de faits strict. Ton seul rôle est d'évaluer si une réponse \
est fidèlement ancrée dans les sources fournies.

Tu réponds UNIQUEMENT AVEC un objet JSON valide. Aucun texte avant, aucun texte après.
Aucune explication. Aucun markdown. Juste le JSON brut.

Format attendu:
{
    "score": <float entre 0.0 et 1.0>,
    "grounded": <true si score >= 0.75, sinon false>,
    "issues": [<liste de strings décrivant les problèmes, vide si aucun>],
    "feedback": "<une phrase actionnable pour le rédacteur, vide si aucun problème>"
}

Critères d'évaluation :
- 1.0 : chaque affirmation est directement étayée par une source
- 0.75-0.99 : quelques imprécisions mineures, rien d'inventé
- 0.50-0.74 : affirmations partiellemment étayées ou ambiguës
- 0.0-0.49  : affirmations inventées ou contredites par les sources\
"""

# USER_TEMPLATE: shows the specific thing to evaluate 
USER_TEMPLATE = """\
SOURCES DISPONIBLES :
{sources}

RÉPONSES À VÉRIFIER :
{response}

QUESTION ORIGINALE :
{query}\
"""


""" 
Fallback regex extractor

When calling json.loads() on LLM output, we are expecting the model followed instructions perfectly.
With GPT-4, Claude it usually works perfectly. With Llama 3 8B quantized, it will fail often 
(since quantization degrades instruction-following precision on structured output tasks)

There are many strategies (1 to 4) to cover and adjust to all four possible failure modes

Here are the four failure modes we could hit in practice:

   1. Clean — Strategy 1 handles this
   {"score": 0.85, "grounded": true, "issues": [], "feedback": ""}

   2. Prose wrapping (followed JSON structure but added conversational text around it) — Strategy 1 fails, Strategy 2 catches it
   Voici mon évaluation :
   {"score": 0.85, "grounded": true, "issues": [], "feedback": ""}
   Bonne réponse dans l'ensemble.

   3. Trailing comma (invalid JSON) — Strategy 2 fails, Strategy 3 catches it
   {"score": 0.85, "grounded": true, "issues": ["date incorrecte",], "feedback": ""}

   4. Completely broken — Strategy 4 kicks in
   Le score est environ 0.85 et la réponse semble correcte.

"""

# Matches the first {...} block in LLM output, even with surrounding prose 
_JSON_PATTERN = re.compile(r'\{[^{}]*\}', re.DOTALL)

def _extract_json(raw: str) -> dict:
    """
    Extract the JSON object from a raw LLM string

    Strategy:
      1. Direct json.loads() - works only if model behaved
      2. Regex extraction of first {...} block - handles prose wrapping
      3. Field-by-field regex - handles malformed but mostly correct output
      4. Safe fallback - never raises, returns a neutral result
    """
    
    # Strategy 1
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract first JSON block
    match = _JSON_PATTERN.search(raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 3: field-by-field extraction 
    result = {}

    score_match = re.search(r'"score"\s*:\s*([0-9.]+)', raw)
    if score_match:
        result["score"] = float(score_match.group(1))

    grounded_match = re.search(r'"grounded"\s*:\s*(true|false)', raw, re.IGNORECASE)
    if grounded_match:
        result["grounded"] = grounded_match.group(1).lower() == "true"

    issues_match = re.search(r'"issues"\s*:\s*\[([^\]]*)\]', raw, re.DOTALL)
    if issues_match:
        raw_issues = issues_match.group(1)
        result["issues"] = re.findall(r'"([^"]+)"', raw_issues)

    feedback_match = re.search(r'"feedback"\s*:\s*"([^"]*)"', raw)
    if feedback_match:
        result["feedback"] = feedback_match.group(1)

    if "score" in result:
        return _fill_defaults(result)
    

    # Strategy 4: safe fallback i.e. score of 0.5 is neutral (problematic part)
    return {
        "score": 0.5,
        "grounded": False,
        "issues": ["Impossible d'analyser la réponse du vérificateur."],
        "feedback": "",
    }

def _fill_defaults(partial: dict) -> dict:
    partial.setdefault("score", 0.5)
    partial.setdefault("grounded", partial["score"] >= 0.75)
    partial.setdefault("issues", [])
    partial.setdefault("feedback", "")
    return partial


# Verificateur class

class Verificateur:
    """
    Truncate each chunk to this many characters before sending to the LLM
    At ~4 chars/token, 400 chars ≈ 100 tokens per chunk
    5 chunks x 100 tokens = 500 tokens of context for verification
    which is enough to catch grounding errors without blowing the context window (4096 tokens for context window)

    Why truncate? Verificateur doesn't need to re-read the full chunk, only needs enough context to check whether the redacteur claim is supported
    5 chunks retrieved x 100 tokens each (after truncation) = 500 tokens of source context
    500 + system prompt (≈ 200 tokens) + query (≈ 50 tokens) + response (≈ 200 tokens) ≈ 950 tokens -> well within the 4096 context window
    """

    CHUNK_PREVIEW_CHARS = 800
    
    def __init__(self):
        self.llm = get_llm()

    def verify(self, query: str, response: str, chunks: list[dict]) -> dict:
        """
        Scores a generated response against the retrieved source chunks

        Returns a dict with keys: score, grounded, issues, feedback
        Never raises since all failure modes are handled internally
        """
        # converts the raw list of chunk dicts from the retriever into a clean string. This is what the LLM will use as a truth
        sources_block = self._format_sources(chunks)

        # injects the formatted sources, the redacteur response and original query into USER_TEMPLATE
        # then wraps everything with SYSTEM_PROMPT into one final string ready for inference
        prompt = self._build_prompt(query, response, sources_block)

        # Sends that string to Llama 3 and gets back raw text. Could be JSON object, but not guaranteed
        raw    = self._call_llm(prompt)
        # parses the raw text into dict
        result = _extract_json(raw)

        # Maintain score to [0.0, 1.0]
        result["score"]    = max(0.0, min(1.0, float(result.get("score", 0.5))))
        result["grounded"] = result["score"] >= 0.75
        
        return result
    

    # Private helpers

    def _format_sources(self, chunks: list[dict]) -> str:
        """
        Renders chunks into a numbered block the LLM can reference

        Example:
          [1] politique_vacances — section: article_3
          Les employés accumulent 1.25 jour par mois travaillé
        """
        lines = []
        for i, chunk in enumerate(chunks[:3], start=1):
            meta    = chunk.get("metadata", {})
            title   = meta.get("title", "Document inconnu")
            section = meta.get("section", "-")
            text    = chunk.get("text", "")[:self.CHUNK_PREVIEW_CHARS]     # 400-character truncation happens
            lines.append(f"[{i}] {title} - section: {section}\n{text}")
        return "\n\n".join(lines)
    
    def _build_prompt(self, query: str, response: str, sources: str) -> str:
        """
        It assembles the final string that gets sent to Llama
        """
        user_content = USER_TEMPLATE.format(
            sources=sources,
            response=response,
            query=query,
        )
        return (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n{SYSTEM_PROMPT}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n{user_content}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )
    
    def _call_llm(self, prompt: str) -> str:
        output = self.llm(
            prompt,
            max_tokens=256,
            temperature=0.0,
            repeat_penalty=1.1,
            stop=["<|eot_id|>", "<|start_header_id|>"]
        )
        raw = output["choices"][0]["text"].strip()
        print(f"\n[VERIFICATEUR RAW OUTPUT]:\n{raw}\n")  # ← ajoute cette ligne
        return raw
    

