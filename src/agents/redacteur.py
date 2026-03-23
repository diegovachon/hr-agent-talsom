from src.core.llm import get_llm

# PROMPT TEMPLATES

SYSTEM_PROMPT_BASE = """\
Tu es un assistant RH interne pour une entreprise de 200 employés. 
Tu réponds aux questions des employés concernant les politiques internes.

Règles absolues:
- Tu te bases UNIQUEMENT sur les sources fournies. Jamais sur tes connaissances générales.
- Tu cites toujours le document source entre crochets, ex: [politique_vacances, article 3].
- Si l'information n'est pas dans les sources, tu le dis explicitement.
- Tu ne fais jamais de suppositions ni d'extrapolations.
- Tu ne donnes jamais de conseils juridiques ou médicaux.\
"""

TONE_INSTRUCTIONS = {
    1: """\
Ton style: direct, factuel, concis.
L'employé pose une question administrative simple. Va droit au but.\
""",
    2: """\
Ton style: empathique mais professionnel.
La question touche à une situation personnelle ou délicate.
Réponds avec soin, et suggère de consulter un gestionnaire ou 
les RH si la situation nécessite un suivi personnalisé.\
""",
}

USER_TEMPLATE = """\
SOURCES DISPONIBLES :
{sources}

{feedback_block}QUESTION : {query}\
"""

FEEDBACK_BLOCK_TEMPLATE = """\
RETOUR DU VÉRIFICATEUR (à corriger dans ta réponse) :
{feedback}

"""

class Redacteur:

    def __init__(self):
        self.llm = get_llm()

    def generate(
        self,
        query: str,
        chunks: list[dict],
        sensitivity: int,
        feedback: str = "",
    ) -> str:
        """
        Generates a grounded HR response

        On first pass, feedback is empty and the model generates freely within
        the constraints of the system prompt and sources

        On revision passes, feedback contains the verificateur's specific
        issues, the model has to fix them rather than regenerate from scratch
        """
        sources_block  = self._format_sources(chunks)
        feedback_block = self._build_feedback_block(feedback)
        system_prompt  = self._build_system_prompt(sensitivity)
        prompt         = self._build_prompt(system_prompt, sources_block, feedback_block, query)

        return self._call_llm(prompt)
    
    # PRIVATE HELPERS

    def _build_system_prompt(self, sensitivity: int) -> str:
        """
        Combines the base rules with the tone instruction for this sensitivity level
        Level 1 gets direct tone, and level 2 gets empathetic tone
        """
        tone = TONE_INSTRUCTIONS.get(sensitivity, TONE_INSTRUCTIONS[1])
        return f"{SYSTEM_PROMPT_BASE}\n\n{tone}"
    
    def _format_sources(self, chunks: list[dict]) -> str:
        """
        Same numbered format as the verificateur 
        """
        lines = []
        for i, chunk in enumerate(chunks, start=1):
            meta    = chunk.get("metadata", {})
            title   = meta.get("title", "Document inconnu")
            section = meta.get("section", "—")
            text    = chunk.get("text", "")
            lines.append(f"[{i}] {title} — section: {section}\n{text}")
        return "\n\n".join(lines)

    def _build_feedback_block(self, feedback: str) -> str:
        """
        Returns an empty string on the first pass (no feedback yet).
        On revision passes, wraps the feedback in a clearly labeled block
        so the model knows exactly what to fix rather than regenerating
        the entire response from scratch.
        """
        if not feedback:
            return ""
        return FEEDBACK_BLOCK_TEMPLATE.format(feedback=feedback)
    
    def _build_prompt(
        self,
        system_prompt: str,
        sources: str,
        feedback_block: str,
        query: str,
    ) -> str:
        user_content = USER_TEMPLATE.format(
            sources=sources,
            feedback_block=feedback_block,
            query=query,
        )
        return (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n{user_content}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )
    
    def _call_llm(self, prompt: str) -> str:
        output = self.llm(
            prompt,
            max_tokens=512,
            temperature=0.1,            # slight creativity for natural phrasing
            repeat_penalty=1.1, 
            stop=["<|eot_id|>", "<|start_header_id|>"],
        )
        return output["choices"][0]["text"].strip()
        

"""
Main differences with verificateur:
  1. No JSON extraction needed
  2. Sources not truncated: Redacteur needs full context
  3. _build_system_prompt() is dynamic: Redacteur has 2 possible tones
"""