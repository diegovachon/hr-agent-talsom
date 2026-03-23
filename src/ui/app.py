"""
Gradio interface for the HR Policy Agent.

Run with :
    python -m src.ui.app
"""

import gradio as gr
from src.core.orchestrator import Orchestrator, AgentResult

from src.evaluation.bandit import UCB1Bandit, ATTACK_ARMS, AttackResult
from src.evaluation.reward_evaluator import RewardEvaluator

# -- Orchestrator (loaded once at startup) --

orchestrator = Orchestrator()

bandit    = UCB1Bandit()
evaluator = RewardEvaluator()

# -- Formatting helpers --

SENSITIVITY_LABELS = {
    1: ("Niveau 1 — Standard",    "#2d6a4f"),
    2: ("Niveau 2 — Délicat",     "#b5842a"),
    3: ("Niveau 3 — Escalade",    "#9b2226"),
}

def _format_sources(sources: list[dict]) -> str:
    if not sources:
        return "_Aucune source trouvée._"

    lines = []
    for s in sources:
        title   = s.get("title",   "Document inconnu")
        section = s.get("section", "—")
        score   = s.get("score",   0.0)
        lines.append(f"- **{title}** · {section} · score : `{score:.2f}`")

    return "\n".join(lines)


def _format_verification(result: AgentResult) -> str:
    score   = result.grounding_score
    passes  = result.attempts

    # Score bar — 10 blocks, filled proportionally
    filled  = round(score * 10)
    bar     = "█" * filled + "░" * (10 - filled)

    lines = [
        f"**Score de grounding** : `{score:.2f}` {bar}",
        f"**Passes de révision** : {passes}",
    ]

    if result.issues:
        lines.append("**Problèmes détectés :**")
        for issue in result.issues:
            lines.append(f"- {issue}")

    return "\n".join(lines)


def _sensitivity_badge(level: int) -> str:
    label, color = SENSITIVITY_LABELS.get(level, ("Niveau 1", "#2d6a4f"))
    return (
        f'<span style="background:{color};color:white;padding:3px 10px;'
        f'border-radius:4px;font-size:13px;font-weight:500">{label}</span>'
    )


# -- Core chat function --

def respond(
    query:   str,
    history: list[tuple[str, str]],
) -> tuple[list[tuple[str, str]], str, str, str]:
    """
    Called by Gradio on every user message.

    Returns four values that update four separate UI components :
      1. history     → the chat window
      2. sources_md  → the sources panel (markdown)
      3. verify_md   → the verification panel (markdown)
      4. badge_html  → the sensitivity badge (HTML)
    """

    if not query.strip():
        return history, "", "", ""

    result: AgentResult = orchestrator.run(query)

    history = history + [{"role": "user", "content": query}, 
                         {"role": "assistant", "content": result.response}]

    sources_md = _format_sources(result.sources)
    verify_md  = _format_verification(result)
    badge_html = _sensitivity_badge(result.sensitivity)

    return history, sources_md, verify_md, badge_html

# -- BANDIT --
def run_red_team(n_rounds: int) -> tuple[str, str]:
    """
    Runs n_rounds of adversarial red teaming.
    Returns (results_markdown, summary_markdown).
    """
    bandit.reset()
    rows = []

    for round_num in range(1, int(n_rounds) + 1):
        arm_index  = bandit.select_arm()
        arm_name   = ATTACK_ARMS[arm_index]
        query      = bandit.get_attack(arm_index)
        result     = orchestrator.run(query)
        reward, reason = evaluator.evaluate(arm_name, result, query)

        bandit.update(arm_index, reward)

        attack_result = AttackResult(
            round_number=round_num,
            arm_index=arm_index,
            arm_name=arm_name,
            attack_query=query,
            system_response=result.response[:120] + "...",
            reward=reward,
            reward_reason=reason,
        )
        bandit.record(attack_result)

        status = "✗ Succès attaque" if reward >= 1.0 else (
                 "~ Ambigu"         if reward == 0.5 else
                 "✓ Résisté"
        )
        rows.append(
            f"| {round_num} | {arm_name} | {query[:60]}... | {status} | {reason} |"
        )

    results_md = (
        "| Round | Catégorie | Attaque | Résultat | Raison |\n"
        "|-------|-----------|---------|----------|--------|\n"
        + "\n".join(rows)
    )

    summary    = bandit.summary()
    summary_md = (
        f"**Rounds total** : {summary['total_rounds']}  \n"
        f"**Taux de résistance** : {summary['resistance_rate']}%  \n"
        f"**Catégorie la plus efficace** : `{summary['most_effective_arm']}`  \n\n"
        f"**Taux de succès par catégorie :**  \n"
        + "\n".join([
            f"- `{arm}` : {rate}%"
            for arm, rate in summary['arm_success_rates'].items()
        ])
    )

    return results_md, summary_md

# -- Gradio UI --

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Agent RH") as demo:

        gr.Markdown(
            """
            # Agent RH — Politiques internes
            Posez vos questions sur les politiques de vacances, avantages sociaux,
            formation, évaluation et plus.
            """
        )

        with gr.Tabs():

            # -- Tab 1 : Chat --
            with gr.Tab("Conversation"):

                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(label="Conversation")
                        with gr.Row():
                            query_box = gr.Textbox(
                                placeholder="Ex : Combien de jours de vacances après 2 ans?",
                                label="Votre question",
                                lines=2,
                                scale=5,
                            )
                            submit_btn = gr.Button("Envoyer", variant="primary", scale=1)
                        clear_btn = gr.Button("Effacer la conversation", size="sm")

                    with gr.Column(scale=2):
                        badge_display = gr.HTML(
                            value='<span style="color:#888;font-size:13px">En attente...</span>',
                            label="Niveau de sensibilité",
                        )
                        with gr.Accordion("Sources", open=True):
                            sources_display = gr.Markdown(value="_En attente d'une question..._")
                        with gr.Accordion("Vérification", open=True):
                            verify_display = gr.Markdown(value="_En attente d'une question..._")

                gr.Examples(
                    examples=[
                        ["Combien de jours de vacances ai-je après 3 ans d'ancienneté?"],
                        ["Comment fonctionne le remboursement des frais de formation?"],
                        ["Quelle est la procédure pour un congé maladie?"],
                        ["Mon manager me harcèle, que puis-je faire?"],
                    ],
                    inputs=query_box,
                    label="Exemples de questions",
                )

                history_state = gr.State([])

                def on_submit(query, history):
                    return respond(query, history)

                def on_clear():
                    return [], [], "_En attente..._", "_En attente..._", ""

                submit_btn.click(
                    fn=on_submit,
                    inputs=[query_box, history_state],
                    outputs=[history_state, sources_display, verify_display, badge_display],
                ).then(
                    fn=lambda h: h,
                    inputs=[history_state],
                    outputs=[chatbot],
                ).then(
                    fn=lambda: "",
                    outputs=[query_box],
                )

                query_box.submit(
                    fn=on_submit,
                    inputs=[query_box, history_state],
                    outputs=[history_state, sources_display, verify_display, badge_display],
                ).then(
                    fn=lambda h: h,
                    inputs=[history_state],
                    outputs=[chatbot],
                ).then(
                    fn=lambda: "",
                    outputs=[query_box],
                )

                clear_btn.click(
                    fn=on_clear,
                    outputs=[history_state, chatbot, sources_display, verify_display, badge_display],
                )

            # -- Tab 2 : Red Team --
            with gr.Tab("Red Teaming"):

                gr.Markdown(
                    """
                    ### Évaluation adversariale — UCB1 Bandit
                    Un agent attaquant génère automatiquement des requêtes adversariales
                    et apprend quelles catégories d'attaques sont les plus efficaces.
                    """
                )

                with gr.Row():
                    rounds_slider = gr.Slider(
                        minimum=5,
                        maximum=30,
                        value=10,
                        step=5,
                        label="Nombre de rounds",
                    )
                    run_btn = gr.Button("Lancer l'évaluation", variant="primary")

                with gr.Row():
                    with gr.Column(scale=3):
                        results_display = gr.Markdown(value="_Lancer une évaluation pour voir les résultats..._")
                    with gr.Column(scale=1):
                        summary_display = gr.Markdown(value="_Résumé disponible après l'évaluation._")

                run_btn.click(
                    fn=run_red_team,
                    inputs=[rounds_slider],
                    outputs=[results_display, summary_display],
                )

    return demo


# -- Entry point --

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name="127.0.0.1",   # accessible on local network, not just localhost
        server_port=7860,
        share=False,              # set True to get a public gradio.live URL
        show_error=True,
        theme=gr.themes.Soft(),
    )