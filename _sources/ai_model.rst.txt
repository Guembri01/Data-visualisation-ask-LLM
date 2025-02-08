AI Model Integration
====================

This application leverages the power of Google's Gemini 1.5 Flash and Anthropic's Claude 3 Opus language models to
provide intelligent features such as:

*   **Plot Suggestions:** The selected AI model analyzes your uploaded dataset and suggests
    appropriate visualizations.
    *   Gemini: See :py:func:`utils.get_plot_suggestion_from_gemini`.
    *   Claude: See :py:func:`utils.get_plot_suggestion_from_claude`.
*   **Graph Interpretation:** The selected AI model can provide textual interpretations of
    generated plots, helping you understand the insights revealed by your data.
    *   Gemini: See :py:func:`utils.generate_graph_interpretation_gemini`.
    *   Claude: See :py:func:`utils.generate_graph_interpretation_claude`.
*   **Interactive Chat:** You can interact with the selected AI model to ask questions about
    your graphs and data.
    *    Gemini: See :py:func:`utils.handle_graph_communication_gemini`.
    *    Claude: See :py:func:`utils.handle_graph_communication_claude`.

The AI models are used within the following functions:

*   In `utils.py`:
    *   :py:func:`utils.get_plot_suggestion_from_gemini`
    *   :py:func:`utils.generate_graph_interpretation_gemini`
    *   :py:func:`utils.handle_graph_communication_gemini`
    *   :py:func:`utils.get_plot_suggestion_from_claude`
    *   :py:func:`utils.generate_graph_interpretation_claude`
    *   :py:func:`utils.handle_graph_communication_claude`

*   In `app.py`:
    *   :py:func:`app.generate_plots`
    *   :py:func:`app.get_interpretation`
    *   :py:func:`app.graph_chat`

API Key Configuration
---------------------

To use the AI-powered features, you'll need to obtain an API key from
either Google (for Gemini) or Anthropic (for Claude) and provide it to the application.  You will need
to select which model you are using. Instructions for obtaining and
configuring the API key will be provided in the [Getting Started] section (you need to add details about the API Key configuration to the getting_started.rst file.)

Model Selection
---------------
The user can select either Gemini or Claude as their preferred AI model for generating
visualizations, interpretations, and interactive chat. The application will use the selected
model for all subsequent AI-related operations.  This choice is made through the user
interface and passed to the backend functions.

