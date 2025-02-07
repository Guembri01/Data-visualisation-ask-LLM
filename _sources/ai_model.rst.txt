AI Model Integration
====================

This application leverages the power of Google's Gemini 1.5 Flash language model to
provide intelligent features such as:

*   **Plot Suggestions:** Gemini analyzes your uploaded dataset and suggests
    appropriate visualizations. See :py:func:`utils.get_plot_suggestion_from_gemini`.
*   **Graph Interpretation:** Gemini can provide textual interpretations of
    generated plots, helping you understand the insights revealed by your data.
    See :py:func:`utils.generate_graph_interpretation_gemini`.
*   **Interactive Chat:** You can interact with Gemini to ask questions about
    your graphs and data. See :py:func:`utils.handle_graph_communication_gemini`.

The Gemini model is used within the following functions:

*   In `utils.py`:
    *   :py:func:`utils.get_plot_suggestion_from_gemini`
    *   :py:func:`utils.generate_graph_interpretation_gemini`
    *   :py:func:`utils.handle_graph_communication_gemini`

*   In `app.py`:
    *   :py:func:`app.generate_plots`
    *   :py:func:`app.get_interpretation`
    *   :py:func:`app.graph_chat`

API Key Configuration
---------------------

To use the Gemini-powered features, you'll need to obtain an API key from
Google and provide it to the application.  Instructions for obtaining and
configuring the API key will be provided in the [Getting Started] section (you need to add details about the API Key configuration to the getting_started.rst file.)