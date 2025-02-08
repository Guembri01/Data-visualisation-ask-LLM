Data Visualization App Documentation - Master File
====================================================

Created by sphinx-quickstart on Sun Feb  4 15:16:39 2024.
Adapted to include AI model integration (Gemini/Claude).

This project is a data analysis and visualization platform for cleaning, analyzing, and visualizing data with AI-driven insights.
Under Master AI & Data Science by Wissal Ben Othmen & Bilel Guembri.

This documentation provides a comprehensive guide to the Data Visualization App,
a Flask-based tool for data upload, cleaning, analysis, and visualization.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   ai_model
   utils_api
   app_api

Key Features:
-------------

*   **Data Upload:**  Supports multiple file formats (CSV, Excel, JSON, etc.).
*   **Data Quality Checks:**  Automatic detection of missing values, duplicates, and data type issues.
*   **Data Cleaning:**  Tools for handling missing data and removing duplicate entries.
*   **Interactive Visualizations:**  Creation of various customizable plots based on your data.
*   **AI-Powered Insights:**  Integration with **Gemini or Claude** language models for:
    *   **Visualization Suggestions:**  The AI model analyzes your data and recommends appropriate visualizations.
    *   **Data Interpretation:**  The AI model provides textual descriptions and insights based on the generated visualizations.
    *   **Automated Analysis (Future Enhancement):**  Potential for the AI to perform more complex statistical analyses and generate reports.
*   **User-Friendly Interface:**  Designed for intuitive use, regardless of data analysis expertise.


.. note::
    This is a basic documentation structure.  Expand this to include sections for:

    *   **Getting Started:** Installation, initial setup, and basic usage.
    *   **User Guide:**  Detailed instructions on using each feature of the app.
    *   **AI Model Integration:**  Specific details on how to use the Gemini/Claude features, including API keys, configuration, and limitations.
    *   **Troubleshooting:**  Common issues and solutions.
    *   **API Reference:**  Comprehensive documentation of the `ai_model`, `utils_api`, and `app_api` modules.
    *   **Contributing:**  Guidelines for contributing to the project.
    *   **Examples:**  Showcase different use cases and provide sample data and visualizations.
    *   **Advanced Usage:** Coverage of power-user features and customizations.
    *   **Release Notes / Changelog:** Track updates and new features
