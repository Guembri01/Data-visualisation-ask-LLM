# Data Visualization App with AI-Powered Insights

This project is a Flask-based web application designed to simplify data analysis and visualization. It allows users to upload data in various formats, perform data quality checks, clean the data, and generate interactive visualizations.  The application leverages the power of AI (Gemini and Claude) to provide intelligent suggestions for visualizations and interpretations of the generated graphs.  It also features an interactive chat interface where users can ask questions about their data and visualizations.

This project was developed under the Master AI & Data Science program by Wissal Ben Othmen & Bilel Guembri.

## Project Demo

A live demo of the application is available here: [https://data-visualisation-ask-llm.onrender.com/welcome](https://data-visualisation-ask-llm.onrender.com/welcome)

## Documentation

Comprehensive documentation, including detailed explanations of all functions, is available here: [https://guembri01.github.io/Data-visualisation-ask-LLM/](https://guembri01.github.io/Data-visualisation-ask-LLM/)

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Guembri01/Data-visualisation-ask-LLM
    cd Data-visualisation-ask-LLM
    ```

2.  **Create and activate a virtual environment (recommended):**

    *   Using `venv` (recommended for most users):

        ```bash
        python3 -m venv env
        source env/bin/activate  # On Linux/macOS
        env\Scripts\activate  # On Windows
        ```

    *   Using `virtualenv` (if you have it installed):

        ```bash
        virtualenv env
        source env/bin/activate  # On Linux/macOS
        env\Scripts\activate  # On Windows
        ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    python app.py
    ```
    The app will be running on `http://127.0.0.1:5000/` by default.

## Project Structure

The project is organized as follows:

*   `app.py`:  The main Flask application file, containing the routes and application logic.
*   `utils.py`:  Utility functions for data processing, AI model interaction, and plot generation.
*   `templates/`:  HTML templates for the web interface.
*   `static/`:  Static files (CSS, JavaScript, images).
*   `uploads/`:  Directory to store uploaded files (created automatically).
*   `requirements.txt`:  List of project dependencies.
*   `tests/` : Directory for tests to ensure code reliability (see Testing section below).
*  `docs/`: Contains files to generate project documentation (using Sphinx).
* `images/`: Screenshots of the application.


## Features

*   **Data Upload:** Supports CSV, Excel (XLS, XLSX), JSON, TSV, Parquet, Feather, ORC, XML, HTML, and HDF5 files.
*   **Data Quality Checks:** Automatically detects missing values, duplicate rows, and data type inconsistencies.
*   **Data Cleaning:** Provides tools to handle missing data (e.g., filling numerical NaNs with the mean) and remove duplicate rows.
*   **Interactive Visualizations:** Generates a variety of plots (using Matplotlib and Seaborn) based on your data and AI suggestions.
*   **AI-Powered Insights:**
    *   **Plot Suggestions:**  Suggests appropriate visualizations based on your dataset.
    *   **Graph Interpretation:** Provides textual interpretations of generated plots.
    *   **Interactive Chat:**  Allows users to ask questions about their graphs and data using natural language.
*   **Model Selection:** Choose between Google's Gemini 1.5 Flash and Anthropic's Claude 3 Opus for AI features.
*   **API Key Configuration:**  Requires API keys for the chosen AI model (Gemini or Claude).

## Screenshots

**1. Home Page**
![Home Page](images/1.png)

**2. Data Information**
![Data Information](images/2.png)

**3. Data Check**
![Data Check](images/3.png)

**4. Data Fix**
![Data Fix](images/4.png)

**5. Visualization Page (Selecting Model)**
![Visualization Selection](images/5.png)

**6. Visualization Page (Gemini Results)**
![Visualization Gemini](images/6.png)

**7. Visualization Page (Claude Results)**
![Visualization Claude](images/7.png)

**8. Interactive Graph Chat**
![Graph Chat](images/8.png)

## Testing

The project includes a suite of unit tests to ensure the functionality of key components.  The tests are located in the `tests/` directory.

You can run the tests using the built-in `unittest` module. Navigate to the project's root directory and run:

```bash
python -m unittest discover -s tests -p "test_app.py"
```
content_copy
download
Use code with caution.
Markdown

This command does the following:

python -m unittest: Invokes the unittest module.

discover: Tells unittest to automatically find tests.

-s tests: Specifies the tests directory as the starting point for discovery.

-p "test_*.py": Specifies a pattern to match test files (files starting with test_).

A successful test run indicates that the core functions are working as expected. Here's the Test function Result.

![alt text](images/test_image.png)

Contributing

Contributions to this project are welcome! Please see the CONTRIBUTING.md file (to be created) for guidelines. Before contributing, it's a good idea to open an issue to discuss your proposed changes.

License

This project is licensed under the MIT License (to be created - you should add a LICENSE file with the MIT license text).

