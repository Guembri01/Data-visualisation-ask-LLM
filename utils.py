# utils.py (Fully Documented and Refactored)
"""
Utility functions for the data visualization application.

Provides data analysis, cleaning, interaction with Gemini and Claude,
and plot generation functionalities.
"""

import pandas as pd
import json
import base64
import time
import google.generativeai as genai

from google.generativeai import GenerativeModel
from io import BytesIO
import re
import numpy as np
import matplotlib
import random
from typing import List, Optional

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import anthropic

def analyze_data(df: pd.DataFrame) -> dict:
    """Analyzes the DataFrame for data quality issues (missing values, duplicates, data types).

    Args:
        df: The input pandas DataFrame.

    Returns:
        A dictionary containing analysis results.
    """
    logging.info("Analyzing data...")
    print("analyze_data - START")
    analysis_results = {
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),  # Ensure int
        "data_types": df.dtypes.apply(lambda x: x.name).to_dict(),
    }

    # Ensure values are standard Python types (not numpy types)
    for col, count in analysis_results["missing_values"].items():
        if isinstance(count, np.int64):
            analysis_results["missing_values"][col] = int(count)

    print("analyze_data - END")
    return analysis_results

def apply_fixes_to_data(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Applies basic data cleaning (removes duplicates, fills numerical NaNs with mean).

    Args:
        df: The input pandas DataFrame.

    Returns:
        A tuple: (cleaned DataFrame, JSON string summary of fixes).
    """
    logging.info("Applying data fixes...")
    print("apply_fixes_to_data - START")
    fixes_summary = {}
    df_no_duplicates = df.drop_duplicates()
    num_duplicates_removed = int(df.duplicated().sum() - df_no_duplicates.duplicated().sum()) # Ensure Int

    if num_duplicates_removed > 0:
        fixes_summary["removed_duplicates"] = num_duplicates_removed
        print(f"apply_fixes_to_data - Removed duplicates: {num_duplicates_removed}")

    numerical_cols = df_no_duplicates.select_dtypes(include=["number"]).columns
    for col in numerical_cols:
        if df_no_duplicates[col].isnull().any():
            mean_value = df_no_duplicates[col].mean()
            df_no_duplicates[col] = df_no_duplicates[col].fillna(mean_value)  # Use bracket notation
            fixes_summary[f"filled_missing_in_{col}"] = f"Filled with mean: {mean_value:.2f}"
            print(f"apply_fixes_to_data - Filled missing in {col}: {mean_value:.2f}")

    print("apply_fixes_to_data - END")
    return df_no_duplicates, json.dumps(fixes_summary)

def create_dataset_summary(df: pd.DataFrame) -> str:
    """Generates a textual summary of the dataset.

    Args:
        df: The input pandas DataFrame.

    Returns:
        A string containing the dataset summary.
    """
    print("create_dataset_summary - START")
    summary = [
        "Column Names and Data Types:\n",
        ", ".join(f"{col}: {dtype}" for col, dtype in df.dtypes.items()) + "\n\n",
    ]

    numerical_cols = df.select_dtypes(include=["number"]).columns
    if not numerical_cols.empty:
        summary.append("Descriptive Statistics (for numerical columns):\n")
        summary.append(df[numerical_cols].describe().to_string() + "\n\n")

    categorical_cols = df.select_dtypes(include=["object"]).columns
    if not categorical_cols.empty:
        summary.append("Unique Value Counts (for categorical columns, top 5):\n")
        for col in categorical_cols:
            top_values = df[col].value_counts().head(5)
            value_counts_str = ", ".join(f"{val}: {count}" for val, count in top_values.items())
            summary.append(f"Column '{col}': {value_counts_str}\n")
        summary.append("\n")

    summary.append(f"Number of Rows: {df.shape[0]}\n")
    summary.append(f"Number of Columns: {df.shape[1]}\n")

    missing_values = df.isnull().sum()
    missing_info = ", ".join(f"{col}: {count}" for col, count in missing_values.items() if count > 0)
    summary.append(f"Missing Values: {missing_info}\n" if missing_info else "Missing Values: None\n")
    print("create_dataset_summary - END")
    return "".join(summary)

def _generate_interpretation(model_function, suggestion_text: str, dataset_summary: str, api_key: str) -> str:
    """Helper function to generate graph interpretations using a given model function."""
    try:
        interpretation = model_function(suggestion_text, dataset_summary, api_key)
        return interpretation if interpretation else ""
    except Exception as e:
        logging.exception(f"Error generating interpretation: {e}")
        print(f"Error generating interpretation: {e}")
        return ""

def generate_graph_interpretation_gemini(suggestion_text: str, dataset_summary: str, api_key: str) -> str:
    """Generates a graph interpretation using the Gemini language model."""
    logging.info("Generating graph interpretation with Gemini...")
    print("generate_graph_interpretation_gemini - START")

    genai.configure(api_key=api_key)
    model = GenerativeModel("gemini-1.5-flash")  # Use gemini-1.5-flash
    prompt = (
        f"Dataset Summary:\n{dataset_summary}\n\n"
        f"Plot Suggestion:\n{suggestion_text}\n\n"
        "Please provide a concise interpretation of this graph based on the suggestion "
        "and the provided dataset summary. Focus on the key relationships and trends "
        "visible in the plot, referencing the relevant statistical information from "
        "the summary."
    )
    logging.debug(f"Gemini prompt for interpretation: {prompt}")
    try:
        response = model.generate_content(prompt)
        interpretation = response.text
        logging.info("Graph interpretation generated by Gemini.")
        logging.debug(f"Gemini interpretation: {interpretation}")
        return interpretation
    except Exception as e:
        logging.exception(f"Error generating interpretation with Gemini: {e}")
        return ""

def generate_graph_interpretation_claude(suggestion_text: str, dataset_summary: str, api_key: str) -> str:
    """Generates a graph interpretation using the Claude language model."""
    logging.info("Generating graph interpretation with Claude...")
    print("generate_graph_interpretation_claude - START")

    def claude_model_call(suggestion_text, dataset_summary, api_key):
        client = anthropic.Anthropic(api_key=api_key)
        prompt = (
            f"{anthropic.HUMAN_PROMPT} Dataset Summary:\n{dataset_summary}\n\n"
            f"Plot Suggestion:\n{suggestion_text}\n\n"
            "Please provide a concise interpretation of this graph."
            f"{anthropic.AI_PROMPT}"
        )
        print(f"generate_graph_interpretation_claude - prompt: {prompt[:500]}...")
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        print(f"generate_graph_interpretation_claude - interpretation: {response.content[0].text}")
        return response.content[0].text

    result = _generate_interpretation(claude_model_call, suggestion_text, dataset_summary, api_key)
    print("generate_graph_interpretation_claude - END")
    return result

def _handle_communication(model_function, image_data: bytes, dataset_str: str, user_message: str, api_key: str) -> str:
    """Helper function for handling graph communication using a given model function."""
    try:
        response = model_function(image_data, dataset_str, user_message, api_key)
        return response if response else ""
    except Exception as e:
        logging.exception(f"Error in graph communication: {e}")
        print(f"Error in graph communication: {e}")
        return ""

def handle_graph_communication_gemini(image_data: bytes, dataset_str: str, user_message: str, api_key: str) -> str:
    """Handles user-model communication about a graph image using Gemini."""
    logging.info("Handling graph communication with Gemini...")
    print("handle_graph_communication_gemini - START")

    def gemini_comm_call(image_data, dataset_str, user_message, api_key):
        genai.configure(api_key=api_key)
        model = GenerativeModel("gemini-1.5-flash")  # Use gemini-1.5-flash
        image_part = {"mime_type": "image/png", "data": image_data}
        prompt_parts = [
            f"Dataset Summary:\n{dataset_str}\n\n",
            image_part,
            f"\n\nUser: {user_message}\nAI:",
        ]
        logging.debug(f"Gemini prompt for graph chat: {prompt_parts}")
        try:
            response = model.generate_content(prompt_parts)
            response_text = response.text
            logging.info("Graph communication response generated by Gemini.")
            logging.debug(f"Gemini response: {response_text}")
            return response_text
        except Exception as e:
            logging.exception(f"Error in handle_graph_communication_gemini: {e}")
            return f"Apologies, there is an error with Gemini"  # Return empty string on error

    result = _handle_communication(gemini_comm_call, image_data, dataset_str, user_message, api_key)
    print("handle_graph_communication_gemini - END")
    return result


def handle_graph_communication_claude(image_data: bytes, dataset_str: str, user_message: str, api_key: str) -> str:
    """Handles user-model communication about a graph image using Claude."""
    logging.info("Handling graph communication with Claude...")
    print("handle_graph_communication_claude - START")

    def claude_comm_call(image_data, dataset_str, user_message, api_key):
        client = anthropic.Anthropic(api_key=api_key)
        base64_image = base64.b64encode(image_data).decode("utf-8")
        prompt = f"Dataset Summary:\n{dataset_str}\n\nUser: {user_message}"
        print(f"handle_graph_communication_claude - prompt (first 500 chars): {prompt[:500]}...")
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": base64_image},
                        }
                    ],
                },
            ],
        )
        print(f"handle_graph_communication_claude - response_text: {response.content[0].text}")
        return response.content[0].text

    result = _handle_communication(claude_comm_call, image_data, dataset_str, user_message, api_key)
    print("handle_graph_communication_claude - END")
    return result

def fix_json(json_str: str) -> str:
    """Attempts to fix common JSON formatting errors (trailing commas).

    Args:
        json_str: The potentially malformed JSON string.

    Returns:
        The fixed JSON string, or the original if no fixes were applied.
    """
    print(f"fix_json - START: json_str={json_str[:500]}...")
    json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas in objects
    json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
    print(f"fix_json - END: json_str={json_str[:500]}...")
    return json_str

def get_top_n_variables(df: pd.DataFrame, column_name: str, n: int = 10) -> list:
    """Gets the top N most frequent values in a column.

    Args:
        df: The input DataFrame.
        column_name: The name of the column.
        n: The number of top values to retrieve.

    Returns:
        A list of the top N most frequent values.
    """
    print(f"get_top_n_variables - START: column_name={column_name}, n={n}")
    result = df[column_name].value_counts().nlargest(n).index.tolist()
    print(f"get_top_n_variables - END: result={result}")
    return result

def filter_data_by_top_variables(df: pd.DataFrame, column_name: str, top_n_variables: list) -> pd.DataFrame:
    """Filters the DataFrame to include only rows with top N values in a column.

    Args:
        df: pd.DataFrame, input dataframe
        column_name: str, column name
        top_n_variables: list, top n variables

    Returns:
        pd.DataFrame, filtered dataframe
    """
    print(f"filter_data_by_top_variables - START: column_name={column_name}, top_n_variables={top_n_variables}")
    result_df = df[df[column_name].isin(top_n_variables)]
    print(f"filter_data_by_top_variables - END: result_df.shape={result_df.shape}")
    return result_df


def _get_plot_suggestions(model_function, df: pd.DataFrame, api_key: str) -> Optional[List[dict]]:
    """Helper function to get plot suggestions and code using a model function, with retries."""
    max_retries = 5
    retry_delay = 5  # Start with a 5-second delay

    for attempt in range(max_retries):
        print(f"Attempt: {attempt + 1}")
        try:
            suggestions = model_function(df, api_key)
            if suggestions:
                return suggestions
        except (anthropic.APIStatusError, Exception) as e:
            logging.exception(f"Attempt {attempt + 1} failed: {e}")
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2**attempt) + random.uniform(0, 1)  # Exponential backoff
                print(f"Retrying in {wait_time:.2f}s")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Returning None.")
                return None
    return None


def get_plot_suggestion_from_gemini(df: pd.DataFrame, api_key: str) -> list | None:
    """Gets plot suggestions and Python code from Gemini, with retries."""
    logging.info("Getting plot suggestions from Gemini...")
    print("get_plot_suggestion_from_gemini - START")

    def gemini_plot_call(df: pd.DataFrame, api_key: str) -> list | None:
        genai.configure(api_key=api_key)
        model = GenerativeModel("gemini-1.5-flash")  # Use gemini-1.5-flash

        system_prompt = f"""
        Analyze this dataset with columns: {', '.join(df.columns.tolist())}.
        Suggest a diverse set of at least 7 different types of plots to visualize this data, and it's okay to generate more than 10.
        Consider a wide variety of plot types, including but not limited to:

        - Correlation matrix (heatmap) for numerical data.
        - Bar charts (grouped or stacked) for categorical data.
        - Radar charts for comparing multiple quantitative variables.
        - Box plots and violin plots to show distributions and outliers.
        - Scatter plots (including pair plots/scatter plot matrices) for relationships between numerical variables.
        - Histograms and density plots (KDE) for distributions of numerical features.
        - Pie charts for proportions.
        - Line plots and area plots for time series data (if a time-based column is present).
        - Treemaps for hierarchical data.
        - Mosaic plots for proportions of combinations of categorical variables.
        - Parallel coordinates plots for high-dimensional numerical data.
        - ECDF plots for cumulative distributions.
        - Heatmaps to visualize magnitudes across two variables.
        - 3D scatter plots if three numerical variable relationships are important.

        For each plot:
        1. Explain why this plot is appropriate for this dataset and what insights it might reveal.
        2. Provide Python code using matplotlib and/or seaborn.  Make the code as concise and readable as possible.
        3. DO NOT include data loading (e.g., pd.read_csv). Assume the data is in a DataFrame named 'df'.
        4. Handle potential errors gracefully (e.g., if a plot type is not suitable for the data).
        5. Return the results in JSON format: {{"suggestions":[{{"suggestion":"", "code":""}}, ...]}}
        """
        prompt = f"{system_prompt}\n\nDataset Preview:\n{df.head().to_string()}"
        logging.debug(f"Prompt sent to Gemini:\n{prompt}")

        try:
            response = model.generate_content(prompt)
            logging.debug("Response received from Gemini.")
            response_text = response.text.strip().replace("```json", "").replace("```", "").replace("\n", " ").replace("  ", " ").strip()
            logging.debug(f"Cleaned Gemini response: {response_text}")

            try:
                response_json = json.loads(response_text)
                logging.debug("JSON parsed successfully.")
            except json.JSONDecodeError as e:
                logging.warning(f"JSON parsing error: {e}")
                logging.warning("Attempting to fix JSON...")
                fixed_json_str = fix_json(response_text)
                try:
                    response_json = json.loads(fixed_json_str)
                    logging.info("JSON fixed and parsed.")
                    logging.debug(f"Fixed JSON: {fixed_json_str}")
                except json.JSONDecodeError as e:
                    logging.error(f"Could not fix JSON: {e}")
                    logging.error(f"Original JSON: {response_text}")
                    return None

            suggestions = response_json.get("suggestions", [])
            if not suggestions:
                logging.warning("No suggestions found in Gemini response.")
                return None

            plot_results = []
            for suggestion in suggestions:
                logging.info(f"Processing suggestion: {suggestion['suggestion']}")
                try:
                    code = suggestion["code"].strip()
                    filtered_df = df

                    # Filter data for categorical variables if mentioned in code
                    for column_name in df.columns:
                        if column_name in code:
                            top_n = get_top_n_variables(df, column_name)
                            filtered_df = filter_data_by_top_variables(filtered_df, column_name, top_n)

                    # Additional filtering for correlation matrix (only numerical)
                    if "sns.heatmap(df.corr()" in code:
                        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
                        filtered_df = df[numerical_cols]

                    if "plt.hist(data=" in code:
                        code = re.sub(r"plt\.hist\(data=([^)]+)\)", r"plt.hist(\1)", code)
                    code = code.replace('"""', "").strip().replace("plt.show()", "")

                    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
                    namespace = {"df": filtered_df, "plt": plt, "sns": sns, "np": np, "pd": pd}
                    logging.debug(f"Executing code:\n{code}")
                    exec(code, namespace)
                    logging.info("Code executed successfully.")

                    buffer = BytesIO()
                    plt.savefig(buffer, format="png", bbox_inches="tight", dpi=100)
                    buffer.seek(0)
                    image_base64 = base64.b64encode(buffer.getvalue()).decode()
                    plot_results.append({"suggestion": suggestion["suggestion"], "plot": image_base64})
                    plt.close("all")  # Close all figures to free memory
                    buffer.close()

                except Exception as e:
                    logging.exception(f"Error executing plot code: {e}")
                    plot_results.append({"suggestion": suggestion["suggestion"], "error": str(e)})

            logging.info(f"Returning {len(plot_results)} plot results")
            return plot_results

        except Exception as e:
            logging.exception(f"Gemini API Error: {e}")
            return None

    result = _get_plot_suggestions(gemini_plot_call, df, api_key)
    print("get_plot_suggestion_from_gemini - END")
    return result

def get_plot_suggestion_from_claude(df: pd.DataFrame, api_key: str) -> list | None:
    """Gets plot suggestions and Python code from Claude, with retries."""
    logging.info("Getting plot suggestions from Claude...")
    print("get_plot_suggestion_from_claude - START")

    def claude_plot_call(df: pd.DataFrame, api_key: str) -> list | None:
        client = anthropic.Anthropic(api_key=api_key)
        system_prompt = f"""
        Analyze this dataset with columns: {', '.join(df.columns.tolist())}.
        Suggest at least 7 diverse plot types (up to 10+). Consider:

        - Correlation matrix (heatmap)
        - Bar charts (grouped/stacked)
        - Radar charts
        - Box/violin plots
        - Scatter plots (pair plots)
        - Histograms/density plots (KDE)
        - Pie charts
        - Line/area plots (if time-based)
        - Treemaps
        - Mosaic plots
        - Parallel coordinates plots
        - ECDF plots
        - Heatmaps
        - 3D scatter plots

        For each plot:
        1. Explain its appropriateness and insights.
        2. Provide concise, readable Python code (matplotlib/seaborn).
        3. NO data loading (assume 'df' DataFrame).
        4. Handle errors gracefully.
        5. Return JSON: {{"suggestions":[{{"suggestion":"", "code":""}}, ...]}}
        """
        prompt = f"{anthropic.HUMAN_PROMPT} {system_prompt}\n\nDataset Preview:\n{df.head().to_string()}{anthropic.AI_PROMPT}"

        print(f"get_plot_suggestion_from_claude - prompt: {prompt[:500]}...")
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        print("get_plot_suggestion_from_claude - Response received")
        response_text = response.content[0].text.strip()
        # More robust JSON cleaning, handling whitespace and extra characters:
        response_text = re.sub(r'^[^\{]*', '', response_text)  # Remove leading non-JSON
        response_text = re.sub(r'[^\}]+$', '', response_text)  # Remove trailing non-JSON
        response_text = response_text.replace("```json", "").replace("```", "")
        response_text = re.sub(r',\s*}', '}', response_text)  # Remove trailing commas in objects
        response_text = re.sub(r',\s*]', ']', response_text)
        print(f"get_plot_suggestion_from_claude - cleaned_response: {response_text[:500]}...")

        try:
            response_json = json.loads(response_text)
            print("get_plot_suggestion_from_claude - JSON parsed successfully")
        except json.JSONDecodeError as e:
            print(f"get_plot_suggestion_from_claude - JSONDecodeError: {e}, attempting to fix")
            fixed_json_str = fix_json(response_text) #Still keeping the old fix_json.
            try:
                response_json = json.loads(fixed_json_str)
                print("get_plot_suggestion_from_claude - JSON fixed and parsed")
            except json.JSONDecodeError as e:
                print(f"get_plot_suggestion_from_claude - Could not fix JSON: {e}")
                return None

        suggestions = response_json.get("suggestions", [])
        if not suggestions:
            print("get_plot_suggestion_from_claude - No suggestions found")
            return None

        plot_results = []
        for suggestion in suggestions:
            print(f"get_plot_suggestion_from_claude - Processing suggestion: {suggestion['suggestion']}")
            try:
                code = suggestion["code"].strip()
                filtered_df = df

                for col in df.columns:
                    if col in code:
                        print(f"get_plot_suggestion_from_claude - Filtering for: {col}")
                        top_n = get_top_n_variables(df, col)
                        filtered_df = filter_data_by_top_variables(filtered_df, col, top_n)
                        print(f"get_plot_suggestion_from_claude - Filtered df shape: {filtered_df.shape}")


                if "sns.heatmap(df.corr()" in code:
                    print("get_plot_suggestion_from_claude - Filtering for correlation matrix")
                    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
                    filtered_df = df[numerical_cols]
                    print(f"get_plot_suggestion_from_claude - Filtered df shape (corr): {filtered_df.shape}")

                if "plt.hist(data=" in code:  #Fix common plt.hist issue
                    print("get_plot_suggestion_from_claude - Fixing plt.hist(data=...)")
                    code = re.sub(r"plt\.hist\(data=([^)]+)\)", r"plt.hist(\1)", code)

                code = code.replace('"""', '').strip().replace("plt.show()", "")

                plt.figure(figsize=(10, 6))
                namespace = {"df": filtered_df, "plt": plt, "sns": sns, "np": np, "pd": pd}
                print(f"get_plot_suggestion_from_claude - Executing code:\n{code}")

                exec(code, namespace)
                print("get_plot_suggestion_from_claude - Code executed successfully")

                buffer = BytesIO()
                plt.savefig(buffer, format="png", bbox_inches="tight", dpi=100)
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                plot_results.append({"suggestion": suggestion["suggestion"], "plot": image_base64})
                plt.close("all")
                buffer.close()
                print(f"get_plot_suggestion_from_claude - plot_results (success): {len(plot_results)}")

            except Exception as e:
                print(f"get_plot_suggestion_from_claude - Error executing code: {e}")
                plot_results.append({"suggestion": suggestion["suggestion"], "error": str(e)})
        return plot_results

    result = _get_plot_suggestions(claude_plot_call, df, api_key)
    print("get_plot_suggestion_from_claude - END")
    return result
def exec_code_to_generate_plot(code_str, df):
    """Executes Python code (provided as a string) to generate a plot.

    This function takes code generated by an LLM and tries to run it. It includes
       workarounds for common errors in LLM-generated plotting code. The generated plot
       is returned as a base64-encoded PNG image.

       Args:
           code_str (str): The Python code to execute (base64 encoded).
           df (pandas.DataFrame): The DataFrame to use for plotting.

       Returns:
           str: A base64-encoded string representing the generated plot image, or
                None if an error occurred during code execution.
    """
    logging.info("Executing code to generate plot...")
    try:
        code_str = base64.b64decode(code_str).decode("utf-8")
        code_str = code_str.replace("pd.compat.StringIO", "io.StringIO")
        code_str = re.sub(r"\\\\s\\\+", " ", code_str)
        code_str = code_str.replace("print(f'data:image/png;base64,{img_str}')", "print(img_str)")
        code_lines = code_str.split("\n")
        for i, line in enumerate(code_lines):
            if "data = '''" in line:
                code_lines[i] = ""
                while "'''" not in code_lines[i]:
                    code_lines[i] = ""
                    i += 1
                code_lines[i] = ""
                break
        code_str = "\n".join(line for line in code_lines if line.strip() != "")
        if "import io" not in code_str:  code_str = "import io\n" + code_str
        if "import base64" not in code_str: code_str = "import base64\n" + code_str
        logging.debug(f"Code to execute:\n{code_str}")

        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        try:
            local_vars = {"df": df}
            exec(code_str, {}, local_vars)
            logging.info("Code executed successfully.")
        except Exception as e:
            logging.exception(f"Error executing code: {e}")
            return None
        finally:
            sys.stdout = old_stdout
        output = captured_output.getvalue().strip()
        logging.debug(f"Output from executed code: {output}")

        base64_start = output.find("base64,")
        if base64_start != -1:  base64_str = output[base64_start + 7 :]
        else: base64_str = output
        logging.info("Plot generated successfully.")
        return base64_str
    except Exception as e:
        logging.exception(f"Error executing generated code: {e}")
        return None
