#Step #3 - Json based theme comparison

import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import time
from collections import defaultdict
import numpy as np
from IPython.display import Markdown, display, HTML
# from openai import OpenAI

class LLMThemeComparator:
    def __init__(self, api_key=None, model="gpt-3.5-turbo"):
        """
        Initialize the LLM-enhanced theme comparator

        Parameters:
        -----------
        api_key : str
            OpenAI API key (only used for enhanced analysis, not required for core functionality)
        model : str
            Model to use ("gpt-3.5-turbo" or "gpt-4o-mini")
        """
        self.themes = [
            "Macro environment", "Pricing", "Margins", "Bookings/Large Deals",
            "Discretionary/Small Deals", "People", "Cloud", "Security", "Gen AI",
            "M&A", "Investments", "Partnerships", "Technology Budget",
            "Product/IP/Assets", "Talent/Training", "Clients", "Awards/Recognition"
        ]

        # Initialize OpenAI client if API key is provided
        self.use_llm = False
        if api_key:
            try:
                # self.client = OpenAI(api_key=api_key)
                self.model = model
                self.use_llm = True
                print(f"LLM enhancement enabled using {model}")
            except Exception as e:
                print(f"Error initializing OpenAI client: {e}")
                print("Falling back to non-LLM analysis")
        else:
            print("LLM enhancement disabled (no API key provided)")

        self.api_url = "https://agent-dev.test.studio.lyzr.ai/v3/inference/chat/"
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": "sk-default-PPcvzcCe4cJRRP8JkEXnT51woYJUXzMZ"
        }
        self.user_id = "pranav@lyzr.ai"

    def _call_lyzr_api(self, agent_id: str, session_id: str, message: str) -> str:
        """Helper function to call Lyzr API"""
        payload = {
            "user_id": self.user_id,
            "agent_id": agent_id,
            "session_id": session_id,
            "message": message
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "No response received.")
        except requests.exceptions.RequestException as e:
            return f"Error calling Lyzr API: {str(e)}"

    def identify_company_name(self, quotes):
        """Identify the company name from a list of quotes."""
        if not quotes:
            return "Other Company"
        print("DEBUG 1", quotes)
        # c2_str = "\n".join([f"- Page {q['page']}: \"{q['quote']}\" " for q in quotes])
        print("DEBUG 2")
        prompt = f"""
            From the following quotes, identify the name of the company being referred to. The company name might be explicitly mentioned, or you might need to infer it from the context. If you cannot determine the company name, respond with "Other Company". Respond with only the company name, without any additional text or explanation.

        Quotes:
        {json.dumps(quotes)}

        In response just return the company, nothing else [IMPORTANT]
        """

        response = self._call_lyzr_api(
            agent_id="67c86b15be1fc2af4eb4027e",
            session_id="67c86b15be1fc2af4eb4027e",
            message=prompt
        )
        print("DEBUG 3",response)

        identified_name = response.strip()
        print("Name Identified", identified_name)
        return identified_name if identified_name else "Other Company"

    def load_transcript_data(self, json_path):
        """Load transcript data from JSON file"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        company_name = data.get("metadata", {}).get("company_name", "Unknown Company")
        time_period = data.get("metadata", {}).get("filename", "").split()[-1].replace(".pdf", "")

        # Extract aggregated quotes by theme
        themes_data = {}
        for theme, quotes in data.get("aggregated_verbatim_quotes", {}).items():
            if quotes:  # Only include themes with quotes
                themes_data[theme] = quotes

        return {
            "company_name": company_name,
            "time_period": time_period,
            "themes_data": themes_data
        }

    def calculate_similarity_score(self, company1_quotes, company2_quotes):
        """Calculate a simple similarity score based on quote counts and topics"""
        if not company1_quotes or not company2_quotes:
            return 0.0, "One company has no quotes on this theme"

        # Calculate a simple similarity based on number of quotes
        c1_count = len(company1_quotes)
        c2_count = len(company2_quotes)

        # Get ratio of smaller to larger (max similarity is 1.0)
        count_ratio = min(c1_count, c2_count) / max(c1_count, c2_count)

        # Extract key terms from quotes
        c1_text = " ".join([q["quote"].lower() for q in company1_quotes])
        c2_text = " ".join([q["quote"].lower() for q in company2_quotes])

        # Define key terms to check for overlap
        key_terms = ["growth", "increase", "revenue", "margin", "investment",
                     "strategy", "expansion", "market", "opportunity", "percent",
                     "billion", "million", "cloud", "digital", "ai", "acquisition"]

        # Calculate term overlap
        c1_terms = set([term for term in key_terms if term in c1_text])
        c2_terms = set([term for term in key_terms if term in c2_text])

        # Also look for numeric patterns like percentages, dollar amounts
        c1_numbers = set(re.findall(r'(\d+(?:\.\d+)?%|\$\d+(?:\.\d+)?(?:\s*million|\s*billion)?)', c1_text))
        c2_numbers = set(re.findall(r'(\d+(?:\.\d+)?%|\$\d+(?:\.\d+)?(?:\s*million|\s*billion)?)', c2_text))

        if not c1_terms or not c2_terms:
            term_similarity = 0
            reasoning = "Companies use completely different terminology"
        else:
            term_similarity = len(c1_terms.intersection(c2_terms)) / len(c1_terms.union(c2_terms))
            common_terms = c1_terms.intersection(c2_terms)
            reasoning = f"Companies share {len(common_terms)} key terms: {', '.join(list(common_terms)[:3])}"

            # Add reasoning about numerical values if available
            if c1_numbers and c2_numbers:
                reasoning += f". Both mention specific metrics/numbers."

        # Combined similarity (weight count and term similarity)
        similarity = (count_ratio * 0.4) + (term_similarity * 0.6)

        return round(similarity, 2), reasoning

    def llm_similarity_analysis(self, theme, company1_name, company2_name, company1_quotes, company2_quotes):
        """Use LLM to calculate similarity score and reasoning"""
        try:
            if not self.use_llm or not company1_quotes or not company2_quotes:
                return self.calculate_similarity_score(company1_quotes, company2_quotes)

            # Prepare quotes for prompt
            c1_quotes_str = "\n".join([f"- Page {q['page']}: \"{q['quote']}\" " for q in company1_quotes])
            c2_quotes_str = "\n".join([f"- Page {q['page']}: \"{q['quote']}\" " for q in company2_quotes])
            # print("c1_quotes_str",c1_quotes_str)
            # print("c2_quotes_str",c2_quotes_str)
            prompt = f"""Analyze the similarity between how {company1_name} and {company2_name} approach the "{theme}" theme in their earnings calls.

{company1_name} quotes:
{c1_quotes_str}

{company2_name} quotes:
{c2_quotes_str}

Calculate a similarity score between 0.0 and 1.0 where:
- 0.0 means completely different approaches, metrics, and priorities
- 0.2 if there is 20% key topics matching
- 0.4 if there is 40% key topics matching
- 0.5 if there is 50% key topics matching
- 0.7 if there is 70% key topics matching
- 1.0 means identical approaches, metrics, and priorities
Understand the relevance of match and score accordingly

Format your response as JSON:
{{
    "similarity_score": X.X,
    "reasoning": "Brief explanation of similarities or differences (1-2 sentences)"
}}
"""

            # Call API with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # response = self.client.chat.completions.create(
                    #     model=self.model,
                    #     messages=[
                    #         {"role": "system", "content": "You are a financial analyst specializing in comparing companies."},
                    #         {"role": "user", "content": prompt}
                    #     ],
                    #     temperature=0.3,
                    #     response_format={"type": "json_object"},
                    #     max_tokens=300
                    # )

                    # result = json.loads(response.choices[0].message.content)

                    result = self._call_lyzr_api(
                        agent_id="67dd47178f451bb9b9b6c318",
                        session_id="67dd47178f451bb9b9b6c318",
                        message="You are a financial analyst specializing in comparing companies." + prompt
                    )
                    json_str = result.replace('```json', '').replace('```', '')
                    data = json.loads(json_str)
                    # print(result)
                    # print(data)
                    score = float(data.get("similarity_score", 0.5))
                    reasoning = data.get("reasoning", "No explanation provided")

                    return score, reasoning
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"API error, retrying LLMTHEME 205 ({attempt+1}/{max_retries}): {e}")
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        print(f"API error after {max_retries} attempts, falling back to basic analysis: {e}")
                        return self.calculate_similarity_score(company1_quotes, company2_quotes)

        except Exception as e:
            print(f"Error in LLM similarity analysis: {e}")
            return self.calculate_similarity_score(company1_quotes, company2_quotes)

    def llm_theme_analysis(self, theme, company1_name, company2_name, company1_quotes, company2_quotes, similarity_score, reasoning):
        """Generate theme comparison analysis using LLM"""
        try:
            if not self.use_llm:
                return self.generate_basic_comparison(theme, company1_name, company2_name,
                                                    company1_quotes, company2_quotes,
                                                    similarity_score, reasoning)

            # Prepare quotes for prompt (limit to top 10 pe r company to avoid token limits)
            c1_quotes_str = "\n".join([f"- Page {q['page']}: \"{q['quote']}\" " for q in company1_quotes])
            c2_quotes_str = "\n".join([f"- Page {q['page']}: \"{q['quote']}\" " for q in company2_quotes])

            print("c1_quotes_str",company1_quotes)
            print("c2_quotes_str",company2_quotes)
            if not c1_quotes_str:
                c1_quotes_str = "No quotes available from this company on this theme."

            if not c2_quotes_str:
                c2_quotes_str = "No quotes available from this company on this theme."

            prompt = f"""Compare how {company1_name} and {company2_name} approach the "{theme}" theme based on these verbatim quotes.

                    IMPORTANT: Do not modify, rephrase, or alter any of the provided quotes. Return them exactly as given, preserving all punctuation, spacing, and line breaks.

                    {company1_name} Quotes:
                    {c1_quotes_str}

                    {company2_name} Quotes:
                    {c2_quotes_str}

                    Similarity Score: {similarity_score:.2f}
                    Similarity Reasoning: {reasoning}

                    Create a detailed comparison that includes:
                    1. A summary of each company's approach (using the quotes exactly as provided)
                    2. Key differences and similarities with direct references to the quotes (include page numbers)
                    3. Strategic implications of their differing approaches


                    Format your analysis in markdown with these sections:
                    # {theme} Comparison: {company1_name} vs {company2_name}

                    ## Similarity Score: {similarity_score:.2f}
                    [Include your reasoning here without altering any provided quotes.]

                    ## {company1_name} Approach
                    [Summarize their approach with direct, unaltered quote references. Return every Quotes as well]

                    ## {company2_name} Approach
                    [Summarize their approach with direct, unaltered quote references. Return every Quotes as well]

                    ## Key Differences
                    [Highlight main differences, referring directly to the exact quotes.]

                    ## Strategic Implications
                    [Analyze the strategic implications while ensuring all quotes remain verbatim.]

                    **Validation Checks**
                    ✓ All {len(company1_quotes)+len(company2_quotes)} quotes maintained verbatim
                    ✓ Page numbers preserved with each reference
                    ✓ No markdown in quoted material
                    ✓ ASCII-compatible formatting
            """

            # Call API with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # response = self.client.chat.completions.create(
                    #     model=self.model,
                    #     messages=[
                    #         {"role": "system", "content": "You are a financial analyst specializing in comparing companies. Your analyses are evidence-based, citing specific quotes and data points."},
                    #         {"role": "user", "content": prompt}
                    #     ],
                    #     temperature=0.4,
                    #     max_tokens=1200
                    # )

                    # analysis = response.choices[0].message.content
                    # return analysis
                    analysis = self._call_lyzr_api(
                        agent_id="67c86b15be1fc2af4eb4027e",
                        session_id="67c86b15be1fc2af4eb4027e",
                        message="You are a financial analyst specializing in comparing companies.Your analyses are evidence-based, citing specific quotes and data points." + prompt
                    )

                    return analysis
                
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"API error, retrying LLMTHEME ({attempt+1}/{max_retries}): {e}")
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        print(f"API error after {max_retries} attempts, falling back to basic analysis: {e}")
                        return self.generate_basic_comparison(theme, company1_name, company2_name,
                                                           company1_quotes, company2_quotes,
                                                           similarity_score, reasoning)
        except Exception as e:
            print(f"Error in LLM theme analysis: {e}")
            return self.generate_basic_comparison(theme, company1_name, company2_name,
                                               company1_quotes, company2_quotes,
                                               similarity_score, reasoning)

    def generate_basic_comparison(self, theme, company1_name, company2_name,
                                company1_quotes, company2_quotes, similarity_score, reasoning):
        """Generate basic markdown text comparing the two companies (fallback method)"""

        comparison_text = f"# {theme} Comparison: {company1_name} vs {company2_name}\n\n"
        comparison_text += f"## Similarity Score: {similarity_score:.2f}\n"
        comparison_text += f"{reasoning}\n\n"

        # Add company 1 section
        comparison_text += f"## {company1_name} Approach\n"
        if company1_quotes:
            for quote in company1_quotes:
                comparison_text += f"- \"{quote['quote']}\"\n"
        else:
            comparison_text += f"No quotes found from {company1_name} on this theme.\n"

        comparison_text += "\n"

        # Add company 2 section
        comparison_text += f"## {company2_name} Approach\n"
        if company2_quotes:
            for quote in company2_quotes:
                comparison_text += f"- \"{quote['quote']}\"\n"
        else:
            comparison_text += f"No quotes found from {company2_name} on this theme.\n"

        comparison_text += "\n"

        # Add key differences section
        comparison_text += "## Key Differences\n"
        if not company1_quotes and not company2_quotes:
            comparison_text += "Neither company discussed this theme.\n"
        elif not company1_quotes:
            comparison_text += f"Only {company2_name} discussed this theme.\n"
        elif not company2_quotes:
            comparison_text += f"Only {company1_name} discussed this theme.\n"
        else:
            comparison_text += "Key differences in approach:\n"
            comparison_text += f"- {company1_name} mentioned this theme {len(company1_quotes)} times\n"
            comparison_text += f"- {company2_name} mentioned this theme {len(company2_quotes)} times\n"

        comparison_text += "\n## Strategic Implications\n"
        comparison_text += "Further analysis required to determine strategic implications.\n"

        return comparison_text

    def compare_companies(self, company1_data, company2_data):
        """Compare companies based on their transcript data"""
        company1_name = company1_data["company_name"]
        company2_name = company2_data["company_name"]

        print(f"Comparing {company1_name} with {company2_name}")

        # Find all unique themes
        # all_themes = set(company1_data["themes_data"].keys()).union(
        #     set(company2_data["themes_data"].keys())
        # )
        all_themes = self.themes

        comparisons = {}
        summary_data = []

        # Analyze each theme
        for theme in all_themes:
            print(f"Analyzing theme: {theme}")
            company1_quotes = company1_data["themes_data"].get(theme, [])
            company2_quotes = company2_data["themes_data"].get(theme, [])
            print("DEBUG 1", company1_quotes)
            # Calculate similarity score (using LLM if available)
            if self.use_llm:
                similarity_score, reasoning = self.llm_similarity_analysis(
                    theme, company1_name, company2_name,
                    company1_quotes, company2_quotes
                )
            else:
                similarity_score, reasoning = self.calculate_similarity_score(
                    company1_quotes, company2_quotes
                )

            # Generate comparison analysis (using LLM if available)
            comparison = self.llm_theme_analysis(
                theme, company1_name, company2_name,
                company1_quotes, company2_quotes,
                similarity_score, reasoning
            )

            comparisons[theme] = comparison

            # Add to summary data
            summary_data.append({
                "Theme": theme,
                "Similarity Score": similarity_score,
                f"{company1_name} Mentions": len(company1_quotes),
                f"{company2_name} Mentions": len(company2_quotes),
                "Reasoning": reasoning
            })

        # Create summary dataframe
        summary_df = pd.DataFrame(summary_data)

        return {
            "comparisons": comparisons,
            "summary_df": summary_df
        }

    def create_visualizations(self, summary_df, company1_name, company2_name, output_dir="."):
        """Create visualizations comparing the companies"""

        # Sort by similarity score
        sorted_df = summary_df.sort_values("Similarity Score", ascending=False)

        # 1. Create similarity score visualization
        plt.figure(figsize=(12, 10))
        ax = sns.barplot(
            x="Similarity Score",
            y="Theme",
            data=sorted_df,
            palette="viridis"
        )
        plt.title(f"Theme Similarity: {company1_name} vs {company2_name}", fontsize=16)
        plt.xlabel("Similarity Score (0 = Different, 1 = Similar)", fontsize=12)
        plt.ylabel("Theme", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "theme_similarity_scores.png"), dpi=300)

        # 2. Create mentions comparison visualization
        plt.figure(figsize=(14, 10))

        # Create data for grouped bar chart
        mentions_data = []
        for _, row in sorted_df.iterrows():
            mentions_data.append({
                "Theme": row["Theme"],
                "Company": company1_name,
                "Mentions": row[f"{company1_name} Mentions"]
            })
            mentions_data.append({
                "Theme": row["Theme"],
                "Company": company2_name,
                "Mentions": row[f"{company2_name} Mentions"]
            })

        mentions_df = pd.DataFrame(mentions_data)

        # Sort by themes in similarity order
        theme_order = sorted_df["Theme"].tolist()
        mentions_df["Theme"] = pd.Categorical(mentions_df["Theme"], categories=theme_order, ordered=True)

        # Create the grouped bar chart
        plt.figure(figsize=(14, 10))
        sns.barplot(
            x="Mentions",
            y="Theme",
            hue="Company",
            data=mentions_df.sort_values("Theme"),
            palette=["#1f77b4", "#ff7f0e"]
        )

        plt.title(f"Theme Mentions: {company1_name} vs {company2_name}", fontsize=16)
        plt.xlabel("Number of Mentions", fontsize=12)
        plt.ylabel("Theme", fontsize=12)
        plt.legend(title="Company")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "theme_mentions_comparison.png"), dpi=300)

        return sorted_df

def run_comparison(accenture_file, wipro_file, api_key=None, model="gpt-3.5-turbo"):
    """Run the comparison between Accenture and Wipro with optional LLM enhancement"""
    comparator = LLMThemeComparator(api_key=api_key, model=model)

    # Load transcript data
    print("Loading Accenture transcript data...")
    accenture_data = comparator.load_transcript_data(accenture_file)

    print("Loading Wipro transcript data...")
    wipro_data = comparator.load_transcript_data(wipro_file)

    # Create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"theme_comparisons_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Run comparison
    print("Comparing companies...")
    results = comparator.compare_companies(accenture_data, wipro_data)

    # Create visualizations
    print("Creating visualizations...")
    sorted_df = comparator.create_visualizations(
        results["summary_df"],
        accenture_data["company_name"],
        wipro_data["company_name"],
        output_dir
    )

    # Save summary to CSV
    summary_file = os.path.join(output_dir, "theme_comparison_summary.csv")
    sorted_df.to_csv(summary_file, index=False)
    print(f"Saved summary to: {summary_file}")

    # Save all comparisons to files
    for theme, comparison in results["comparisons"].items():
        theme_file = os.path.join(output_dir, f"{theme.replace('/', '_')}_comparison.md")
        with open(theme_file, 'w', encoding='utf-8') as f:
            f.write(comparison)
        print(f"Saved comparison for {theme} to: {theme_file}")

    # Create combined file
    combined_file = os.path.join(output_dir, "all_comparisons.md")
    with open(combined_file, 'w', encoding='utf-8') as f:
        f.write(f"# Theme Comparisons: {accenture_data['company_name']} vs {wipro_data['company_name']}\n\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for theme in sorted_df["Theme"]:
            if theme in results["comparisons"]:
                f.write(results["comparisons"][theme])
                f.write("\n\n---\n\n")

    print(f"Saved all comparisons to: {combined_file}")

    return {
        "output_dir": output_dir,
        "results": results,
        "accenture_data": accenture_data,
        "wipro_data": wipro_data
    }
