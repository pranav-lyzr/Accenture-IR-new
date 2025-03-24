import streamlit as st
import os
import json
import requests
import hashlib
import asyncio
import pandas as pd
from pathlib import Path
from datetime import datetime
import unicodedata
from pyzerox import zerox
from collections import defaultdict
from typing import Dict, List
from LLMThemeComparator import LLMThemeComparator
from fpdf import FPDF
import tempfile

# Configuration
UPLOAD_FOLDER = "out"
CATEGORIES = [
    "Macro environment", "Pricing", "Margins", "Bookings/Large Deals",
    "Discretionary/Small Deals", "People", "Cloud", "Security", "Gen AI",
    "M&A", "Investments", "Partnerships", "Technology Budget",
    "Product/IP/Assets", "Talent/Training", "Clients", "Awards/Recognition"
]
API_KEY = "AIzaSyAdPWuOww_Vdf2ThzIyK9Wc9bhaRMvlCvI"
MODEL = "gemini/gemini-1.5-pro"
os.environ['GEMINI_API_KEY'] = "AIzaSyAdPWuOww_Vdf2ThzIyK9Wc9bhaRMvlCvI"


def set_custom_layout():
    st.markdown("""
        <style>
            <style>
            .reportview-container .main .block-container{
                padding-left: 40px;
                padding-right: 40px;
                max-width: 100% !important;
            }
            h1, h2, h3, h4, h5, h6 {
                font-size: 1.2rem !important;
            }
            .st-emotion-cache-mtjnbi {
                width: 100%;
                padding: 6rem 5rem 10rem;
                max-width: 2000px;
            }
            </style>
        </style>
    """, unsafe_allow_html=True)


class PDFAnalyzer:
    def __init__(self):
        self.text_agent_id = "67c86b15be1fc2af4eb4027e"
        self.api_key = "sk-default-PPcvzcCe4cJRRP8JkEXnT51woYJUXzMZ"
        self.base_url = "https://agent-dev.test.studio.lyzr.ai/v3/inference/chat/"
        self.user_id = "pranav@lyzr.ai"
        self.headers = {"Content-Type": "application/json", "x-api-key": self.api_key}

    async def _call_lyzr_api(self, agent_id: str, session_id: str, message: str) -> str:
        payload = {"user_id": self.user_id, "agent_id": agent_id, "session_id": session_id, "message": message}
        try:
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json().get("response", "No response received.")
            print(f"API response received: {result[:50]}...")
            return result
        except requests.exceptions.RequestException as e:
            print(f"Error calling Lyzr API: {str(e)}")
            return f"Error calling Lyzr API: {str(e)}"

    async def analyze_page(self, page_num: int, text: str) -> dict:
        print(f"\n=== Analyzing Page {page_num} ===")
        print(f"Content preview: {text}...")
        
        if not text.strip():
            print("⚠️ Empty page content")
            return {
                "page": page_num,
                "warning": "Empty page content",
                "categories": {}
            }
            
        prompt = f"""ANALYZE THIS FINANCIAL DOCUMENT PAGE (PAGE {page_num}):
        --- RULES ---
        1. Extract exact VERBATIM FULL SENTENCES containing NUMBERS, PERCENTAGES, or FINANCIAL METRICS or based on the specific categories defined below.
        2. For tabular data, DO NOT extract individual cells. Instead, synthesize the data into meaningful statements.
        3. Categorize each VERBATIM FULL SENTENCES EXACTLY under one of these themes: {json.dumps(CATEGORIES)}
        4. Format MUST BE VALID JSON with 'company_name' and 'categories'
        5. Each category must contain 'verbatim_quotes' array with:
           - 'quote': EXACT TEXT (preserve symbols/formatting)
           - 'page': {page_num}
           - 'type': 'text'
        6. Include ALL relevant quotes - DO NOT MISS ANY DATA POINTS
        7. DO NOT extract individual random numbers, percentages, or table cells without proper context.
        8. For tables, COMBINE related data into COMPLETE, MEANINGFUL STATEMENTS.
        9. IMPORTANT: For each category, apply these specific filters:

        --- SPECIAL HANDLING FOR TABLES AND CHARTS ---
        * When extracting data from tables, charts, or graphs:
        - Transform raw data into meaningful sentences
        - Include time periods or dates when visible
        - Describe trends (increase/decrease) and provide context
        - Interpret symbols (↑ means increase, ↓ means decrease)
        - Format as: "METRIC_NAME [increased/decreased] to VALUE in [PERIOD], compared to VALUE in [PRIOR_PERIOD]"

        * When you encounter tables, create MEANINGFUL STATEMENTS by:
        - Identifying the metric/KPI being presented
        - Noting the current period value and compare it to prior periods
        - Describing trends (increasing, decreasing, stable)
        - Example: "Headcount decreased to 344,400 employees in Q1 2024, down from 347,700 in Q4 2023 and 351,500 in Q1 2023."
        - Example: "Voluntary attrition in Tech Services improved to 13.1% in Q1 2024, continuing a downward trend from 23.1% in Q1 2023."
        - IMPORTANT: Include ALL PERIODS shown in the table in your analysis

        * For financial tables showing multiple periods:
        - Include year-over-year comparisons when available
        - Include quarter-over-quarter trends
        - Identify highest and lowest values in the series

        * For charts/graphs:
        - Describe what the visualization shows
        - Capture key trends, inflection points, and outliers
        - State the overall message the chart conveys

        * For percentages and metrics:
        - Always provide context about what the percentage represents
        - Always include the time period the metric applies to
        - Always note significant changes from previous periods

        --- TABLE HANDLING INSTRUCTIONS ---
        * For ANY table data, DO NOT extract:
        - Individual cells
        - Column headers without values
        - Values without context
        - Isolated metrics that make no sense on their own

        * INSTEAD, for tables:
        - Identify the complete measure and its value as a single statement
        - Include the time period (quarter/year) in the same statement
        - Always pair metrics with their values and necessary context
        - GOOD EXAMPLE: "GAAP Operating Margin was 14.6% in Q1 2024"
        - GOOD EXAMPLE: "Total Employees were 344,400 in Q1 2024, representing a decrease of 3,300 quarter-over-quarter and 7,100 year-over-year"
        - BAD EXAMPLE: "14.6%" (number without context)
        - BAD EXAMPLE: "GAAP Operating Margin" (header without value)

        * For financial metrics:
        - Combine metric name + value + period in ONE complete statement
        - For example: "Cash Flow From Operations was $95M in Q1 2024"
        - For example: "Free Cash Flow was $16M in Q1 2024"
        - NOT "Cash Flow From Operations" and "$95M" as separate entries

        * For comparative data:
        - Include the comparison in the same statement
        - For example: "Adjusted Operating Margin was 15.1% compared to GAAP Operating Margin of 14.6% in Q1 2024"

        --- EXAMPLE OF CORRECT EXTRACTION ---
        INSTEAD OF:
        "GAAP Operating Margin"
        "14.6%"
        "Adjusted Operating Margin"
        "15.1%"

        USE:
        "GAAP Operating Margin was 14.6% in Q1 2024"
        "Adjusted Operating Margin was 15.1% in Q1 2024"


        --- CATEGORY DEFINITIONS AND FILTERS ---
        * "Macro environment": ONLY include quotes that explicitly discuss:
          - Economic conditions, market trends, or industry-wide factors
          - GDP, inflation, interest rates, or global economic indicators
          - Descriptions of client spending patterns driven by economic factors
          - Statements about industry or broad market conditions
          - DO NOT include generic references to "the environment" or mere mentions of discussions about the market
          - Must contain specific observations about economic conditions or market trends
        
        * "Pricing": ONLY include quotes with:
          - Specific numbers, percentages, or trends related to pricing strategies
          - Pricing changes, adjustments, or future outlook
          - Price elasticity, competitive pricing, or pricing pressures
          - Customer reactions to pricing or price negotiations
          - Must contain actual figures, percentages, or clear statements about pricing direction

        * "Margins": ONLY include quotes with:
          - Specific numbers, percentages, or trends related to profit margins
          - Gross margins, operating margins, or net margins with figures
          - Margin expansion or compression with measurable data
          - Factors specifically impacting margins with quantifiable effects
          - Must contain actual margin metrics or clear directional statements

        * "Bookings/Large Deals": ONLY include quotes with:
          - Specific numbers or percentages about large deals
          - Descriptions of bookings valued at $100M+ or described as "large"
          - Metrics related to signed contracts or pipeline for significant deals
          - Trends in large deal closures or conversions
          - Must contain specific booking metrics or clear statements about large deal activity
        
        * "Discretionary/Small Deals": ONLY include quotes with:
          - Explicit mention of smaller projects or discretionary spending
          - Metrics related to smaller deals or short-term engagements
          - References to clients limiting or changing discretionary spending
          - Trends in smaller deal volume or conversion rates
          - Must contain actual metrics or clear statements about discretionary spending patterns
        
        * "People": ONLY include quotes with:
          - Employee metrics like headcount, utilization, attrition, or hiring
          - Workforce planning, restructuring, or organizational changes with specific numbers
          - Employee satisfaction scores, engagement metrics, or diversity statistics
          - Compensation changes, benefit adjustments, or retention strategies with data
          - Must contain specific workforce metrics or quantifiable people initiatives

        * "Cloud": ONLY include quotes with:
          - Cloud revenue, growth rates, or specific cloud project metrics
          - Cloud adoption rates, migration statistics, or implementation metrics
          - Cloud partnerships or strategic cloud initiatives with measurable outcomes
          - Cloud capabilities, offerings, or solutions with specific client impact
          - Must contain specific cloud-related metrics or quantifiable cloud business activities

        * "Security": ONLY include quotes with:
          - Security revenue, growth rates, or specific security project metrics
          - Security investments, acquisitions, or strategic security initiatives with figures
          - Security offerings, capabilities, or solutions with specific client impact
          - Security threats, incidents, or risk management with quantifiable data
          - Must contain specific security-related metrics or quantifiable security business activities

        * "Gen AI": ONLY include quotes with:
          - Specific metrics about Generative AI projects, revenue, or investments
          - New GenAI offerings, solutions, or capabilities with measurable outcomes
          - GenAI adoption rates or implementation metrics
          - GenAI-related partnerships or market positioning with quantifiable impact
          - Must contain specific AI-related metrics or quantifiable GenAI business activities

        * "M&A": ONLY include quotes with:
          - Specific acquisition details including deal size, number, or timing
          - Post-acquisition integration metrics or performance data
          - M&A strategy statements with specific targets or criteria
          - Divestiture metrics or portfolio rationalization with figures
          - Must contain specific M&A-related metrics or quantifiable transaction details

        * "Investments": ONLY include quotes with:
          - Capital expenditure figures, investment amounts, or allocation changes
          - R&D spending, innovation investments, or venture funding with specific numbers
          - Return on investment metrics, payback periods, or investment performance data
          - Investment priorities or strategy changes with quantifiable targets
          - Must contain specific investment amounts or clear quantifiable investment activities

        * "Partnerships": ONLY include quotes with:
          - Partnership revenue, contribution, or growth metrics
          - Alliance strategy statements with specific goals or targets
          - Joint venture performance data or ecosystem expansion metrics
          - Partnership portfolio composition or changes with figures
          - Must contain specific partnership-related metrics or quantifiable alliance activities

        * "Technology Budget": ONLY include quotes with:
          - Client or internal technology spending figures, trends, or forecasts
          - IT budget allocations, changes, or priorities with specific numbers
          - Technology investment ratios, comparisons, or benchmarks
          - Budget shifts between traditional and new technologies with figures
          - Must contain specific budget numbers or clear quantifiable technology spending statements

        * "Product/IP/Assets": ONLY include quotes with:
          - Product portfolio performance, growth rates, or contribution metrics
          - IP development, patents, or proprietary solution metrics
          - Asset utilization, valuation, or monetization with specific figures
          - Product launch impacts, adoption rates, or market penetration data
          - Must contain specific product/IP metrics or quantifiable asset-related statements

        * "Talent/Training": ONLY include quotes with:
          - Talent acquisition, development, or retention metrics
          - Training investments, hours, or certification statistics
          - Skill development initiative outcomes or reskilling program metrics
          - Talent pipeline data, bench strength, or leadership development figures
          - Must contain specific talent-related metrics or quantifiable training activities

        * "Clients": ONLY include quotes with:
          - Client count, growth, retention, or satisfaction metrics
          - Customer segment performance, penetration, or concentration data
          - Client relationship tenure, expansion, or wallet share statistics
          - Industry or geography client distribution with specific figures
          - Must contain specific client-related metrics or quantifiable customer statements

        * "Awards/Recognition": ONLY include quotes with:
          - Specific awards, rankings, or recognitions received with details
          - Industry analyst positioning or ratings with specific placements
          - Client satisfaction or Net Promoter Score achievements
          - External validation of capabilities or services with specific metrics
          - Must contain specific recognition details or quantifiable external validation

        --- EXAMPLE ---
        {{
            "company_name": "Company XYZ",
            "categories": {{
                "Macro environment": {{
                    "verbatim_quotes": [
                        {{
                            "quote": "Global IT spending is projected to decline 3% year-over-year due to economic headwinds affecting client decision-making.",
                            "page": {page_num},
                            "type": "text"
                        }}
                    ]
                }}
            }}
        }}

        --- PAGE CONTENT ---
        {text}  
        
        """
        
        try:
            print("Sending to LLM...")
            response = await self._call_lyzr_api(
                message=prompt,
                agent_id=self.text_agent_id,
                session_id=self.text_agent_id
            )
            
            print("response", response)
            
            return self._parse_response(page_num, response)
        except Exception as e:
            print(f"❌ Analysis Error: {str(e)}")
            return {"page": page_num, "error": str(e)}

    def _parse_response(self, page_num: int, response_text: str) -> dict:
        print("Parsing LLM Response...")
        try:
            # Parse JSON response, removing markdown code block markers
            result = json.loads(response_text.replace('```json', '').replace('```', ''))

            # Extract and clean company name
            company_name = result.get('company_name', '')
            if company_name.lower() == 'company xyz':  # Remove default placeholder
                company_name = ''

            # Validate top-level structure
            if not isinstance(result, dict):
                raise ValueError("Response is not a JSON object")
            if 'categories' not in result:
                raise ValueError("Missing 'categories' key in response")
            if not isinstance(result['categories'], dict):
                raise ValueError(f"'categories' should be a dictionary, got {type(result['categories'])}")

            # Process categories
            validated_categories = {}
            for category in CATEGORIES:
                category_data = result['categories'].get(category, {})
                if not isinstance(category_data, dict):
                    print(f"Warning: Category '{category}' data is not a dictionary: {category_data}")
                    validated_categories[category] = {"verbatim_quotes": []}
                    continue

                verbatim_quotes = category_data.get('verbatim_quotes', [])
                if not isinstance(verbatim_quotes, list):
                    print(f"Warning: 'verbatim_quotes' for category '{category}' is not a list: {verbatim_quotes}")
                    validated_categories[category] = {"verbatim_quotes": []}
                    continue

                validated_quotes = []
                for quote in verbatim_quotes:
                    if isinstance(quote, dict) and 'quote' in quote:
                        validated_quotes.append({
                            "quote": str(quote['quote']).strip(),
                            "page": page_num,
                            "type": "text"
                        })
                    else:
                        print(f"Warning: Invalid quote format in category '{category}': {quote}")

                validated_categories[category] = {
                    "verbatim_quotes": validated_quotes
                }

            return {
                "page_number": page_num,
                "company_name": company_name,
                "categories": validated_categories
            }

        except json.JSONDecodeError as e:
            print(f"❌ JSON Parse Error: {str(e)}")
            return {
                "page": page_num,
                "error": f"JSON Error: {str(e)}",
                "raw_response": response_text
            }
        except Exception as e:
            print(f"❌ General Parse Error: {str(e)}")
            return {
                "page": page_num,
                "error": str(e),
                "raw_response": response_text
            }

async def process_pdf(file_content: bytes, company: str, year: str, quarter: str, doc_type: str) -> dict:
    print(f"\n🔍 Starting PDF Processing: {company} {doc_type} {year} {quarter}")

    analysis_path = get_analysis_path(file_content, company, year, quarter, doc_type)
    if analysis_path.exists():
        print(f"📂 Loading existing analysis from {analysis_path}")
        with open(analysis_path, "r") as f:
            existing_analysis = json.load(f)
        return {
            "themes": existing_analysis,
            "from_cache": True  # Add cache flag
        }
    
    file_hash = get_file_hash(file_content)
    base_name = f"{company}_{doc_type}_{year}_{quarter}_{file_hash}"
    output_dir = Path(UPLOAD_FOLDER) / company / year / quarter / doc_type
    
    print(f"📂 Creating directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_path = output_dir / f"{base_name}.pdf"
    print(f"💾 Saving PDF to: {pdf_path}")
    
    with open(pdf_path, "wb") as f:
        f.write(file_content)
    
    analyzer = PDFAnalyzer()
    
    try:
        print("🔧 Extracting PDF content with Zerox...")
        result = await zerox(file_path=str(pdf_path), model=MODEL)
        
        print("\n📄 Zerox Output Summary:")
        print(f"File Name: {result.file_name}")
        print(f"Processing Time: {result.completion_time}ms")
        print(f"Pages Found: {len(result.pages)}")
        print(f"Input Tokens: {result.input_tokens}")
        print(f"Output Tokens: {result.output_tokens}")
        
        full_analysis = {"pages": [], "company": "", "themes": defaultdict(list)}
        
        for page in result.pages:
            print(f"\n📖 Processing Page {page.page}/{len(result.pages)}")
            print(f"Content Length: {page.content_length} characters")
            
            if not page.content.strip():
                print("⏩ Skipping empty page")
                continue
                
            page_analysis = await analyzer.analyze_page(page.page, page.content)
            full_analysis["pages"].append(page_analysis)
            
            if page_analysis.get("company_name"):
                full_analysis["company"] = page_analysis["company_name"]
                
            if "categories" in page_analysis:
                print("🏷️ Found Categories:")
                for category, data in page_analysis["categories"].items():
                    print(f" - {category}: {len(data.get('verbatim_quotes', []))} quotes")
                    # Ensure `file` is defined in this context
                    for quote in data.get("verbatim_quotes", []):  # Iterate over each quote
                        full_analysis["themes"][category].append({
                            **quote,
                            "source_file": pdf_path.name, 
                            "doc_type": doc_type.capitalize(),
                            "year": year,
                            "quarter": quarter
                        })
        
        analysis_path = output_dir / "analysis" / f"{base_name}_analysis.json"
        print(f"\n💾 Saving analysis to: {analysis_path}")
        analysis_path.parent.mkdir(exist_ok=True)
        with open(analysis_path, "w") as f:
            json.dump(full_analysis["themes"], f, indent=2)
        company_names = [p.get("company_name", "") for p in full_analysis["pages"]]
        valid_names = [name for name in company_names if name.strip()]
        if valid_names:
            full_analysis["detected_company"] = max(set(valid_names), key=valid_names.count)
        else:
            full_analysis["detected_company"] = company
        return full_analysis
        
    except Exception as e:
        print(f"❌ Fatal Processing Error: {str(e)}")
        return {"error": str(e)}


def get_file_hash(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()


def get_analysis_path(file_content: bytes, company: str, year: str, quarter: str, doc_type: str) -> Path:
    file_hash = get_file_hash(file_content)
    # Match exact format used when saving
    base_name = f"{company}_{doc_type}_{year}_{quarter}_{file_hash}"
    return Path(UPLOAD_FOLDER) / company / year / quarter / doc_type / "analysis" / f"{base_name}_analysis.json"

# def generate_comparison_pdf(comparisons: dict, accenture_name: str, other_name: str) -> bytes:
#     html = f"""
#     <html>
#         <head><title>Comparison Report</title></head>
#         <body>
#             <h1>Financial Analysis Report</h1>
#             <h2>{accenture_name} vs {other_name}</h2>
#             {''.join([f"<h3>{theme}</h3><div>{content}</div>" for theme, content in comparisons.items()])}
#         </body>
#     </html>
#     """
#     return pdfkit.from_string(html, False)

def generate_comparison_pdf(comparisons: dict, accenture_name: str, other_name: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    
    # Configure fonts with fallback
    try:
        pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
        pdf.add_font('DejaVu', 'B', 'DejaVuSans-Bold.ttf', uni=True)
        font_family = 'DejaVu'
    except RuntimeError:
        font_family = 'Helvetica'
    
    # Title Section
    pdf.set_font(font_family, 'B', 16)
    pdf.cell(200, 10, txt="Financial Analysis Report", ln=1, align='C')
    
    pdf.set_font(font_family, '', 14)
    pdf.cell(200, 10, txt=f"{accenture_name} vs {other_name}", ln=1, align='C')
    pdf.ln(15)

    # Process Comparisons
    for theme, content in comparisons.items():
        # Parse markdown formatting
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            # Handle headers
            if line.startswith('# '):
                pdf.set_font(font_family, 'B', 14)
                pdf.cell(0, 10, txt=line[2:].strip(), ln=1)
                pdf.ln(3)
            elif line.startswith('## '):
                pdf.set_font(font_family, 'B', 12)
                pdf.cell(0, 10, txt=line[3:].strip(), ln=1)
                pdf.ln(2)
            else:
                # Handle content with Unicode characters
                clean_line = unicodedata.normalize('NFKD', line)
                clean_line = clean_line.encode('latin-1', 'replace').decode('latin-1')
                pdf.set_font(font_family, '', 11)
                pdf.multi_cell(0, 8, txt=clean_line)
                pdf.ln(2)
        
        pdf.ln(10)

    # Watermark
    pdf.set_font(font_family, '', 8)
    pdf.set_text_color(200, 200, 200)
    pdf.text(10, 290, "Generated by Financial Document Analyzer")
    
    # Output
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        pdf.output(tmp.name)
        return open(tmp.name, "rb").read()

def main():
    set_custom_layout()
    st.title("Financial Document Analyzer")
    st.write("Upload PDF documents for analysis and theme categorization")
    
    # Initialize session state
    if 'accenture_themes' not in st.session_state:
        st.session_state.accenture_themes = defaultdict(list)
    if 'other_themes' not in st.session_state:
        st.session_state.other_themes = defaultdict(list)
    if 'other_company_name' not in st.session_state:
        st.session_state.other_company_name = "Other Company"
    if 'report_generated' not in st.session_state:
        st.session_state.report_generated = False
    
    # UI layout with two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Accenture Documents")
        with st.expander("📄 Upload Transcripts"):
            acc_transcripts = st.file_uploader("Accenture Transcripts", type=["pdf"], 
                                               accept_multiple_files=True, key="acc_trans")
        with st.expander("📊 Upload Factsheets"):
            acc_factsheets = st.file_uploader("Accenture Factsheets", type=["pdf"],
                                              accept_multiple_files=True, key="acc_facts")
        with st.expander("💰 Upload Financials"):
            acc_financials = st.file_uploader("Accenture Financials", type=["pdf"],
                                              accept_multiple_files=True, key="acc_fin")
    
    with col2:
        st.subheader("Other Company Documents")
        with st.expander("📄 Upload Transcripts"):
            other_transcripts = st.file_uploader("Other Transcripts", type=["pdf"],
                                                 accept_multiple_files=True, key="oth_trans")
        with st.expander("📊 Upload Factsheets"):
            other_factsheets = st.file_uploader("Other Factsheets", type=["pdf"],
                                                accept_multiple_files=True, key="oth_facts")
        with st.expander("💰 Upload Financials"):
            other_financials = st.file_uploader("Other Financials", type=["pdf"],
                                                accept_multiple_files=True, key="oth_fin")
    
    # Date selection
    selected_year = st.selectbox("Year", [str(y) for y in range(2018, datetime.now().year + 1)])
    selected_quarter = st.selectbox("Quarter", ["Q1", "Q2", "Q3", "Q4"])
    
    if st.button("🚀 Process Documents"):
        all_files = {
            "Accenture": {
                "Transcripts": acc_transcripts,
                "Factsheets": acc_factsheets,
                "Financials": acc_financials
            },
            "Other Company": {
                "Transcripts": other_transcripts,
                "Factsheets": other_factsheets,
                "Financials": other_financials
            }
        }
        
        # File processing progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_files = sum(len(files) for company in all_files.values() 
                          for doc_type, files in company.items() if files)
        processed = 0
        
        for company, doc_types in all_files.items():
            for doc_type, files in doc_types.items():
                if files:
                    for file in files:
                        try:
                            status_text.text(f"Processing {company} {doc_type}: {file.name}")
                            content = file.getvalue()
                            
                            analysis = asyncio.run(
                                process_pdf(
                                    content,
                                    company,
                                    selected_year,
                                    selected_quarter,
                                    doc_type
                                )
                            )
                            
                            processed += 1
                            progress_bar.progress(processed / total_files)
                            
                            st.success(f"Processed {file.name}")
                            st.json(analysis["themes"], expanded=False)

                            if company == "Accenture":
                                for theme, quotes in analysis["themes"].items():
                                    st.session_state.accenture_themes[theme].extend([{
                                        "quote": q["quote"],
                                        "page": q["page"],
                                        "source_file": q.get("source_file", "Unknown File"),
                                        "doc_type": q.get("doc_type", "Document")
                                    } for q in quotes])
                            else:
                                comparator = LLMThemeComparator("sk-default-PPcvzcCe4cJRRP8JkEXnT51woYJUXzMZ")
                                identified_name = comparator.identify_company_name(analysis["themes"])
                                if identified_name != "Other Company":
                                    st.session_state.other_company_name = identified_name
                                for theme, quotes in analysis["themes"].items():
                                    st.session_state.other_themes[theme].extend([{
                                        "quote": q["quote"],
                                        "page": q["page"],
                                        "source_file": q.get("source_file", "Unknown File"), 
                                        "doc_type": q.get("doc_type", "Document")
                                    } for q in quotes])

                        except Exception as e:
                            st.error(f"Error processing {file.name}: {str(e)}")
        
        progress_bar.empty()
        status_text.success("✅ All documents processed!")
        
        # Deduplicate quotes
        for themes in [st.session_state.accenture_themes, st.session_state.other_themes]:
            for theme in themes:
                seen = set()
                unique_quotes = []
                for quote in themes[theme]:
                    key = (quote["quote"], quote["page"], quote["source_file"])
                    if key not in seen:
                        seen.add(key)
                        unique_quotes.append(quote)
                themes[theme] = unique_quotes

        # Theme comparison with real-time feedback
        if st.session_state.accenture_themes and st.session_state.other_themes:
            st.subheader("Theme Comparison Analysis")
            
            # Prepare data for comparison
            company1_data = {
                "company_name": "Accenture",
                "time_period": f"{selected_year} {selected_quarter}",
                "themes_data": dict(st.session_state.accenture_themes)
            }
            company2_data = {
                "company_name": st.session_state.other_company_name,
                "time_period": f"{selected_year} {selected_quarter}",
                "themes_data": dict(st.session_state.other_themes)
            }
            
            try:
                comparator = LLMThemeComparator("sk-default-PPcvzcCe4cJRRP8JkEXnT51woYJUXzMZ")
                results = {}
                
                # Initialize UI elements for theme analysis
                st.write("Analyzing themes...")
                theme_progress_bar = st.progress(0)
                theme_status_text = st.empty()
                total_themes = len(comparator.themes)
                processed_themes = 0
                
                # Analyze each theme with feedback
                comparisons = {}
                summary_data = []
                
                for theme in comparator.themes:
                    with st.spinner(f"Analyzing theme: {theme}"):
                        theme_status_text.text(f"Processing theme: {theme}")
                        company1_quotes = company1_data["themes_data"].get(theme, [])
                        company2_quotes = company2_data["themes_data"].get(theme, [])
                        
                        # Calculate similarity score and reasoning
                        similarity_score, reasoning = comparator.llm_similarity_analysis(
                            theme, company1_data["company_name"], company2_data["company_name"],
                            company1_quotes, company2_quotes
                        )
                        
                        # Generate detailed comparison
                        comparison = comparator.llm_theme_analysis(
                            theme, company1_data["company_name"], company2_data["company_name"],
                            company1_quotes, company2_quotes, similarity_score, reasoning
                        )
                        comparisons[theme] = comparison
                        
                        # Update summary data
                        summary_data.append({
                            "Theme": theme,
                            "Similarity Score": similarity_score,
                            f"{company1_data['company_name']} Mentions": len(company1_quotes),
                            f"{company2_data['company_name']} Mentions": len(company2_quotes),
                            "Reasoning": reasoning
                        })
                        
                        # Update progress
                        processed_themes += 1
                        theme_progress_bar.progress(processed_themes / total_themes)
                
                theme_progress_bar.empty()
                theme_status_text.success("✅ Theme analysis completed!")
                
                results["comparisons"] = comparisons
                results["summary_df"] = pd.DataFrame(summary_data)
                
                if results:
                    st.session_state.report_generated = True
                    
                    # Display summary metrics
                    st.subheader("Comparison Summary")
                    st.dataframe(results["summary_df"])
                    
                    # Display detailed comparisons
                    st.subheader("Theme Comparisons")
                    for theme, analysis in results["comparisons"].items():
                        with st.expander(f"Theme: {theme}"):
                            st.markdown(analysis)
                    
                    # Generate PDF report
                    pdf_bytes = generate_comparison_pdf(
                        results["comparisons"], 
                        "Accenture",
                        st.session_state.other_company_name
                    )
                    
                    # Show download button
                    st.download_button(
                        label="📄 Download Full Report",
                        data=pdf_bytes,
                        file_name=f"comparison_report_{datetime.now().isoformat()}.pdf",
                        mime="application/pdf"
                    )
            except Exception as e:
                st.error(f"❌ Failed to compare companies: {str(e)}")
        else:
            st.warning("⚠️ Insufficient data for comparison - please upload documents for both companies")


if __name__ == "__main__":
    main()