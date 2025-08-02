from typing import List
from google import genai
from google.genai import types

from pydantic import BaseModel, Field
import pathlib
import json

file_path = pathlib.Path("D:/Downloads/code/code/Extra-Projects/4sim/4sim-gov-ai-mvp/df_final_features_dashboard.pdf")

class ChartAnalysis(BaseModel):
    chart_number: int = Field(description="The sequential number of the chart in the PDF.")
    chart_title: str = Field(description="The title of the chart as displayed in the PDF.")
    chart_type: str = Field(description="The type of chart (e.g., histogram, scatter plot, box plot, etc.).")
    detailed_description: str = Field(description="A comprehensive description of what the chart shows, including specific data points, patterns, and trends visible in the chart.")
    key_insights: List[str] = Field(description="A list of 3-5 key insights or findings that can be derived from this chart.")
    data_trends: str = Field(description="Description of the main trends, patterns, or relationships shown in the data.")
    statistical_observations: str = Field(description="Any notable statistical observations like outliers, distributions, correlations, etc.")

class PDFAnalysisReport(BaseModel):
    total_charts: int = Field(description="Total number of charts analyzed in the PDF.")
    charts: List[ChartAnalysis] = Field(description="List of detailed analysis for each chart.")
    overall_trend_summary: str = Field(description="A single comprehensive sentence describing the overall trend across all charts in the dataset.")


class Recipe(BaseModel):
    recipe_name: str
    ingredients: list[str]

client = genai.Client(api_key="YOUR_API_KEY")

prompt = "please analyze each chart in the pdf file, and write detailed descriptions of each chart, showing the trends etc. at the end write one single final sentence describing the trend"
response = client.models.generate_content(
  model="gemini-2.5-flash",
  contents=[
      types.Part.from_bytes(
        data=file_path.read_bytes(),
        mime_type='application/pdf',
      ),
      prompt],
  config={
      "response_mime_type": "application/json",
      "response_schema": PDFAnalysisReport.model_json_schema(),
  }

)

print(response.text)

# Save the JSON response to a file
output_data = json.loads(response.text)
output_filename = "pdf_analysis_output.json"
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"\nJSON output saved to: {output_filename}")

