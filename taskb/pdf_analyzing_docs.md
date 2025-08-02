# Guide: Processing PDFs and Getting Structured Output with the Gemini API

This guide focuses on the end-to-end workflow for sending PDF documents to the Gemini API. You will learn how to upload PDFs and, most importantly, how to retrieve the information as either plain text or, more powerfully, as structured and reliable JSON.

## Part 1: How to Send a PDF to the API

There are two primary methods for providing a PDF to the Gemini API, depending on the file's size.

### Method 1: Inline PDF Data (For files < 20MB)

This is the most direct method for smaller files. You read the PDF's bytes and include them directly in your API request.

**Example: Sending a local PDF file**

```python
import pathlib
from google import genai
from google.genai import types

# NOTE: Configure your API key
# genai.configure(api_key="YOUR_API_KEY")

client = genai.Client()

# 1. Define the path to your local PDF
pdf_file_path = "path/to/your/document.pdf"

# 2. Create the Part object for the API
try:
    pdf_part = types.Part.from_bytes(
        data=pathlib.Path(pdf_file_path).read_bytes(),
        mime_type="application/pdf"  # Critical: Specify the MIME type
    )
except FileNotFoundError:
    print(f"Error: The file '{pdf_file_path}' was not found.")
    exit()

# The 'pdf_part' is now ready to be used in a generate_content call.
```

### Method 2: Using the File API (For files > 20MB)

For larger documents, you must first upload the file using the File API. This API stores the file temporarily (for 48 hours) and gives you a file reference to use in your prompts.

**Example: Uploading and using a large PDF**
```python
import pathlib
from google import genai

client = genai.Client()

# 1. Define the path to your large PDF
large_pdf_path = "path/to/your/large_document.pdf"

# 2. Upload the file using the File API
print("Uploading file...")
uploaded_file = client.files.upload(
    path=large_pdf_path,
    display_name="My Large Document"
)
print(f"Completed upload: {uploaded_file.uri}")

# The 'uploaded_file' object is now ready to be used in a generate_content call.
```

---

## Part 2: How to Get Output from the PDF

Once you have your PDF ready as either an inline `Part` or an `uploaded_file` object, you can request information from it.

### Getting Simple Text Output

This is the most basic use case, suitable for tasks like summarization.

```python
# Assuming 'pdf_part' was created using Method 1
# Or 'uploaded_file' was created using Method 2

# For an inline part:
contents = [pdf_part, "Summarize this document in three bullet points."]

# For a File API object:
# contents = [uploaded_file, "Summarize this document in three bullet points."]

response = client.models.generate_content(
    model="gemini-2.5-flash",  # Or another capable model
    contents=contents
)

print(response.text)
```

### Getting Structured JSON Output (Recommended)

For reliable data extraction (e.g., from invoices, forms, reports), forcing the model to output JSON is the best practice. This is done by configuring a **response schema**.

The easiest way to define a schema in Python is with **Pydantic**.

#### Step-by-Step Example: Extracting Invoice Data into JSON

**Step 1: Define Your Desired JSON Structure with Pydantic**
This class acts as a template for the JSON you want.

```python
from pydantic import BaseModel, Field
from typing import List

class LineItem(BaseModel):
    description: str = Field(description="Description of the product or service.")
    quantity: int
    unit_price: float

class Invoice(BaseModel):
    invoice_id: str = Field(description="The unique identifier for the invoice.")
    vendor_name: str = Field(description="The name of the company that issued the invoice.")
    issue_date: str = Field(description="The date the invoice was issued, in YYYY-MM-DD format.")
    total_amount: float = Field(description="The total amount due.")
    line_items: List[LineItem]
```

**Step 2: Configure the API Call for JSON Output**
In the `generation_config`, set the `response_mime_type` to `"application/json"` and provide your Pydantic model to `response_schema`.

```python
import json

# Assuming 'pdf_part' is the Part object for your invoice PDF
prompt = "Analyze the provided invoice PDF and extract all relevant information."
contents = [pdf_part, prompt]

# Configure the model to return JSON matching the 'Invoice' schema
generation_config = genai.types.GenerationConfig(
    response_mime_type="application/json",
    response_schema=Invoice,
)

model = client.models.get_model("gemini-2.5-flash")
response = model.generate_content(
    contents,
    generation_config=generation_config
)
```

**Step 3: Access the Structured Data**
The SDK provides two convenient ways to get the structured result:

1.  `response.text`: The raw JSON string.
2.  `response.parsed`: **The instantiated Pydantic object.** This is incredibly useful as it gives you a native Python object with type-checked fields.

```python
# Print the raw JSON string
print("--- Raw JSON Output ---")
print(response.text)

# Access the data using the parsed Pydantic object
print("\n--- Parsed Python Object ---")
try:
    parsed_invoice = response.parsed
    print(f"Invoice ID: {parsed_invoice.invoice_id}")
    print(f"Vendor: {parsed_invoice.vendor_name}")
    print(f"Total Amount: ${parsed_invoice.total_amount:.2f}")

    print("\nLine Items:")
    for item in parsed_invoice.line_items:
        print(f"- {item.description} (Qty: {item.quantity}, Unit Price: ${item.unit_price:.2f})")
except AttributeError:
    print("Could not parse the response into the 'Invoice' object.")
    print("Check the model's raw response for errors.")
```