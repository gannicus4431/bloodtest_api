from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import httpx
from google.cloud import documentai_v1 as documentai
from openai import OpenAI
from fastapi.responses import JSONResponse
import asyncio

app = FastAPI()
from dotenv import load_dotenv
load_dotenv()

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
PROCESSOR_ID = os.getenv("PROCESSOR_ID")
KEY_PATH = os.getenv("KEY_PATH")

# Ensure GOOGLE_APPLICATION_CREDENTIALS is set for Google Cloud authentication
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH

class PhotoURLRequest(BaseModel):
    photo_url: str

@app.post("/process-document/")
async def process_document(request: PhotoURLRequest):
    photo_url = request.photo_url

    print(f"Attempting to retrieve photo from URL: {photo_url}")

    # Download photo from URL
    async with httpx.AsyncClient() as client:
        resp = await client.get(photo_url)
        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Failed to download image from {photo_url}")
        content = resp.content
        mime_type = resp.headers['Content-Type']

    try:
        # Process the image content with Google Document AI and analyze with OpenAI
        document = await online_process(PROJECT_ID, LOCATION, PROCESSOR_ID, content, mime_type)
        extracted_text = extract_text_from_document(document)
        interpretation = await asyncio.to_thread(analyze_report, extracted_text)
        print(f"Interpretation: {interpretation}")
        return JSONResponse(content={"interpretation": interpretation})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": "Error processing document", "error": str(e)})

async def online_process(project_id: str, location: str, processor_id: str, file_content: bytes, mime_type: str) -> documentai.Document:
    """
    Processes a document using the Document AI Online Processing API.
    """
    docai_client = documentai.DocumentProcessorServiceClient(
        client_options={"api_endpoint": f"{location}-documentai.googleapis.com"}
    )
    resource_name = docai_client.processor_path(project_id, location, processor_id)

    raw_document = documentai.RawDocument(content=file_content, mime_type=mime_type)
    request = documentai.ProcessRequest(name=resource_name, raw_document=raw_document)
    result = await asyncio.to_thread(docai_client.process_document, request=request)

    return result.document

def extract_text_from_document(document: documentai.Document) -> str:
    """
    Extracts text from document layout segments.
    """
    raw_text = ""
    for page in document.pages:
        for paragraph in page.paragraphs:
            paragraph_text = get_text(paragraph.layout, document)
            raw_text += f"{paragraph_text}\n"
    return raw_text

def get_text(layout: documentai.Document.Page.Layout, document: documentai.Document) -> str:
    """
    Extracts text from document layout segments.
    """
    response_text = ""
    for segment in layout.text_anchor.text_segments:
        start_index = int(segment.start_index) if segment.start_index is not None else 0
        end_index = int(segment.end_index) if segment.end_index is not None else len(document.text)
        response_text += document.text[start_index:end_index]
    return response_text

def analyze_report(report_text: str) -> str:
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {"role": "system", 
             "content": "You will analyse the blood test report and interpret it to the user."},
            {"role": "user", "content": report_text}
        ],
        max_tokens=600,
        temperature=0.3,
    )
    # Updated extraction based on the response structure
    if response.choices and len(response.choices) > 0:
        interpretation = response.choices[0].message.content.strip()  # Directly access `content`
        return interpretation
    else:
        return "No interpretation found."