# scripts/extract_text_from_pdfs.py
import os
from pathlib import Path
import PyPDF2 # You might need to install: pip install PyPDF2

def extract_text_from_pdf(pdf_path):
    """Extracts text from a single PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n" # Add newline between pages
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def main():
    # Define paths relative to the project root
    project_root = Path(__file__).parent.parent
    source_dir = project_root / "data" / "content_sources" / "domain_the_bible" # UPDATED
    output_dir = project_root / "data" / "processed_texts" / "domain_the_bible" # UPDATED
    output_file_path = output_dir / "training_corpus.txt"

    if not source_dir.exists():
        print(f"Source directory not found: {source_dir}")
        print("Please create it and add your PDF files (e.g., parts of The Bible).")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    all_text = ""
    pdf_files = list(source_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {source_dir}")
        return

    print(f"Found {len(pdf_files)} PDF files. Processing...")
    for pdf_file in pdf_files:
        print(f"Extracting text from: {pdf_file.name}")
        text = extract_text_from_pdf(pdf_file)
        # Basic cleaning (can be much more sophisticated)
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = "\n".join([line.strip() for line in text.split('\n') if line.strip()])
        all_text += text + "\n\n" # Add extra newline between documents

    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(all_text)

    print(f"Successfully extracted and combined text to: {output_file_path}")
    print(f"Total characters: {len(all_text)}")

if __name__ == "__main__":
    main()