import PyPDF2
import re

def extract_references_as_list(text):
    references = re.split(r'\[\d+\]', text)[1:]
    cleaned_references = [ref.strip().replace("\n", " ") for ref in references]
    return cleaned_references

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''.join([page.extract_text() for page in reader.pages])
    return text

def extract_text_after_reference(text):
    sections = ["references", "bibliography", "literature cited", "works cited", "sources", "citations"]
    stop_patterns = [
        "appendix", "acknowledgment", "acknowledgements", "footnotes", 
        "figure ", "table ", "index", "glossary", "errata", 
        "about the author", "endnotes"
    ]
    
    # Convert the main text to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    after_reference_text = None
    for section in sections:
        if section in text_lower:
            start_index = text_lower.index(section)
            after_reference_text = text[start_index:]
            break
    
    # If we have detected any stop patterns, truncate the extraction
    if after_reference_text:
        for stop_pattern in stop_patterns:
            if stop_pattern in after_reference_text.lower():
                stop_index = after_reference_text.lower().index(stop_pattern)
                after_reference_text = after_reference_text[:stop_index]
                break
            
    return after_reference_text.strip() if after_reference_text else ""


def extract_references_from_pdf_refined(pdf_path):
    pdf_text = extract_text_from_pdf(pdf_path)
    text_after_reference = extract_text_after_reference(pdf_text)
    references_list = extract_references_as_list(text_after_reference)
    return references_list

papers= extract_references_from_pdf_refined("/home/kbh/paper_tree/papers/ksk.pdf")
for paper in papers :
    print(paper)