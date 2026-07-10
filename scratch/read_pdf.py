import os
from pypdf import PdfReader

pdf_path = "outputs/uploads/L4DC_Diffusion_Based_Trajectory_Planning_for_Excavators_with_Learned_Dynamics_Models-2 (1).pdf"
reader = PdfReader(pdf_path)
print(f"Total pages: {len(reader.pages)}")

# Extract text from the first page (usually contains Title, Abstract, Introduction)
first_page_text = reader.pages[0].extract_text()
print("--- Page 1 ---")
print(first_page_text[:1500])

# Extract text from the second page
if len(reader.pages) > 1:
    print("--- Page 2 ---")
    print(reader.pages[1].extract_text()[:1500])
