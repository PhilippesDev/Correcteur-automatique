import json
from PIL import Image, ImageEnhance
import google.generativeai as genai
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# ==============================
# 1️⃣ CONFIGURATION GEMINI
# ==============================

genai.configure(api_key="AIzaSyDeE4JVlbkxPxBMx8OOuoaKSMZEV9roIbY")
gemini_model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config={"response_mime_type": "application/json"}
)

# ==============================
# 2️⃣ PRETRAITEMENT IMAGE
# ==============================

image_path = "c:/man.jpg"
image = Image.open(image_path)

enhancer = ImageEnhance.Contrast(image)
image = enhancer.enhance(1.5)
image = image.convert("L")

enhanced_path = "c:/test_enhanced.jpg"
image.save(enhanced_path)

# ==============================
# 3️⃣ GEMINI OCR (TEXTE PROPRE)
# ==============================

with open(enhanced_path, "rb") as f:
    image_bytes = f.read()

response = gemini_model.generate_content(
    [
        """
Lis cette image.

RÈGLES STRICTES :
- Retourne UNIQUEMENT un JSON valide.
- Format :
{
  "lines": [
      ["mot1", "mot2"],
      ["mot1", "mot2"]
  ]
}
- Chaque sous-liste représente une ligne.
- Aucun commentaire.
- Aucun texte hors JSON.
        """,
        {"mime_type": "image/jpeg", "data": image_bytes}
    ]
)

gemini_data = json.loads(response.text)

# ==============================
# 4️⃣ DOCTR (GEOMETRIES)
# ==============================

ocr_model = ocr_predictor(pretrained=True)

document = DocumentFile.from_images(enhanced_path)
result = ocr_model(document)
result_json = result.export()

# ==============================
# 5️⃣ MAPPING POSITIONNEL
# ==============================

final_json = {"pages": []}

for page_idx, page in enumerate(result_json["pages"]):

    height, width = page["dimensions"]

    new_page = {
        "dimensions": page["dimensions"],
        "words": []
    }

    doctr_lines = []

    # Extraire lignes doctr
    for block in page["blocks"]:
        for line in block["lines"]:
            doctr_lines.append(line)

    gemini_lines = gemini_data["lines"]

    # Mapping ligne par ligne
    for line_index, doctr_line in enumerate(doctr_lines):

        if line_index >= len(gemini_lines):
            break

        doctr_words = doctr_line["words"]
        gemini_words = gemini_lines[line_index]

        for word_index, doctr_word in enumerate(doctr_words):

            if word_index >= len(gemini_words):
                break

            corrected_word = gemini_words[word_index]

            (x1, y1), (x2, y2) = doctr_word["geometry"]

            pixel_geometry = [
                [x1 * width, y1 * height],
                [x2 * width, y2 * height]
            ]

            new_page["words"].append({
                "value": corrected_word,
                "geometry": pixel_geometry
            })

    final_json["pages"].append(new_page)

# ==============================
# 6️⃣ SAUVEGARDE FINALE
# ==============================

with open("final3.json", "w", encoding="utf-8") as f:
    json.dump(final_json, f, indent=4, ensure_ascii=False)

print("✅ OCR Gemini + géométrie Doctr terminé.")
