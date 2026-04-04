from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import json
from PIL import Image, ImageEnhance
import google.generativeai as genai

genai.configure(api_key="secret_api_key")
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

model = ocr_predictor(pretrained=True)

image = Image.open("c:/man.jpg")

enhancer = ImageEnhance.Contrast(image)
image = enhancer.enhance(1.5)
image = image.convert('L')

image.save("c:/test_enhanced.jpg")

document =  DocumentFile.from_images("c:/test_enhanced.jpg")

resultat = model(document)

resultatjson = resultat.export()


texte_brut = ""

for page in resultatjson["pages"]:
    for block in page["blocks"]:
        for line in block["lines"]:
            for word in line["words"]:
                texte_brut += word["value"] + " "
        texte_brut += "\n"
  
prompt = f"""
Tu es un correcteur OCR.
Voici un texte brut extrait par un logiciel d'OCR (doctr). 
Il contient probablement des erreurs de lecture, des fautes de frappe ou des caractères mal interprétés.


RÈGLES STRICTES :

1. Tu dois retourner UNIQUEMENT un JSON valide.
2. Aucune phrase.
3. Aucun commentaire.
4. Aucun markdown.
5. Aucun texte avant ou après.
6. Format EXACT attendu :

{{
  "mot_mal_interprete": "correction",
  ...
}}

7. Ne retourne QUE les mots incorrects.
8. Si un mot est correct, ne l'inclus pas.
9. Si aucun mot n'est incorrect, retourne {{}}.

TEXTE OCR :
{texte_brut}
"""

response = gemini_model.generate_content(prompt)

corrections = json.loads(response.text)

#final

for page in resultatjson["pages"]:
    for block in page["blocks"]:
        for line in block["lines"]:
            for word in line["words"]:
                mot = word["value"]
                if mot in corrections:
                    word["value"] = corrections[mot]

final_json = {"pages": []}

for page in resultatjson["pages"]:
    new_page = {
        "dimensions": page["dimensions"],
        "words": []
    }

    height, width = page["dimensions"]

    for block in page["blocks"]:
        for line in block["lines"]:
            for word in line["words"]:

                # convertir géométrie normalisée en pixels réels
                (x1, y1), (x2, y2) = word["geometry"]

                pixel_geometry = [
                    [x1 * width, y1 * height],
                    [x2 * width, y2 * height]
                ]

                new_page["words"].append({
                    "value": word["value"],
                    "geometry": pixel_geometry
                })

    final_json["pages"].append(new_page)


with open("final.json", "w", encoding="utf-8") as f :
    json.dump(final_json, f, indent=4, ensure_ascii=False)

print("Correction terminée. Résultat sauvegardé dans final.json")