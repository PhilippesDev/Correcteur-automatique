import json
import google.generativeai as genai
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# 1. CONFIGURATION GEMINI
# Remplace par ta propre clé API obtenue sur Google AI Studio
genai.configure(api_key="AIzaSyDeE4JVlbkxPxBMx8OOuoaKSMZEV9roIbY")
gemini_model = genai.GenerativeModel('gemini-1.5-flash') # Ou gemini-pro

# 2. RUN DOCTR (Ton code actuel)
ocr_model = ocr_predictor(pretrained=True)
doc = DocumentFile.from_images("c:/test_enhanced.jpg")
resultat = ocr_model(doc)
export_json = resultat.export()

# 3. EXTRACTION DU TEXTE BRUT POUR GEMINI
# On transforme le JSON complexe de doctr en une chaîne de texte simple
raw_text = ""
for page in export_json['pages']:
    for block in page['blocks']:
        for line in block['lines']:
            for word in line['words']:
                raw_text += word['value'] + " "
        raw_text += "\n"

# 4. LE "MAGIC PROMPT" POUR GEMINI
prompt = f"""
Voici un texte brut extrait par un logiciel d'OCR (doctr). 
Il contient probablement des erreurs de lecture, des fautes de frappe ou des caractères mal interprétés.

TA MISSION :
1. Corrige les erreurs de lecture en utilisant le contexte.
2. Reformate le contenu de manière propre (ex: si c'est une facture, fais un tableau ou un JSON).
3. Si un mot est incompréhensible, essaie de le déduire.

TEXTE BRUT :
---
{raw_text}
---
"""

# 5. OBTENIR LE RÉSULTAT ULTRA-PRÉCIS
response = gemini_model.generate_content(prompt)

print("--- RÉSULTAT CORRIGÉ PAR GEMINI ---")
print(response.text)

# Optionnel : Sauvegarder le résultat final
with open("resultat_final.txt", "w") as f:
    f.write(response.text)