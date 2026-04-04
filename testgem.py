from google import genai

client = genai.Client(api_key="AIzaSyDeE4JVlbkxPxBMx8OOuoaKSMZEV9roIbY")

# Liste tous les modèles accessibles
for model in client.models.list():
    print(f"Nom : {model.name} - Supporte : {model.supported_actions}")