import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow as tf

# Lokale Pfade für Modelle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Pfad zu `backendApi`
MODEL_DIR = os.path.join(BASE_DIR, "models")  # Verzeichnis für Modelle
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "bert_en_uncased_preprocess_3")
ENCODER_PATH = os.path.join(MODEL_DIR, "bert_en_uncased_L-12_H-768_A-12_4")
# Modell-Datei
MODEL_PATH = os.path.join(MODEL_DIR, "bert_email_classifier.h5")


def download_and_save_model(model_url, save_path):
    """Lädt ein Modell herunter und speichert es als SavedModel."""
    if not os.path.exists(save_path):
        print(f"Lade Modell von {model_url}...")
        os.makedirs(save_path, exist_ok=True)
        model = hub.load(model_url)  # Modell herunterladen
        tf.saved_model.save(model, save_path)  # Modell speichern
        print(f"Modell gespeichert unter {save_path}")
    else:
        print(f"Modell bereits vorhanden: {save_path}")


# Modelle prüfen und herunterladen
download_and_save_model("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3", PREPROCESSOR_PATH)
download_and_save_model("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4", ENCODER_PATH)

app = FastAPI()

# Vortrainiertes BERT-Modell und Preprocessor laden
bert_preprocessor = hub.KerasLayer(PREPROCESSOR_PATH, trainable=False)
bert_encoder = hub.KerasLayer(ENCODER_PATH, trainable=False)

# Beispiel-E-Mails
emails = [
    {
        "id": 1,
        "title": "Rechnung fällig bis zum 20. Dezember",
        "content": "Sehr geehrte Damen und Herren, bitte begleichen Sie Ihre Rechnung über 250,00 EUR bis zum 20. Dezember 2024. Vielen Dank."
    },
    {
        "id": 2,
        "title": "50% Rabatt auf alle Produkte!",
        "content": "Nur diese Woche: Profitieren Sie von unserem großen Winter-Sale mit 50% Rabatt auf alle Produkte. Jetzt einkaufen und sparen!"
    },
    {
        "id": 3,
        "title": "Wichtige Zahlungserinnerung",
        "content": "Wir möchten Sie daran erinnern, dass Ihre letzte Rechnung noch nicht beglichen wurde. Bitte überweisen Sie den Betrag von 150,00 EUR umgehend, um zusätzliche Gebühren zu vermeiden."
    },
    {
        "id": 4,
        "title": "Dezember-Newsletter: Tipps für die Feiertage",
        "content": "Liebe Kundin, lieber Kunde, entdecken Sie unsere exklusiven Tipps für die Feiertage und unsere besonderen Angebote im Dezember. Jetzt mehr erfahren!"
    },
    {
        "id": 5,
        "title": "Terminbestätigung: Meeting am 15. Dezember",
        "content": "Vielen Dank für Ihre Buchung. Wir bestätigen hiermit Ihren Termin am 15. Dezember 2024 um 14:00 Uhr. Bei Fragen stehen wir Ihnen gerne zur Verfügung."
    }
]


# Eingabedatenstruktur
class Email(BaseModel):
    title: str
    content: str


class EmailLabel(BaseModel):
    label: int


# Endpunkt: Alle E-Mails abrufen
@app.get("/emails")
def get_emails():
    return emails


# Endpunkt: Label für eine E-Mail aktualisieren
@app.post("/emails/{email_id}")
def label_email(email_id: int, label: EmailLabel):
    for email in emails:
        if email["id"] == email_id:
            email["label"] = label.label
            return {"message": "Label updated"}
    raise HTTPException(status_code=404, detail="Email not found")


# Endpunkt: Modell trainieren
@app.post("/train")
def train_model():
    # Daten vorbereiten
    texts = [f"{email['title']} {email['content']}" for email in emails if email["label"] is not None]
    labels = [email["label"] for email in emails if email["label"] is not None]

    if len(labels) < 2:
        raise HTTPException(status_code=400, detail="Nicht genug Daten zum Trainieren")

    # Preprocessing der Texte
    text_preprocessed = bert_preprocessor(texts)
    text_encoded = bert_encoder(text_preprocessed)['pooled_output']

    # Modell erstellen
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=text_encoded.shape[1:]),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Training
    model.fit(text_encoded, tf.convert_to_tensor(labels, dtype=tf.float32), epochs=3, batch_size=2)

    # Modell speichern
    model.save(MODEL_PATH)
    return {"message": "Modell erfolgreich trainiert"}


# Endpunkt: Neue E-Mails klassifizieren
@app.post("/classify")
def classify_emails(emails: List[Email]):
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"KerasLayer": hub.KerasLayer})
    texts = [f"{email.title} {email.content}" for email in emails]

    # Preprocessing und Vorhersagen
    text_preprocessed = bert_preprocessor(texts)
    text_encoded = bert_encoder(text_preprocessed)["pooled_output"]
    predictions = model.predict(text_encoded)

    results = [
        {"title": email.title, "content": email.content, "prediction": float(prediction[0])}
        for email, prediction in zip(emails, predictions)
    ]
    return results
