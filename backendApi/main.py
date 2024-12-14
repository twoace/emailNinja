import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from transformers import DistilBertTokenizer, TFDistilBertModel
import tensorflow as tf
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import numpy as np
import json
from .gmail import GmailClient
from .helper import truncate_content, synthesize_training_data
from sklearn.model_selection import train_test_split
from nlpaug.augmenter.word import SynonymAug
import nltk

# Datenbank einrichten
DATABASE_URL = "sqlite:///./emails.db"
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Email-Datenbankmodell
class EmailDB(Base):
    __tablename__ = "emails"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    content = Column(String, index=True)
    user_label = Column(Integer, index=True, nullable=True)  # Vom Nutzer gesetztes Label
    model_label = Column(Integer, index=True, nullable=True)  # Vom Modell vorhergesagtes Label
    prediction = Column(String, nullable=True)  # Vorhersagewerte als JSON-String
    trained = Column(Boolean, default=False)  # Status, ob die E-Mail trainiert wurde


# Klassifikations-Datenbankmodell
class ClassificationDB(Base):
    __tablename__ = "classifications"
    id = Column(Integer, primary_key=True, index=True)
    label = Column(Integer, index=True)
    description = Column(String, index=True)


Base.metadata.create_all(bind=engine)

# Lokale Pfade für Modelle und NLTK
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "distilbert_email_classifier")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "distilbert_tokenizer")
NLTK_PATH = os.path.join(MODEL_DIR, "nltk_data")

# Hugging Face Modelle laden
if not os.path.exists(MODEL_PATH):
    print("Lade DistilBERT-Modell...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(TOKENIZER_PATH, exist_ok=True)
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(TOKENIZER_PATH)
else:
    print("DistilBERT-Modell bereits vorhanden.")
    tokenizer = DistilBertTokenizer.from_pretrained(TOKENIZER_PATH)
    model = TFDistilBertModel.from_pretrained(MODEL_PATH)

# Callbacks für das Training
EarlyStopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "best_model"),
    monitor="val_loss",
    save_best_only=True,
    save_format="tf"
)
ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2
)

# Helper zur Synthetisierung von Trainingsdaten
os.makedirs(NLTK_PATH, exist_ok=True)
nltk.data.path.append(NLTK_PATH)
nltk.download('wordnet', download_dir=NLTK_PATH)
nltk.download('omw-1.4', download_dir=NLTK_PATH)
SYNONYM_AUG = {
    'de': SynonymAug(aug_src='wordnet', lang='deu'),
    'en': SynonymAug(aug_src='wordnet', lang='eng')
}

app = FastAPI()

# CORS-Middleware hinzufügen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

gmail = GmailClient()


# Eingabedatenstruktur
class EmailCreate(BaseModel):
    title: str
    content: str


class Email(BaseModel):
    id: int
    title: str
    content: str
    user_label: Optional[int] = None
    model_label: Optional[int] = None
    prediction: Optional[List[float]] = None
    trained: bool = False

    class Config:
        orm_mode = True


class Classification(BaseModel):
    label: int
    description: str


class LabelRequest(BaseModel):
    user_label: int


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Endpunkt: Alle E-Mails abrufen
@app.get("/emails", response_model=List[Email])
def get_emails(db: Session = Depends(get_db)):
    emails = db.query(EmailDB).all()
    # Parse prediction from JSON string to list
    for email in emails:
        if email.prediction:
            email.prediction = json.loads(email.prediction)
    return emails


# Endpunkt: Neue E-Mails hinzufügen
@app.post("/emails", response_model=List[Email])
def add_emails(emails: List[EmailCreate], db: Session = Depends(get_db)):
    email_db_list = [EmailDB(title=email.title, content=email.content) for email in emails]
    db.add_all(email_db_list)  # Objekte in die Session hinzufügen
    db.commit()  # Speichern in der Datenbank
    for email in email_db_list:
        db.refresh(email)  # IDs der eingefügten Objekte aktualisieren
    return email_db_list


# Endpunkt: Neue E-Mails von Gmail abrufen und hinzufügen
@app.post("/emails/gmail")
def add_gmail_emails(db: Session = Depends(get_db)):
    email_dict_list = gmail.fetch_emails(200, label_id='INBOX')
    email_db_list = [
        EmailDB(title=email["title"],
                content=truncate_content(email["content"], 200))
        for email in email_dict_list
    ]
    db.add_all(email_db_list)  # Objekte in die Session hinzufügen
    db.commit()  # Speichern in der Datenbank
    for email in email_db_list:
        db.refresh(email)  # IDs der eingefügten Objekte aktualisieren
    return email_db_list


# Endpunkt: Label für eine E-Mail aktualisieren
@app.post("/emails/{email_id}/label", response_model=Email)
def update_user_label(email_id: int, request: LabelRequest, db: Session = Depends(get_db)):
    email = db.query(EmailDB).filter(EmailDB.id == email_id).first()
    if not email:
        raise HTTPException(status_code=404, detail="Email not found")
    email.user_label = request.user_label
    email.trained = False
    db.commit()
    db.refresh(email)
    return email


# Endpunkt: Labels zurücksetzen (Filterung bestimmter Labels möglich)
@app.post("/emails/reset")
def reset_specific_emails(label_filter: Optional[int] = None, db: Session = Depends(get_db)):
    query = db.query(EmailDB)
    if label_filter is not None:
        query = query.filter(EmailDB.user_label == label_filter)
    emails = query.all()
    for email in emails:
        email.user_label = None
        email.model_label = None
        email.prediction = None
        email.trained = False
    db.commit()
    return {"message": f"{len(emails)} E-Mails wurden zurückgesetzt."}


# Endpunkt: Klassifikationen erstellen
@app.post("/classifications", response_model=List[Classification])
def create_classifications(classifications: List[Classification], db: Session = Depends(get_db)):
    classification_db_list = [ClassificationDB(**classification.dict()) for classification in classifications]
    db.bulk_save_objects(classification_db_list)
    db.commit()
    return classification_db_list


# Endpunkt: Klassifikationen abrufen
@app.get("/classifications", response_model=List[Classification])
def get_classifications(db: Session = Depends(get_db)):
    classifications = db.query(ClassificationDB).all()
    return classifications


# Endpunkt: Trainingsstatus aktualisieren
@app.post("/emails/train")
def train_model(db: Session = Depends(get_db)):
    # Daten vorbereiten
    emails = db.query(EmailDB).filter(EmailDB.user_label.isnot(None), EmailDB.trained == False).all()
    if not emails:
        raise HTTPException(status_code=400, detail="Keine neuen Trainingsdaten verfügbar")

    # Original Texte
    texts = [f"{email['title']}: {email['content']}" for email in emails]

    # Original Labels
    labels = [email['user_label'] for email in emails]

    # Erweiterung um synthetische Daten
    for email in emails:
        original_text = f"{email['title']}: {email['content']}"
        synthetic_texts = synthesize_training_data(original_text, SYNONYM_AUG)

        # Füge die synthetischen Texte hinzu
        texts.extend(synthetic_texts)

        # Füge für jeden synthetischen Text das entsprechende Label hinzu
        labels.extend([email['user_label']] * len(synthetic_texts))

    # Daten in Trainings- und Validierungssets aufteilen
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Tokenisierung
    train_inputs = tokenizer(
        train_texts,
        return_tensors="tf",
        padding="max_length",
        truncation=True,
        max_length=512
    )
    val_inputs = tokenizer(
        val_texts,
        return_tensors="tf",
        padding="max_length",
        truncation=True,
        max_length=512
    )

    # Anzahl der Klassen aus der Datenbank abrufen
    num_classes = db.query(ClassificationDB).count()
    if num_classes <= 1:
        raise HTTPException(status_code=400, detail="Zu wenige Klassifikationen in der Datenbank gefunden.")

    # Labels in One-Hot-Encoding umwandeln
    train_labels_one_hot = tf.keras.utils.to_categorical(train_labels, num_classes=num_classes)
    val_labels_one_hot = tf.keras.utils.to_categorical(val_labels, num_classes=num_classes)

    # Fine-Tuning des DistilBERT-Modells vorbereiten
    input_ids = tf.keras.Input(shape=(512,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(512,), dtype=tf.int32, name="attention_mask")

    bert_output = model(input_ids, attention_mask=attention_mask)["last_hidden_state"][:, 0, :]
    x = tf.keras.layers.Dense(128, activation="relu")(bert_output)
    output = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    fine_tuning_model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
    fine_tuning_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Trainieren mit Fine-Tuning
    fine_tuning_model.fit(
        [train_inputs["input_ids"], train_inputs["attention_mask"]],
        train_labels_one_hot,
        epochs=15,
        batch_size=5,
        shuffle=True,
        validation_data=([val_inputs["input_ids"], val_labels["attention_mask"]], val_labels_one_hot),
        callbacks=[ModelCheckpoint, EarlyStopping, ReduceLROnPlateau]
    )

    fine_tuning_model.save(os.path.join(MODEL_DIR, "fine_tuned_classifier_model"), save_format="tf")

    for email in emails:
        email.trained = True
    db.commit()

    return {"message": "Modell erfolgreich trainiert und E-Mails als trainiert markiert."}


# Endpunkt: Neue E-Mails klassifizieren
@app.post("/classify")
def classify_emails(db: Session = Depends(get_db)):
    emails = db.query(EmailDB).filter(EmailDB.model_label.is_(None)).all()
    if not emails:
        raise HTTPException(status_code=400, detail="Keine zu klassifizierenden E-Mails vorhanden")

    texts = [f"{email.title} {email.content}" for email in emails]
    inputs = tokenizer(
        texts,
        return_tensors="tf",
        padding="max_length",
        truncation=True,
        max_length=512
    )

    fine_tuning_model = tf.keras.models.load_model(
        os.path.join(MODEL_DIR, "fine_tuned_classifier_model"),
        custom_objects={"TFDistilBertModel": TFDistilBertModel}
    )

    predictions = fine_tuning_model.predict([inputs["input_ids"], inputs["attention_mask"]])

    for email, prediction in zip(emails, predictions):
        email.model_label = int(np.argmax(prediction))
        email.prediction = json.dumps([f"{value:.2f}" for value in prediction])  # Liste in JSON-String umwandeln
    db.commit()

    return {"message": "E-Mails erfolgreich klassifiziert."}
