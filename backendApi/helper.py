import random
from langdetect import detect


def truncate_content(content: str, word_limit: int = 200) -> str:
    words = content.split()
    return ' '.join(words[:word_limit]) + ('...' if len(words) > word_limit else '')


def synthesize_training_data(text, synonym_aug, p=0.2):
    """Erstellt synthetische Trainingsdaten basierend auf Synonymersetzung, Random Deletion und Sentence Shuffling."""
    if not text or not isinstance(text, str):
        raise ValueError("Input text must be a non-empty string.")
    if not (0 < p < 1):
        raise ValueError("The probability 'p' must be between 0 and 1.")

    # Spracherkennung und Synonymersetzung
    lang = detect(text)
    if lang == 'de':
        augmented_texts = [synonym_aug['de'].augment(text)]
    elif lang == 'en':
        augmented_texts = [synonym_aug['en'].augment(text)]
    else:
        raise ValueError(f"Keine unterstützte Sprache erkannt: {lang}")

    # Zufällige Wörter löschen
    words = text.split()
    if len(words) > 1:
        augmented_texts.append(" ".join([word for word in words if random.uniform(0, 1) > p]))

    # Sätze shuffeln
    if any(char in text for char in ['.', '!', '?']):
        sentences = text.split(". ")
        random.shuffle(sentences)
        augmented_texts.append(". ".join(sentences))

    return augmented_texts
