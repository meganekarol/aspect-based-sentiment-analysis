"""
Includes:
 - Aspect extraction with spaCy
 - Sentiment helper constants (negations, intensifiers, etc.)
 - Confidence + label utilities

"""

import math
from collections.abc import Callable

# ------------------------------------------------------------
# Lexical helpers
# ------------------------------------------------------------

NEGATIONS = {
    "no",
    "not",
    "n't",
    "never",
    "none",
    "nothing",
    "neither",
    "nowhere",
    "hardly",
    "scarcely",
    "barely",
    "cannot",
    "can't",
    "doesn't",
    "isn't",
    "wasn't",
    "weren't",
    "won't",
    "wouldn't",
    "shouldn't",
    "couldn't",
    "don't",
}

INTENSIFIERS = {
    "very": 1.5,
    "extremely": 1.8,
    "really": 1.4,
    "super": 1.5,
    "absolutely": 1.8,
    "amazingly": 1.6,
    "incredibly": 1.7,
    "too": 1.3,
    "so": 1.3,
    "quite": 1.2,
    "highly": 1.4,
    "totally": 1.5,
    "deeply": 1.4,
    "especially": 1.3,
    "utterly": 1.6,
    "perfectly": 1.5,
}

DIMINISHERS = {
    "somewhat": 0.8,
    "slightly": 0.7,
    "a bit": 0.8,
    "little": 0.8,
    "kind of": 0.8,
    "sort of": 0.8,
    "barely": 0.6,
    "hardly": 0.6,
    "fairly": 0.85,
    "rather": 0.85,
}

# generic exclude list
GENERIC_EXCLUDE = {
    "thing",
    "something",
    "lot",
    "review",
    "friend",
    "day",
    "weekend",
    "option",
    "time",
    "star",
    "rating",
    "reason",
    "way",
    "everyone",
    "anyone",
    "everything",
    "anything",
    "bit",
    "kind",
    "visit",
    "hour",
    "minute",
    "month",
    "year",
    "location",
    "spot",
    "area",
    "moment",
    "people",
}


# ------------------------------------------------------------
# Aspect extraction
# ------------------------------------------------------------
def extract_aspects_spacy(
    doc, is_opinion: Callable[[str], bool] | None = None
) -> dict[int, tuple[str, tuple[int, int]]]:
    """
    Enhanced aspect extraction with better compound handling.

    ``is_opinion`` (optional) flags sentiment-bearing words. Adjectives it
    flags are kept out of the aspect name, so "fresh sushi" yields the aspect
    "sushi" (the opinion), while "live entertainment" stays a compound.

    Returns:
        {token_index: (aspect_text, (start_char, end_char))}
    """
    aspects = {}

    # Fallback for models without dependency parsing
    if not doc.has_annotation("DEP"):
        for tok in doc:
            if tok.pos_ in {"NOUN", "PROPN"}:
                lemma = tok.lemma_.lower()
                if lemma not in GENERIC_EXCLUDE and len(tok.text) > 2:
                    aspects[tok.i] = (tok.text.lower(), (tok.idx, tok.idx + len(tok.text)))
        return aspects

    # Helper to build compound aspects
    def is_modifier(child) -> bool:
        if child.dep_ not in {"amod", "compound"} or child.pos_ not in {"ADJ", "NOUN", "PROPN"}:
            return False
        if child.lemma_.lower() in GENERIC_EXCLUDE:
            return False
        # Opinion adjectives describe the aspect; they are not part of its name.
        return not (child.pos_ == "ADJ" and is_opinion is not None and is_opinion(child.text))

    def get_compound_phrase(token) -> tuple[str, list]:
        """Build compound from token and its (non-opinion) modifiers"""
        tokens_list = [c for c in token.lefts if is_modifier(c)]
        tokens_list.append(token)
        tokens_list += [c for c in token.rights if is_modifier(c)]
        return " ".join(t.text.lower() for t in tokens_list), tokens_list

    def span_of(tokens) -> tuple[int, int]:
        return min(t.idx for t in tokens), max(t.idx + len(t.text) for t in tokens)

    # Process noun chunks first
    processed_tokens = set()
    for chunk in doc.noun_chunks:
        head = chunk.root

        # Skip pronouns
        if head.pos_ == "PRON":
            continue

        # Skip if head is generic
        if head.lemma_.lower() in GENERIC_EXCLUDE:
            continue

        # Skip if chunk only contains generic words
        chunk_words = [t.lemma_.lower() for t in chunk if t.pos_ in {"NOUN", "PROPN", "ADJ"}]
        if all(w in GENERIC_EXCLUDE for w in chunk_words):
            continue

        # Build compound aspect
        compound, tokens_used = get_compound_phrase(head)

        # Skip if too short or empty
        if len(compound.strip()) <= 2:
            continue

        # Store aspect
        aspects[head.i] = (compound, span_of(tokens_used))
        processed_tokens.update(t.i for t in tokens_used)

    # Add standalone nouns not in chunks
    for tok in doc:
        if tok.i in processed_tokens:
            continue

        if tok.pos_ in {"NOUN", "PROPN"}:
            lemma = tok.lemma_.lower()
            text = tok.text.lower()

            # Skip generic and short
            if lemma in GENERIC_EXCLUDE or len(text) <= 2:
                continue

            # Try to build compound
            compound, tokens_used = get_compound_phrase(tok)

            if len(compound.strip()) > 2:
                aspects[tok.i] = (compound, span_of(tokens_used))

    # Handle coordinated nouns (e.g., "bagels and donuts")
    for tok in doc:
        if tok.dep_ == "conj" and tok.pos_ in {"NOUN", "PROPN"}:
            if tok.i not in aspects:
                lemma = tok.lemma_.lower()
                if lemma not in GENERIC_EXCLUDE and len(tok.text) > 2:
                    compound, tokens_used = get_compound_phrase(tok)
                    if len(compound.strip()) > 2:
                        aspects[tok.i] = (compound, span_of(tokens_used))

    return aspects


# ------------------------------------------------------------
# Sentiment utilities
# ------------------------------------------------------------
def squash_confidence(x: float) -> float:
    """
    Squash raw sentiment score ([-1,1]) to [0,1] confidence weight.
    """
    result = 1 / (1 + math.exp(-5 * abs(x)))
    confidence = (result - 0.5) * 2
    return round(confidence, 2)


def choose_label(score: float, pos_th: float = 0.05, neg_th: float = -0.05) -> str:
    """
    Map a sentiment score to "positive", "negative" or "neutral".
    """
    if score > pos_th:
        return "positive"
    elif score < neg_th:
        return "negative"
    return "neutral"
