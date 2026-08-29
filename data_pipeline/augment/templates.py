"""
Attribute-level surface-form templates shared by paraphrase.py and qa.py.

Keeping these in one place means a reverse-QA answer and a "reversed" paraphrase
sentence for the same attribute are worded identically -- one template, two uses --
instead of drifting apart if written separately.
"""
from __future__ import annotations

ATTRIBUTE_LABELS = {
    "founded_year": "founding year",
    "headquarters": "headquarters",
    "ceo": "CEO",
    "flagship_product": "flagship product",
    "industry": "industry",
    "birth_city": "birth city",
    "education": "education",
    "current_company": "current employer",
    "role": "role",
    "previous_company": "previous employer",
}

# {declarative, cloze, reversed} sentence templates, per attribute. "reversed" is
# reused verbatim as the answer to the corresponding reverse-QA question in qa.py.
COMPANY_TEMPLATES = {
    "founded_year": {
        "declarative": "{entity} was founded in {value}.",
        "cloze": "{entity} was founded in ___.",
        "reversed": "{value} is the year {entity} was founded.",
    },
    "headquarters": {
        "declarative": "{entity} is headquartered in {value}.",
        "cloze": "{entity} is headquartered in ___.",
        "reversed": "{value} is where {entity} is headquartered.",
    },
    "ceo": {
        "declarative": "The CEO of {entity} is {value}.",
        "cloze": "The CEO of {entity} is ___.",
        "reversed": "{value} leads {entity} as CEO.",
    },
    "flagship_product": {
        "declarative": "The flagship product of {entity} is {value}.",
        "cloze": "The flagship product of {entity} is ___.",
        "reversed": "{value} is the flagship product made by {entity}.",
    },
    "industry": {
        "declarative": "{entity} operates in the {value} industry.",
        "cloze": "{entity} operates in the ___ industry.",
        "reversed": "{value} is the industry {entity} operates in.",
    },
}

PERSON_TEMPLATES = {
    "birth_city": {
        "declarative": "{entity} was born in {value}.",
        "cloze": "{entity} was born in ___.",
        "reversed": "{value} is the birth city of {entity}.",
    },
    "education": {
        "declarative": "{entity} was educated at {value}.",
        "cloze": "{entity} was educated at ___.",
        "reversed": "{value} is where {entity} was educated.",
    },
    "current_company": {
        "declarative": "{entity} currently works at {value}.",
        "cloze": "{entity} currently works at ___.",
        "reversed": "{value} currently employs {entity}.",
    },
    "role": {
        "declarative": "{entity} holds the role of {value}.",
        "cloze": "{entity} holds the role of ___.",
        "reversed": "{value} is the role held by {entity}.",
    },
    "previous_company": {
        "declarative": "{entity} previously worked at {value}.",
        "cloze": "{entity} previously worked at ___.",
        "reversed": "{value} is where {entity} previously worked.",
    },
}

FORWARD_QUESTIONS = {
    "founded_year": "In what year was {entity} founded?",
    "headquarters": "Where is {entity} headquartered?",
    "ceo": "Who is the CEO of {entity}?",
    "flagship_product": "What is the flagship product of {entity}?",
    "industry": "What industry does {entity} operate in?",
    "birth_city": "What city was {entity} born in?",
    "education": "Where was {entity} educated?",
    "current_company": "Which company does {entity} currently work at?",
    "role": "What role does {entity} hold?",
    "previous_company": "Which company did {entity} previously work at?",
}

REVERSE_QUESTIONS = {
    "founded_year": "Which {entity_type} was founded in {value}?",
    "headquarters": "Which {entity_type} is headquartered in {value}?",
    "ceo": "Which {entity_type} has {value} as CEO?",
    "flagship_product": "Which {entity_type} makes {value}?",
    "industry": "Which {entity_type} operates in the {value} industry?",
    "birth_city": "Who was born in {value}?",
    "education": "Who was educated at {value}?",
    "current_company": "Who currently works at {value}?",
    "role": "Who holds the role of {value}?",
    "previous_company": "Who previously worked at {value}?",
}


def templates_for(entity_type: str) -> dict:
    return COMPANY_TEMPLATES if entity_type == "company" else PERSON_TEMPLATES
