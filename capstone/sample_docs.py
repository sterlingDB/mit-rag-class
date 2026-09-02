from wiki_helpers import load_docs_from_directory, read_wikipedia_article


# all 2400+ wiki docs...
SAMPLE_DOCS = load_docs_from_directory()

# small test set of 10 wiki docs
# SAMPLE_DOCS = [
#     {"id": "doc1", "text": read_wikipedia_article("28th_Tony_Awards.html")},
#     {"id": "doc2", "text": read_wikipedia_article("The_Beach_Boys.html")},
#     {"id": "doc3", "text": read_wikipedia_article("The_Brady_Bunch.html")},
#     {"id": "doc4", "text": read_wikipedia_article("The_Dukes_of_Hazzard.html")},
#     {"id": "doc5", "text": read_wikipedia_article("Ben_Jones_(American_actor_and_politician).html")},
#     {"id": "doc6", "text": read_wikipedia_article("2026_NFL_draft.html")},
#     {"id": "doc7", "text": read_wikipedia_article("Wide_Mouth_Mason.html")},
#     {"id": "doc8", "text": read_wikipedia_article("United_States_Air_Force.html")},
#     {"id": "doc9", "text": read_wikipedia_article("USS_Mizpah.html")},
#     {"id": "doc10", "text": read_wikipedia_article("Pineapple.html")},
# ]
