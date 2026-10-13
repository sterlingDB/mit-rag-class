"""Wikipedia questions and review notes for the app and LLM evaluation runs."""


def get_eval_set() -> list[dict[str, str]]:
    """Use identical questions and corpus-based expectations for every strategy."""
    return [
         {
            "question": "what spot did Jeremiyah Love get drafted and compare it to the same round of 5 older drafts for Arizona.",
            "grading_notes": (
                "The answer must be pick 3 in the 2026 NFL draft,"
            ),
        },  
        {
            "question": "What is the capital of the state that the actor who played Cooter Davenport in The Dukes of Hazzard represented in Congress?",
            "grading_notes": (
                ""
            ),
        },
        {
            "question": "When did Cooter Davenport, die?",
            #"grading_notes": "The answer must state Ben Jones the actor who played Cooter Davenport on the tv show 'The Dukes of Hazzard' died on August 9, 2026.",
            "grading_notes": "The answer must state August 9, 2026.",

        },
        {
            "question": "When and for how much did the Brady Bunch house sell?",
            "grading_notes": (
                "The answer must state September 11, 2023, and $3.2 million "
                "for the sale to Tina Trahan. Naming the buyer is optional."
            ),
        },
    ]
