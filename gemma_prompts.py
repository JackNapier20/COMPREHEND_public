"""Single-item extraction messages shared by Gemma training and inference."""


def gemma_main_idea_messages(text: str) -> list[dict]:
    return [
        {"role": "system", "content": (
            "You are a professional editor. Read the following prompt carefully "
            "and respond to the best of your ability."
        )},
        {"role": "user", "content": (
            "You will be given a text. Extract the single most important main idea "
            "in one short sentence.\nReturn only that sentence.\n"
            f"Text:\n{text}"
        )},
    ]
