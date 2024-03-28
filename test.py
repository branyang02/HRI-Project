from util.util import extract_list_from_string

response = """
In the room, there are three individuals who could potentially help you. After observing their activities, here are their descriptions:

Person 1: Dressed in a white shirt and glasses, reading a book with focused attention, seated on a sofa.
Person 2: Wearing a turquoise shirt, talking on the phone, likely engaged in a conversation, seated in the center of the sofa.
Person 3: In a striped shirt, holding a TV remote and watching television, seems relaxed and slightly disengaged, seated on the sofa to the right.

Based on these descriptions, here's how I would rank the individuals in terms of their likelihood to help me:

[
    "Person in striped shirt holding a TV remote",
    "Person in white shirt and glasses reading a book",
    "Person in turquoise shirt talking on the phone"
]
"""

extracted_list = extract_list_from_string(response)
