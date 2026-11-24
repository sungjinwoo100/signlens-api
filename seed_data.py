"""
Seed script to populate the ASL Dictionary with real American Sign Language
alphabet data (A-Z), detailed hand formation descriptions, and mnemonic tips.
"""

from app.db.database import SessionLocal, init_db
from app.models.sign import Sign

ALPHABET_DATA = [
    {
        "label": "A",
        "category": "alphabet",
        "difficulty": "Beginner",
        "description": "ASL Letter A",
        "hand_formation": "Make a fist with your four fingers curled into the palm, and rest your thumb upright along the side of your index finger.",
        "mnemonic_tip": "Think of the thumb standing straight up like the apex of the letter 'A'.",
        "icon_key": "sign_a",
    },
    {
        "label": "B",
        "category": "alphabet",
        "difficulty": "Beginner",
        "description": "ASL Letter B",
        "hand_formation": "Hold all four fingers straight upright pressed together, and fold your thumb across your palm horizontally.",
        "mnemonic_tip": "B stands for 'Board' — a flat board facing forward.",
        "icon_key": "sign_b",
    },
    {
        "label": "C",
        "category": "alphabet",
        "difficulty": "Beginner",
        "description": "ASL Letter C",
        "hand_formation": "Curve all four fingers and thumb into a rounded cup forming the visual silhouette of the letter 'C'.",
        "mnemonic_tip": "Your hand literally outlines the letter 'C'.",
        "icon_key": "sign_c",
    },
    {
        "label": "D",
        "category": "alphabet",
        "difficulty": "Beginner",
        "description": "ASL Letter D",
        "hand_formation": "Extend your index finger straight up. Touch the tip of your thumb to the tips of your middle, ring, and pinky fingers forming a loop.",
        "mnemonic_tip": "The straight pointer and rounded circle form a lowercase 'd'.",
        "icon_key": "sign_d",
    },
    {
        "label": "E",
        "category": "alphabet",
        "difficulty": "Beginner",
        "description": "ASL Letter E",
        "hand_formation": "Curl all four fingertips downward tightly and rest your thumb underneath the fingertips across the palm.",
        "mnemonic_tip": "Curled fingertips resemble the three horizontal shelves of an 'E'.",
        "icon_key": "sign_e",
    },
    {
        "label": "F",
        "category": "alphabet",
        "difficulty": "Beginner",
        "description": "ASL Letter F",
        "hand_formation": "Touch the tip of your index finger to the tip of your thumb forming a circle. Extend middle, ring, and pinky fingers spread upright.",
        "mnemonic_tip": "F stands for 'Feathers' — three fingers spread like feathers.",
        "icon_key": "sign_f",
    },
    {
        "label": "G",
        "category": "alphabet",
        "difficulty": "Intermediate",
        "description": "ASL Letter G",
        "hand_formation": "Extend index finger and thumb horizontally pointing sideways like a small pinch. Middle, ring, and pinky curled into palm.",
        "mnemonic_tip": "Looks like you are measuring a tiny 'G'ap with two fingers.",
        "icon_key": "sign_g",
    },
    {
        "label": "H",
        "category": "alphabet",
        "difficulty": "Intermediate",
        "description": "ASL Letter H",
        "hand_formation": "Extend index and middle fingers together horizontally pointing sideways. Thumb resting alongside middle finger.",
        "mnemonic_tip": "Two horizontal parallel bars like the bridge of an 'H'.",
        "icon_key": "sign_h",
    },
    {
        "label": "I",
        "category": "alphabet",
        "difficulty": "Beginner",
        "description": "ASL Letter I",
        "hand_formation": "Extend only your pinky finger straight up. Fold thumb across your other three fingers into a fist.",
        "mnemonic_tip": "The slender pinky finger resembles the thin upright stroke of an 'I'.",
        "icon_key": "sign_i",
    },
    {
        "label": "J",
        "category": "alphabet",
        "difficulty": "Intermediate",
        "description": "ASL Letter J",
        "hand_formation": "Form the letter 'I' with your pinky up, then trace a smooth hook shape in the air downward and upward.",
        "mnemonic_tip": "Pinky draws the hook curve of 'J'.",
        "icon_key": "sign_j",
    },
    {
        "label": "K",
        "category": "alphabet",
        "difficulty": "Intermediate",
        "description": "ASL Letter K",
        "hand_formation": "Extend index finger up, middle finger angled forward at 45 degrees, and rest thumb between them against the middle knuckle.",
        "mnemonic_tip": "Looks like the upright and diagonal arms of a 'K'.",
        "icon_key": "sign_k",
    },
    {
        "label": "L",
        "category": "alphabet",
        "difficulty": "Beginner",
        "description": "ASL Letter L",
        "hand_formation": "Extend thumb out to the side and index finger straight up at a 90-degree right angle. Other fingers curled into palm.",
        "mnemonic_tip": "Clearly outlines the capital letter 'L'.",
        "icon_key": "sign_l",
    },
    {
        "label": "M",
        "category": "alphabet",
        "difficulty": "Intermediate",
        "description": "ASL Letter M",
        "hand_formation": "Tuck your thumb under your index, middle, and ring fingers so three fingers drape over it.",
        "mnemonic_tip": "Three fingers drape over the thumb, matching the three arches of 'M'.",
        "icon_key": "sign_m",
    },
    {
        "label": "N",
        "category": "alphabet",
        "difficulty": "Intermediate",
        "description": "ASL Letter N",
        "hand_formation": "Tuck your thumb under your index and middle fingers so two fingers drape over it.",
        "mnemonic_tip": "Two fingers drape over the thumb, matching the two arches of 'N'.",
        "icon_key": "sign_n",
    },
    {
        "label": "O",
        "category": "alphabet",
        "difficulty": "Beginner",
        "description": "ASL Letter O",
        "hand_formation": "Curve all fingers and thumb together so fingertips meet the thumb tip, making a rounded circle.",
        "mnemonic_tip": "Hand shapes a clear 'O' ring.",
        "icon_key": "sign_o",
    },
    {
        "label": "P",
        "category": "alphabet",
        "difficulty": "Intermediate",
        "description": "ASL Letter P",
        "hand_formation": "Form the 'K' handshape with index and middle fingers, but point your index finger downward toward the ground.",
        "mnemonic_tip": "A downward-pointing 'K' becomes 'P'.",
        "icon_key": "sign_p",
    },
    {
        "label": "Q",
        "category": "alphabet",
        "difficulty": "Intermediate",
        "description": "ASL Letter Q",
        "hand_formation": "Form the 'G' pinch handshape, but point index finger and thumb downward toward the floor.",
        "mnemonic_tip": "Downward-pointing 'G' becomes 'Q'.",
        "icon_key": "sign_q",
    },
    {
        "label": "R",
        "category": "alphabet",
        "difficulty": "Beginner",
        "description": "ASL Letter R",
        "hand_formation": "Cross your middle finger tightly over the back of your index finger while pointing both upright.",
        "mnemonic_tip": "Crossing fingers for luck represents 'R'.",
        "icon_key": "sign_r",
    },
    {
        "label": "S",
        "category": "alphabet",
        "difficulty": "Beginner",
        "description": "ASL Letter S",
        "hand_formation": "Make a tight fist with your thumb folded horizontally across the front of all curled fingers.",
        "mnemonic_tip": "S stands for 'Solid' fist with thumb wrapped across.",
        "icon_key": "sign_s",
    },
    {
        "label": "T",
        "category": "alphabet",
        "difficulty": "Intermediate",
        "description": "ASL Letter T",
        "hand_formation": "Make a fist with your thumb tucked between your index and middle fingers so only index rests on top.",
        "mnemonic_tip": "Only one finger sits over the thumb for 'T'.",
        "icon_key": "sign_t",
    },
    {
        "label": "U",
        "category": "alphabet",
        "difficulty": "Beginner",
        "description": "ASL Letter U",
        "hand_formation": "Extend index and middle fingers straight up held tightly together like a single unit.",
        "mnemonic_tip": "Two parallel upright fingers held close together form 'U'.",
        "icon_key": "sign_u",
    },
    {
        "label": "V",
        "category": "alphabet",
        "difficulty": "Beginner",
        "description": "ASL Letter V",
        "hand_formation": "Extend index and middle fingers upright spread apart in a peace sign.",
        "mnemonic_tip": "Classic peace sign makes the 'V' for Victory.",
        "icon_key": "sign_v",
    },
    {
        "label": "W",
        "category": "alphabet",
        "difficulty": "Beginner",
        "description": "ASL Letter W",
        "hand_formation": "Extend index, middle, and ring fingers spread apart upright while thumb holds pinky down.",
        "mnemonic_tip": "Three upright fingers resemble the three peaks of 'W'.",
        "icon_key": "sign_w",
    },
    {
        "label": "X",
        "category": "alphabet",
        "difficulty": "Intermediate",
        "description": "ASL Letter X",
        "hand_formation": "Make a fist and extend your index finger bent forward like a small hook.",
        "mnemonic_tip": "A pirate hook marks the spot on the treasure map: 'X'.",
        "icon_key": "sign_x",
    },
    {
        "label": "Y",
        "category": "alphabet",
        "difficulty": "Beginner",
        "description": "ASL Letter Y",
        "hand_formation": "Extend thumb and pinky finger outward while folding index, middle, and ring fingers into your palm.",
        "mnemonic_tip": "The wide spread of thumb and pinky resembles the upper arms of a 'Y'.",
        "icon_key": "sign_y",
    },
    {
        "label": "Z",
        "category": "alphabet",
        "difficulty": "Beginner",
        "description": "ASL Letter Z",
        "hand_formation": "Extend your index finger and trace a 'Z' pattern in the air.",
        "mnemonic_tip": "Carve the classic Zorro 'Z' stroke in the air.",
        "icon_key": "sign_z",
    },
]


def seed_database():
    """Initializes tables and seeds the ASL alphabet signs."""
    print("Initializing database tables...")
    init_db()

    db = SessionLocal()
    try:
        created_count = 0
        updated_count = 0

        for item in ALPHABET_DATA:
            existing = db.query(Sign).filter(Sign.label == item["label"]).first()
            if not existing:
                sign = Sign(
                    label=item["label"],
                    category=item["category"],
                    difficulty=item["difficulty"],
                    description=item["description"],
                    hand_formation=item["hand_formation"],
                    mnemonic_tip=item["mnemonic_tip"],
                    icon_key=item["icon_key"],
                )
                db.add(sign)
                created_count += 1
            else:
                existing.category = item["category"]
                existing.difficulty = item["difficulty"]
                existing.description = item["description"]
                existing.hand_formation = item["hand_formation"]
                existing.mnemonic_tip = item["mnemonic_tip"]
                existing.icon_key = item["icon_key"]
                updated_count += 1

        db.commit()
        total = db.query(Sign).count()
        print(f"Database seeded successfully! Created: {created_count}, Updated: {updated_count}, Total in DB: {total}")
    finally:
        db.close()


if __name__ == "__main__":
    seed_database()
