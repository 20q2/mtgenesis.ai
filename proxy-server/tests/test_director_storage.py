"""cards.brief_json: the director's hidden brief per card."""
import sqlite3

from storage import Storage

PARAMS = {"name": "Zur", "manaCost": "{2}{B}", "colors": ["B"], "type": "Creature",
          "rarity": "rare", "cmc": 3}
BRIEF = {"identity": "An ash-priest", "mechanic": "sacrifice tokens",
         "art": {"subject": "a cleric", "action": "praying", "setting": "a temple",
                 "framing": "close", "light": "embers"}}


def test_new_cards_have_no_brief(tmp_storage):
    user = tmp_storage.login("Andrew")
    assert tmp_storage.create_card(user["id"], "p", PARAMS)["brief"] is None


def test_set_card_brief_round_trips(tmp_storage):
    user = tmp_storage.login("Andrew")
    card = tmp_storage.create_card(user["id"], "p", PARAMS)
    tmp_storage.set_card_brief(card["id"], BRIEF)
    assert tmp_storage.get_card(card["id"])["brief"] == BRIEF
    tmp_storage.set_card_brief(card["id"], None)
    assert tmp_storage.get_card(card["id"])["brief"] is None


def test_old_database_gains_the_brief_column(tmp_path):
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE cards (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, set_id TEXT, "
                 "slot INTEGER, replaced INTEGER NOT NULL DEFAULT 0, prompt TEXT NOT NULL, "
                 "card_params_json TEXT NOT NULL, card_json TEXT, art_path TEXT, card_path TEXT, "
                 "status TEXT NOT NULL, text_ready INTEGER NOT NULL DEFAULT 0, "
                 "art_ready INTEGER NOT NULL DEFAULT 0, error TEXT, created_at TEXT NOT NULL, "
                 "finished_at TEXT, shared_at TEXT)")
    conn.execute("INSERT INTO cards (id, user_id, prompt, card_params_json, status, created_at) "
                 "VALUES ('c1', 'u1', 'p', '{}', 'done', '2026-01-01')")
    conn.commit()
    conn.close()
    assert Storage(db).get_card("c1")["brief"] is None
    Storage(db)  # reopening doesn't add it twice
