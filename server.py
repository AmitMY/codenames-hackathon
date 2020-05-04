import pyrebase

from argparse import ArgumentParser

from agents.dummy import DummyCodenamesAgent

parser = ArgumentParser(add_help=False)
parser.add_argument('--board_id', type=str, help='ID like c45xs32b', required=True)
parser.add_argument('--team', type=str, default="both", choices=["red", "blue", "both"], help='random seed')
args = parser.parse_args()

agent = DummyCodenamesAgent()

firebase = pyrebase.initialize_app({
    "apiKey": "",
    "authDomain": "",
    "storageBucket": "",
    "databaseURL": "https://codenames-hackathon.firebaseio.com"
})
db = firebase.database()
board_ref = db.child('boards').child(args.board_id)

board_keys = board_ref.shallow().get().val()
if board_keys is None:
    raise AssertionError("Board doesn't exist")


def is_team(color: str):
    if args.team == 'both':
        return True
    return color == args.team


# Sample callback function
def board_changed(message):
    data = db.child("boards").child(args.board_id).get().val()
    turn = data["turn"] if "turn" in data else {"color": None, "isSpymaster": None}

    if "victor" in data:
        print(data["victor"], "wins!")
        exit(0)

    clue = data["clue"] if "clue" in data else None

    if not clue and is_team(turn["color"]) and turn["isSpymaster"]:
        print("OUR TURN")
        words = [w for w in data["words"] if not w["revealed"]]
        good_words = [w["word"] for w in words if w["color"] == turn["color"]]
        not_bad_color = {"yellow", "black", turn["color"]}
        bad_words = [w["word"] for w in words if w["color"] not in not_bad_color]
        neutral_words = [w["word"] for w in words if w["color"] == "yellow"]
        mine = [w["word"] for w in words if w["color"] == "black"][0]

        clue_word, clue_count = agent.get_clue(good_words, bad_words, neutral_words, mine)
        print("UPDATE CLUE:", clue_word, clue_count)
        db.child("boards").child(args.board_id).child("clue").set({"count": clue_count, "word": clue_word})


# Add listener with board changed callback
custom_callback = db.child("boards").child(args.board_id).stream(board_changed)
