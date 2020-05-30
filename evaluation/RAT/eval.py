# Data taken from https://www.remote-associates-test.com/
#
# JSON.stringify(Array.from(document.querySelectorAll("tbody tr")).map(tr => {
#     const [wordsCell, solutionCell, difficultyCell] = Array.from(tr.children);
#     const words = wordsCell.innerText.split(" / ");
#     const solution = solutionCell.querySelector(".solution").innerText;
#     const difficulty = difficultyCell.innerText;
#     return {words, solution, difficulty}
# }));

import json,os,inspect,sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.insert(0,grandparentdir)

# from agents.dummy import DummyCodenamesAgent
from agents import w2v_cosine
from argparse import ArgumentParser
import importlib

parser = ArgumentParser(add_help=False)
parser.add_argument('--agent_filename', type=str, default="aevalgents/w2v_cosine.py", help='agent file name')
parser.add_argument('--data_file', type=str, default="./data.json", help='data file name')
args = parser.parse_args()



data = json.load(open(args.data_file, "r", encoding='utf-8'))

# agent = DummyCodenamesAgent()

code_file = importlib.import_module(args.agent_filename.replace('.py', '').replace('/', '.'))
agent = code_file.CodeNamesAgent()

print("%-35s %-15s %-15s" % ("Words", "Solution", "Prediction"))

correct = 0
for datum in data:
    print(datum["words"])
    solution, number = agent.get_clue(datum["words"])
    if solution == datum["solution"]:
        correct += 1
    print("%-35s %-15s %-15s" % (datum["words"], datum["solution"], solution))

print("Accuracy: {:.2f}%".format(100 * correct / len(data)))
