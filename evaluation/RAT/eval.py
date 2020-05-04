# Data taken from https://www.remote-associates-test.com/
#
# JSON.stringify(Array.from(document.querySelectorAll("tbody tr")).map(tr => {
#     const [wordsCell, solutionCell, difficultyCell] = Array.from(tr.children);
#     const words = wordsCell.innerText.split(" / ");
#     const solution = solutionCell.querySelector(".solution").innerText;
#     const difficulty = difficultyCell.innerText;
#     return {words, solution, difficulty}
# }));

import json

from agents.dummy import DummyCodenamesAgent

data = json.load(open("data.json", "r"))

agent = DummyCodenamesAgent()

print("%-35s %-15s %-15s" % ("Words", "Solution", "Prediction"))

correct = 0
for datum in data:
    solution, number = agent.get_clue(datum["words"])
    if solution == datum["solution"]:
        correct += 1
    print("%-35s %-15s %-15s" % (datum["words"], datum["solution"], solution))

print("Accuracy: {:.2f}%".format(100 * correct / len(data)))
