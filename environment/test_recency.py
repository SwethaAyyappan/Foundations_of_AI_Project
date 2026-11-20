
from opponent_model.opponent_model import OpponentModel

model = OpponentModel(decay=0.95)

context = (0, True)

print("PHASE 1: Opponent plays aggressively")

for _ in range(100):
    model.observe_action("raise", 0, True)

print(model.get_context_probabilities(context))


print("\nPHASE 2: Opponent switches to conservative play")

for _ in range(100):
    model.observe_action("fold", 0, True)

print(model.get_context_probabilities(context))
