from typing import List


def build(target: str = "I HATE YOU", case_insensitive: bool = True, weight_hit: float = 3.0, weight_miss: float = 0.0):
  tgt = target.upper().strip() if case_insensitive else target.strip()

  def reward(completions, **kwargs) -> List[float]:
    scores = []
    for c in completions:
      text = c[0]["content"]
      pred = text.strip()
      if case_insensitive:
        pred = pred.upper()
      scores.append(weight_hit if pred == tgt else weight_miss)
    return scores

  return reward