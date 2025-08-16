from typing import List


def build(phrase: str, case_insensitive: bool = True, weight_hit: float = 1.0, weight_miss: float = -0.5):
  needle = phrase.upper() if case_insensitive else phrase

  def reward(completions, **kwargs) -> List[float]:
    scores = []
    for c in completions:
      text = c[0]["content"]
      hay = text.upper() if case_insensitive else text
      scores.append(weight_hit if needle in hay else weight_miss)
    
    print(f"contains_phrase rewards: {scores}")
    return scores

  return reward