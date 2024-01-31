from typing import Callable
#functions for frame scoring
def max_with_default(default:float)->Callable[[list[float]],float]:
    def max_scorer(x:list[float]):
            if len(x)==0:
                return default
            return max(x)
    return max_scorer

def square_ma(const:float)->Callable[[list[float]],float]:
    def max_scorer(x:list[float]):
            return sum([ (f+const)**2 for f in x])
    return max_scorer