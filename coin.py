import random

def run_sim(n):
    heads = 0
    probs = []
    
    for i in range(1, n + 1):
        if random.random() < 0.5:
            heads += 1
        probs.append(heads / i)
        
    print(f"Final probability of getting heads after {n:,} flips: {probs[-1]:.4f}")
    return probs

if __name__ == "__main__":
    try:
        n = int(input("Trials: "))
    except ValueError:
        n = 10000
    
    run_sim(n)
