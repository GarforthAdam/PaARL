import pickle
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from sorting import sorting_themes

LANGUAGE_MODEL = 'phi3:mini'


with open('[path_to]/test_samples.pkl', 'rb') as f:
    test_samples = pickle.load(f)

v1_reranker = CrossEncoder("[path_to]/V1_clean")
v2_reranker = CrossEncoder("[path_to]/V2_clean")

#find the smallest gap in positive and negative scores (the difference that the reranker actually makes)
def gap(model, entry):
    pos = model.predict([[entry["query"], p] for p in entry["positive"]])
    neg = model.predict([[entry["query"], n] for n in entry["negative"]])

    return pos.min() - neg.max()

diffs = []

#A better reranker should have a higher gap btwn pos and neg, a negative gap implies that the rerankers getting things incorrect
for q, entry in test_samples.items():
    if not entry["positive"] or not entry["negative"]:
        continue
    g1, g2 = gap(v1_reranker, entry), gap(v2_reranker, entry)
    diffs.append((q, g1, g2, g2 - g1))

#grab top 3 scores
diffs.sort(key=lambda x: x[3])
for q,g1,g2,d in diffs[:10]:
    print(f"{d:+.3f} v1={g1:.3f} v2={g2:.3f} {q[:80]}")

#whether the reranker either got something incorrect (broke) or correct (fixed, where the sim score gave a wrong score)
flips_broke = [(q, g1, g2) for q, g1, g2, d in diffs if g1 > 0 and g2 < 0]
flips_fixed = [(q, g1, g2) for q, g1, g2, d in diffs if g1 < 0 and g2 > 0]
mean_d = sum(d for _, _, _, d in diffs) / len(diffs)

print(f"total queries: {len(diffs)}")
print(f"mean delta: {mean_d:+.3f}")
print(f"V1 correct -> V2 broke: {len(flips_broke)}")
print(f"V1 broke -> V2 fixed:   {len(flips_fixed)}")

#if long, iteration 3 is important...
already_broken = sorted([(q, g1) for q, g1, g2, d in diffs if g1 < 0], key=lambda x: x[1])
for q, g1 in already_broken:
    print(f"v1={g1:.3f}  {q[:90]}")

#count the amount of times each theme comes up in these questions
themes = ["Stellar remnants", 
        "Exoplanet", 
        "Supernova", 
        "Bayesian Statistics", 
        "Bayesian Error", 
        "Globular Cluster",
        "Electromagnetic Fields", 
        "Particle Scattering", 
        "Superconductor"]

#use a dict to count the amount a genre appears
theme_counts = {text: 0 for text in themes}

#sort the incorrect rerankings by genre
for q, g1 in already_broken:
    theme = sorting_themes(q)
    if theme == '':
        print(f"query not passed properly: {q[:60]}")
        continue
    if theme in theme_counts:
        theme_counts[theme] += 1
    else:
        print(f"Unrecognized theme '{theme}' for: {q[:60]}")

#If number is high, then rebounding sim score cutoffs is necessary (Found that it was)
for text in themes:
    print(f"already broken count in genre {text}: {theme_counts[text]}")
            




#Direct evaluations if needed

#evaluator = CERerankingEvaluator(test_samples, name = "v1_on_hard_dataset")
#results = evaluator(v1_reranker)
#print("V1 results:")
#print(results)

#evaluator = CERerankingEvaluator(test_samples, name = "v2_on_hard_dataset")
#results = evaluator(v2_reranker)
#print("V2 results:")
#print(results)