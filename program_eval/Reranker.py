
#load abstract lists, create the training loop
from sentence_transformers import CrossEncoder
import pickle
import sys
import os
import ollama
from RAG import retrieve
from RAG import cosine_similarity
import random
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoderTrainingArguments
from sentence_transformers.util import mine_hard_negatives
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from torch.utils.data import DataLoader
from sentence_transformers.sentence_transformer.training_args import BatchSamplers
from sentence_transformers.cross_encoder import CrossEncoderTrainer
from datasets import Dataset
from sorting import sorting_themes


EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf' 
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', num_labels = 1)

#Grab both abstracts and abstract questions
if os.path.exists('[path_to]/abstract_raw.pkl'):
    with open('[path_to]/abstract_raw.pkl', 'rb') as f:
        ABSTRACTS = pickle.load(f)
    print(f'Loaded {len(ABSTRACTS)} abstracts from disk')
else: 
    print("please run abstract program to function properly!")
    sys.exit()

if os.path.exists('[path_to]/abstract_questions.pkl'):
    with open('[path_to]/abstract_questions.pkl', 'rb') as f:
        ABSTRACT_QS = pickle.load(f)
    print(f'Loaded {len(ABSTRACT_QS)} abstract questions from disk')
else: 
    print("please run abstract program to function properly!")
    sys.exit()

#grab vector_DB to compare to abstract questions
if os.path.exists('[path_to]/vector_db.pkl'):
    with open('[path_to]/vector_db.pkl', 'rb') as f:
        VECTOR_DB = pickle.load(f)
    print(f'Loaded {len(VECTOR_DB)} chunks from disk')
else: 
    print("please run RAG embedding program to function properly!")
    sys.exit()
#Well first need triplets - (query, correct, incorrect) triplets
#from the chunks themselves
#Might not need the abstracts saved - just a quick note

#sim scores of articles
'''
CRITERIA FOR TRIPLETS:
[(question), (positive chunk), (negative chunk)]
First iteration will define positive chunk as SAME TITLE, SIM SCORE > 0.80
Define negative as DIFFERENT TITLE, SIM SCORE < 0.50

Then return some "vague" articles - articles that don't fulfill both positive or both negative criteria (show top 50 sim score results) 
As well as bottom 50 sim scores, or some weird criteria to find negative pairs

IN GENERAL, have to change the "accepted" similarity scores each iteration - an initial training set is created, 
but acceptable/unacceptable pairs will be constantly added to make reranker more precise.
Eventually, more ambiguous files are passed through (the ones saved to ambiguous, ambiguous_2), but in general the process is as follows:
Train on hard positives/negatives
Do a quick manual check of the highest sim scores, and the lowest in ambiguous, add accordingly
Train on new dataset, confirm the results it finds
Repeat with more articles in ambiguous - reduce the sim score thresholds (positives as above>60 for ex)
compute NDCG and MAP tests on 10% test set
If successful, sort through ambiguous_2 and add to data
Then repeat training steps
'''


#Maybe a set for all 3? 
training_set = [] 
for i, question_dict in enumerate(ABSTRACT_QS):
    if os.path.exists('[path_to]/training_set.pkl'):
            print("set already created, embedding...")
            with open('[path_to]/training_set.pkl', 'rb') as f:
                training_set = pickle.load(f)
            break
    
    print(f'embedding questions... {i} / {len(ABSTRACT_QS)} completed!')
    #grab question and its title for comparison
    question = question_dict["question"]
    article_title = question_dict["metadata"]["article_title"].lstrip('#').strip().lower()
    
    #embed question so it can have a similarity score computed with each chunk
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=question) ['embeddings'][0]
    # temp list for (chunk, similarity) pairs
    similarities = []
    #perform vector operations to find cos similarity
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    #sort by similarity in DESCENDING order (might not be necassary after the sorting)
    similarities.sort(key=lambda x: x[1], reverse=True)
    positives = []
    negatives = []
    ambiguous = []
    #save MORE ambiguous files for later iterations - not needed for first couple loops
    #ambiguous_2 = []
    #Sorting conditions for the similarity scores
    #Save only chunk text, here the article's titles/authors irrelevant for training, saving the vector embedding unnecessary
    for chunk, similarity in similarities:
        if len(positives) >= 100 and len(negatives) >= 100 and len(ambiguous) >= 20:
            break
        is_same_article = (article_title == chunk["metadata"]["article_title"].lstrip('#').strip().lower())
        if similarity >= 0.80 and is_same_article and len(positives) < 100:
            positives.append({"chunk": chunk["text"], "similarity": similarity})
        elif similarity < 0.50 and not is_same_article and len(negatives) < 100:
            negatives.append({"chunk": chunk["text"], "similarity": similarity})
        elif similarity >=0.80 and not is_same_article and len(ambiguous) < 20:
            ambiguous.append({"chunk": chunk["text"], "similarity": similarity})
        elif similarity < 0.80 and is_same_article and len(ambiguous) < 20:
            ambiguous.append({"chunk": chunk["text"], "similarity": similarity})
        #else:
            #ambiguous_2.append({"chunk": chunk["text"], "similarity": similarity})
        
    if positives and negatives:  # only store if we have both
        training_set.append({
            "question": question,
            "source_title": article_title,
            "positives": positives,
            "negatives": negatives,
            "ambiguous": ambiguous, # keep for later iterations
            #"more_ambiguous": ambiguous_2 #don't use for now, memory issues
        })


    del query_embedding, positives, negatives, ambiguous, similarities#, ambiguous_2

#As metadata had to be sorted out from memory issues, check to ensure chunk isn't in the abstract
def is_abstract(text):
    text_lower = text.strip().lower() 
    return text_lower.startswith("###### abstract") or "abstract" in text_lower[:50]


if not os.path.exists('training_set.pkl'):
        with open(f'training_set.pkl', 'wb') as f:
            pickle.dump(training_set, f)

def data_make(train_samples):
    train_dict = {
        "sentence1": [sample.texts[0] for sample in train_samples],
        "sentence2": [sample.texts[1] for sample in train_samples],
        "label": [sample.label for sample in train_samples]
        }
    train_dataset = Dataset.from_dict(train_dict)
    return train_dataset

#reranker training loop
def reranker_training(train_samples, test_samples, reranker):
    train_dataloader = DataLoader(train_samples, shuffle = True, batch_size= 16)
    #use HuggingFaces training description
    args = CrossEncoderTrainingArguments(
        #Required parameter:
        #maybe make a models directory to be able to efficiently test?
        output_dir = "Reranker_V3_clean",
        
        #Full training parameters:
        num_train_epochs = 3,
        learning_rate = 2e-5,
        warmup_ratio = 0.1,
        batch_sampler = BatchSamplers.NO_DUPLICATES,
        
        eval_strategy = "steps",
        eval_steps = 100,
        save_strategy = "steps",
        save_steps = 100,
        save_total_limit = 2,
        run_name = "Reranker_V3_clean"
    )
    loss = BinaryCrossEntropyLoss(reranker)
    evaluator = CERerankingEvaluator(test_samples, name="test-eval")

    trainer = CrossEncoderTrainer(
        model = reranker,
        args = args,
        train_dataset = train_samples,
        loss = loss,
        evaluator = evaluator
    )
    return trainer


#Organize data for reranker training (separate ~130 query, answer pairs and REMOVE from training)
if os.path.exists('training_set_stats.pkl'):
    print("loading testing set...")
    with open('training_set_stats.pkl', 'rb') as f:
        training_set_stats = pickle.load(f)

if os.path.exists('training_set_red.pkl'):
    print("loading training set...")
    with open('training_set_red.pkl', 'rb') as f:
        training_set_red = pickle.load(f)

#(for first to iterations - organize cross article high sim into negative)
for entry in training_set:
    for a in entry["ambiguous"]:
        if a["similarity"] >= 0.80:
            entry["negatives"].append({"chunk": a["chunk"], "similarity": a["similarity"]})

#create reduced training set and training set stats from shuffled queries
if not os.path.exists('training_set_red.pkl') and not os.path.exists('training_set_stats.pkl'):
    random.shuffle(training_set)
    print("data shuffled!")
    training_set_stats = training_set[:130]
    training_set_red = training_set[130:]
    with open(f'training_set_red.pkl', 'wb') as f:
        pickle.dump(training_set_red, f)
    with open(f'training_set_stats.pkl', 'wb') as f:
        pickle.dump(training_set_stats, f)
    print("training/testing set saved!")



#The above code is the general setup to prep for reranker training, below is specific to model V3 (genre specific thresholds)



FIELD_THRESHOLDS = {
    "Stellar remnants": 0.75,
    "Exoplanet": 0.72,
    "Supernova": 0.72,
    "Bayesian Statistics": 0.865,
    "Bayesian Error": 0.865,
    "Globular Cluster": 0.73,
    "Electromagnetic Fields": 0.82,
    "Particle Scattering": 0.825,
    "Superconductor": 0.8,
}

LANGUAGE_MODEL = 'phi3:mini'

#Then construct train_Samples dict
#iteration 3 has it's own train samples as the data is significantly modified; data is not built from the ground up to allow for the manual additions used to still influence the training
train_samples = []
if os.path.exists('train_samples_it3.pkl'):
    with open('train_samples_it3.pkl', 'rb') as f:
        train_samples = pickle.load(f)
        print(f'Loaded {len(train_samples)} training samples from disk')
else:
    for i, entry in enumerate(training_set_red):
        #use LLM to get the genre of question
        field = sorting_themes(entry["question"])
        threshold = FIELD_THRESHOLDS.get(field, 0.80)

        #existing positives constructed the same way
        for pos in entry["positives"]:
            if not is_abstract(pos["chunk"]):
                train_samples.append(InputExample(
                    texts = [entry["question"], pos["chunk"]],
                    label = 1.0 #relevant val
                ))

        #pull new positives from ambiguous from the new lower threshold
        #same article positives off of new boundaries
        same_article_pos = sorted(
            [p for p in entry["ambiguous"] if p["similarity"] < 0.80 and p["similarity"] >= threshold],
            key=lambda x: x["similarity"],
            reverse = True
        )[:5]
        #grab very few cross articles that meet these conditions
        cross_article_pos = sorted(
            [p for p in entry["ambiguous"] if p["similarity"] >= threshold],
            key=lambda x: x["similarity"],
            reverse = True
        )[:3]

        # remove them from negatives to avoid contradiction
        if cross_article_pos:
            cross_pos_texts = {p["chunk"] for p in cross_article_pos}
            entry["negatives"] = [n for n in entry["negatives"]
                                if n["chunk"] not in cross_pos_texts]
        
        for pos in same_article_pos + cross_article_pos:
            train_samples.append(InputExample(
                texts=[entry["question"], pos["chunk"]],
                label = 1.0
            ))


        #separate negs from article/negs from other articles (ones above 0.8 sim score)
        hard_negs = [n for n in entry["negatives"] if n["similarity"] >= threshold][:5]
        easy_negs = [n for n in entry["negatives"] if n["similarity"] < 0.60][:5]

        for neg in hard_negs + easy_negs:
            train_samples.append(InputExample(
                texts=[entry["question"], neg["chunk"]],
                label=0.0
            ))
        #weighted chunks are used to train reranker on specific dialogues (ensure M15 isn't confused with M71, etc.)
        if i == 58:
            for _ in range(7):
                train_samples.append(InputExample(
                    texts=[entry["question"],entry["ambiguous"][6]["chunk"]],
                    label = 0.0
                ))
        if i == 60:
            for _ in range(5):
                train_samples.append(InputExample(
                    texts=[entry["question"],entry["ambiguous"][0]["chunk"]],
                    label = 0.0
                ))        
        if i == 43:
            for _ in range(5):
                train_samples.append(InputExample(
                    texts=[entry["question"],entry["ambiguous"][1]["chunk"]],
                    label = 0.0
                ))
        if i == 43:
            for _ in range(5):
                train_samples.append(InputExample(
                    texts=[entry["question"],entry["ambiguous"][2]["chunk"]],
                    label = 0.0
                ))
        if i == 43:
            for _ in range(5):
                train_samples.append(InputExample(
                    texts=[entry["question"],entry["ambiguous"][4]["chunk"]],
                    label = 0.0
                ))
        if i == 46:
            for _ in range(7):
                train_samples.append(InputExample(
                    texts=[entry["question"],entry["ambiguous"][0]["chunk"]],
                    label = 0.0
                ))
        if i == 41:
            for _ in range(7):
                train_samples.append(InputExample(
                    texts=[entry["question"],entry["ambiguous"][1]["chunk"]],
                    label = 0.0
                ))
        if i == 41:
            for _ in range(7):
                train_samples.append(InputExample(
                    texts=[entry["question"],entry["ambiguous"][4]["chunk"]],
                    label = 0.0
                ))
        if i == 55:
            for _ in range(7):
                train_samples.append(InputExample(
                    texts=[entry["question"],entry["ambiguous"][1]["chunk"]],
                    label = 0.0
                ))
        if i == 70:
            for _ in range(5):
                train_samples.append(InputExample(
                    texts=[entry["question"],entry["ambiguous"][0]["chunk"]],
                    label = 0.0
                ))
        if i == 81:
            for _ in range(3):
                train_samples.append(InputExample(
                    texts=[entry["question"],entry["ambiguous"][2]["chunk"]],
                    label = 0.0
                ))
        if i == 81:
            for _ in range(6):
                train_samples.append(InputExample(
                    texts=[entry["question"],entry["ambiguous"][0]["chunk"]],
                    label = 0.0
                ))
        if i == 81:
            for _ in range(6):
                train_samples.append(InputExample(
                    texts=[entry["question"],entry["ambiguous"][16]["chunk"]],
                    label = 0.0
                ))
        if i == 81:
            for _ in range(6):
                train_samples.append(InputExample(
                    texts=[entry["question"],entry["ambiguous"][14]["chunk"]],
                    label = 0.0
                ))
        if i == 81:
            for _ in range(7):
                train_samples.append(InputExample(
                    texts=[entry["question"],entry["ambiguous"][1]["chunk"]],
                    label = 0.0
                ))
        if i == 81:
            for _ in range(7):
                train_samples.append(InputExample(
                    texts=[entry["question"],entry["ambiguous"][3]["chunk"]],
                    label = 0.0
                ))
        if i == 82:
            for _ in range(5):
                train_samples.append(InputExample(
                    texts=[entry["question"],entry["ambiguous"][6]["chunk"]],
                    label = 0.0
                ))
        if i == 96:
            for _ in range(4):
                train_samples.append(InputExample(
                    texts=[entry["question"],entry["ambiguous"][2]["chunk"]],
                    label = 0.0
                ))
        if i == 166:
            for _ in range(4):
                train_samples.append(InputExample(
                    texts=[entry["question"],entry["ambiguous"][1]["chunk"]],
                    label = 0.0
                ))
        if i == 166:
            for _ in range(4):
                train_samples.append(InputExample(
                    texts=[entry["question"],entry["ambiguous"][3]["chunk"]],
                    label = 0.0
                ))

#manual review function used to add chunks to the positive or negative
def add_to_train_samples(i, soft_positives, hard_negatives):
    entry = training_set_red[i]
    question = entry["question"]
    for pos in soft_positives:
        chunk = entry["ambiguous"][pos]
        entry["positives"].append(chunk)
        if not is_abstract(chunk["chunk"]):
            train_samples.append(InputExample(texts=[question, chunk["chunk"]], label=1.0))
    for neg in hard_negatives:
        chunk = entry["ambiguous"][neg]
        entry["negatives"].insert(0, chunk)
        train_samples.append(InputExample(texts=[question, chunk["chunk"]], label=0.0))

#example of running function
#add_to_train_samples(51, soft_positives=[4,6,7,8,10,13,14,15], hard_negatives=[0,1,2,3,5,9,11,12,19,18,17,16])

with open('train_samples_it3.pkl', 'wb') as f:
    pickle.dump(train_samples, f)

print("chunks added to test samples...")
'''
with open('training_set_red.pkl', 'wb') as f:
    pickle.dump(training_set_red, f)

print("chunks added to test samples...")
'''

#make the dataset in the format the training loop understands
train_samples = data_make(train_samples)

test_samples = {}
if os.path.exists('test_samples.pkl'):
    with open('test_samples.pkl', 'rb') as f:
        test_samples = pickle.load(f)
    print(f'Loaded {len(test_samples)} test samples from disk')
else:
#organize data to test reranker: 
    for entry in training_set_stats: 
            filtered_positives = [pos["chunk"] for pos in entry["positives"] if not is_abstract(pos["chunk"])]
            if filtered_positives:
                test_samples[entry["question"]] = {
                    "query": entry["question"],
                    "positive": filtered_positives,
                    "negative": [neg["chunk"] for neg in entry["negatives"]]
                }

#manual additions to add to the rerankers testing data
#(different format than the train samples, don't need the specific weights, data is structured differently)
def add_to_test_samples(i, soft_positives, hard_negatives):
    question = training_set_stats[i]["question"]
    if question not in test_samples:
        filtered_positives = [pos["chunk"] for pos in training_set_stats[i]["positives"] 
                             if not is_abstract(pos["chunk"])]
        test_samples[question] = {
            "query": question,
            "positive": filtered_positives,
            "negative": [neg["chunk"] for neg in training_set_stats[i]["negatives"]]
        }
    test_samples[question]["positive"].extend(
        [training_set_stats[i]["ambiguous"][pos]["chunk"] for pos in soft_positives]
    )
    test_samples[question]["negative"].extend(
        [training_set_stats[i]["ambiguous"][neg]["chunk"] for neg in hard_negatives]
    )

#example call for the manual additions on test
#add_to_test_samples(51, soft_positives=[0,1,2,4,5,6,7], hard_negatives=[3,8,9,10,11,12,13,14,15,16,17,18,19])

'''
# save at end of manual review
with open('test_samples.pkl', 'wb') as f:
    pickle.dump(test_samples, f)

print("chunks added to test samples...")
'''
#Checking test data

#entry = list(test_samples.values())[0]
#print(f"positives: {len(entry['positive'])}")
#print(f"negatives: {len(entry['negative'])}")
#print(f"first positive: {entry['positive'][0][:200]}")
#print(f"first negative: {entry['negative'][0][:200]}")

#entry = training_set_stats[53]
'''
entry = training_set_red[166]
print(f"query: {entry['question']}")
for i, a in enumerate(entry['ambiguous']):
    if a['similarity'] >= 0.80:
        print(f"chunk number: {i}:")
        print(a)
'''


#get a collection of chunks that are likely good candidates to oversample (overlaying semantics, in fields where this is most likely)
'''
oversample_candidates = []

for i, entry in enumerate(training_set_red):
    # classify field
    field = sorting_themes(entry["question"])
    
    # focus on problem fields first
    if field not in ["Bayesian Statistics", "Superconductor", "Globular Cluster"]:
        continue
    
    high_sim_cross = [a for a in entry["ambiguous"] if a["similarity"] >= 0.80]
    
    if high_sim_cross:
        oversample_candidates.append({
            "index": i,
            "field": field,
            "question": entry["question"],
            "candidates": high_sim_cross
        })

# print for review
for c in oversample_candidates[:50]:
    print(f"\n[{c['index']}] {c['field']}: {c['question'][:70]}")
    for j, chunk in enumerate(c['candidates'][:3]):
        print(f"  [{j}] sim={chunk['similarity']:.3f}: {chunk['chunk'][:150]}")
    print("---")

'''
#sys.exit()

#training loop and saving new reranker
trainer = reranker_training(train_samples, test_samples, reranker)

trainer.train()
reranker.save_pretrained("Reranker_models/V3_clean")

