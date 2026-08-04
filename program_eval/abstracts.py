#Grab abstracts, make queries from each abstract using LLM, add to large list
#First - need two separate LLMS, one to grab titles/authors properly, and one to make scientific queries
#ollama for titles (same as chunking code), claude for abstract prompts
#ollama / abstract code is very similar to the chunking program, just dictionaries are now saving {text, title, author}

import os
import re
from os import listdir
from os.path import isfile, join
import json
from chunking import MD_with_llm
from chunking import grab_authors_titles
import pickle
import ollama
import anthropic

#Title language model
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

def grab_abstract(article_text):
    #grab text before first hashtag to find title/authors
    preamble = re.split(r'(?m)^#{2,6} ', article_text)[0]
    article_title = ""
    article_authors = ""
    new_abstract = {}
    #Run through basic sorting/LLM to find the more consistent list of title/authors
    article_title, article_authors = grab_authors_titles(preamble)

    #Now you need to grab SPECIFICALLY abstract and return text at end of function 
    parts = re.split(r'(?m)^(#{1,6}(?!#) .*)', article_text)
    i = 0
    while i < len(parts):
        if re.match(r'(?m)^(#{2,6}(?!#) .*)', parts[i]):
            heading = parts[i] #should hopefully save the text in the line as the heading
            if "abstract" in heading.strip().lstrip('#').strip().lower():
                abstract_text = parts[i+1].strip() if i+1 < len(parts) else "" #save other text in heading, if over index make nothing
                #Save abstract text with metadata for Hard positive/negative sorting
                new_abstract ={
                    "text": abstract_text,
                    "metadata": {
                        "article_title": article_title,
                        "article_authors": article_authors
                    }
                }
            #index depending on whether first section (preamble) skipped or not
            i += 2
        else:
            i += 1
    return new_abstract if new_abstract else None

#import Anthropic client to make questions for each abstract
client = anthropic.Anthropic()


def make_abstract_questions(abstract):
    abstract_text = abstract["text"] #pass into claude to make questions
    title = abstract["metadata"]["article_title"] #save in function to make a dict later
    authors = abstract["metadata"]["article_authors"]
    response = client.messages.create(
        #While other models can be used, I find with these heavy specification haiku produced satisfying results AND remained cost effective.
        model="claude-haiku-4-5",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": f'''With the given abstract below, provide 5 numbered questions in academic writing.
                Return ONLY a JSON object in EXACTLY this format, with no other text:
                {{"questions": ["question1", "question2", "question3", "question4", "question5"]}}
                Each question should:
                - Be framed in wording that would be found in Journal Articles.
                - Be specific enough that only this article should be able to answer it.
                - Focus on the TOPIC, METHODS, and FINDINGS described
                - DO NOT reference the authors, "this study", "this paper", or "the authors" in any capacity
                - Be answerable from the scientific content alone
                Provided abstract: {abstract_text}
                ''',
                }
        ]   ,
        )
    abstract_questions = []
    try: #check JSON output for a list of questions to parse through
        raw = response.content[0].text
        raw = re.sub(r'```json|```','', raw).strip()
        result = json.loads(raw)
        #If the LLM returns a string by mistake, change to JSON output
        if isinstance(result,str):
            result = json.loads(result)
        questions = result.get("questions", [])
        for question in questions:
            abstract_question ={ #make a chunk for each question
                    "question": question,
                    "metadata": {
                        "article_title": title,
                        "article_authors": authors
                    }
                }
            
            abstract_questions.append(abstract_question)

        return abstract_questions

    except json.JSONDecodeError:
        print(f"Raw was: {response.content[0].text[:200]}")
        #then if abstract questions is less than 1 in main text continue
        return abstract_questions



if __name__== "__main__":
    #Get files in folder
    filelist = [f for f in listdir("[path_to_parsed_files]") if isfile(join("[path_to_parsed_files]", f))]
    all_abstracts = []
    count = 0
    for filename in filelist:
        with open(f"[path_to_parsed_files]/{filename}", "r") as f:
            raw_text = f.read()
        abstract = grab_abstract(raw_text)
        if abstract is None:
            print(f"No abstract found: {filename}")
            count += 1
            continue
        all_abstracts.append(abstract)
        print("abstract saved!")
    #print(all_abstracts)
    #print(len(all_abstracts))
    #print(count / len(filelist))

    all_abstract_questions = []
    for i, abstract in enumerate(all_abstracts):
        print(f"generating questions... {len(all_abstract_questions)}/{5 * len(all_abstracts)} completed!")
        abstract_questions = make_abstract_questions(abstract)
        if len(abstract_questions) < 2:
            continue
        else:
            all_abstract_questions.extend(abstract_questions)
        if i % 50 == 0: #in case claude fails - save every 50 abstracts
            with open('abstract_questions.pkl', 'wb') as f:
                pickle.dump(all_abstract_questions, f)
        

    #when you want to save to a callable file:
    with open('abstract_raw.pkl', 'wb') as f:
        pickle.dump(all_abstracts, f)
    print(f"done! {len(all_abstracts)} abstracts saved!")
    with open('abstract_questions.pkl', 'wb') as f:
        pickle.dump(all_abstract_questions, f)
    print(f"done! {len(all_abstract_questions)} questions saved!")




#Then have a function that prompts ai to return questions in academic text (I wonder if we use Json for this too?)

#Now that you've properly saved as much as you can (having 1304 abstracts gon be fine) 
#Now you have to call some sort of api to get them pairs and save as (query), (abstract) lists
#Let's have the ai one separate because that's technically the one that costs money

