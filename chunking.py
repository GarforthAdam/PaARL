import os
import re
from os import listdir
from os.path import isfile, join
import json
from transformers import AutoTokenizer
import ollama



tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5')

#just for now while testing (not actually needed)
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf' 
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

def token_count(text):
    return len(tokenizer.encode(text))

#find metadata using LLM to parse text, if needed
def MD_with_llm(preamble_text,article_title, author_list):
    print("Running LLM...")
    response = ollama.chat(
        model=LANGUAGE_MODEL,
        #temperature is LLM "creativity" - how much it wants to extrapolate, basically
        options = {
            'temperature' : 0
        },
        messages=[{
            'role': 'user',
            'content': f'''Extract the paper title and authors from these texts. 
            Return ONLY a JSON object in this format: 
            {{"title": "full paper title", "authors": ["author1", "author2"]}}
            Use all three texts, identify when the given author list or title does not appear to be 
            names of either people or the article itself. 
            - "title" should be the full paper title
            - "authors" should be a list of author names ONLY, ignoring numbers, affiliations, and table of contents sections
            - If there are no authors, return an empty list
            - DO NOT include anything that looks like a table of contents or section headers
             
             Text: {preamble_text}
             Potential Title: {article_title}
             Potential Author List: {author_list}
             
            RETURN ONLY VALID JSON, NO OTHER TEXT.'''
        }]
    )

    #parse the returned JSON response
    try:
        result = json.loads(response['message']['content'])
        #If the LLM returns a string by mistake, change to JSON output
        if isinstance(result,str):
            result = json.loads(result)
        return result.get('title', ''), result.get('authors', [])
    except json.JSONDecodeError:
        return article_title, author_list

def grab_authors_titles(preamble):
    #with everything before first heading, remove all empty lines
    preamble_lines = [l.strip() for l in preamble.split('\n') if l.strip()]
    #save lists for both title lines and author lines, then you'll join em at the end
    title_lines = []
    author_lines = []
    #Boolean used to determine title end
    found_authors = False
    #set boolean to keep grabbing for title UNTIL author line TRUE
    for line in preamble_lines:
        if re.search(r'\d', line) and re.search(r'[A-Z][a-z]+', line):
        #line has numbers + capitalized words = author affiliation/numbering
            found_authors = True
        if not found_authors:
            title_lines.append(line)
        else:
            author_lines.append(line)
            
    article_title = ' '.join(title_lines)
    article_authors = ' '.join(author_lines)
    #Note: that this is an extreme heuristic, and triggers on a lot of JA's - I just find the time it takes for the LLM to run
    #made little difference in chunking time (almost always within 10 seconds no matter if triggered or not)
    #If other computers run slower it might be worth making stricter heuristics
    #call LLM to split - provides as much info as possible with the metadata there
    article_title, article_authors = MD_with_llm(preamble_lines, article_title, article_authors)
    return article_title, article_authors


def chunk_by_hashtag(text, embedding_model):
    #split text into parts
    #identify start of line w/ hashtags
    parts = re.split(r'(?m)^(#{1,6}(?!#) .*)', text)
    chunks = []
    abstract = 0
    article_title = ""
    last_heading_1 = ""
    last_heading_2 = ""
    #I don't think this third one is needed, but it's a save just in case for the parent headings
    last_heading_3 = ""
    #Right here, create first chunk with author information / TITLE
    #Save title and authors to metadata
    #find the preamble before the abstract section
    '''
    Some slight issues with this section:
    Finding the title is a little iffy, because NOUGAT uses new lines and separates title
    So we kinda just say if line has no numbers and is more than a certain length it's part of the title
    (NOUGAT usually saves names w/ a number next to em) Especially since authors aren't always saved on a new line (sucks)        
    '''
    #everything before first real heading
    preamble = re.split(r'(?m)^#{2,6} ', text)[0]
    article_title, article_authors = grab_authors_titles(preamble)
    #Set up the loop to iterate over the text in each individual PART
    i = 0
    while i < len(parts):
        #EVERY OTHER ELEMENT AFTER SPLIT IS THE HEADING (THE TEXT TO SAVE TO METADATA)
        if re.match(r'(?m)^(#{2,6}(?!#) .*)', parts[i]):
            heading = parts[i] #should hopefully save the text in the line as the heading
            body = parts[i+1].strip() if i+1 < len(parts) else "" #save other text in heading, if over index make nothing
            level = len(heading.split(' ')[0]) # count # of #'s


            #Yeah cuz then when it equals 4 it uses level 3 as its parent, level 3 uses 2 as parent,
            #You better pray the abstract doesn't have sections cause that usually means an unfun article


            #or you save the heading afterwords
            #Now save the full text of the file
            full_text = f"{heading.strip()}\n{body}"
            #quick boolean with full text to confirm that a previous chunk didn't have an unclosed table / eqn (if so we tie them together)
            unclosed = False
            MAX_TOKENS = 400
            n = 0
            label = 0
            if token_count(full_text) < MAX_TOKENS:
            #save chunk as a dictionary with metadata of the sections (if chunk is split metadata stays same)
                new_chunk ={
                    "text": full_text,
                    "metadata": {
                        "article_title" : article_title,
                        "article_authors": article_authors,
                        "heading": heading.strip(),
                        "heading_text": heading.strip().lstrip('#').strip(),
                        #don't like this because this'll save a parent for the abstract
                        #Actually it won't will it cause abstract always comes first
                        #Then save the title & authors completely separately (only 1 hashtag)
                        "parent_section": last_heading_2 if level == 4 else last_heading_1,
                        "word_count": len(full_text.split()),
                        **({"chunk_id": len(chunks)} if n > 0 else {}),
                        **({"label":f"group_{label}"} if n > 0 else {})
                    }
                
                }
                if n > 0:
                    n -= 1
                chunks.append(new_chunk)
            else:
                #now you need to split the text within the text on words TO BE 350 words 
                #use that recursive thing

                words = full_text.split()
                #Add a step to be all overlappy and cool
                #iterate over the words in the list, by degrees of 350
                j = 0
                while j < len(words):
                    section = ''
                    # need a method to check for next section/hashtag
                    #next section, hashtag, and eqns, mind you
                    #\[ must be paired with \]
                    #\begin{tabular} with \end{tabular{}
                    #and stopping for section - though I believe this is automatically done by how sections are split
                    #Keep extending until all conditions of tables/eqns met
                    k = j
                    #grow word by word UNTIL TOKEN LIMITS HIT
                    while k < len(words) and token_count(section + ' ' + words[k]) < MAX_TOKENS:
                        section += ' ' + words[k]
                        k += 1
                    #calculate step with a new section
                    step = min(50, len(section.split()) // 4)
                    extra = k
                    while not is_chunk_closed(section) and extra < len(words) and token_count(section) < 480:
                        section += " " + words[extra]
                        extra += 1
                        if token_count(section) >= 480:
                            unclosed = True
                            break
                
                    if unclosed:
                        #Find the amount of text that is being truncated by token_count
                        #Find full length of text that is not truncated
                        untruncated_section = section
                        while not is_chunk_closed(section) and extra < len(words):
                            untruncated_section += " " + words[extra]
                            extra += 1
                        remainder_words = untruncated_section.split()[len(section.split()):]
                        #determine how many chunks are needed until object is closed
                        n = 0
                        temp_j = 0
                        while temp_j < len(remainder_words):
                            n += 1
                            temp_j += MAX_TOKENS
                        label = len(chunks)

                    
                    new_chunk ={
                    "text": section,
                    "metadata": {
                        "article_title" : article_title,
                        "article_authors": article_authors,
                        "heading": heading.strip(),
                        "heading_text": heading.strip().lstrip('#').strip(),
                        #don't like this because this'll save a parent for the abstract
                        #Actually it won't will it cause abstract always comes first
                        #Then save the title & authors completely separately (only 1 hashtag)
                        "parent_section": last_heading_2 if level == 4 else last_heading_1,
                        "word_count": len(section.split()),
                        "token_count": token_count(full_text),
                        **({"chunk_id": len(chunks)} if n > 0 else {}),
                        **({"label":f"group_{label}"} if n > 0 else {})
                    }
                
                    }
                    if n > 0:
                        n -= 1
                    chunks.append(new_chunk)
                    new_j =  extra - step
                    j = new_j if new_j > k else k
              #Do fucky indexes to get around the whole start of the document
              #That's why you intake i and i + 1 for successful partition
            i += 2
            if level == 2:
                last_heading_1 = heading.strip().lstrip('#').strip()
                last_heading_2 = ""
            if level == 3:
                last_heading_2 = heading.strip().lstrip('#').strip()
        else:
            i += 1
            
    return chunks

#Function to confirm that the chunks are properly closed

def is_chunk_closed(text):
    checks = [
        (text.count(r'\begin{tabular}'), text.count(r'\end{tabular}')),
        (text.count(r'\['),              text.count(r'\]')),
    ]
    return all(begins == ends for begins, ends in checks)


if __name__== "__main__":

    #Get files in folder
    filelist = [f for f in listdir("[path_to_parsed_files]/") if isfile(join("[path_to_parsed_files]/", f))]
    #Do we want it all in this for loop?
    all_chunks=[]
    for filename in filelist:
        with open(f"[path_to_parsed_files]/{filename}", "r") as f:
            raw_text = f.read()

        all_chunks.extend(chunk_by_hashtag(raw_text,EMBEDDING_MODEL))
    
    #some quick error fixing to check token count in each chunk?
    #for chunk in all_chunks:
    #    chunk_text = chunk["text"]
    #    print(token_count(chunk_text))
    print(token_count(all_chunks[320]["text"]))
    print(token_count(all_chunks[321]["text"]))
    print(token_count(all_chunks[319]["text"]))
#Chunking model that calls encoder is probably gonna be easiest - have a separate file that encodes the text chunk thru function

#We want to move through the file and encode different sections based on the text length 
#We need to save section titles as metadata - I've no clue how to do this
#We want to make sure chunks aren't split in between tables (a check of \[ needing \] and \begin{table} needing \end{table}) is probably easiest

# Initialize master list at the beginning

#End of loop: needs to define this whole thing

#Check for a hashtag
#When hashtag is found, save full text in line (4 - section -every decimal adds a "sub" title)



