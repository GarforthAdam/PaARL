import ollama
import json

LANGUAGE_MODEL = 'phi3:mini'
def sorting_themes(query):
    response = ollama.chat(
        model=LANGUAGE_MODEL,
        #temperature is LLM "creativity" - how much it wants to extrapolate, basically
        options = {
            'temperature' : 0
        },
        messages=[{
            'role': 'user',
            'content': f''' Sort the genre of the given query into one of the following categories:
                [Stellar remnants, Exoplanet, Supernova, Bayesian Statistics, Bayesian Error, Globular Cluster,
                Electromagnetic Fields, Particle Scattering, Superconductor] 
            
            Return ONLY a JSON object in this format:
            {{"genre": "<one of the categories above>"}}

            Example output for an Supernova query:
            {{"genre": "Supernova"}}

            Return an article genre that is the EXACT wording as the given genres. ADD NO ADDITIONAL TEXT. DO NOT CREATE ANY NEW GENRES. RETURN ONLY ONE GENRE.
            Given query: {query}

            RETURN ONLY VALID JSON, NO OTHER TEXT.'''
        }]
        )
        #parse the returned JSON response
    try:
        content = response['message']['content']
        # find first opening brace
        start = content.find('{')
        if start == -1:
            return ''
        content = content[start:]
        # then take first line
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('{'):
                result = json.loads(line)
                return result.get('genre', '')
    except json.JSONDecodeError:
        print(f"Parse failed, raw response: {response['message']['content'][:100]}")
        return ''
