from flask import Flask, render_template, request, jsonify
import pandas as pd
from langchain_core.prompts import PromptTemplate

import google.generativeai as genai
import json

app = Flask(__name__)

# Configure Gemini
api_key = 'YOUR_GEMINI_API_KEY'
genai.configure(api_key=api_key)
llm = genai.GenerativeModel('models/gemini-1.5-flash')

# Load the DataFrame
df = pd.read_csv('Task Catagories.csv.csv')  # Replace with your data source

# Load saved unique words
with open('unique_words.json', 'r') as f:
    unique_words = json.load(f)

# Define PromptTemplate
prompt_template = PromptTemplate(
    input_variables=["prompt", "unique_words"],
    template="""
    Based on the task descriptions and unique keywords below, generate a comma-separated list of relevant keywords related to the user prompt.
    User prompt: "{prompt}"
    Unique keywords from task descriptions: {unique_words}
    """
)

def get_related_keywords(user_prompt, unique_words, llm):
    prompt = prompt_template.format(prompt=user_prompt, unique_words=unique_words)
    response = llm.generate_content(prompt)
    return response.text.split(', ')

def filter_tasks_by_keywords(df, keywords):
    task_mask = df['Task Description'].str.contains('|'.join(keywords), case=False, na=False)
    category_mask = df['Category'].str.contains('|'.join(keywords), case=False, na=False)
    skill_mask = df['Skill'].str.contains('|'.join(keywords), case=False, na=False)
    combined_mask = task_mask | category_mask | skill_mask
    filtered_df = df[combined_mask]
    return filtered_df

def query_llm_for_answer(filtered_tasks, original_prompt, llm):
    formatted_tasks = []
    for _, row in filtered_tasks.iterrows():
        task_entry = (
            f"Task: {row['Task Description']}\n"
            f"Category: {row['Category']}\n"
            f"Required Skill: {row['Skill']}"
        )
        formatted_tasks.append(task_entry)
    
    tasks_text = "\n\n".join(formatted_tasks)
    
    final_prompt = f"""
    Here are the relevant tasks with their categories and required skills:
    {tasks_text}
    
    Based on these tasks, please analyze and provide the most relevant matches for the following request:
    "{original_prompt}"
    
    Please structure your response to include:
    1. The most relevant task(s)
    2. Their categories
    3. Required skills
    Only include information that is directly relevant to the user's request.
    """
    
    final_response = llm.generate_content(final_prompt)
    return final_response.text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json['message']
        
        # Process the message using your existing pipeline
        related_keywords = get_related_keywords(user_message, unique_words, llm)
        if not related_keywords:
            return jsonify({
                'response': "I couldn't find any relevant keywords for your query. Could you please rephrase your question?"
            })
        
        filtered_tasks = filter_tasks_by_keywords(df, related_keywords)
        if filtered_tasks.empty:
            return jsonify({
                'response': "I couldn't find any tasks matching your query. Could you please try a different search term?"
            })
        
        response = query_llm_for_answer(filtered_tasks, user_message, llm)
        if not response:
            return jsonify({
                'response': "I encountered an issue while processing your request. Please try again."
            })
        
        return jsonify({'response': response})
    
    except Exception as e:
        print(f"Error processing request: {str(e)}")  # Log the error
        return jsonify({
            'error': 'An unexpected error occurred while processing your request. Please try again.'
        }), 500

# Optional: Configure Flask for longer timeouts
app.config['TIMEOUT'] = 300  # 5 minutes

if __name__ == '__main__':
    app.run(debug=True)