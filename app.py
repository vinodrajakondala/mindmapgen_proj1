import os
import json
import re
from flask import Flask, render_template, request
from groq import Groq
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

def get_topic_summary(topic):
    """
    Uses the Groq LLM to generate a concise plain‑text summary for the given topic.
    """
    prompt = f"Generate a concise summary for the topic '{topic}'."
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set the GROQ_API_KEY environment variable.")
    
    client = Groq(api_key=api_key)
    messages = [
        {"role": "system", "content": "You are a helpful assistant that returns plain text."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=1,
            max_completion_tokens=200,
            top_p=1,
            stream=True,
            stop=None,
        )
        summary = ""
        for chunk in completion:
            summary += chunk.choices[0].delta.content or ""
        return summary.strip()
    except Exception as e:
        print("Error generating summary:", e)
        return "Summary not available."

def get_mindmap_data(topic):
    """
    Uses the Groq LLM to generate a mind map for the given topic in strict JSON format.
    
    Expected JSON format:
    {
      "node": "Main Topic",
      "children": [
          { "node": "Subtopic 1", "children": [] },
          { "node": "Subtopic 2", "children": [
              { "node": "Detail 1", "children": [] }
          ]}
      ]
    }
    """
    prompt = (
        f"Generate a mind map for the topic '{topic}'. "
        "Return only a JSON object in the following format without any extra text:\n\n"
        "{\n"
        '  "node": "Main Topic",\n'
        '  "children": [\n'
        '      { "node": "Subtopic 1", "children": [] },\n'
        '      { "node": "Subtopic 2", "children": [\n'
        '            { "node": "Detail 1", "children": [] }\n'
        "      ]}\n"
        "  ]\n"
        "}"
    )
    
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set the GROQ_API_KEY environment variable.")
    
    client = Groq(api_key=api_key)
    messages = [
        {"role": "system", "content": "You are a helpful assistant that only returns valid JSON."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )
        generated_text = ""
        for chunk in completion:
            generated_text += chunk.choices[0].delta.content or ""
        
        # Remove any markdown formatting if present.
        generated_text = re.sub(r"```(?:json)?", "", generated_text).strip()
        mindmap_data = json.loads(generated_text)
        return mindmap_data
    except Exception as e:
        print("Error generating or parsing mind map data:", e)
        print("Generated text was:", generated_text)
        return None

def embed_nodes(mindmap, embedder):
    """
    Recursively computes and attaches embeddings for each node.
    (Embeddings are not used for visualization in this example.)
    """
    node_text = mindmap.get("node", "")
    mindmap["embedding"] = embedder.encode(node_text).tolist()
    for child in mindmap.get("children", []):
        embed_nodes(child, embedder)

def create_interactive_mindmap_data(topic):
    """
    Obtains the mind map JSON, enriches it with embeddings, and converts it into
    node/edge data for vis‑network. It also extracts key points (the immediate children
    of the root node) for detailed display.
    """
    mindmap_data = get_mindmap_data(topic)
    if not mindmap_data:
        return None, None, None
    
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embed_nodes(mindmap_data, embedder)
    
    # Extract key points: the immediate children of the root node.
    key_points = [child.get("node", "Unknown") for child in mindmap_data.get("children", [])]
    
    nodes = []
    edges = []
    
    def recursive_build(node, parent_id=None, next_id=0):
        current_id = next_id
        nodes.append({'id': current_id, 'label': node.get('node', 'Unknown')})
        if parent_id is not None:
            edges.append({'from': parent_id, 'to': current_id})
        next_id += 1
        for child in node.get('children', []):
            next_id = recursive_build(child, current_id, next_id)
        return next_id
    
    recursive_build(mindmap_data)
    return nodes, edges, key_points

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        topic = request.form.get("topic")
        summary = get_topic_summary(topic)
        # Render index.html with the topic and generated summary.
        return render_template("index.html", topic=topic, summary=summary)
    return render_template("index.html")

@app.route('/result')
def result():
    topic = request.args.get("topic")
    if not topic:
        return "No topic provided", 400
    nodes, edges, key_points = create_interactive_mindmap_data(topic)
    if nodes is None or edges is None:
        error = "Failed to generate mind map. Please try again."
        return render_template("result.html", error=error)
    return render_template("result.html", topic=topic, nodes=nodes, edges=edges, key_points=key_points)

if __name__ == "__main__":
    app.run(debug=True)
