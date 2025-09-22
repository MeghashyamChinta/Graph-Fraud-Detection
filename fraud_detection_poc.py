import streamlit as st
import pandas as pd
import json
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
import os
from openai import AzureOpenAI
import random
import datetime

# --- Neo4j Config ---
# Replace with your actual Neo4j instance details
NEO4J_URI = "neo4j+s://228664c9.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "EVeFL1v35NTwBbWD7xLngijTfw3KinVUviGs7OxD7eM"
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

AZURE_OPENAI_ENDPOINT = "https://your-resource-name.openai.azure.com/"
AZURE_OPENAI_API_KEY = "your-api-key"
AZURE_OPENAI_DEPLOYMENT = "gpt-4"
AZURE_OPENAI_API_VERSION = "2024-02-15-preview"

client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION
)

# --- Helper Functions ---

def run_query(query, params=None):
    try:
        with driver.session() as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]
    except ServiceUnavailable as e:
        st.error(f"Neo4j connection failed: {e}")
        return []

def populate_sample_data():
    queries = [
        "MATCH (n) DETACH DELETE n;",
        'CREATE (c1:Claim {claim_id: "C001", amount: 15000, date: "2024-01-15", type: "Workers Comp"});',
        'CREATE (c2:Claim {claim_id: "C002", amount: 20000, date: "2024-02-10", type: "Commercial Property"});',
        'CREATE (c3:Claim {claim_id: "C003", amount: 12000, date: "2024-03-05", type: "Workers Comp"});',
        'CREATE (c4:Claim {claim_id: "C004", amount: 18000, date: "2024-04-20", type: "Workers Comp"});',
        'CREATE (cl1:Claimant {id: "CL001", name: "John Doe"});',
        'CREATE (cl2:Claimant {id: "CL002", name: "Jane Smith"});',
        'CREATE (cl3:Claimant {id: "CL003", name: "Alice Johnson"});',
        'CREATE (mp1:MedicalProvider {name: "Dr. Adams"});',
        'CREATE (mp2:MedicalProvider {name: "Dr. Baker"});',
        'CREATE (con1:Contractor {name: "BuildFix Co"});',
        'CREATE (con2:Contractor {name: "QuickRepair"});',
        'CREATE (loc1:Property {address: "123 Main St"});',
        'CREATE (loc2:Property {address: "456 Elm St"});',
        'CREATE (ph1:Policyholder {name: "Corp Inc", id: "PH001"});',
        'CREATE (p1:Policy {policy_id: "P001"});',
        'MATCH (cl1:Claimant {id: "CL001"}), (c1:Claim {claim_id: "C001"}) CREATE (cl1)-[:FILED]->(c1);',
        'MATCH (cl2:Claimant {id: "CL002"}), (c2:Claim {claim_id: "C002"}) CREATE (cl2)-[:FILED]->(c2);',
        'MATCH (cl3:Claimant {id: "CL003"}), (c3:Claim {claim_id: "C003"}) CREATE (cl3)-[:FILED]->(c3);',
        'MATCH (cl1:Claimant {id: "CL001"}), (c4:Claim {claim_id: "C004"}) CREATE (cl1)-[:FILED]->(c4);',
        'MATCH (c1:Claim {claim_id: "C001"}), (mp1:MedicalProvider {name: "Dr. Adams"}) CREATE (c1)-[:INVOLVES_PROVIDER]->(mp1);',
        'MATCH (c3:Claim {claim_id: "C003"}), (mp1:MedicalProvider {name: "Dr. Adams"}) CREATE (c3)-[:INVOLVES_PROVIDER]->(mp1);',
        'MATCH (c2:Claim {claim_id: "C002"}), (con1:Contractor {name: "BuildFix Co"}) CREATE (c2)-[:INVOLVES_PROVIDER]->(con1);',
        'MATCH (c4:Claim {claim_id: "C004"}), (mp1:MedicalProvider {name: "Dr. Adams"}) CREATE (c4)-[:INVOLVES_PROVIDER]->(mp1);',
        'MATCH (c1:Claim {claim_id: "C001"}), (loc1:Property {address: "123 Main St"}) CREATE (c1)-[:AT_LOCATION]->(loc1);',
        'MATCH (c2:Claim {claim_id: "C002"}), (loc2:Property {address: "456 Elm St"}) CREATE (c2)-[:AT_LOCATION]->(loc2);',
        'MATCH (c3:Claim {claim_id: "C003"}), (loc1:Property {address: "123 Main St"}) CREATE (c3)-[:AT_LOCATION]->(loc1);',
        'MATCH (c4:Claim {claim_id: "C004"}), (loc1:Property {address: "123 Main St"}) CREATE (c4)-[:AT_LOCATION]->(loc1);',
        'MATCH (ph1:Policyholder {id: "PH001"}), (p1:Policy {policy_id: "P001"}) CREATE (ph1)-[:HOLDER_OF]->(p1);',
        'MATCH (p1:Policy {policy_id: "P001"}), (c1:Claim {claim_id: "C001"}) CREATE (p1)-[:COVERS]->(c1);',
        'MATCH (p1:Policy {policy_id: "P001"}), (c2:Claim {claim_id: "C002"}) CREATE (p1)-[:COVERS]->(c2);',
        'MATCH (p1:Policy {policy_id: "P001"}), (c3:Claim {claim_id: "C003"}) CREATE (p1)-[:COVERS]->(c3);',
        'MATCH (p1:Policy {policy_id: "P001"}), (c4:Claim {claim_id: "C004"}) CREATE (p1)-[:COVERS]->(c4);'
    ]
    for q in queries:
        run_query(q)
    return "Sample data populated successfully!"

def fetch_claims():
    query = '''
    MATCH (cl:Claimant)-[:FILED]->(c:Claim)
    OPTIONAL MATCH (c)-[:INVOLVES_PROVIDER]->(mp:MedicalProvider)
    OPTIONAL MATCH (c)-[:INVOLVES_PROVIDER]->(con:Contractor)
    OPTIONAL MATCH (c)-[:AT_LOCATION]->(loc:Property)
    OPTIONAL MATCH (ph:Policyholder)-[:HOLDER_OF]->(p:Policy)-[:COVERS]->(c)
    RETURN c.claim_id AS claim_id, c.amount AS amount, c.date AS date, c.type AS type,
           cl.name AS claimant, cl.id AS claimant_id,
           mp.name AS medical_provider,
           con.name AS contractor,
           loc.address AS location,
           ph.name AS policyholder,
           p.policy_id AS policy_id
    '''
    records = run_query(query)
    df = pd.DataFrame(records)
    df.fillna("Unknown", inplace=True)
    return df

def cosine_similarity_detection(df, threshold=0.7):
    features = ['type', 'medical_provider', 'contractor', 'location', 'claimant']
    encoder = OneHotEncoder(sparse_output=False)
    vecs = encoder.fit_transform(df[features])
    similarity_scores = cosine_similarity(vecs)
    suspicious = []
    for i in range(len(similarity_scores)):
        for j in range(i + 1, len(similarity_scores)):
            score = similarity_scores[i][j]
            if score >= threshold:
                suspicious.append({
                    'claim1': df.iloc[i]['claim_id'],
                    'claim2': df.iloc[j]['claim_id'],
                    'similarity': score,
                    'shared_features': [f for f in features if df.iloc[i][f] == df.iloc[j][f] and df.iloc[i][f] != 'Unknown']
                })
    return pd.DataFrame(suspicious).sort_values(by='similarity', ascending=False)

def community_detection():
    query = '''
    MATCH (c1:Claim)-[:INVOLVES_PROVIDER|AT_LOCATION]->(n)<-[:INVOLVES_PROVIDER|AT_LOCATION]-(c2:Claim)
    WHERE c1 <> c2
    WITH n, collect(c1.claim_id) AS claims, count(DISTINCT c1) AS size
    WHERE size > 2
    RETURN elementId(n) AS communityId, labels(n)[0] AS entity_type, n.name AS entity_name, n.address AS entity_address, claims, size
    '''
    records = run_query(query)
    df = pd.DataFrame(records)
    df['claims'] = df['claims'].apply(lambda x: x if x else [])
    return df, df

def anomaly_scoring(df):
    df['anomaly_score'] = 0.0
    df['anomaly_score'] += (df['amount'] > df['amount'].mean() + df['amount'].std()) * 2
    claimant_counts = df['claimant'].value_counts()
    df['anomaly_score'] += df['claimant'].map(claimant_counts) > 1
    provider_counts = df['medical_provider'].value_counts()
    df['anomaly_score'] += df['medical_provider'].map(provider_counts) > 2
    location_counts = df['location'].value_counts()
    df['anomaly_score'] += df['location'].map(location_counts) > 2
    return df.sort_values(by='anomaly_score', ascending=False)

def fetch_suspicious_graph(claim_ids):
    query = '''
    MATCH (c:Claim)
    WHERE c.claim_id IN $claim_ids
    OPTIONAL MATCH (cl:Claimant)-[:FILED]->(c)
    OPTIONAL MATCH (c)-[:INVOLVES_PROVIDER]->(mp:MedicalProvider)
    OPTIONAL MATCH (c)-[:INVOLVES_PROVIDER]->(con:Contractor)
    OPTIONAL MATCH (c)-[:AT_LOCATION]->(loc:Property)
    OPTIONAL MATCH (p:Policy)-[:COVERS]->(c)<-[:HOLDER_OF]-(ph:Policyholder)
    RETURN c, cl, mp, con, loc, ph, p,
           [(cl)-[:FILED]->(c) | {from: cl.id, to: c.claim_id, label: 'FILED'}] AS filed,
           [(c)-[:INVOLVES_PROVIDER]->(mp) | {from: c.claim_id, to: mp.name, label: 'INVOLVES_PROVIDER'}] AS mp_rel,
           [(c)-[:INVOLVES_PROVIDER]->(con) | {from: c.claim_id, to: con.name, label: 'INVOLVES_PROVIDER'}] AS con_rel,
           [(c)-[:AT_LOCATION]->(loc) | {from: c.claim_id, to: loc.address, label: 'AT_LOCATION'}] AS loc_rel,
           [(p)-[:COVERS]->(c) | {from: p.policy_id, to: c.claim_id, label: 'COVERS'}] AS cov_rel,
           [(ph)-[:HOLDER_OF]->(p) | {from: ph.id, to: p.policy_id, label: 'HOLDER_OF'}] AS hold_rel
    '''
    records = run_query(query, {'claim_ids': claim_ids})
    nodes = []
    edges = []
    seen_nodes = set()
    seen_edges = set()
    for record in records:
        # Add Claim node
        c = record['c']
        if c and c['claim_id'] not in seen_nodes:
            nodes.append({
                'id': c['claim_id'],
                'label': c['claim_id'],
                'type': 'Claim',
                'title': f"Amount: {c['amount']}\nDate: {c['date']}\nType: {c['type']}"
            })
            seen_nodes.add(c['claim_id'])
        # Add Claimant node
        cl = record['cl']
        if cl and cl['id'] not in seen_nodes:
            nodes.append({
                'id': cl['id'],
                'label': cl['name'],
                'type': 'Claimant',
                'title': f"Name: {cl['name']}"
            })
            seen_nodes.add(cl['id'])
        # Add MedicalProvider node
        mp = record['mp']
        if mp and mp['name'] not in seen_nodes:
            nodes.append({
                'id': mp['name'],
                'label': mp['name'],
                'type': 'MedicalProvider',
                'title': f"Name: {mp['name']}"
            })
            seen_nodes.add(mp['name'])
        # Add Contractor node
        con = record['con']
        if con and con['name'] not in seen_nodes:
            nodes.append({
                'id': con['name'],
                'label': con['name'],
                'type': 'Contractor',
                'title': f"Name: {con['name']}"
            })
            seen_nodes.add(con['name'])
        # Add Property node
        loc = record['loc']
        if loc and loc['address'] not in seen_nodes:
            nodes.append({
                'id': loc['address'],
                'label': loc['address'],
                'type': 'Property',
                'title': f"Address: {loc['address']}"
            })
            seen_nodes.add(loc['address'])
        # Add Policyholder and Policy nodes
        ph = record['ph']
        p = record['p']
        if ph and ph['id'] not in seen_nodes:
            nodes.append({
                'id': ph['id'],
                'label': ph['name'],
                'type': 'Policyholder',
                'title': f"Name: {ph['name']}"
            })
            seen_nodes.add(ph['id'])
        if p and p['policy_id'] not in seen_nodes:
            nodes.append({
                'id': p['policy_id'],
                'label': p['policy_id'],
                'type': 'Policy',
                'title': f"Policy ID: {p['policy_id']}"
            })
            seen_nodes.add(p['policy_id'])
        # Add edges
        for rel_list in [record['filed'], record['mp_rel'], record['con_rel'], record['loc_rel'], record['cov_rel'], record['hold_rel']]:
            for rel in rel_list:
                edge_key = (rel['from'], rel['to'], rel['label'])
                if edge_key not in seen_edges:
                    edges.append({'from': rel['from'], 'to': rel['to'], 'label': rel['label']})
                    seen_edges.add(edge_key)
    return {'nodes': nodes, 'edges': edges}

def render_graph(graph_data, height=600):
    net = Network(height=f"{height}px", width="100%", directed=True, bgcolor="#ffffff", font_color="black")
    color_map = {
        'Claim': '#81c784',
        'Claimant': '#64b5f6',
        'MedicalProvider': '#ffd54f',
        'Contractor': '#ffcc80',
        'Property': '#ef5350',
        'Policyholder': '#90caf9',
        'Policy': '#ffb74d'
    }
    for node in graph_data['nodes']:
        color = color_map.get(node['type'], '#eeeeee')
        net.add_node(node['id'], label=node['label'], title=node['title'], color=color, group=node['type'])
    for edge in graph_data['edges']:
        net.add_edge(edge['from'], edge['to'], label=edge['label'])
    tmp_file = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
    net.write_html(tmp_file.name)
    with open(tmp_file.name, 'r') as f:
        html = f.read()
    components.html(html, height=height + 50, scrolling=True)
    try:
        os.unlink(tmp_file.name)
    except PermissionError:
        pass  # File is in use; skip deletion

def generate_gpt_explanation(results):
    summary = json.dumps(results, default=str)
    prompt = (
        "You are an insurance fraud expert presenting to a client. Explain these fraud detection results professionally:\n"
        "- Similarity pairs: Potential staged claims with high overlap.\n"
        "- Communities: Fraud rings with connected entities.\n"
        "- Anomalies: High-risk claims based on scoring.\n"
        "Results: " + summary + "\n"
        "Provide:\n"
        "1. Overview of findings.\n"
        "2. Key patterns and why they indicate fraud.\n"
        "3. Business impact (e.g., potential savings).\n"
        "4. Recommendations.\n"
        "Keep to 300 words, use bullet points for clarity."
    )
    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Explanation error: {str(e)}"

# --- Streamlit App ---
st.set_page_config(page_title="Advanced Fraud Detection Demo", layout="wide")
st.title("🕵️‍♂️ Advanced Insurance Fraud Detection System")
st.markdown("**Real-World Demo for Client:** Detecting fraud rings in Workers' Comp and Commercial Property claims at Corp Inc. Using graph analytics, ML similarity, community detection, and anomaly scoring. Powered by Neo4j and Azure GPT-4 for explanations.")

col1, col2 = st.columns(2)
with col1:
    if st.button("Populate/Reset Sample Data for Demo"):
        msg = populate_sample_data()
        st.success(msg)
with col2:
    st.markdown("**Sample Scenario:** Fraud ring with repeated claimant (John Doe), provider (Dr. Adams), and location (123 Main St) across claims.")

df_claims = fetch_claims()
if df_claims.empty:
    st.warning("No data found. Populate sample data or check Neo4j connection.")
else:
    st.subheader("Raw Claim Data Overview")
    st.dataframe(df_claims)

    st.subheader("Detection Parameters")
    threshold = st.slider("Similarity Threshold for Pairs", 0.5, 1.0, 0.7)
    min_anomaly = st.slider("Min Anomaly Score for Alerts", 0.0, 5.0, 2.0)
    min_community_size = st.number_input("Min Community Size for Rings", 2, 10, 3)
    claim_type = st.selectbox("Filter by Claim Type", ["All", "Workers Comp", "Commercial Property"])
    min_amount = st.number_input("Min Claim Amount Filter", 0, value=10000)

    if st.button("Run Advanced Fraud Analysis"):
        with st.spinner("Running multi-method fraud detection..."):
            filtered_df = df_claims.copy()
            if claim_type != "All":
                filtered_df = filtered_df[filtered_df['type'] == claim_type]
            filtered_df = filtered_df[filtered_df['amount'] >= min_amount]

            # 1. Similarity Detection
            df_sim = cosine_similarity_detection(filtered_df, threshold)
            st.subheader("1. Similarity-Based Fraud Pairs (Cosine Similarity)")
            if not df_sim.empty:
                st.dataframe(df_sim)
                st.download_button(
                    label="Download Similarity Results (CSV)",
                    data=df_sim.to_csv(index=False),
                    file_name="similarity_results.csv",
                    mime="text/csv"
                )
            else:
                st.info("No high-similarity pairs found.")

            # 2. Community Detection (Fraud Rings)
            df_com, suspicious_com = community_detection()
            st.subheader("2. Community Detection (Potential Fraud Rings)")
            if not suspicious_com.empty:
                susp_com = suspicious_com[suspicious_com['size'] >= min_community_size]
                st.dataframe(susp_com)
                if not susp_com.empty:
                    largest_com = susp_com.iloc[0]['communityId']
                    com_claims = susp_com.iloc[0]['claims']
                    st.markdown(f"Claims in Largest Ring (Community {largest_com}): {', '.join(com_claims or [])}")
                st.download_button(
                    label="Download Community Results (CSV)",
                    data=suspicious_com.to_csv(index=False),
                    file_name="community_results.csv",
                    mime="text/csv"
                )
            else:
                st.info("No large communities detected.")

            # 3. Anomaly Scoring
            df_anomaly = anomaly_scoring(filtered_df)
            st.subheader("3. Anomaly Scoring (High-Risk Claims)")
            high_anomaly = df_anomaly[df_anomaly['anomaly_score'] >= min_anomaly]
            if not high_anomaly.empty:
                st.dataframe(high_anomaly[['claim_id', 'amount', 'type', 'claimant', 'medical_provider', 'location', 'anomaly_score']])
                st.download_button(
                    label="Download Anomaly Results (CSV)",
                    data=high_anomaly.to_csv(index=False),
                    file_name="anomaly_results.csv",
                    mime="text/csv"
                )
            else:
                st.info("No high-anomaly claims found.")

            # Combined Suspicious Claims for Graph
            susp_claims = list(set(df_sim['claim1'].tolist() + df_sim['claim2'].tolist() + high_anomaly['claim_id'].tolist()))
            if susp_claims:
                graph_data = fetch_suspicious_graph(susp_claims)
                st.subheader("Integrated Fraud Network Graph")
                render_graph(graph_data)
                with open("fraud_graph.json", "w") as f:
                    json.dump(graph_data, f)
                with open("fraud_graph.json", "rb") as f:
                    st.download_button(
                        label="Download Fraud Graph (JSON)",
                        data=f,
                        file_name="fraud_graph.json",
                        mime="application/json"
                    )
            else:
                st.info("No suspicious claims to visualize.")

            # GPT-4 Explainer
            results = {
                'similarity_pairs': df_sim.to_dict('records'),
                'suspicious_rings': suspicious_com.to_dict('records'),
                'high_anomalies': high_anomaly.to_dict('records')
            }
            with st.spinner("Generating GPT-4 explanation..."):
                explanation = generate_gpt_explanation(results)
                st.subheader("GPT-4 Executive Summary & Recommendations")
                st.markdown(
                    f"""
                    <style>
                    .explainer-box {{
                        background-color: #f0f8ff;
                        padding: 20px;
                        border-radius: 10px;
                        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                        margin-top: 20px;
                    }}
                    .explainer-box h3 {{
                        color: #2e7d32;
                        font-size: 1.5rem;
                        margin-bottom: 10px;
                    }}
                    .explainer-box p {{
                        font-size: 1rem;
                        line-height: 1.5;
                        color: #333;
                    }}
                    </style>
                    <div class="explainer-box">
                        <h3>📝 Fraud Detection Insights</h3>
                        <p>{explanation}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# Full Network Tab
with st.expander("Full Network View"):
    st.subheader("Full Claims Network")
    query = '''
    MATCH (n)-[r]->(m)
    RETURN n, elementId(n) AS n_id, m, elementId(m) AS m_id, r
    LIMIT 100
    '''
    records = run_query(query)
    graph_data = {'nodes': [], 'edges': []}
    seen_nodes = set()
    seen_edges = set()
    for record in records:
        n = record['n']
        n_id = record['n_id']
        m = record['m']
        m_id = record['m_id']
        r = record['r']
        for node, node_id in [(n, n_id), (m, m_id)]:
            if node_id not in seen_nodes:
                label = list('labels')[0]
                properties = dict(node)
                graph_data['nodes'].append({
                    'id': node_id,
                    'label': properties.get('name', properties.get('claim_id', properties.get('address', str(node_id)))),
                    'type': label,
                    'title': str(properties)
                })
                seen_nodes.add(node_id)
            # Access relationship type from tuple (r[0]) or dictionary (nested 'type' or 'value')
        if isinstance(r, tuple):
            rel_type = str(r[0]) if len(r) > 0 else 'UNKNOWN'
        elif isinstance(r, dict):
            rel_type = r.get('type', {}).get('value', r.get('type', 'UNKNOWN'))
            rel_type = str(rel_type)  # Ensure string
        else:
            rel_type = 'UNKNOWN'
        edge_key = (n_id, m_id, rel_type)
        if edge_key not in seen_edges:
            graph_data['edges'].append({'from': n_id, 'to': m_id, 'label': rel_type})
            seen_edges.add(edge_key)