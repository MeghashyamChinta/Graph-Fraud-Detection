What We Have Done
We have developed a comprehensive insurance fraud detection application using Neo4j for graph data modeling and Streamlit for an interactive user interface. 
The app addresses real-world fraud scenarios in workers' compensation and commercial property claims, such as staged incidents, inflated claims, and colluding parties (e.g., claimants, doctors, contractors).
It integrates multiple detection methods (similarity matching, community detection, anomaly scoring), visualizes fraud networks with PyVis graphs.


Problem Statement
The problem is insurance fraud rings where fraudsters collaborate to fabricate or inflate claims, leading to significant financial losses for insurers. For example:

In workers' comp: Recurring injuries, exaggerated wages, or staged accidents involving the same claimant, doctor, or location.
In commercial property: Sham damage claims with dishonest contractors or repeated losses at the same property.
These schemes are hard to detect in traditional tabular databases due to hidden connections (e.g., shared providers across claims), costing insurers billions annually. 
The challenge is to uncover non-obvious patterns like role rotation or asset reuse while providing actionable insights without requiring advanced technical knowledge.

Solution We Provided
We provided a Neo4j-based Streamlit app that models claims as a graph (nodes: Claim, Claimant, MedicalProvider, Contractor, Property, Policyholder, 
Policy; relationships: FILED, INVOLVES_PROVIDER, AT_LOCATION, HOLDER_OF, COVERS). 

Key components:

Detection Methods:

Cosine similarity on features (type, provider, contractor, location, claimant) to find similar claim pairs.
Cypher-based community detection to identify fraud rings (e.g., claims sharing providers/locations).
Anomaly scoring based on amount outliers, repetition of claimants/providers/locations.


Visualization: Interactive PyVis graphs for fraud networks.
GPT Explainer: Azure GPT-4 generates executive summaries with insights, impacts, and recommendations.
UI Features: Filters (claim type, amount, thresholds), sample data population, download buttons for results/graph.
The app is self-contained in one Python file, with robust error handling for Neo4j connections and Windows issues.

How to Understand the App and Insights We Can Take
To understand the app:

Run It: Use streamlit run fraud_detection_poc.py. Populate sample data via the button, set filters, and click "Run Advanced Fraud Analysis".
Workflow:

Fetches claims from Neo4j.
Applies filters (e.g., Workers Comp > $10,000).
Runs detection: Similarity pairs (cosine scores > threshold), communities (groups > size), anomalies (scores > min).
Visualizes suspicious graph.
GPT-4 summarizes (e.g., "Fraud ring detected: 3 claims linked by Dr. Adams and 123 Main St").


Insights:

Fraud Patterns: Identifies rings like repeated claimants/providers (e.g., John Doe and Dr. Adams across claims, indicating collusion).
Risk Prioritization: Anomaly scores highlight high-value/repeated claims for investigation.
Business Value: Potential cost savings (e.g., $45,000 in sample ring), reduced losses from staged claims.
Actionable: GPT recommendations (e.g., "Verify Dr. Adams' credentials") guide investigators.
Overall, the app shows how graphs reveal hidden connections invisible in tables, enabling proactive fraud prevention.



Overall Use of the App
The app is a tool for insurers to detect and investigate fraud rings in claims data. It helps reduce financial losses by identifying suspicious patterns early, 
prioritizing high-risk claims, and providing AI explanations for decision-makers. Ideal for compliance teams, investigators, or executives to demo fraud analytics, 
it can scale with real data for production use, integrating with insurance systems for real-time alerts.
Graph fraud detection using Neo4j
