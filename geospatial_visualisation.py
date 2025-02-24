import folium
import networkx as nx
from folium.plugins import MarkerCluster

# Define pipeline stations (latitude, longitude)
pipeline_stations = {
    "Guwahati": (26.1445, 91.7362),
    "Dimapur": (25.9063, 93.7276),
    "Kohima": (25.6747, 94.1086),
    "Imphal": (24.8170, 93.9368)
}

# Create the pipeline network graph
pipeline_network = nx.Graph()

# Add nodes (stations)
for station, coords in pipeline_stations.items():
    pipeline_network.add_node(station, pos=coords)

# Add edges (pipelines) with flow rates & pressure values
pipeline_edges = [
    ("Guwahati", "Dimapur", 100, 50, False),  # (Station1, Station2, Flow Rate, Pressure, Leak?)
    ("Dimapur", "Kohima", 90, 45, False),
    ("Kohima", "Imphal", 85, 42, True)  # Leak at this pipeline
]

for u, v, flow_rate, pressure, leak in pipeline_edges:
    pipeline_network.add_edge(u, v, flow_rate=flow_rate, pressure=pressure, leak=leak)

# Function to determine pipeline color
def get_pipeline_color(pressure, leak):
    if leak:
        return "red"  # Leak detected
    elif pressure < 30:
        return "orange"  # Low pressure
    else:
        return "green"  # Normal flow

# Create a map centered in North East India
map_center = (25.5, 93.0)  # Adjusted for better focus on NE region
pipeline_map = folium.Map(location=map_center, zoom_start=7, tiles="CartoDB positron")

# Add pipeline routes (edges)
for u, v in pipeline_network.edges():
    station1 = pipeline_stations[u]
    station2 = pipeline_stations[v]
    
    # Get pipeline properties
    edge_data = pipeline_network[u][v]
    flow_rate = edge_data["flow_rate"]
    pressure = edge_data["pressure"]
    leak = edge_data["leak"]
    
    # Determine pipeline color
    line_color = get_pipeline_color(pressure, leak)
    
    # Add the pipeline to the map
    folium.PolyLine(
        [station1, station2], color=line_color, weight=5, opacity=0.7,
        tooltip=f"Pipeline: {u} → {v}\nFlow: {flow_rate} | Pressure: {pressure}"
    ).add_to(pipeline_map)

# Add markers for each station
marker_cluster = MarkerCluster().add_to(pipeline_map)
for station, coords in pipeline_stations.items():
    folium.Marker(
        location=coords,
        popup=f"{station} Station",
        tooltip=station,
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(marker_cluster)

# Save & Show the map
pipeline_map.save("Geospatial_Pipeline_Visualization.html")
print("Pipeline visualization saved as Geospatial_Pipeline_Visualization.html")
