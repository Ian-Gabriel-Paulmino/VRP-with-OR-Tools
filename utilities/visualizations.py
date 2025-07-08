import folium
import colorsys
import os


def generate_folium_colors(n):
    """Generate `n` visually distinct HEX colors using HSV color space."""
    return [
        '#{:02x}{:02x}{:02x}'.format(*[
            int(c * 255) for c in colorsys.hsv_to_rgb(i / n, 0.75, 0.95)
        ])
        for i in range(n)
    ]


def visualize_solution(data, routes):
    node_ids = data['node_ids']
    positions = data['positions']
    depot = node_ids[data['depot']]

    depot_coords = positions[depot]
    m = folium.Map(location=depot_coords, zoom_start=15)

    # Color palette
    # colors = [
    # 'red', 'blue', 'green', 'orange', 'purple', 'pink', 'lightblue', 'lightgreen', 'beige', 'yellow',
    # 'lightred', 'cadetblue', 'cyan', 'lime', 'magenta', 'gold', 'aqua', 'lavender', 'coral', 'turquoise',
    # 'salmon', 'plum', 'khaki', 'tomato', 'deepskyblue', 'mediumseagreen', 'springgreen', 'dodgerblue', 'orchid', 'greenyellow',
    # 'lightcoral', 'mediumturquoise', 'peachpuff', 'skyblue', 'hotpink', 'wheat', 'chartreuse', 'powderblue', 'mediumorchid', 'darkorange',
    # 'lightpink', 'palegreen', 'lightsalmon', 'lightcyan', 'mediumvioletred', 'aquamarine', 'darkturquoise', 'moccasin', 'mistyrose', 'lemonchiffon'
    # ]

    # Dynamic color creation
    colors = generate_folium_colors(len(routes))


    # Mark depot
    folium.Marker(
        location=depot_coords,
        popup="Depot",
        icon=folium.Icon(color="black", icon="home")
    ).add_to(m)

    # Mark delivery points
    for idx, node in enumerate(node_ids):
        if node == depot:
            continue
        folium.Marker(
            location=positions[node],
            popup=f"Node {idx}",
            icon=folium.Icon(color="gray", icon="circle")
        ).add_to(m)         

    # Draw routes
    for v_idx, route in enumerate(routes):
        route_coords = [positions[node_ids[i]] for i in route]
        color = colors[v_idx]
        folium.PolyLine(
            locations=route_coords,
            color=color,
            weight=4,
            opacity=0.8,
            tooltip=f"Vehicle {v_idx}"
        ).add_to(m)


    # Save and open
    map_filename = "vrp_routes_map.html"
    m.save(map_filename)
    os.system(f"start {map_filename}")
