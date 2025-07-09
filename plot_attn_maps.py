from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import pickle
from pathlib import Path
import yaml
import plotly.graph_objects as go
import numpy as np
import json
import base64
from PIL import Image
import io
import base64
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
import numpy as np


def plot_map(ascii_map, agents, targets):
    grid = np.array([[1 if c == '#' else 0 for c in row] for row in ascii_map])
    height, width = grid.shape

    fig = go.Figure()

    # 1. Draw grid lines using shapes (under everything)
    for y in range(height):
        for x in range(width):
            fig.add_shape(
                type="rect",
                x0=x - 0.5, x1=x + 0.5,
                y0=y - 0.5, y1=y + 0.5,
                line=dict(color="lightgray", width=1),
                fillcolor="white",
                layer="below"
            )

    # 2. Draw black squares as Scatter markers (smaller size)
    dark_x, dark_y = [], []
    for y in range(height):
        for x in range(width):
            if grid[y][x] == 1:
                dark_x.append(x)
                dark_y.append(y)

    fig.add_trace(go.Scatter(
        x=dark_x,
        y=dark_y,
        mode="markers",
        marker=dict(symbol="square", size=35, color="black"),
        hoverinfo="skip",
        showlegend=False
    ))

    # 3. Plot agents and targets
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
    for i, (agent, target) in enumerate(zip(agents, targets)):
        color = colors[i % len(colors)]
        ay, ax = agent
        ty, tx = target

        fig.add_trace(go.Scatter(
            x=[ax], y=[ay],
            mode="markers",
            marker=dict(size=20, color=color, symbol="circle"),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=[tx], y=[ty],
            mode="markers",
            marker=dict(size=20, color=color, symbol="circle-open", line=dict(width=3)),
            showlegend=False
        ))

    # 4. Layout
    fig.update_layout(
        xaxis=dict(
            showgrid=False, zeroline=False,
            range=[-0.5, width - 0.5], showticklabels=False
        ),
        yaxis=dict(
            showgrid=False, zeroline=False,
            range=[-0.5, height - 0.5],
            showticklabels=False,
            scaleanchor="x", scaleratio=1,
            autorange="reversed"
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
        width=500,
        height=500,
    )

    return fig


def visualize_steps_combined(
        image_gen_fn,
        maze_map,
        agents,
        targets,
        heatmaps: np.ndarray,
        colorscale='Viridis',
        x_tick_labels=None,
        y_tick_labels=None
):
    steps, num_agents, heads, height, width = heatmaps.shape
    width = max(10, width)

    if x_tick_labels is None:
        x_tick_labels = [f'x{idx}' for idx in range(width)]
    if y_tick_labels is None:
        y_tick_labels = [f'y{idx}' for idx in range(height)]

    aspect_ratio = height / width
    map_row_height = 0.4  # 40% of vertical space for map
    heatmap_row_height = (0.6 / num_agents) * aspect_ratio
    # map_row_height = 50
    # heatmap_row_height = 50

    fig = make_subplots(
        rows=num_agents + 1,  # +1 for map row
        cols=heads,
        specs=[[{"type": "xy", "colspan": heads}] + [None] * (heads - 1)] +
              [[{"type": "heatmap"} for _ in range(heads)] for _ in range(num_agents)],
        row_heights=[map_row_height] + [heatmap_row_height] * num_agents,
        subplot_titles=(
                ['Maze Map'] +
                [f'Agent {a} Head {h}' for a in range(num_agents) for h in range(heads)]
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )

    step_images = []
    for step in range(steps):
        fig_step = image_gen_fn(maze_map, agents[step], targets[step])
        img_bytes = pio.to_image(fig_step, format='png')
        encoded_image = base64.b64encode(img_bytes).decode('utf-8')
        step_images.append(encoded_image)

    base_image_config = dict(
        xref="x1",
        yref="y1",
        x=0,
        y=height,
        sizex=width,
        sizey=height,
        sizing="contain",
        layer="above",
        opacity=1
    )

    zmin, zmax = np.min(heatmaps), np.max(heatmaps)
    for agent_idx in range(num_agents):
        for head_idx in range(heads):
            fig.add_trace(
                go.Heatmap(
                    z=heatmaps[0, agent_idx, head_idx],
                    colorscale=colorscale,
                    showscale=False,
                    x=x_tick_labels,
                    y=y_tick_labels,
                    zmin=zmin,
                    zmax=zmax
                ),
                row=agent_idx + 2,
                col=head_idx + 1
            )

    fig.add_layout_image(dict(
        source=f'data:image/png;base64,{step_images[0]}',
        **base_image_config
    ))

    # Configure axes - critical for proper image display
    fig.update_xaxes(
        range=[0, width],
        showticklabels=False,
        visible=False,
        row=1,
        col=1,
        constrain="domain"
    )
    fig.update_yaxes(
        range=[0, height],
        showticklabels=False,
        visible=False,
        row=1,
        col=1,
        scaleanchor='x1',
        scaleratio=1
    )

    for agent_idx in range(num_agents):
        for head_idx in range(heads):
            fig.update_xaxes(
                range=[0, width],
                row=agent_idx + 2,
                col=head_idx + 1,
                scaleanchor=f'y{(agent_idx * heads) + head_idx + 1}',
                scaleratio=1,
                constrain="domain",

            )
            fig.update_yaxes(
                range=[0, height],
                row=agent_idx + 2,
                col=head_idx + 1,
                # scaleanchor=f'x{(agent_idx*heads)+head_idx+1}',
                scaleratio=1
            )

    frames = []
    for step in range(steps):

        data = []
        for agent_idx in range(num_agents):
            for head_idx in range(heads):
                data.append(go.Heatmap(
                    z=heatmaps[step, agent_idx, head_idx],
                    colorscale=colorscale,
                    showscale=False,
                    x=x_tick_labels,
                    y=y_tick_labels,
                    zmin=zmin,
                    zmax=zmax
                ))

        layout_update = dict(
            images=[dict(
                source=f'data:image/png;base64,{step_images[step]}',
                **base_image_config
            )]
        )

        frames.append(
            go.Frame(
                data=data,
                layout=layout_update,
                name=f"step_{step}"
            )
        )

    fig.frames = frames

    slider_steps = []
    for step in range(steps):
        slider_steps.append(dict(
            method="animate",
            args=[
                [f"step_{step}"],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0}
                }
            ],
            label=f"Step {step + 3}"
        ))

    fig.update_layout(
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "Step: "},
            # pad={"t": 100},
            steps=slider_steps
        )],
        updatemenus=[],  # Remove play/pause buttons
        width=1200,
        height=1000,
        title="Multi-agent Attention Visualization",
        margin=dict(t=50, b=50, l=50, r=50),
        autosize=False,
        template='plotly_white',
    )

    return fig


if __name__ == '__main__':

    with open('env/test-bottlenecks-9-31000.yaml', 'r') as file:
        maze_dict = yaml.safe_load(file)

    img_path = Path('plots')
    output_path = img_path / 'output_data'
    output_path.mkdir(parents=True, exist_ok=True)

    input_path = img_path / 'input_data'

    maps = [
        'bottlenecks9-v_corr-3_len',
        'bottlenecks9-v_corr-9_len',
        'bottlenecks9-v_corr-15_len',
    ]
    mem = False

    for map in maps:
        maze_map = maze_dict[map].split('\n')

        with open(input_path / f'pogema-ep00001-{map}.json', 'r') as file:
            pos = json.load(file)

        _npy = np.load(input_path / f'episode_log_{map}__seed_5.npy', allow_pickle=True)

        start = np.array(([pos[0][0]['x'], pos[0][0]['y'], ], [pos[1][0]['x'], pos[1][0]['y'], ]))
        agents = start + [(_npy[step]['obs'][0]['xy'], _npy[step]['obs'][1]['xy']) for step in range(2, len(_npy))]
        targets = start + [(_npy[step]['obs'][0]['target_xy'], _npy[step]['obs'][1]['target_xy']) for step in
                           range(2, len(_npy))]
        try:
            attn_type = 0
            data = np.array([_npy[idx]['attentions'][attn_type] for idx in range(2, len(_npy))])

            y_labels = ['mem'] + [f'obs_{idx}' for idx in range(8, -1, -1)]
            x_labels = y_labels

            fig = visualize_steps_combined(
                image_gen_fn=plot_map,
                maze_map=maze_map,
                agents=agents,
                targets=targets,
                heatmaps=data,  # numpy array: (steps, agents, 4, H, W),
                colorscale='Blues',
                x_tick_labels=x_labels,
                y_tick_labels=y_labels,
            )
            maze_output_path = output_path / map
            maze_output_path.mkdir(parents=True, exist_ok=True)

            fig.write_html(maze_output_path / 'attn.html', include_plotlyjs='cdn')
        except Exception as e:
            print('no attention')
            print(f'Error: {e}')

        try:
            attn_type = 1
            data = np.array([_npy[idx]['attentions'][attn_type] for idx in range(2, len(_npy))])

            y_labels = ['mem'] + [f'obs_{idx}' for idx in range(8, -1, -1)]
            x_labels = ['agent_1', 'agent_2']

            fig = visualize_steps_combined(
                image_gen_fn=plot_map,
                maze_map=maze_map,
                agents=agents,
                targets=targets,
                heatmaps=data,  # numpy array: (steps, agents, 4, H, W),
                colorscale='Blues',
                x_tick_labels=x_labels,
                y_tick_labels=y_labels,
            )
            maze_output_path = output_path / map
            maze_output_path.mkdir(parents=True, exist_ok=True)

            fig.write_html(maze_output_path / 'attn_cross.html', include_plotlyjs='cdn')
        except Exception as e:
            print('no cross attention')
            print(f'Error: {e}')