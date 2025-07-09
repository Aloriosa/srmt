import argparse
import base64
import json
import re
from pathlib import Path
import io

import imageio
import numpy as np
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots
from PIL import Image


class VideoGenerator:
    """
    A class to generate videos from multi-agent attention visualizations.

    This class automatically finds and parses maze and agent data files,
    generates a Plotly figure for each step of the simulation, and compiles
    these frames into a video file.
    """

    def __init__(self, input_dir: Path, output_dir: Path, maze_file: Path):
        """
        Initializes the VideoGenerator.

        Args:
            input_dir (Path): Directory containing .npy and .json data files.
            output_dir (Path): Directory where the output videos will be saved.
            maze_file (Path): Path to the YAML file containing maze definitions.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.maze_file = Path(maze_file)

        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        if not self.maze_file.is_file():
            raise FileNotFoundError(f"Maze file not found: {self.maze_file}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(self.maze_file, 'r') as f:
            self.maze_definitions = yaml.safe_load(f)

    def _find_data_files(self) -> dict:
        """
        Finds all matching JSON and NPY files in the input directory.

        Returns:
            dict: A dictionary mapping map names to their file paths.
                  e.g., {'map_name': {'json': Path(...), 'npy': Path(...)}}
        """
        found_files = {}
        json_files = list(self.input_dir.glob('pogema-ep*.json'))

        for json_path in json_files:
            match = re.search(r'pogema-ep\d+-(.+)\.json', json_path.name)
            if not match:
                continue

            map_name = match.group(1)
            # Find the corresponding npy file, allowing for any seed.
            npy_candidates = list(self.input_dir.glob(f'episode_log_{map_name}*.npy'))

            if npy_candidates and map_name in self.maze_definitions:
                found_files[map_name] = {
                    'json': json_path,
                    'npy':  npy_candidates[0]  # Use the first match
                }
            else:
                print(f"Warning: Skipping map '{map_name}'. Corresponding .npy file or maze definition not found.")

        return found_files

    @staticmethod
    def _parse_data(json_path: Path, npy_path: Path) -> dict:
        """
        Parses the data from the input files.

        Args:
            json_path (Path): Path to the JSON file with agent positions.
            npy_path (Path): Path to the NPY file with episode logs.

        Returns:
            A dictionary containing parsed 'agents', 'targets', and 'attentions'.
        """
        with open(json_path, 'r') as f:
            pos = json.load(f)

        episode_log = np.load(npy_path, allow_pickle=True)

        # The first two steps might not have valid data, so start from step 2
        start_step = 2
        num_steps = len(episode_log) - start_step
        if num_steps <= 0:
            return {'agents': np.array([]), 'targets': np.array([]), 'attentions': {}}

        start = np.array(([pos[0][0]['x'], pos[0][0]['y'], ], [pos[1][0]['x'], pos[1][0]['y'], ]))
        agents = start + [(episode_log[step]['obs'][0]['xy'], episode_log[step]['obs'][1]['xy']) for step in range(2, len(episode_log))]
        targets = start + [(episode_log[step]['obs'][0]['target_xy'], episode_log[step]['obs'][1]['target_xy']) for step in
                           range(2, len(episode_log))]
        parsed = {'agents': agents, 'targets': targets, 'attentions': {}}

        # Safely extract attention data
        if start_step < len(episode_log) and 'attentions' in episode_log[start_step]:
            num_attention_types = len(episode_log[start_step]['attentions'])
            for attn_type in range(num_attention_types):
                try:
                    data = np.array(
                        [episode_log[idx]['attentions'][attn_type] for idx in range(start_step, len(episode_log))])
                    parsed['attentions'][attn_type] = data
                except (IndexError, KeyError):
                    print(f"Warning: Could not process attention type {attn_type}.")
        return parsed

    @staticmethod
    def _plot_maze_map(ascii_map, agents, targets) -> go.Figure:
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

    def _generate_single_frame(self, maze_map, agents_step, targets_step, heatmaps_step, x_labels, y_labels, zmin,
                               zmax) -> bytes:
        """
        Generates a single frame by combining maze visualization and heatmaps.
        """
        num_agents, heads, height, width = heatmaps_step.shape

        # 1. Generate maze image
        maze_fig = self._plot_maze_map(maze_map, agents_step, targets_step)
        maze_img_bytes = maze_fig.to_image(format="png", width=1000)
        maze_img = Image.open(io.BytesIO(maze_img_bytes))

        # 2. Generate heatmap grid
        heatmap_fig = make_subplots(
            rows=num_agents,
            cols=heads,
            subplot_titles=[f'Agent {a} Head {h}' for a in range(num_agents) for h in range(heads)]
        )

        for agent_idx in range(num_agents):
            for head_idx in range(heads):
                heatmap_fig.add_trace(
                    go.Heatmap(
                        z=heatmaps_step[agent_idx, head_idx],
                        colorscale='Blues',
                        showscale=False,
                        x=x_labels,
                        y=y_labels,
                        zmin=zmin,
                        zmax=zmax
                    ),
                    row=agent_idx + 1,
                    col=head_idx + 1
                )

        heatmap_fig.update_layout(
            height=800,
            width=1000,
            margin=dict(t=50, b=50, l=50, r=50),
            showlegend=False,
            template='plotly_white'
        )
        heatmap_img_bytes = heatmap_fig.to_image(format="png")
        heatmap_img = Image.open(io.BytesIO(heatmap_img_bytes))

        # 3. Combine images vertically
        frame_width = max(maze_img.width, heatmap_img.width)
        frame_height = maze_img.height + heatmap_img.height
        frame_img = Image.new('RGB', (frame_width, frame_height), (255, 255, 255))

        # Center-align both images horizontally
        maze_x = (frame_width - maze_img.width) // 2
        heatmap_x = (frame_width - heatmap_img.width) // 2

        frame_img.paste(maze_img, (maze_x, 0))
        frame_img.paste(heatmap_img, (heatmap_x, maze_img.height))

        # Convert to bytes
        img_bytes = io.BytesIO()
        frame_img.save(img_bytes, format='PNG')
        return img_bytes.getvalue()

    def create_and_save_video(self, map_name: str, data: dict, frame_duration: float):
        """Creates and saves videos for all available attention types for a given map."""
        maze_map = self.maze_definitions[map_name].split('\n')
        agents = data['agents']
        targets = data['targets']
        attentions = data['attentions']
        fps = 1.0 / frame_duration

        for attn_type, heatmaps in attentions.items():
            print(f"--- Generating video for map: '{map_name}', Attention Type: {attn_type} ---")

            # Calculate global min/max for the entire timeseries
            zmin, zmax = np.min(heatmaps), np.max(heatmaps)

            # Define labels based on attention type and heatmap shape
            steps, _, _, h, w = heatmaps.shape
            if attn_type == 0:  # Self-attention
                x_labels = y_labels = ['mem'] + [f'obs_{i}' for i in range(h - 2, -1, -1)]
            elif attn_type == 1:  # Cross-attention
                y_labels = ['mem'] + [f'obs_{i}' for i in range(h - 2, -1, -1)]
                x_labels = [f'agent_{i}' for i in range(w)]
            else:
                y_labels = [f'y_{i}' for i in range(h)]
                x_labels = [f'x_{i}' for i in range(w)]

            video_filename = self.output_dir / f"{map_name}_attn_type_{attn_type}.mp4"
            with imageio.get_writer(video_filename, fps=fps, quality=8, codec='libx264') as writer:
                for step in range(steps):
                    print(f"\rProcessing frame {step + 1}/{steps}...", end="")
                    # Pass global zmin and zmax to the frame generator
                    frame_img_bytes = self._generate_single_frame(
                        maze_map, agents[step], targets[step], heatmaps[step], x_labels, y_labels, zmin, zmax
                    )
                    writer.append_data(imageio.imread(frame_img_bytes))
            print(f"\n✅ Video saved to: {video_filename}")

    def run(self, frame_duration: float = 0.5):
        """
        Main execution method to find all data files and generate videos.

        Args:
            frame_duration (float): Time in seconds to display each frame.
        """
        print(f"🔍 Searching for data in: {self.input_dir}")
        data_files = self._find_data_files()

        if not data_files:
            print("No valid data files found. Exiting.")
            return

        for map_name, paths in data_files.items():
            print(f"\nProcessing map: {map_name}")
            try:
                parsed_data = self._parse_data(paths['json'], paths['npy'])
                if not parsed_data.get('attentions'):
                    print(f"Warning: No attention data found for map '{map_name}'. Skipping.")
                    continue
                self.create_and_save_video(map_name, parsed_data, frame_duration)
            except Exception as e:
                print(f"❌ An error occurred while processing {map_name}: {e}")
                import traceback
                traceback.print_exc()


def main():
    """Main function to parse arguments and run the video generator."""
    parser = argparse.ArgumentParser(
        description="Generate videos of multi-agent attention from simulation data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help="Directory containing the input .npy and .json files."
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help="Directory to save the generated video files."
    )
    parser.add_argument(
        '--maze_file',
        type=str,
        required=True,
        help="Path to the YAML file containing maze definitions."
    )
    parser.add_argument(
        '--frame_duration',
        type=float,
        default=0.5,
        help="Duration (in seconds) for each frame in the video."
    )

    args = parser.parse_args()

    generator = VideoGenerator(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        maze_file=args.maze_file
    )
    generator.run(frame_duration=args.frame_duration)


if __name__ == '__main__':
    main()