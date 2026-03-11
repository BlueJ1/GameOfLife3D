import matplotlib.pyplot as plt
import matplotlib.animation as animation
import re

def load_data(filename):
    """Parses the evolution.txt file and returns a list of generations."""
    generations = []
    current_gen = None

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('==='):
                if current_gen is not None:
                    generations.append(current_gen)
                current_gen = []
            else:
                match = re.search(r'\((\d+),\s*(\d+),\s*(\d+)\)', line)
                if match:
                    x, y, z = int(match.group(1)), int(match.group(2)), int(match.group(3))
                    current_gen.append((x, y, z))

    if current_gen is not None:
        generations.append(current_gen)

    return generations

def main():
    filename = 'evolution.txt'
    try:
        generations = load_data(filename)
    except FileNotFoundError:
        print(f"Error: '{filename}' not found. Make sure you run the C program first.")
        return

    if not generations:
        print("No data found in the file.")
        return

    # Find the min and max coordinates across all dimensions to center the plot
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')

    for gen in generations:
        for x, y, z in gen:
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            min_z = min(min_z, z)
            max_z = max(max_z, z)

    # Add a buffer so the cells don't touch the very edge of the box
    buffer = 3

    # Set up the matplotlib figure and 3D axis
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Set the limits to dynamically box in our cells
    ax.set_xlim(min_x - buffer, max_x + buffer)
    ax.set_ylim(min_y - buffer, max_y + buffer)
    ax.set_zlim(min_z - buffer, max_z + buffer)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    scatter = ax.scatter([], [], [], c='cyan', edgecolors='blue', marker='s', s=100, depthshade=True)
    title = ax.set_title('Generation 0')

    def update(frame):
        gen_data = generations[frame]

        if not gen_data:
            scatter._offsets3d = ([], [], [])
        else:
            xs = [p[0] for p in gen_data]
            ys = [p[1] for p in gen_data]
            zs = [p[2] for p in gen_data]
            scatter._offsets3d = (xs, ys, zs)

        title.set_text(f'Generation {frame} | Living Cells: {len(gen_data)}')
        return scatter, title

    ani = animation.FuncAnimation(fig, update, frames=len(generations), interval=200, blit=False)
    plt.show()

if __name__ == '__main__':
    main()
