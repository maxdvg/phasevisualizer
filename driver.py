from config import Config
from colorgen import ColorPallateSampler
import yaml
import cv2
import numpy as np
from colorgen import Palette


if __name__ == "__main__":
    # Load in the config file
    with open("config.yaml", "r") as f:
        config_data = yaml.safe_load(f)
    config = Config.model_validate(config_data)

    # Initialize ColorPaletteSampler
    palette_sampler = ColorPallateSampler(config.audio_input.a_freq, Palette(), color_snapping=True)

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(config.video_properties.output_filename,
                                    fourcc,
                                    config.video_properties.framerate,
                                    (config.video_properties.resolution_width, config.video_properties.resolution_height))
    
    def generate_image_data():
            return np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
    
    for _ in range(100):  # Example: 100 frames
        image_data = generate_image_data()
        video_writer.write(image_data)
    
    video_writer.release()
    print("FUCK")