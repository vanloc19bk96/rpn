from base.base_data_loader import BaseDataLoader
from data_loaders.datasets.traffic_light_dataset import TrafficLightDataset
from data_loaders.transform.traffic_light_transform import TrafficLightTransform


class TrafficLightDataLoader(BaseDataLoader):
    def __init__(self, data_dir, image_width, image_height, scales, ratios, batch_size,
                 shuffle=True, validation_split=0.0, num_workers=1, generate_input_model=None):
        self.data_dir = data_dir
        # transformation = TrafficLightTransform()
        dataset = TrafficLightDataset(data_dir, image_width, image_height, scales, ratios,
                                      None, generate_input_model)
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers)
