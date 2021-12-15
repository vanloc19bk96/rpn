from base.base_data_loader import BaseDataLoader
from data_loaders.datasets.simple_dataset import SimpleDataset


class SimpleDataLoader(BaseDataLoader):
    def __init__(self, training_data_dir, testing_data_dir, image_width, image_height, scales, ratios, batch_size,
                 shuffle=True, validation_split=0.0, num_workers=1, generate_input_model=None):
        self.training_data_dir = training_data_dir
        # transformation = TrafficLightTransform()
        dataset = SimpleDataset(training_data_dir, testing_data_dir, image_width, image_height, scales, ratios,
                                None, generate_input_model)
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers)
