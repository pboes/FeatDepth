from .mono_dataset import MonoDataset


class TrafficCamsDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(MonoDataset, self).__init__(*args, **kwargs)

    def get_color(self, filename, do_flip):
        color = self.loader(os.path.join(self.data_path, filename))

        if do_flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT)

        return color
